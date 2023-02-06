import os
import copy
import random
import cv2
import torch
import numpy as np
import pandas as pd
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from ipywidgets import interact
import warnings
warnings.filterwarnings('ignore')

random_seed = 2022

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Model():
    def __init__(self):
        self.model = models.vgg19(pretrained=True) # vgg19 pretrained 모델 사용

        # face dataset에 맞게 모델 아키텍처 수정
        # output 사이즈를 지정하여 연산을 수행할 수 있음
        self.class_list = ['real', 'fake']
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), # fully-connected
            nn.ReLU(), # activation function
            nn.Dropout(0.1), # 정규화
            nn.Linear(256, len(self.class_list)), # real, fake 두 개 클래스
            nn.Sigmoid() # 시그모이 함수로 확률 출력
        )

        ckpt = torch.load("./model/model_20.pth", map_location=torch.device('cpu'))

        self.model = self.build_vgg19_based_model(device_name="cpu")
        self.model.load_state_dict(ckpt)
        self.model.eval()

    # pretrained vgg19 모델에 avgpool과 classifier를 추가하여 새로운 model 생성
    def build_vgg19_based_model(self, device_name='cuda'): 
        device = torch.device(device_name)
        self.model = models.vgg19(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.class_list)),
            nn.Softmax(dim=1)
        )
        return self.model.to(device)


    def get_RGB_image(self, dir): 
        image = cv2.imread(dir) # opencv로 이미지를 읽어옴(bgr 형식)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # bgr 형식을 rgb 형식으로 변환
        return image


    def preprocess_image(self, image):
        ori_H, ori_W = image.shape[:2]
    
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
            ])
    
        tensor_image = transformer(image)
        tensor_image = tensor_image.unsqueeze(0)
        return tensor_image


    def test_model(self, image_dir):
        image = self.get_RGB_image(image_dir)
        tensor_image = self.preprocess_image(image)

        with torch.no_grad():
            prediction = self.model(tensor_image)

        _, pred_label = torch.max(prediction.detach(), dim=1)

        pred_label = pred_label.squeeze(0)

        # 이미지가 real일 확률
        prob_list = prediction.tolist()
        prob_reduce_list = np.array(prob_list).flatten().tolist()
    
        label = None
        if pred_label.item() == 0:
            label = 'REAL'
        if pred_label.item() == 1:
            label = 'FAKE'
    
        prob = format(prob_reduce_list[0] * 100, '.3f')

        return label, prob