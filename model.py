import os
import copy
import random
import cv2
import torch
import numpy as np
from PIL import Image
import pandas as pd
import collections
import torch
from torch import nn
from torchvision import transforms, models
from src.transform import SSDTransformer
from src.utils import generate_dboxes, Encoder, colors, coco_classes
from src.model import SSD, ResNet
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
        self.model_RF = models.resnet50(pretrained=True) # vgg19 pretrained 모델 사용

        # face dataset에 맞게 모델 아키텍처 수정
        # output 사이즈를 지정하여 연산을 수행할 수 있음
        self.class_list = ['real', 'fake']
        self.model_RF.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.model_RF.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), # fully-connected
            nn.ReLU(), # activation function
            nn.Dropout(0.1), # 정규화
            nn.Linear(256, len(self.class_list)), # real, fake 두 개 클래스
            nn.Sigmoid() # 시그모이 함수로 확률 출력
        )

        ckpt_RF = torch.load("./model/resnet50_model_34.pth", map_location=torch.device('cpu'))

        self.model_RF = self.build_resnet_based_model(device_name="cpu")
        self.model_RF.load_state_dict(ckpt_RF)
        self.model_RF.eval()
        
        self.model_Obj = "ssd"
        self.model_Obj = SSD(backbone=ResNet()) # backbone 모델 mobileNet으로 변경 가능
        ckpt_Obj = torch.load("./model/SSD.pth", map_location=torch.device('cpu')) # 모델 불러오기
        self.model_Obj.load_state_dict(ckpt_Obj["model_state_dict"])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            self.model_Obj.cuda()
        else:
            self.model_Obj.to(self.device) # cpu
        self.model_Obj.eval()
        self.dboxes = generate_dboxes() # default bounding box의 크기를 정하고 생성 함수(utils.py)
        self.transformer = SSDTransformer(self.dboxes, (300, 300), val=True) # 데이터에 적용할 transform 


    # pretrained vgg19 모델에 avgpool과 classifier를 추가하여 새로운 model 생성
    def build_resnet_based_model(self, device_name='cuda'): 
        # device = torch.device(device_name)
        # self.model_RF = models.vgg19(pretrained=True)
        # self.model_RF.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        # self.model_RF.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, len(self.class_list)),
        #     nn.Softmax(dim=1)
        device = torch.device(device_name)
        self.model_RF = models.resnet50(pretrained=True)
        self.model_RF.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        fc_inputs = self.model_RF.fc.in_features
        self.model_RF.fc = nn.Sequential(
            nn.Linear(fc_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(self.class_list)), 
            nn.Softmax(dim=1) # For using NLLLoss()
        )
        return self.model_RF.to(device)


    def set_image(self, dir): 
        self.img_RF = cv2.imread(dir) # opencv로 이미지를 읽어옴(bgr 형식)
        self.img_RF = cv2.cvtColor(self.img_RF, cv2.COLOR_BGR2RGB) # bgr 형식을 rgb 형식으로 변환
        self.img_Obj = Image.open(dir).convert("RGB")


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
        self.labels = None
        self.label = None
        self.prob = None
        
        self.set_image(image_dir)
        
        self.labels = self.test_model_Obj(image_dir)
        if len(self.labels) == 1 and self.labels.get("person") == 1:
            self.label, self.prob = self.test_model_RF()
            self.labels = None
        else:
            self.label = None
            self.prob = None
            
        print(self.labels)
        print(self.label)
        print(self.prob)
            
    
    def test_model_Obj(self, dir_path):
        self.img_Obj, _, _, _ = self.transformer(self.img_Obj, None, torch.zeros(1,4), torch.zeros(1))
        encoder = Encoder(self.dboxes)

        if torch.cuda.is_available():
            self.img_Obj = self.img_Obj.cuda()
        else:
            self.img_Obj = self.img_Obj.to(self.device)

        labels = []

        cls_threshold=0.3
        nms_threshold=0.5

        with torch.no_grad():
            ploc, plabel = self.model_Obj(self.img_Obj.unsqueeze(dim=0)) # predicted location, predicted label
            result = encoder.decode_batch(ploc, plabel, nms_threshold, 20)[0] # encoding
            loc, label, prob = [r.cpu().numpy() for r in result]
            best = np.argwhere(prob > cls_threshold).squeeze(axis=1) # best값
            loc = loc[best]
            label = label[best]
            prob = prob[best]
            output_img = cv2.imread(dir_path)
            if len(loc) > 0:
                height, width, _ = output_img.shape
                loc[:, 0::2] *= width
                loc[:, 1::2] *= height
                loc = loc.astype(np.int32)
                for box, lb, pr in zip(loc, label, prob):
                    category = coco_classes[lb]
                    color = colors[lb]
                    xmin, ymin, xmax, ymax = box
                    cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                    text_size = cv2.getTextSize(category + " : %.2f" % pr, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    cv2.rectangle(output_img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color,
                        -1)
                    cv2.putText(
                        output_img, category + " : %.2f" % pr,
                        (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 2)
            
                    labels.append(category)
                    
            labels = collections.Counter(labels)
            
            return labels
    

    def test_model_RF(self):
        tensor_image = self.preprocess_image(self.img_RF)

        with torch.no_grad():
            prediction = self.model_RF(tensor_image)

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
    
        prob = format(prob_reduce_list[0] * 100, '.1f')
        
        return label, prob