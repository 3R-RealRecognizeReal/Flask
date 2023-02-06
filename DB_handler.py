import pyrebase
import json

class DBModule():
    def __init__(self):
        with open("./auth/firebaseAuth.json") as f:
            config = json.load(f)
        
        firebase = pyrebase.initialize_app(config)
        self.db = firebase.database()
        self.storage = firebase.storage()
        
        
    def login(self, uid, pwd):
        users = self.db.child("users").get().val()
        try:
            userinfo = users[uid]
            if userinfo["pwd"] == pwd:
                print("가입되어 있습니다.")
                return True
            else:
                print("비밀번호를 잘못 입력했습니다.")
                return False
        except:
            print("등록되지 않은 아이디입니다.")
            return False	
        
    
    def signin_verification(self, uid):
        users = self.db.child("users").get().val()
        for i in users:
            if uid == i:
                return False
        return True	
    
    
    def signin(self, _id_, pwd, name, email):
        # print(_id_, pwd, name, email)
        information = {
        	"pwd": pwd,
			"name": name,
			"email": email
        }
        if self.signin_verification(_id_):
            self.db.child("users").child(_id_).set(information)
            return True
        else:
            return False
    
    
    def check_upload(self, uid, filename):
        upload_lists = self.db.child("uploads").child(uid).get().val()
        if upload_lists == None:
            return True
        else:
            newfile = filename.split('/')[-1].split('.')[0]
            # 파일명 중복
            for file in upload_lists:
                if file == newfile:
                    return False
            # 한 id당 10회
            if len(upload_lists) > 10:
                return False
            return True
    
    
    def upload(self, uid, filename, path_local, label, prob):
        path_on_cloud = "images/" + uid + '/' + filename
        if self.check_upload(uid, path_on_cloud):
            information = {
                "path_on_cloud": path_on_cloud,
                "label": label,
                "probability": prob
            }
            self.db.child("uploads").child(uid).child(filename.split('.')[0]).set(information)
            self.storage.child(path_on_cloud).put("./" + path_local)
            return True
        else:
            False
    
    
    def upload_list(self, uid):
        upload_lists = self.db.child("uploads").child(uid).get().val()
        print(upload_lists)
        return upload_lists
    
    
    def upload_detail(self, uid, pid):
        post = self.db.child("uploads").child(uid).get().val()[pid]
        path_local = 'static/downloads/' + post['path_on_cloud'].split('/', maxsplit=1).pop().replace('/', '_')
        self.storage.child(post['path_on_cloud']).download("", path_local)
        return post, path_local
    
    
    def get_user(self, uid):
        pass