import pyrebase
import json
import uuid

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
    
    
    def write_post(self, title, contents, uid):
        pid = str(uuid.uuid4())[:10]
        print(pid)
        information = {
            "title": title,
            "contents": contents,
            "uid": uid
        }
        self.db.child("posts").child(pid).set(information)
        
    
    def post_list(self):
        post_lists = self.db.child("posts").get().val()
        return post_lists
    
    
    def post_detail(self, pid):
        post = self.db.child("posts").get().val()[pid]
        return post
    
    
    def upload(self, uid, filename, path_local, label, prob):
        path_on_cloud = "images/" + filename
        pid = str(uuid.uuid4())[:10]
        print(pid)
        information = {
            "path_on_cloud": path_on_cloud,
            "label": label,
            "probability": prob
        }
        self.db.child("uploads").child(uid).child(pid).set(information)
        self.storage.child(path_on_cloud).put("./" + path_local)
    
    
    def upload_list():
        pass
    
    
    def upload_detail():
        pass
    
    
    def get_user(self, uid):
        pass