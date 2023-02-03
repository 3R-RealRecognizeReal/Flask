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
                return True
            else:
                return False
        except:
            return False        
    
    
    def signin_verification(self, uid):
        users = self.db.child("users").get().val()
        for i in users:
            if uid == i:
                return False
        
        return True
    
    
    def singin(self, _id_, pwd, name, email):
        information = {
            "pwd": pwd,
            "uname": name,
            "email": email
        }
        if self.signin_verification(_id_):
            self.db.child("users").child(_id_).set(information)
            return True
        
        else:
            return False
    
    def upload(self, user, contents):
        pass
    
    
    def upload_list(self):
        pass
    
    
    def upload_detail(self, pid):
        pass
    
    
    def get_user(self, uid):
        pass