from DB_handler import DBModule
from model import Model
from flask import Flask, render_template, request, url_for, redirect, flash, session
import os
import time
from PIL import Image, ImageOps
import cv2

app = Flask(__name__)
app.secret_key = "wjddusdlek@!!@wjddusdlek!!"

DB = DBModule()
M = Model()


def model_prediction(img_src):
    M.test_model(img_src)
    return M.labels, M.label, M.prob


@app.route("/")
def index(): 
    if "uid" in session:
        user = session["uid"]
    else:
        user = "Login"   
    return render_template("landingpage.html", user=user)


@app.route('/demo')
def demo():
    if "uid" in session:
        user, items, length = upload_list()
        return render_template("demo.html", user=user, upload_list=items, length=length)
    else:
        user = "Login"
        return redirect(url_for("login"))


app.config['UPLOAD_FOLDER'] = 'static/uploads'
@app.route("/upload_done", methods=["POST"])
def upload_done():
    file = request.files['file']
    uid = session.get("uid")
    
    name = file.filename
    filename = uid + "_" + file.filename
    
    # file = Image.open(file)
    # file.thumbnail((400, 400))
    
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # 로컬 저장
    
    path_local = url_for('static', filename = 'uploads/' + filename)
    
    start_time = time.time()
    model_prediction('./' + path_local)
    end_time = time.time()
    app.logger.info(end_time-start_time)
    
    if M.labels != None:
        flash("label 여러개")
        user, items, length = upload_list()
        return render_template('demo.html', user=user, upload_list=items, length=length, filename=path_local, labels=M.labels, label=None, probability=None)
        
    elif DB.upload(uid, name, path_local, M.labels, M.label, M.prob):
        user, items, length = upload_list()
        return render_template('demo.html', user=user, upload_list=items, length=length, filename=path_local, labels=None, label=M.label, probability=M.prob)
    
    else:
        user, items, length = upload_list()
        flash("이미 있는 파일 혹은 10개 이상")
        return render_template('demo.html', user=user, upload_list=items, length=length, filename=path_local, labels=None, label=None, probability=None)        


@app.route("/upload_list", methods = {"GET"})
def upload_list():
    if "uid" in session:
        user = session.get("uid")
        ckb = request.args.get("menuicon")
        upload_list = DB.upload_list(user)
        
        if upload_list == None:
            length = 0
            items = None
        else:
            length = len(upload_list)
            items = upload_list.items()
        return user, items, length
    else:
        return redirect(url_for("login"))


@app.route("/post/<string:fid>")
def post(fid):
    if "uid" in session:
        uid = session.get("uid")
        path_local, label, prob = DB.upload_detail(uid, fid)
        user, items, length = upload_list()
        return render_template("demo.html", user=user, upload_list=items, length=length, filename=path_local, label=label, probability=prob)
    else:
        return redirect(url_for("login"))


@app.route("/logout") 
def logout(): 
    if "uid" in session:
        session.pop("uid")
        return redirect(url_for("index"))
    else:
        return redirect(url_for("login"))


@app.route("/login") 
def login(): 
    if "uid" in session:
        return redirect(url_for("index"))
    return render_template("login.html")


@app.route("/login_done", methods = {"GET"}) 
def login_done(): 
    if "uid" in session:
        return redirect(url_for("index"))
    uid = request.args.get("uid")
    pwd = request.args.get("pwd")
    print(uid, pwd)

    if DB.login(uid, pwd): 
        session["uid"] = uid
        return redirect(url_for("index"))
    else:
        flash("INVALID")
        return redirect(url_for("login"))


@app.route("/signin")
def signin():
    return render_template('signin.html')


@app.route("/signin_done", methods=["get"])
def signin_done():
    email = request.args.get("email")
    uid = request.args.get("id")
    pwd = request.args.get("pwd")
    name = request.args.get("name")
    if DB.signin(_id_=uid, pwd=pwd, name=name, email=email):
        return redirect(url_for("index"))
    else:
        flash("INVALID")
        return redirect(url_for("signin"))


@app.route("/user/<uid>")
def user(uid):
    pass


if __name__=='__main__':
    app.run(debug=True)