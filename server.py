from DB_handler import DBModule
from flask import Flask, render_template, request, url_for, redirect, flash, session
import os
import time


app = Flask(__name__)
app.secret_key = "wjddusdlek@!!@wjddusdlek!!"

DB = DBModule()

## 승일
def solution2(img_src):
    return 'True', 0.74


@app.route("/")
def index(): 
    if "uid" in session:
        user = session["uid"]
    else:
        user = "Login"   
    return render_template("index.html", user = user)


@app.route('/upload')
def upload():
    if "uid" in session:
        return render_template("upload.html")
    else:
        return redirect(url_for("login"))


app.config['UPLOAD_FOLDER'] = 'static/uploads'
@app.route("/upload_done", methods=["POST"])
def upload_done():
    file = request.files['file']
    uid = session.get("uid")
    
    filename = uid + "_" + file.filename
        
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  # 로컬 저장
    
    path_local = url_for('static', filename = 'uploads/' + filename)
    
    start_time = time.time()
    label, prob = solution2(path_local)
    end_time = time.time()
    app.logger.info(end_time-start_time)
    
    if DB.upload(uid, file.filename, path_local, label, prob):
        return render_template('upload.html', filename=path_local, label=label, probability=prob)
    
    else:
        flash("INVALID")
        return render_template('upload.html', filename=None, label=None, probability=None)


@app.route("/upload_list")
def upload_list():
    if "uid" in session:
        uid = session.get("uid")
        upload_list = DB.upload_list(uid)
        
        if upload_list == None:
            length = 0
            items = None
        else:
            length = len(upload_list)
            items = upload_list.items()
        return render_template("upload_list.html", upload_list=items, length=length)
    else:
        return redirect(url_for("login"))


@app.route("/post/<string:fid>")
def post(fid):
    if "uid" in session:
        uid = session.get("uid")
        post, path_local = DB.upload_detail(uid, fid)
        return render_template("upload_detail.html", post=post, filename=path_local)
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
        flash("SUCCESS")
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