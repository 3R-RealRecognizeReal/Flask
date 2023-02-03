from DB_handler import DBModule
from flask import Flask, render_template, request, url_for, redirect, flash, session
import os
import time


app = Flask(__name__)
app.secret_key = "wjddusdlek@!!@wjddusdlek!!"


DB = DBModule()

@app.route('/')
def index():
    if "uid" in session:
        user = session["uid"]
    else:
        user = "Login"
    return render_template('index.html', user=user)


## 승일
def solution2(img_src):
    return 'True', 0.74


app.config['UPLOAD_FOLDER'] = 'static/uploads'
@app.route('/upload', methods = ['POST'])
def upload():
    file = request.files['file']
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    path_local = url_for('static', filename = 'uploads/' + filename)
    path_on_cloud = "images/" + filename
    DB.storage.child(path_on_cloud).put("./" + path_local)
    
    start_time = time.time()
    label, prob = solution2(path_local)
    end_time = time.time()
    app.logger.info(end_time-start_time)
    return render_template('index.html', filename=path_local, label=label, probability=prob)


@app.route("/list")
def upload_list():
    pass


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


@app.route("/login_done", methods=["get"])
def login_done():
    uid = request.args.get("id")
    pwd = request.args.get("pwd")
    if DB.login(uid, pwd):
        session["id"] = uid
        return redirect(url_for("index"))
    else:
        flash("INVALID ID OR PASSWORD")
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
    if DB.singin(_id_=uid, pwd=pwd, name=name, email=email):
        return redirect(url_for("index"))
    else:
        flash("INVALID ID")
        return redirect(url_for("signin"))

@app.route("/user/<uid>")
def user(uid):
    pass


if __name__=='__main__':
    app.run(debug=True)