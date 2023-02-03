### firebase 사용 안 함
from flask import Flask, render_template, request, url_for
import os
import time

def solution2(img_src):
    return 'True', 0.74

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
@app.route('/templates')
def index():
    return render_template('idx.html')

@app.route('/upload', methods = ['POST'])
def upload():
    file = request.files['file']
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img_src = url_for('static', filename = 'uploads/' + filename)
    
    start_time = time.time()
    label, prob = solution2(img_src)
    end_time = time.time()
    app.logger.info(end_time-start_time)
    return render_template('idx.html', filename=img_src, label=label, probability=prob)

if __name__=='__main__':
    app.run(debug=True)
