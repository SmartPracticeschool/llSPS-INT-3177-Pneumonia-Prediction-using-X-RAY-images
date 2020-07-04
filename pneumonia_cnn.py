import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

global model
app = Flask(__name__)
model = load_model("pneumonia.h5")


@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (512,512))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        preds = model.predict_classes(x)
        print("prediction",preds)
        
        if (preds[0]==0):
            result = 'The patient is NORMAL'
        else:
            result = 'The patient has PNEUMONIA'
        text = "Analysis Result : " + result
        
    return text

if __name__ == '__main__':
    app.run(debug = True, threaded = False)
        
        
        
    
    
    