from flask import Flask,render_template,Response
import numpy as np
import cv2
from utils import Camera
import torch




app = Flask(__name__)

camera = Camera()


@app.route('/home',methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/streamCam',methods=['GET'])
def streamCam():
    return Response(camera.get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == "__main__":
    app.run(debug=True)
