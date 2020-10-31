import cv2
import numpy as np
import base64
from pwcnn import PWCNN
import torch


class Camera:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        self.profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

        self.scaleFactor =  1.13

        self.frame  = None

        self.cnet = PWCNN()

        self.cnet.load_state_dict(torch.load('maskVSNoMask_checkpoints.pth',map_location=torch.device('cpu'))['cnn'])

    def get_frames(self):
        while True:
            self.success,self.frame = self.camera.read()

            self.frame = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)

            if self.success:
                faces = self.face_cascade.detectMultiScale(self.frame,scaleFactor=self.scaleFactor,minNeighbors=4,minSize=(30,30))

                real_face = None

                if len(faces) > 0:
                    real_face = faces
                    for (x,y,w,h) in real_face:
                        cv2.rectangle(self.frame,(x,y),(x+w,y+h),(163,243,0),3)
                        pred_img=self.frame[y:y+h,x:x+w]
                        pred_img=cv2.cvtColor(pred_img,cv2.COLOR_BGR2GRAY) / 255
                        pred_img = cv2.resize(pred_img,(50,50))
                        pix = torch.tensor(pred_img).float().view(1,1,50,50)
                        num_pred = torch.argmax(self.cnet(pix))
                        if num_pred == 1:
                            cv2.putText(self.frame,"MASK DETECTED",(x,y+h+50),cv2.FONT_HERSHEY_SIMPLEX,1,(163,243,0),6)
                        elif num_pred == 0:
                            cv2.putText(self.frame,"NO MASK",(x,y+h+50),cv2.FONT_HERSHEY_SIMPLEX,1,(163,243,0),6)
                        print(num_pred)

                # profile_faces = self.profile_cascade.detectMultiScale(self.frame,scaleFactor=self.scaleFactor,minNeighbors=4,minSize=(30,30))
                success,buffer = cv2.imencode('.jpg',cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB))
                yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')






if __name__ == "__main__":
    camera = Camera()
    print(camera.get_frames())







