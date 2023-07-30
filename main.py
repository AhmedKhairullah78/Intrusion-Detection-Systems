# install firebase_admin
# link our db

import urllib
#import pyrebase


import pickle
import re
from pathlib import Path
from IPython.display import Image ,display # for displaying images
import cv2
import numpy as np
import requests
import io
import torch
import json
from ultralytics import YOLO
from firebase_admin import credentials,initialize_app,db
cred = credentials.Certificate("./kinai.json")
app = initialize_app(cred,{
   
   'databaseURL':'https://graduation-project-bb3e2-default-rtdb.firebaseio.com/'
})


BASE_DIR = Path(__file__).resolve(strict=True).parent



#img = cv2.imread('fire.jpg')
model1 = YOLO('Fire.pt')
model2 = YOLO('weabon2 (2).pt')
model3 = YOLO('mask.pt')
#model.predict(source = 'WhatsApp Video 2023-03-25 at 4.22.40 PM.MP4' ,conf=0.25 , save = True)


# model1 = torch.hub.load('ultralytics/yolov5', 'custom', 
#  path='Fire.pt', force_reload=True) 
# model2 = torch.hub.load('ultralytics/yolov5', 'custom', 
#  path='weabon2.pt', force_reload=True)
# model3 = torch.hub.load('ultralytics/yolov5', 'custom', 
#  path='mask.pt', force_reload=True) 


###############################################3
#################################################
##################################################
###################################################
import math
import cv2
import numpy as np
import face_recognition
import os

path = 'person'
images = []
classNames = []
personsList = os.listdir(path)

for cl in personsList:
    curPersonn = cv2.imread(f'{path}/{cl}')
    images.append(curPersonn)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodeings(image):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodeings(images)
print('Encoding Complete.')



ress=[]
end=[]
from mtcnn import MTCNN
#detector = MTCNN()


ip="http://192.168.1.3:8080"


import datetime
cap=cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc(*'XVID')
outi=cv2.VideoWriter('out.mp4',fourcc,20.0,(640,480))

cap = cv2.VideoCapture('http://192.168.1.3:8080/video')
url = ip + "/shot.jpg"
cap.set(3,100)
cap.set(4,100)
while True:
    img_request = requests.get(url)
    img_arrint = np.array(bytearray(img_request.content), dtype=np.uint8)
    frame = cv2.imdecode(img_arrint, -1)
    ress.append(0)

    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        
    imgS = cv2.resize(frame, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurentFrame = face_recognition.face_locations(imgS)
    encodeCurentFrame = face_recognition.face_encodings(imgS, faceCurentFrame)
    
    res_f = model1(frame)  
    for r in res_f:        
                 boxes = r.boxes
                 for box in boxes:
                     conf = math.ceil((box.conf[0])* 100) / 100
                     if conf > 0.6:
                         ress.append(1)
                         
    
    
    for encodeface, faceLoc in zip(encodeCurentFrame, faceCurentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.rectangle(frame, (x1,y2-35), (x2,y2), (0,0,255), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            ress.append(0)
            
            
        else:
            name='other'
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.rectangle(frame, (x1,y2-35), (x2,y2), (0,0,255), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            roi=frame[y1:y2,x1:x2]
            cv2.imwrite(f'Pic{name}.png', roi)
            ress.append(4)
            
            res_w = model2(frame)  
            for r in res_w:        
                 boxes = r.boxes
                 for box in boxes:
                     conf = math.ceil((box.conf[0])* 100) / 100
                     if conf > 0.6:
                         ress.append(2)

            res_m = model3(frame)  
            for r in res_m:        
                 boxes = r.boxes
                 for box in boxes:
                     conf = math.ceil((box.conf[0])* 100) / 100
                     if conf > 0.6:
                         ress.append(3)             

    

        
   
    for x in ress[:]:
        
        if x==0:
            print("no problem")
        elif x==4:
            print("unknown_person")
        elif x==2:
            end.append("gun_or_knife")
        elif x==3:
            end.append("masked_person")
        elif x==1:
            end.append("fire")
        # update data
        ref=db.reference()
        result=ref.get()
        ref.update({"dd":x})
        # for key,value in result.items():
        #    if(value["detect"] == "NON"):
        #      ref.child(key).update({"dd":x})
             


           
            
    if len(end)!=0:
        print(end)       
    ress.clear()
    end.clear()
    cv2.imshow('Face Recogntion', frame)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
outi.release()
cv2.destroyAllWindows()








################################################################
################################################################
################################################################





















# # img="./114.jpg"
# # results = model1(img)
# # ss=str(results)
# # img_res= ss[18:35]
# ip="http://192.168.1.3:8080"
# def predict_img(ip):
    
#     cap = cv2.VideoCapture('http://192.168.1.3:8080/video')
#     url = ip + "/shot.jpg"
#     cap.set(3,100)
#     cap.set(4,100)
#     while True:
#         img_request = requests.get(url)
#         img_arrint = np.array(bytearray(img_request.content), dtype=np.uint8)
#         img = cv2.imdecode(img_arrint, -1)
#         fin_res=""
    
#         results = model1(img)
#         ss=str(results)
#         rr=ss[18:35] 
#         fin_res=fin_res+"&&"+rr
#         #* update data
#         ref=db.reference("/books/")
#         books=ref.get()
# #         for key,value in books.items():
# #            if(value["author"] == "Potato"):
# #             ref.child(key).update({"price":fin_res})
           
        

# #     #return fin_res

# # predict_img(ip)



# # install firebase_admin
# # link our db

# import urllib
# #import pyrebase


# import pickle
# import re
# from pathlib import Path
# from IPython.display import Image ,display # for displaying images
# import cv2
# import numpy as np
# import requests
# import io
# import torch
# import json
# from ultralytics import YOLO
# from firebase_admin import credentials,initialize_app,db
# cred = credentials.Certificate("./kinai.json")
# app = initialize_app(cred,{
   
#    'databaseURL':'https://kinai-79465-default-rtdb.europe-west1.firebasedatabase.app/'
# })


# BASE_DIR = Path(__file__).resolve(strict=True).parent



# #img = cv2.imread('fire.jpg')
# model1 = YOLO('Fire.pt')
# model2 = YOLO('weabon2 (2).pt')
# model3 = YOLO('mask.pt')
# #model.predict(source = 'WhatsApp Video 2023-03-25 at 4.22.40 PM.MP4' ,conf=0.25 , save = True)


# # model1 = torch.hub.load('ultralytics/yolov5', 'custom', 
# #  path='Fire.pt', force_reload=True) 
# # model2 = torch.hub.load('ultralytics/yolov5', 'custom', 
# #  path='weabon2.pt', force_reload=True)
# # model3 = torch.hub.load('ultralytics/yolov5', 'custom', 
# #  path='mask.pt', force_reload=True) 


# ###############################################3
# #################################################
# ##################################################
# ###################################################

# import cv2
# import numpy as np
# import face_recognition
# import os

# path = 'person'
# images = []
# classNames = []
# personsList = os.listdir(path)

# for cl in personsList:
#     curPersonn = cv2.imread(f'{path}/{cl}')
#     images.append(curPersonn)
#     classNames.append(os.path.splitext(cl)[0])
# print(classNames)

# def findEncodeings(image):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encode = face_recognition.face_encodings(img)[0]
#         encodeList.append(encode)
#     return encodeList

# encodeListKnown = findEncodeings(images)
# print('Encoding Complete.')



# ress=[]
# end=[]
# from mtcnn import MTCNN
# #detector = MTCNN()


# ip="http://192.168.1.3:8080"


# import datetime
# cap=cv2.VideoCapture(0)
# fourcc=cv2.VideoWriter_fourcc(*'XVID')
# outi=cv2.VideoWriter('out.mp4',fourcc,20.0,(640,480))

# cap = cv2.VideoCapture('http://192.168.1.3:8080/video')
# url = ip + "/shot.jpg"
# cap.set(3,100)
# cap.set(4,100)
# while True:
#     img_request = requests.get(url)
#     img_arrint = np.array(bytearray(img_request.content), dtype=np.uint8)
#     frame = cv2.imdecode(img_arrint, -1)


#     ret, frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
        
#     imgS = cv2.resize(frame, (0,0), None, 0.25, 0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

#     faceCurentFrame = face_recognition.face_locations(imgS)
#     encodeCurentFrame = face_recognition.face_encodings(imgS, faceCurentFrame)
    
#     res_f = model1(frame)  
#     re_f=res_f.pandas().xyxy[0]
#     for i in range (len(re_f)) :
#         xmin,ymin,xmax,ymax,confidence,_,_=re_f.iloc[i,:]
#         xmin,ymin,xmax,ymax=int(xmin),int(ymin),int(xmax),int(ymax)
#         if confidence > 0.6 :
#             cv2.rectangle(frame, (xmin , ymin), (xmax , ymax), (255, 0, 0), 2)
#             cv2.putText(frame, str(re_f.name[i]), (xmin , ymin), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA) 
#             cv2.putText(frame, str(round(confidence,2)), (xmin+50 , ymin), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA) 
#             ress.append(4)

#     for encodeface, faceLoc in zip(encodeCurentFrame, faceCurentFrame):
#         matches = face_recognition.compare_faces(encodeListKnown, encodeface)
#         faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
#         matchIndex = np.argmin(faceDis)

#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper()
#             y1, x2, y2, x1 = faceLoc
#             y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
#             cv2.rectangle(frame, (x1,y2-35), (x2,y2), (0,0,255), cv2.FILLED)
#             cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
#             ress.append(0)
            
            
#         else:
#             name='other'
#             y1, x2, y2, x1 = faceLoc
#             y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
#             cv2.rectangle(frame, (x1,y2-35), (x2,y2), (0,0,255), cv2.FILLED)
#             cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
#             roi=frame[y1:y2,x1:x2]
#             cv2.imwrite(f'Pic{name}.png', roi)
#             ress.append(1)
            
#             res_w = model2.predict(source =frame, size=640,conf=0.25 ,save =True)   
#             re_w=res_w.pandas().xyxy[0]
#             for i in range (len(re_w)) :
#                 xmin,ymin,xmax,ymax,confidence,_,_=re_w.iloc[i,:]
#                 xmin,ymin,xmax,ymax=int(xmin),int(ymin),int(xmax),int(ymax)
#                 if confidence > 0.4 :
#                     cv2.rectangle(frame, (xmin , ymin), (xmax , ymax), (255, 0, 0), 2)
#                     cv2.putText(frame, str(re_w.name[i]), (xmin , ymin), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA) 
#                     cv2.putText(frame, str(round(confidence,2)), (xmin+50 , ymin), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA) 
#                     ress.append(2)
#             res_m = model3.predict(source =frame, size=640,conf=0.25 ,save =True)  
#             re_m=res_m.pandas().xyxy[0]
#             for i in range (len(re_m)) :
#                 xmin,ymin,xmax,ymax,confidence,_,_=re_m.iloc[i,:]
#                 xmin,ymin,xmax,ymax=int(xmin),int(ymin),int(xmax),int(ymax)
#                 if confidence > 0.4 :
#                     cv2.rectangle(frame, (xmin , ymin), (xmax , ymax), (255, 0, 0), 2)
#                     cv2.putText(frame, str(re_m.name[i]), (xmin , ymin), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA) 
#                     cv2.putText(frame, str(round(confidence,2)), (xmin+50 , ymin), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2, cv2.LINE_AA) 
#                     ress.append(3)
        
   
#     for x in ress[:]:
#         if x==0:
#             print("no problem")
#             break
            
        
#         elif x==1:
#             print("unknown_person")
#         elif x==2:
#             end.append("gun_or_knife")
#         elif x==3:
#             end.append("masked_person")
#         elif x==4:
#             end.append("fire")
#         # update data
#         ref=db.reference("/Books/")
#         Books=ref.get()
#         for key,value in Books.items():
#            if(value["detect"] == "NON"):
#              ref.child(key).update({"dd":x})
      

           
            
#     if len(end)!=0:
#         print(end)       
#     ress.clear()
#     end.clear()
#     cv2.imshow('Face Recogntion', frame)
#     if cv2.waitKey(1)==ord('q'):
#         break
# cap.release()
# outi.release()
# cv2.destroyAllWindows()








################################################################
################################################################
################################################################





















# # img="./114.jpg"
# # results = model1(img)
# # ss=str(results)
# # img_res= ss[18:35]
# ip="http://192.168.1.3:8080"
# def predict_img(ip):
    
#     cap = cv2.VideoCapture('http://192.168.1.3:8080/video')
#     url = ip + "/shot.jpg"
#     cap.set(3,100)
#     cap.set(4,100)
#     while True:
#         img_request = requests.get(url)
#         img_arrint = np.array(bytearray(img_request.content), dtype=np.uint8)
#         img = cv2.imdecode(img_arrint, -1)
#         fin_res=""
    
#         results = model1(img)
#         ss=str(results)
#         rr=ss[18:35] 
#         fin_res=fin_res+"&&"+rr
#         #* update data
#         ref=db.reference("/books/")
#         books=ref.get()
#         for key,value in books.items():
#            if(value["author"] == "Potato"):
#             ref.child(key).update({"price":fin_res})
           
        

#     #return fin_res

# predict_img(ip)
