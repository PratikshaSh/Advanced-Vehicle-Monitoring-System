#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import pytesseract
import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from yolo_sih import YOLO_Plate
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from mouseclick import video_click
import pyrebase
import random
import datetime
import pandas as pd
import csv
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
config = {
  "apiKey": "AIzaSyDaYILIFsZWzCl54rbQQMrGT5ET3o8Yj6U",
  "authDomain": "vechiledetection",
  "databaseURL": "https://vechiledetection.firebaseio.com/",
  "storageBucket": "vechiledetection.appspot.com",
  "serviceAccount": "C:/Users/TusharGoel/Desktop/Vehicle Object Tracking/db/vechiledetection-firebase-adminsdk-kiblg-ddb38b4b7f.json"
}
firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
db = firebase.database()

currentDTKey = datetime.datetime.now() #For retreiving the key from The system date

DTKey = currentDTKey.strftime("%d%m%Y")

warnings.filterwarnings('ignore')

def RTO(vehicleTime,text):
    vehicleOwner = ['Arif','Tushar','Pratiksha','Shobhit','Praveen','Pranav']
    vehicleName = ['Honda','Activa','Hero','Ford','Maruti','WagonR']
    vehicleRegion = ['Ghaziabad','Meerut','Punjab','Gwalior','Jharkhand','Kanpur']
    vehicleClass = ['Car','Scooter','Bike','Truck','Bus']
    
    currentDT = datetime.datetime.now()
    
    
    vehicleOwner = random.choice(vehicleOwner)
    vehicleName = random.choice(vehicleName)
    vehicleRegion = random.choice(vehicleRegion)
    vehicleClass = random.choice(vehicleClass)
    
#Start-of-Firebase-Operations
     #for retrieving the time from the system 
    
    data = {"vno": str(text),
            "name": str(vehicleOwner),
            "make": str(vehicleName),
            "region": str(vehicleRegion),
            "vclass": str(vehicleClass)
             }

    print(f'Added {data}')
    
    db.child("Entry").child(DTKey).child(vehicleTime).set(data)

# transform newbboxs of (n_object,4,2) np array s.t. return_boxs = bbox_transform(newbboxs)
# newbboxs[i,:,:] = np.array([[xmin,ymin],[xmin+boxw,ymin],[xmin,ymin+boxh],[xmin+boxw,ymin+boxh]]).astype(float)
# return_boxs = [], return_boxs.append([x,y,w,h])
parent_dir = 'C:/Users/TusharGoel/Desktop/Vehicle Object Tracking'
Test_Media_Path = os.path.join(parent_dir,"Data","Source_Images","Test_Images")
Test_Results_Path = os.path.join(parent_dir,"Data",'Source_Images','Test_Image_Detection_Results')
media = os.listdir(Test_Media_Path)
MediaPaths = []
for i in range(len(media)):
    
    MediaPaths.append(os.path.join(Test_Media_Path,media[i]))
img_endings = (".jpg", ".jpg", ".png")
vid_endings = (".mp4", ".mpeg", ".mpg", ".avi")
input_image_paths = []
input_video_paths = []
for item in MediaPaths:
    if item.endswith(img_endings):
        input_image_paths.append(item)
    elif item.endswith(vid_endings):
        input_video_paths.append(item)

def image_preprocess(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("grayim",img_gray)
    _, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_erode = cv2.erode(img_binary, (3,3))
    img_dilate = cv2.dilate(img_erode, (3,3))
    
    LP_WIDTH = img_dilate.shape[0]
    LP_HEIGHT = img_dilate.shape[1]
    
    # Make borders white
    img_dilate[0:3,:] = 255
    img_dilate[:,0:3] = 255
    img_dilate[72:75,:] = 255
    img_dilate[:,330:333] = 255
    
    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/10, 2*LP_HEIGHT/3]
    # cv2.imshow("im_dil",img_dilate)
    # cv2.waitKey(3000)
    # cv2.destroyAllWindows()
    return img_dilate
def bbox_transform(newbboxs):
    return_boxs = []
    for i in range(newbboxs.shape[0]):
        [x,y,w,h] = [newbboxs[i,0,0],newbboxs[i,0,1],newbboxs[i,3,0]-newbboxs[i,0,0],newbboxs[i,3,1]-newbboxs[i,0,1]]
        return_boxs.append([x,y,w,h])
    return return_boxs

def main(yolo,yolo_plate):
    # For Images:
    for images in input_image_paths:
        print('Detecting Image')
        img = cv2.imread(images)
        image_name = images.split('/')[-1].split('\\')[-1]
        image = Image.fromarray(img)
        predicted_class,car_boxes,plate_pred,image = yolo.detect_image(image)
        image = np.asarray(image)
        cv2.imwrite(r'C:\Users\TusharGoel\Desktop\Vehicle Object Tracking\Data\Source_Images\Test_Image_Detection_Results\{}.jpg'.format(image_name),image)
        file = open('result_detection.txt','a')

        for i,box in enumerate(car_boxes):
            print(predicted_class[i])
            if predicted_class[i] is None:
                continue
            else:
                file.write(str(images)+'\t'+str(box[0])+'\t'+str(box[1])+'\t'+str(box[0]+box[3])+'\t'+str(box[1]+box[4])+'\t'+str(predicted_class[i]))
            
        for pred in plate_pred:
            left = pred[0]
            top = pred[1]
            right = pred[2]
            bottom = pred[3]
            roi = image[top:bottom,left:right]
            img_dilate = image_preprocess(roi)
            cv2.imwrite('C:/Users/TusharGoel/Desktop/Extracted_Plate/{}.jpg'.format(image_name),img_dilate)
            img = cv2.imread('C:/Users/TusharGoel/Desktop/Extracted_Plate/{}.jpg'.format(image_name))
            file.write(str(images)+'\t'+str(left)+'\t'+str(top)+'\t'+str(right)+'\t'+str(bottom)+'\t'+'plate')
                                    
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("grayim",img_gray)
            _, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            img_erode = cv2.erode(img_binary, (3,3))
            img_dilate = cv2.dilate(img_erode, (3,3))
            
            LP_WIDTH = img_dilate.shape[0]
            LP_HEIGHT = img_dilate.shape[1]
            
            # Make borders white
            img_dilate[0:3,:] = 255
            img_dilate[:,0:3] = 255
            img_dilate[72:75,:] = 255
            img_dilate[:,330:333] = 255
            
            # Estimations of character contours sizes of cropped license plates
            dimensions = [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/10, 2*LP_HEIGHT/3]
            cv2.imshow("im_dil",img_dilate)
            config = ('-l eng+hin --oem 1 --psm 3')
            text = pytesseract.image_to_string(img_dilate,config=config)
            print('License Plate Recognised: {}'.format(text))
            vehicleTime = currentDTKey.strftime("%H%M%S")
            
            a = open('result_recog.txt','a')
            if text=='':
                a.write(str(images)+'\t'+'')
            else:
                a.write(str(images)+'\t'+str(text))
            RTO(vehicleTime,text)
    df_detect = pd.read_csv('result_detection.txt',sep='\t')
    df_detect.to_csv('detection.tsv')
    df = pd.read_csv('result_recog.txt',sep='\t')
    df.to_csv('recognition.tsv')
    
    # For Video:
    for videos in input_video_paths:
   
        max_cosine_distance = 0.3
        nn_budget = None
        nms_max_overlap1 = 1.0
        
        # deep_sort 
        model_filename = 'model_data/mars-small128.pb'
        encoder = gdet.create_box_encoder(model_filename,batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        tracker = Tracker(metric)
    
        writeVideo_flag = True 
     
        video_capture = cv2.VideoCapture(videos)
        mousepoints = video_click(videos)
         

        if writeVideo_flag:
       
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
            
            
            
        fps = 0.0
        firstflag = 1
        while True:
            ok, frame = video_capture.read()  
            if ok != True:
                break;
            t1 = time.time()
            cv2.line(frame,mousepoints[0],mousepoints[1],(0,255,255),2)
            cv2.line(frame,mousepoints[2],mousepoints[3],(0,255,255),2)
            # imgpts = np.array(convexpoints)
            # mask = cv2.fillConvexPoly(frame, imgpts, (0,255,255))
            # cv2.addWeighted(frame,0.8,mask,0.2,0,frame)
            image = Image.fromarray(frame)
            boxs,plate_prediction,image = yolo.detect_image(image) 
            
            image = np.asarray(image)
            features = encoder(image,boxs)
            
            # score to 1.0 here).
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
            
           
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap1, scores)
            detections = [detections[i] for i in indices]
    
            ### Call the tracker
            tracker.predict()
            tracker.update(detections)
            
    
            ### Add one more step of optical flow
            # convert detections to bboxs for optical flow
            n_object = len(detections)
            bboxs = np.empty((n_object,4,2), dtype=float)
            i = 0
            
            boxes_tracking = np.array([track.to_tlwh() for track in tracker.tracks])
            ### Deep sort tracker visualization
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                print('Vehicle ID {} is On Tracker'.format(str(track.track_id)))
                # cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv2.putText(image, 'ID:'+str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
                if (int(bbox[3]))>=(mousepoints[3][1]):
                    cv2.line(image,mousepoints[2],mousepoints[3],(0,255,0),2)
                    #if (int(bbox[3]))>=(mousepoints[3][1]) and (int(bbox[3]))<=(mousepoints[1][1]) and int(bbox[2])<=mousepoints[1][0] and int(bbox[0])>=mousepoints[0][0]:
                    cv2.line(image,mousepoints[0],mousepoints[1],(0,255,0),2)
                    
                    roi_cars = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                    
                    cv2.imwrite('C:/Users/TusharGoel/Desktop/Car_Extracted/car_{}.png'.format(str(track.track_id)),roi_cars)
                    roi_cars = Image.fromarray(roi_cars)
                    plate_pred,plate_image = yolo_plate.detect_image(roi_cars)
                    plate_image = np.asarray(plate_image)
                    roi_cars = np.asarray(roi_cars)
                    if len(plate_pred)!=0:
                        for i in range(len(plate_pred)):
                            roi_plate = roi_cars[plate_pred[i][1]:plate_pred[i][3],plate_pred[i][0]:plate_pred[i][2]]
                            # roi_plate = image_preprocess(roi_plate)
                            cv2.imwrite('C:/Users/TusharGoel/Desktop/Extracted_Plate/plate_{}.png'.format(str(track.track_id)),roi_plate)
                            
                        if (int(bbox[3]))>int(mousepoints[1][1]):
                            cv2.line(image,mousepoints[2],mousepoints[3],(0,255,255),2)
                            vehicle_text = 'Vehicle is Entering'
                            cv2.putText(image,vehicle_text,(100,100),cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color = (0,255,255),thickness = 2)
                            vehicleTime = currentDTKey.strftime("%H%M%S")
                            img = cv2.imread('C:/Users/TusharGoel/Desktop/Extracted_Plate/plate_{}.png'.format(str(track.track_id)))       
                            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            # cv2.imshow("grayim",img_gray)
                            _, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                            img_erode = cv2.erode(img_binary, (3,3))
                            img_dilate = cv2.dilate(img_erode, (3,3))
                            
                            LP_WIDTH = img_dilate.shape[0]
                            LP_HEIGHT = img_dilate.shape[1]
                            
                            # Make borders white
                            img_dilate[0:3,:] = 255
                            img_dilate[:,0:3] = 255
                            img_dilate[72:75,:] = 255
                            img_dilate[:,330:333] = 255
                            
                            # Estimations of character contours sizes of cropped license plates
                            dimensions = [LP_WIDTH/6, LP_WIDTH/2, LP_HEIGHT/10, 2*LP_HEIGHT/3]
                            cv2.imshow("im_dil",img_dilate)
                            config = ('-l eng+hin --oem 1 --psm 3')
                            text = pytesseract.image_to_string(img_dilate,config=config)
                            print('License Plate Recognised: {}'.format(text))
                            RTO(vehicleTime,text)
                            storage.child("Entry").child(DTKey).child(vehicleTime).child('Car').put('C:/Users/TusharGoel/Desktop/Car_Extracted/car_{}.png'.format(str(track.track_id)))
                            storage.child("Entry").child(DTKey).child(vehicleTime).child('Plate').put('C:/Users/TusharGoel/Desktop/Extracted_Plate/plate_{}.png'.format(str(track.track_id)))
                            print('File Uploaded on FireBase')

            for det in detections:
                bbox = det.to_tlbr()
                cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2) # BGR color
             
            cv2.imshow('result', image)
            
            out.write(image)
            firstflag = 0
                
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            
            
            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
        video_capture.release()
        
        out.release()
            
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO(),YOLO_Plate())
