import numpy as np
import cv2
import requests
from pprint import pprint
import pandas as pd
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import json
import requests

# Pybase config 
config = {
  "apiKey": "AIzaSyDaYILIFsZWzCl54rbQQMrGT5ET3o8Yj6U",
  "authDomain": "vechiledetection",
  "databaseURL": "https://vechiledetection.firebaseio.com/",
  "storageBucket": "gs://vechiledetection.appspot.com"
}

tokenNO = 'Token e9fa430fd4d5081b4ff35e3c634a8e0162ae961b'
    # Preprocess cropped license plate image
df = pd.read_csv('C:/Users/Tushar Goel/Desktop/LicensePlateWithYOLO/Data/Source_Images/Test_Image_Detection_Results/Detection_Results.csv')
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
for i in range(df.shape[0]):
    #img = cv2.imread('{}'.format(df['image_path'][i]))
    # image = img[df['ymin'][i]:df['ymax'][i],df['xmin'][i]:df['xmax'][i]]
    image = cv2.imread('C:/Users/Tushar Goel/Desktop/Vehicle plate detection with Tracker/Data/Source_Images/extracted_plates/6 (2).png')
    cv2.imshow("im",image)
    img = cv2.resize(image, (333, 75))
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
    config = ('-l eng --oem 1 --psm 3')
    text = pytesseract.image_to_string(img_dilate,config=config)
    print(text)
    pathSavedLP = 'C:/Users/Tushar Goel/Desktop/objects/test.png'
    cv2.imwrite( pathSavedLP, img_dilate)
   # print(img_dilate.format)
    
#Star-of API
    
    
    regions = ['fr', 'it', 'in']
    with open('C:/Users/Tushar Goel/Desktop/objects/test.png', 'rb') as fp:
        response = requests.post(
            'https://api.platerecognizer.com/v1/plate-reader/',
            data=dict(regions=regions),  # Optional
            files=dict(upload=fp),
            headers={'Authorization': 'Token e9fa430fd4d5081b4ff35e3c634a8e0162ae961b' })
    
    
    #pprint(response.json().get('results')[0].get('plate'))

    resultsPlate = response.json().get('results')[0].get('plate').upper()
    
    country_code = resultsPlate[:len(resultsPlate)-4]
    plate_number = resultsPlate[len(resultsPlate)-4:]
     
    login_data={"r1[]":"PB22G",
                "r2":"4565",
                "auth":"Y29tLmRlbHVzaW9uYWwudmVoaWNsZWluZm8="}
    
    def RTO(country,plate_no):
        login_data["r1[]"]=country
        login_data["r2"]=plate_no
        r = requests.post("https://rtovehicle.info/batman.php",login_data)
        response = r.content
        my_json = response.decode('utf8').replace("'", '"')
        #print(my_json)
     
        # Load the JSON to a Python list & dump it back out as formatted JSON
        data = json.loads(my_json)
        s = json.dumps(data, indent=4, sort_keys=True)
        #print(s)
        
        global vehicleOwner
        global vehicleName
        global vehicleRegion
        global vehicleClass
        vehicleOwner = data.get('owner_name')
        vehicleName = data.get('vehicle_name')
        vehicleRegion = data.get('regn_auth')
        vehicleClass = data.get('vh_class')
        print('Vehicle ' + str(i+1) + ': \n' + vehicleOwner + '\n' + vehicleName + '\n' + vehicleClass + '\n' + vehicleRegion + '\n' + resultsPlate)
     
    RTO(country_code,plate_number)
#--End of API
    cv2.waitKey(1)
    cv2.destroyAllWindows()
