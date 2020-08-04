# AVSAM ( Advanced Vehicle Survelliance and Monitoring )
<hr>
### Update v1.1

- Improved retrained model with a vast dataset of 5000+ 
- FR secured was 0.97 on 100 samples (previously 0.96)
- Improved system for handling unrecognized number plate by sending a push notification to the concerned personnel with the image and the text fields for manually entering the plate number by looking at the picture, before the data is retreived from the RTO.
- Implemented DeepSORT to keep track of vehicles "coming in" or "goin out"
- Implemented SafeSPEED parameter to keep track of the vehicle's speed while at the entry point
- Speed Detector dependencies varies from camera to camera.It can be implemented by changing the variables.
- Improved best Confidence Score refining mechanism 
- Improved filter for image processing and deskewing algorithms used

<h4>Using YOLOv3 and Pytesseract (Team: Error_404) (College: NIT Jalandhar)</h4>

**Problem Statement :** Create an affordable Solution through image processing of the number plates of vehicles for the **Detection , Identification and Monitoring** of Vehicles in Different scenario.

### Sample Annotation
Please find the annotations using this model on the below link 
https://www.youtube.com/watch?v=A1jiDW4kLoo

# Table of Contents

 - <a href="#Prerequisites">Prerequisites</a>
				 - Detection
				 - OCR
				 - Storage
 - <a href="#steps">How to run</a>
 
 - <a href = "#Samples"> Samples From the Code</a>
 - <a href="#App">Vehicle Detection App</a>
 - <a href="#working">How it Works</a>

 
 ## <h1 id = "Prerequisites">PREREQUISITES</h1>
<H4><U>DETECTION</U></H4>
                        
 - Run Requirements File for Installing the Required Packages :
		 **If you want to run on CPU:**
					`pip install -r requirements-cpu.txt`
		**If you want to run on GPU:**
					`pip install -r requirements-gpu.txt`
<h4><u>OCR<u></h4>
 - Install Pytesseract using following steps:
			 
 - Download Pytesseract using the Below link:
			 - [Tesseract-ocr-Download](https://sourceforge.net/projects/tesseract-ocr-alt/files/)
			-Add **C:/Program Files/Tesseract-OCR/tesseract.exe'** to your  system variables
			- Then Install tesseract on our system by using:
 	
                 `pip install pytesseract-ocr`
  - For Hindi OCR , Download [hin.trainedata](	https://github.com/tesseract-ocr/tessdata/blob/master/hin.traineddata) and put it `C:\Program Files\Tesseract-OCR\tessdata`
<h4><u>STORAGE<u></h4>
		
 - We are using FireBase for our Storage system
 - It can be installed by following command:
		 - `pip install pyrebase`


<h1 id="steps">How to Run</h1>
		

 - **<u>Step 0:</u>** Change the locations of the file in the code wherever required.
 - **<u>Step 1:</u>** Images and Video to Detect should be kept in **Data/Source Images/Test Images/**
 - **<u>Step 2:</u>** If you want to Detect the Files just go to **training/src/keras_yolo3/Tracking.py**
 - **<u>Step 3:</u>** If you have put **Video on Detection** , It would be open for first **20 seconds to click on the four points in frame**...for creating Two line, which will help **model to Track the vehicles to focus on specific vehicle in crowd of vehicles.**

Detection Image Before crossing the line</img>
<hr color='red' >
<img src="Screenshot (191).png">Detection and Tracking ID on each vehicle before crossing the vehicle.</img>
<hr color='red'>
<img src="Screenshot (192).png">After Crossing the line, it turns green and Vehicle is entering message will be flashed and Update will be made on database with image of vehicle and plate</img>

<h1 id="App">Vehicle Detection App<h1>

 - Starting of the app ("https://i.imgur.com/h9ytWbJ.jpg")
 - Database of the Admin ("https://i.imgur.com/Xd94ZMZ.jpg")
 - [NOT REQUIRED] You are requested to install the app on your android device from http://github.com/praveen4030/VechileDetection FOR FULL FUNCTIONALITY


<h1 id="working">How It Works</h1>

 - When we run the model , Number plate will be detected with an average **accuracy of 97%** of the vehicles crossing the Line drawn , Then the Number Plate will be cropped and filtered and passed through The OCR and we will get the Number plate. Then the Numbe Plate we get , then Processed through the RTO Database and If the Data is Found , Then it will be updated in the Storage as well as App too. But If No data is Found in the RTO database , then it will cause an Alert in the Admin Database in the app , and can only be resolved by Manually entering the Number plate in the App .
 

## Troubleshooting

0. If you encounter any error, please make sure you follow the instructions **exactly** (word by word). Once you are familiar with the code, you're welcome to modify it as needed but in order to minimize error, I encourage you to not deviate from the instructions above.  

1. If you are using [pipenv](https://github.com/pypa/pipenv) and are having trouble running `python3 -m venv env`, try:
    ```
    pipenv shell
    ```

2. If you are having trouble getting cv2 to run, try:

    ```
    apt-get update
    apt-get install -y libsm6 libxext6 libxrender-dev
    pip install opencv-python
    ```

3. If you are a Linux user and having trouble installing `*.snap` package files try:
    ```
    snap install --dangerous vott-2.1.0-linux.snap
    ```
    See [Snap Tutorial](https://tutorials.ubuntu.com/tutorial/advanced-snap-usage#2) for more information.



## Stay Up-to-Date

- **star** this repo to get notifications on future improvements and
- **fork** this repo if you like to use it as part of your own project.

