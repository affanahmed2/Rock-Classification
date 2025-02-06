import sys
import ctypes
import numpy as np
import cv2
import csv
from sklearn.neighbors import KNeighborsClassifier  # Importing KNN classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


sys.path.append("MvImport")
from MvCameraControl_class import *



### GLOBAL VARIABLES
edge = 100
box_x1, box_y1, box_x2, box_y2 = 600, 600, 700, 700

# Initialize Camera SDK
MvCamera().MV_CC_Initialize()

# Enumerate Devices
deviceList = MV_CC_DEVICE_INFO_LIST()
ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)

if ret != 0:
    print(f"Device enumeration failed! Error code: 0x{ret:X}")
    sys.exit()

if deviceList.nDeviceNum == 0:
    print("No camera devices found.")
    sys.exit()

print(f"Found {deviceList.nDeviceNum} device(s).")

# Get First Device
stDeviceList = ctypes.cast(deviceList.pDeviceInfo[0], ctypes.POINTER(MV_CC_DEVICE_INFO)).contents

# Create Camera Object
cam = MvCamera()
ret = cam.MV_CC_CreateHandle(stDeviceList)
if ret != 0:
    print(f"Failed to create handle! Error code: 0x{ret:X}")
    sys.exit()

# Open Device
ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
if ret != 0:
    print(f"Failed to open device! Error code: 0x{ret:X}")
    cam.MV_CC_DestroyHandle()
    sys.exit()

# Set Camera Parameters
cam.MV_CC_SetFloatValue("ExposureTime", 10000.0)  # eet exposure time
cam.MV_CC_SetEnumValue("GainAuto", 0)  # enable auto gain

# Start Grabbing
ret = cam.MV_CC_StartGrabbing()
if ret != 0:
    print(f"Failed to start grabbing! Error code: 0x{ret:X}")
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    sys.exit()

print("Camera is grabbing frames... Press ESC to exit.")


# Set Camera Parameters
cam.MV_CC_SetFloatValue("ExposureTime", 100000.0)  # eet exposure time
cam.MV_CC_SetEnumValue("GainAuto", 0)  # enable auto gain

# Collect Training Data
rock_types = ['rock_type_1', 'rock_type_2', 'rock_type_3']  # Replace with actual rock types
training_images = []
training_labels = []



def save_features_to_csv(features, label, filename="rock_features.csv"):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([*features, label])


def load_features_from_csv(filename="rock_features.csv"):
    features = []
    labels = []
    try:
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                features.append([float(x) for x in row[:-1]])  # Convert features to float
                labels.append(int(row[-1]))  # Convert label to int
    except FileNotFoundError:
        print("Feature file not found. Train the model first.")
    return features, labels


def extract_hsv_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert image to HSV
    cv2.imshow("HSV", hsv_image)
    #region = hsv_image[box_y1:box_y2, box_x1:box_x2]  # Get the region within the box
    avg = np.mean(hsv_image, axis=(0, 1))

    print(avg)
    return avg  # Return average HSV values for the region


def getOpenCVImage():
    # Initialize frame buffer
    stOutFrame = MV_FRAME_OUT()
    ctypes.memset(ctypes.byref(stOutFrame), 0, ctypes.sizeof(stOutFrame))

    ret = cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
    if ret != 0:
        print(f"Failed to get image buffer! Error code: 0x{ret:X}")
        exit()

    # Convert to OpenCV Image
    buf_cache = (ctypes.c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
    ctypes.memmove(ctypes.byref(buf_cache), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)

    width, height = stOutFrame.stFrameInfo.nWidth, stOutFrame.stFrameInfo.nHeight
    scale_factor = min(1920 / width, 1080 / height)

    np_image = np.ctypeslib.as_array(buf_cache).reshape(height, width)
    cv_image = cv2.cvtColor(np_image, cv2.COLOR_BayerRG2RGB)
    cv_image = cv2.resize(cv_image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    cam.MV_CC_FreeImageBuffer(stOutFrame)  # Free buffer after use
    
    return cv_image


# Mouse callback function to select region for training
def select_region(event, x, y, flags, param):
    global training_images, training_labels
    if event == cv2.EVENT_LBUTTONDOWN:
        region = param[box_y1:box_y2, box_x1:box_x2]
        cv2.imshow("Selected Region", region)
        print("Press 1 for Rock Type 1, 2 for Rock Type 2, 3 for Rock Type 3.")
        key = cv2.waitKey(0)
        label = None
        if key == ord('1'):
            label = 0
        elif key == ord('2'):
            label = 1
        elif key == ord('3'):
            label = 2
        if label is not None:
            features = extract_hsv_features(region)
            save_features_to_csv(features, label)
            print(f"Captured image for Rock Type {label + 1}.")


# Collect images and labels using mouse clicks
def collect_training_data():
    global training_images, training_labels

    while True:
        cv_image = getOpenCVImage()
        cv2.rectangle(cv_image, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)  # Green rectangle
        cv2.putText(cv_image, "Place the rock here", (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Label above box
            
        # Display image and set mouse callback
        cv2.imshow("Capture Image for Labeling", cv_image)
        cv2.setMouseCallback("Capture Image for Labeling", select_region, param=cv_image)

        if cv2.waitKey(1) == 27:  # Exit loop if ESC is pressed
            break


def train_model():
    features, labels = load_features_from_csv()
    if not features:
        print("No training data available.")
        return None
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(features, labels)
    return model

def classify_rock_type(model, cv_image):
    features = extract_hsv_features(cv_image)
    prediction = model.predict([features])
    print(f"Predicted rock type: Rock Type {prediction[0] + 1}")



def classify_region(event, x, y, flags, param):
    global training_images, training_labels
    model = train_model()
    if event == cv2.EVENT_LBUTTONDOWN:
        region = param[box_y1:box_y2, box_x1:box_x2]
        cv2.imshow("Selected Region", region)
        classify_rock_type(model, region)


def classify():
       while True:
        cv_image = getOpenCVImage()
        cv2.rectangle(cv_image, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
        cv2.putText(cv_image, "Place the rock here", (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capture Image for Classification", cv_image)
        cv2.setMouseCallback("Capture Image for Classification", classify_region, param=cv_image)
        if cv2.waitKey(1) == 27:  # Exit loop if ESC is pressed
            break
    
    

# main loop
inp = int(input("Press 1 if you want to train, 2 if you want to classify"))

if (inp == 1):
    collect_training_data()
else:
    classify()


    


# stop grabbing & release resources
cam.MV_CC_StopGrabbing()
cam.MV_CC_CloseDevice()
cam.MV_CC_DestroyHandle()
cv2.destroyAllWindows()
print("Camera resources released.")
