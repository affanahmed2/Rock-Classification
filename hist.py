import sys
import ctypes
import numpy as np
import cv2
import os
import csv
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize


sys.path.append("MvImport")
from MvCameraControl_class import *


### GLOBAL VARIABLES
edge = 100
box_x1, box_y1, box_x2, box_y2 = 600, 600, 700, 700
csv_file = "histogram.csv"

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
cam.MV_CC_SetFloatValue("ExposureTime", 80000.0)  # eet exposure time
cam.MV_CC_SetEnumValue("GainAuto", 0)  # enable auto gain

# Collect Training Data
rock_types = ['rock_type_1', 'rock_type_2', 'rock_type_3']  # Replace with actual rock types
training_images = []
training_labels = []


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


 # Function to compute HSV histograms
def extract_hsv_histogram_classify(image):
     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
     h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 179])
     s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
     v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
     hist_features = np.concatenate([h_hist, s_hist, v_hist]).flatten()
     return normalize(hist_features.reshape(1, -1))[0]  # Normalize


def extract_hsv_histogram_train(image, class_name, image_name, output_dir="histograms"):
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Compute histograms for H, S, and V
    h_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 179])  # Hue
    s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])  # Saturation
    v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])  # Value

    # Normalize histograms for feature extraction
    hist_features = np.concatenate([h_hist, s_hist, v_hist]).flatten()
    normalized_features = normalize(hist_features.reshape(1, -1))[0]

    # Create output directory
    class_folder = os.path.join(output_dir, class_name)
    os.makedirs(class_folder, exist_ok=True)

    # Plot and save histograms
    plt.figure(figsize=(10, 4))
    
    # Hue Histogram
    plt.subplot(1, 3, 1)
    plt.plot(h_hist, color='red')
    plt.title("Hue Histogram")
    
    # Saturation Histogram
    plt.subplot(1, 3, 2)
    plt.plot(s_hist, color='green')
    plt.title("Saturation Histogram")
    
    # Value Histogram
    plt.subplot(1, 3, 3)
    plt.plot(v_hist, color='blue')
    plt.title("Value Histogram")

    # Save the figure
    hist_path = os.path.join(class_folder, f"{image_name}_hist.png")
    plt.savefig(hist_path)
    plt.close()

    return normalized_features  


# Save features and labels to CSV
def save_to_csv(dataset_path="dataset"):
    features, labels = [], []
    class_map = {"class_1": 0, "class_2": 1, "class_3": 2}
    
    for class_name, label in class_map.items():
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            print(f"Skipping folder {class_path} as it's missing.")
            continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            if image is not None:
                hist_features = extract_hsv_histogram_train(image, class_name, img_name)
                features.append(hist_features)
                labels.append(label)
    
    # Save to CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["feature_" + str(i) for i in range(len(features[0]))] + ["label"])
        for feature, label in zip(features, labels):
            writer.writerow(list(feature) + [label])
    print(f"Data saved to histogram.csv")
    


# Load features and labels from CSV
def load_from_csv(csv_file):
    features, labels = [], []
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header
        for row in reader:
            features.append([float(x) for x in row[:-1]])  # All except the last column
            labels.append(int(row[-1]))  # Last column is the label
    return np.array(features), np.array(labels)

# Train KNN model
def train_model():
    X, y = load_from_csv(csv_file)
    if len(X) == 0:
        print("No training data available.")
        return None
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    return model


def train_and_evaluate_model():
    X, y = load_from_csv(csv_file)
    if len(X) == 0:
        print("No training data available.")
        return None

    # Split data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train KNN model
    for k in [1,2, 3, 4, 5,6, 7,8,9,10]:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        # Test accuracy on test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"K = {k}, Model Accuracy on Test Set: {accuracy * 100:.2f}%")

    return model


def classify_rock_type(model, cv_image):
    features = extract_hsv_histogram_classify(cv_image)
    prediction = model.predict([features])[0]
    print(f"Predicted rock type: Rock Type {prediction + 1}")


def classify_rock(event, x, y, flags, param):
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
        cv2.setMouseCallback("Capture Image for Classification", classify_rock, param=cv_image)
        if cv2.waitKey(1) == 27:  # Exit loop if ESC is pressed
            break


# main loop
inp = int(input("Press 1 if you want to train, 2 if you want to classify, 3 to test accuracy"))

if (inp == 1):
    save_to_csv()
elif (inp == 2):
    classify()
else:
    train_and_evaluate_model()


# Cleanup
cam.MV_CC_StopGrabbing()
cam.MV_CC_CloseDevice()
cam.MV_CC_DestroyHandle()
cv2.destroyAllWindows()    