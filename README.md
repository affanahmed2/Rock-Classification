# Rock-Classification
Rock classification project at Aremak

# Rock Classification Using Machine Vision

## 1️⃣ Project Overview
This project is designed to classify different types of rocks using a Hikrobot industrial camera and computer vision techniques. The system captures images of rocks, extracts their features using HSV color space, and classifies them using a K-Nearest Neighbors (KNN) machine learning model.

### Key Objectives:
- Capture images of rocks using a Hikrobot camera.
- Extract HSV color features from the captured images.
- Train a KNN classifier with labeled rock data.
- Classify unknown rocks based on the trained model.
- Provide an interactive interface for users to label training images and classify rocks in real-time.

## 2️⃣ Equipment and Technologies Used

### Hardware Components:
-  **Camera Model:** Hikrobot MV-CS060-10UC-PRO
-  **Lens:** MVL-HF0828M-6MPE
-  **Camera Stand:** Aremak Adjustable Machine Vision Test Stand
-  **Lighting:** Hikrobot Bar light (MV-LLDS-H-250-40-W)
-  **Operating System:** Windows

### Software Tools:
-  **Programming Language:** Python
-  **Libraries:** OpenCV, NumPy, Scikit-learn, CSV
-  **SDK:** Hikrobot MVS SDK

## 3️⃣ Setup Photos 📸
![Setup Image](Images/Setup.jpg)

## 4️⃣ Installation and Running Instructions 
### Installation:
Ensure you have Python installed (preferably version 3.7+), then install the necessary dependencies:

```sh
pip install opencv-python numpy scikit-learn
```

### Running the Project:
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/rock-classification.git
   cd rock-classification
   ```
2. Connect the Hikrobot camera and ensure the drivers are installed.
3. Run the script:
   ```sh
   python Classification.py
   ```
4. Follow the on-screen instructions to train the model or classify rocks.

## 5️⃣ Code Documentation 

Example function documentation:
```python
def extract_hsv_features(image):
    """
    Extracts the average HSV values from an image.

    Parameters:
    image (numpy array): Input image in BGR format.

    Returns:
    numpy array: Average HSV values.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(hsv_image, axis=(0, 1))
    return avg_hsv
```
For more details, check the inline comments in the source code.

## 6️⃣ Internship Acknowledgment
---
🏢 This project was developed during an internship at [Aremak Bilişim Teknolojileri](https://www.aremak.com.tr) under the supervision of Emrah Bala.


