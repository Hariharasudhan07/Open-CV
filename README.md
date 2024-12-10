# Open-CV
OpenCV intern task by mowito 

# Image Scratch Detection

## Objective
The goal of this project is to train a model capable of detecting images with scratches on text and classifying them as either good (clear text) or bad (scratched text). Bonus features include creating bounding boxes or masks around scratches, as well as providing user-defined thresholds for scratch detection.

## Dataset
The dataset is provided privately via Google Drive. It contains two types of images:
- **Good Images**: Clear text.
- **Bad Images**: Text with scratches.

You can download the dataset from the following links:
- **Non-Zipped Folder**: [Link]
- **Zipped Folder**: [Link]



You can aslo try my model using this link : [https://open-cv.streamlit.app/]
### Evaluation Requirement
The model needs to:
1. Classify input images into good (clear text) or bad (scratched text).
2. Optionally, create bounding boxes or masks on the detected scratches.

## Approaches Explored

### 1. Classical CNN-Based Approach
**Method**:
- A simple Convolutional Neural Network (CNN) was implemented with layers designed for feature extraction.
- The model was trained using binary classification for good vs. bad images.

**Results**:
- **Accuracy**: ~80%
- **Recall for bad images**: ~85%
- **Precision**: ~65%

**Observations**:
- The model performed well on simpler cases but struggled with subtle scratches.
- Overfitting was observed when adding more convolutional layers without augmentation.

![Loss images](/assets/classify.png)

---

### 2. Edge Detection + Contour Detection
**Method**:
- Applied **Canny edge detection** to identify strong edges in the images.
- Used **contour detection** to isolate potential scratches and classify images based on contour density and size metrics.

**Results**:
- **Scratch Detection Accuracy**: ~70%

**Observations**:
- This approach worked well for large, prominent scratches.
- It struggled with smaller or subtle scratches due to noise.

---

### 3. ResNet Classification
**Method**:
- Fine-tuned a **ResNet18** model pre-trained on ImageNet for binary classification of good vs. bad images.

**Results**:
- **Accuracy**: ~98.05%
- **Recall for bad images**: ~99%
- **Precision**: ~98%

**Observations**:
- **Transfer learning** improved performance significantly.
- The model was sensitive to the small dataset size, requiring careful augmentation to avoid overfitting.

![Confussion matix](/assets/resnet.png)
---

### 4. YOLOv11 Classification
**Method**:
- Fine-tuned **YOLOv11** for binary classification while focusing on localizing scratches and classifying images simultaneously.

**Results**:
- **Accuracy**: ~95%
- **Recall for bad images**: ~94%
- **Precision**: ~83%

**Observations**:
- Strong performance in localizing scratches and classifying images.
- Faster inference compared to ResNet.

---

### 5. YOLOv11 Object Detection
**Method**:
- Fine-tuned **YOLOv11** for object detection, treating scratches as objects and generating bounding boxes around scratches.

**Results**:
- **Scratch Detection mAP50-95**: ~95%
- **Recall for bad images**: ~98%
- **Precision**: ~92%

**Observations**:
- Excellent performance in generating bounding boxes for detected scratches.
- Some false positives for artifacts resembling scratches.

---

### 6. VGG-16 (Pytorch)
**Method**:
- A **VGG-16** model was used for scratch detection but struggled due to limited hardware resources. The model was trained on image size `(128, 128)` to optimize training time.

**Observations**:
- Training was not performed without access to a proper GPU, which affected the overall training efficiency.

---

## Bonus Features Explored

### Bounding Box Creation
- **YOLOv11 object detection** was utilized to generate bounding boxes around detected scratches.

### Scratch Threshold Control
- Implemented a threshold for scratch size to classify images as bad.
- Added a **user-configurable slider** to adjust the sensitivity of scratch detection.

### Dataset Augmentation
- **Synthetic Scratches** were generated using texture overlays.
- Combined synthetic scratches with traditional augmentation techniques, including:
  - Rotation
  - Flipping
  - Noise addition

---


