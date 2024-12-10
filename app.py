import streamlit as st
import tensorflow as tf
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torch
from torchvision import models
import tempfile

# Define the model architecture (example: ResNet18)


# Load the models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Class labels for binary classification
class_labels = ["bad", "good"]
cnn="/home/hari07/workspace/intern/models/cnn.h5"
resnet='/home/hari07/workspace/intern/models/resnet_binary_classification.pth'
yolo_cls='/home/hari07/workspace/intern/models/classify_best.pt'
yolo_obj='/home/hari07/workspace/intern/yolo/yolo_object_detection/runs/detect/train/weights/best.pt'
cnn_model = tf.keras.models.load_model(cnn)

yolo_class_model = YOLO(yolo_cls)
yolo_obj_model = YOLO(yolo_obj)

# Load the pretrained ResNet model
resnet_model = models.resnet18(pretrained=False)
num_features = resnet_model.fc.in_features
resnet_model.fc = torch.nn.Linear(num_features, 2)  # Assuming binary classification
resnet_model.load_state_dict(torch.load(resnet, map_location=torch.device('cpu')))
resnet_model = resnet_model.to(device)
resnet_model.eval() 


def edge_detection(image, area_threshold=500):
    
    image = cv2.imread(image)
    if image is None:
        print("Failed to load the image. Check the file path.")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform adaptive edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Perform morphological operations to close small gaps
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours of potential scratches
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area to focus on scratches
    scratch_mask = np.zeros_like(gray)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 10 < area < area_threshold:  # Filter small and large non-scratch regions
            cv2.drawContours(scratch_mask, [contour], -1, 255, -1)
    
    # Calculate the percentage of scratch area
    scratch_area = np.sum(scratch_mask > 0)
    total_area = gray.shape[0] * gray.shape[1]
    scratch_percentage = (scratch_area / total_area) * 100

    # Classify based on scratch percentage
    classification = "Bad" if scratch_percentage > 0.5 else "Good"
    

    text=f"Classification: {classification}\n\n"f"Scratch Area: {scratch_area}\n\n"f"Scratch Percentage: {scratch_percentage:.2f}%"
    return text,scratch_mask,edges,image  # Return the scratch mask for further use

def classify_with_cnn(image):
    img_height = 180
    img_width = 180

    # Load and preprocess the image
    img = image.resize((img_height, img_width)) 
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make a prediction
    predictions = cnn_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Assuming `class_names` is defined as the list of your class names
    class_names = ["bad", "good"]  # Update to match your model's class names

    # Print the prediction
    result="This image most likely belongs to '{}' with {:.2f}% confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    return result,img





def classify_with_resnet(image):
    # Step 1: Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Step 2: Apply transformations and add batch dimension
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Step 3: Perform inference (no gradient computation)
    with torch.no_grad():
        outputs = resnet_model(input_tensor)

    # Step 4: Convert logits to probabilities and get predicted class
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

    # Get predicted label and confidence score for the predicted class
    predicted_label = class_labels[predicted_class.item()]
    confidence_score = probabilities[0, predicted_class.item()].item()  # Confidence score of the predicted label

    # Print results
    print(f"Predicted Class: {predicted_label}")
    print(f"Confidence Score: {confidence_score * 100:.2f}%")
    
    return image ,predicted_label, confidence_score


def classify_with_yolo(image):
    results = yolo_class_model(image)
    
    # Print results to understand its structure
    print(type(results))  # Check the type of the results object
    print(results)        # Print the actual content of the results

    # After inspecting, proceed with the appropriate handling of the results object
    probs = results[0].probs
    conf= probs.top1conf  
    label=probs.top1 
    
    class_lab=class_labels[label] # Confidence scores are in the second last column
    
    print(probs)



    return conf ,class_lab


def detect_with_yolo(image):
    results = yolo_obj_model(image,conf=0.1)
    boxes = results[0].boxes  # Extract bounding boxes
    # print(results)
    print(boxes)
    return boxes, results[0].plot()

# Streamlit UI
st.title('Model Prediction Application')

# Choose model
model_choice = st.selectbox("Select a Model", [
    "CNN (Keras)",
    "Edge Detection",
    "ResNet Binary Classification",
    "YOLO Classification",
    "YOLO Object Detection"
])

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    image = Image.open(uploaded_image)
    if uploaded_image:
    # Save the uploaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(uploaded_image.getvalue())  # Write the uploaded file to the temporary file
            temp_file_path = temp_file.name  

    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Run the appropriate model based on the user's selection
    if model_choice == "CNN (Keras)":
        st.write("Running CNN Model for Classification...")
        result,img  = classify_with_cnn(image)
        st.write(result)
        st.image(img, caption='classifed image', use_column_width=True)

    elif model_choice == "Edge Detection":
        st.write("Running Edge Detection...")
        image = temp_file_path  # Load the image from Streamlit uploader
        text,scratch_mask,edges,img = edge_detection(image,area_threshold=800)  # Call the updated edge_detection function
        st.write(text)
        st.image(img, caption='Actual Image', use_column_width=True)
        st.image(edges,caption='Edegs Image', use_column_width=True)
        # st.image(scratch_mask,caption='Scrached Mask', use_column_width=True)


    elif model_choice == "ResNet Binary Classification":
        st.write("Running ResNet Binary Classification...")
        image,predicted_label, confidence_scores = classify_with_resnet(image)
        st.image(image,caption=f"{predicted_label} with {confidence_scores * 100:.2f}")
        st.write(f"Predicted Class: {predicted_label}")

    elif model_choice == "YOLO Classification":
        st.write("Running YOLO Classification...")
        conf ,label = classify_with_yolo(temp_file_path)
        st.image(image,caption=f"Classifed image  as {label}",use_column_width=True)
        st.write(f"{label} with {conf*100:.2f} of the image")

    elif model_choice == "YOLO Object Detection":
        st.write("Running YOLO Object Detection...")
        
        image_np = np.array(image)
    # Perform detection
        boxes, detected_image = detect_with_yolo(image_np)

        # Display the results
        st.image(detected_image, caption="Detected Image", use_column_width=True)
        st.write(f"Detected boxes: {len(boxes)}")
        for box in boxes:
            st.write(f"Box: {box}")