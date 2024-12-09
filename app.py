import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tkinter import Tk, filedialog

# Load the trained model
model_path = r'C:\Users\azadk\OneDrive\Desktop\projects\bmi_detection\log\model_epoch_07_val_loss_29.54.keras'
model = load_model(model_path)
print("Model loaded successfully.")

# Function to preprocess a single image
def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))
        return img_array[0]
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        return None

# Function to categorize BMI
def categorize_bmi(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal"
    else:
        return "Overweight"

# Function to predict BMI and gender
def predict_bmi_and_sex(front_image_path, side_image_path):
    # Preprocess images
    front_image = preprocess_image(front_image_path)
    side_image = preprocess_image(side_image_path)

    if front_image is None or side_image is None:
        print("One or both images could not be processed.")
        return None, None, None

    # Expand dimensions to match batch size
    front_image = np.expand_dims(front_image, axis=0)
    side_image = np.expand_dims(side_image, axis=0)

    # Predict
    predictions = model.predict([front_image, side_image])
    bmi_prediction = predictions[0][0][0]
    sex_prediction = 1 if predictions[1][0][0] > 0.5 else 0

    # Decode gender
    sex_decoded = 'Male' if sex_prediction == 1 else 'Female'
    bmi_category = categorize_bmi(bmi_prediction)

    return bmi_prediction, sex_decoded, bmi_category

# Main function
if __name__ == "__main__":
    # Use tkinter file dialog to select images
    Tk().withdraw()  # Hide the root window
    print("Select the front view image.")
    front_image_path = filedialog.askopenfilename(title="Select Front View Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

    print("Select the side view image.")
    side_image_path = filedialog.askopenfilename(title="Select Side View Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])

    # Check if images were selected
    if front_image_path and side_image_path:
        print(f"Selected Front Image: {front_image_path}")
        print(f"Selected Side Image: {side_image_path}")

        # Predict
        bmi_prediction, sex_decoded, bmi_category = predict_bmi_and_sex(front_image_path, side_image_path)
        if bmi_prediction is not None and sex_decoded is not None and bmi_category is not None:
            print(f"Predicted BMI: {bmi_prediction:.2f}")
            print(f"Predicted Gender: {sex_decoded}")
            print(f"BMI Category: {bmi_category}")
        else:
            print("Prediction could not be completed due to missing or invalid images.")
    else:
        print("Image selection cancelled.")
