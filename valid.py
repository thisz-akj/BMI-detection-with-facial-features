import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# Paths
model_path = r'C:\Users\azadk\OneDrive\Desktop\projects\bmi_detection\log\model_epoch_07_val_loss_29.54.keras'
bmi_csv_path = r'C:\Users\azadk\OneDrive\Desktop\projects\bmi_detection\bmi.csv'
front_image_dir = r'C:\Users\azadk\OneDrive\Desktop\projects\bmi_detection\illinois_doc_dataset\front\front'
side_image_dir = r'C:\Users\azadk\OneDrive\Desktop\projects\bmi_detection\illinois_doc_dataset\side\side'

# Load the model
model = load_model(model_path)

# Load BMI data
bmi_df = pd.read_csv(bmi_csv_path)

# Function to preprocess a single image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return preprocess_input(image_array)

# Predict BMI and sex
def predict_bmi_and_sex(front_image_path, side_image_path):
    front_image = preprocess_image(front_image_path)
    side_image = preprocess_image(side_image_path)
    predictions = model.predict([front_image, side_image])
    predicted_bmi = predictions[0][0][0]
    predicted_sex_prob = predictions[1][0][0]
    predicted_sex = 'Male' if predicted_sex_prob >= 0.5 else 'Female'
    return predicted_bmi, predicted_sex

# Categorize BMI
def grade_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    else:
        return 'Overweight'

# Test on 720 random IDs
random_ids = bmi_df.sample(n=300)
bmi_true = []
bmi_pred = []

print("Testing on 720 random IDs from bmi.csv:")
for _, row in random_ids.iterrows():
    front_image_path = os.path.join(front_image_dir, f"{row['id']}.jpg")
    side_image_path = os.path.join(side_image_dir, f"{row['id']}.jpg")
    true_bmi = row['BMI']
    try:
        predicted_bmi, predicted_sex = predict_bmi_and_sex(front_image_path, side_image_path)
        bmi_true.append(true_bmi)
        bmi_pred.append(predicted_bmi)
    except Exception as e:
        print(f"Error processing ID {row['id']}: {e}")

# Compute metrics
mae = mean_absolute_error(bmi_true, bmi_pred)
mse = mean_squared_error(bmi_true, bmi_pred)
r2 = r2_score(bmi_true, bmi_pred)
pearson_corr, _ = pearsonr(bmi_true, bmi_pred)


print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")
print(f"Pearson Coefficient: {pearson_corr}")