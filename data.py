import os
import pandas as pd
from PIL import Image

# Load your dataset
dataset_path = r'C:\Users\azadk\OneDrive\Desktop\projects\bmi_detection\bmi.csv'  # Update with the actual path
df = pd.read_csv(dataset_path)

# Paths to the image folders
sides_folder = r'C:\Users\azadk\OneDrive\Desktop\projects\bmi_detection\illinois_doc_dataset\side\side'
front_folder = r'C:\Users\azadk\OneDrive\Desktop\projects\bmi_detection\illinois_doc_dataset\front\front'

# Function to delete corrupted images
def delete_corrupted_images(folder_path):
    corrupted_ids = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        file_id, _ = os.path.splitext(filename)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify the image is not corrupted
        except (IOError, SyntaxError):
            print(f"Corrupted image found and deleted: {file_path}")
            corrupted_ids.append(file_id.strip())
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting corrupted image {file_path}: {e}")
    return corrupted_ids

# Remove corrupted images from both folders
corrupted_side_ids = delete_corrupted_images(sides_folder)
corrupted_front_ids = delete_corrupted_images(front_folder)

# Combine corrupted IDs
corrupted_ids = set(corrupted_side_ids + corrupted_front_ids)

# Check for missing images and add their IDs to the removal list
missing_ids = []
for index, row in df.iterrows():
    image_id = str(row['id']).strip()
    side_image_path = os.path.join(sides_folder, f"{image_id}.jpg")
    front_image_path = os.path.join(front_folder, f"{image_id}.jpg")
    if not os.path.exists(side_image_path) or not os.path.exists(front_image_path):
        missing_ids.append(image_id)

# Combine corrupted and missing IDs
ids_to_remove = corrupted_ids.union(missing_ids)

# Remove rows with corrupted or missing images
df = df[~df['id'].astype(str).isin(ids_to_remove)]

# Count the number of males and females
gender_counts = df['sex'].value_counts()
num_males = gender_counts.get('Male', 0)
num_females = gender_counts.get('Female', 0)

print(f"Number of males: {num_males}")
print(f"Number of females: {num_females}")

# Balance the dataset by removing extra male rows
if num_males > num_females:
    males_to_drop = num_males - num_females
    male_ids_to_drop = df[df['sex'] == 'Male'].sample(n=males_to_drop, random_state=42)['id'].tolist()
    df = df[~df['id'].isin(male_ids_to_drop)]
    ids_to_remove.update(map(str, male_ids_to_drop))
    print(f"Dropped {len(male_ids_to_drop)} male rows to balance genders.")

# Save the updated dataset
df.to_csv(dataset_path, index=False)
print("Updated dataset saved.")

# Function to delete images by IDs
def delete_images(folder_path, ids_to_delete):
    for filename in os.listdir(folder_path):
        file_id, _ = os.path.splitext(filename)
        if file_id.strip() in ids_to_delete:
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

# Delete images corresponding to removed IDs
delete_images(sides_folder, ids_to_remove)
delete_images(front_folder, ids_to_remove)

print("Image deletion process completed.")


import os
import pandas as pd
from PIL import Image

# Load your dataset
dataset_path = r'C:\Users\azadk\OneDrive\Desktop\projects\bmi_detection\bmi.csv'  # Update with the actual path
df = pd.read_csv(dataset_path)


print(df.isnull().all())
# Debugging during training
# Encode 'Sex' column as binary (if not already)
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})  # Update as per your dataset

print(df)

# Save the cleaned dataframe as a CSV file
output_path = r'C:\Users\azadk\OneDrive\Desktop\projects\bmi_detection\bmi.csv'
df.to_csv(output_path, index=False)

print(f"Cleaned dataset saved to {output_path}")
# Save the cleaned dataframe as a CSV file
output_path = r'C:\Users\azadk\OneDrive\Desktop\projects\bmi_detection\cleaned_bmi.csv'
df.to_csv(output_path, index=False)

print(f"Cleaned dataset saved to {output_path}")


# Load the CSV file
df = pd.read_csv('bmi.csv')

# Convert height to inches (assumes height is in meters)
df['height'] = df['height'] * 100

# Calculate BMI
df['BMI'] = (df['weight'] * 703) / (df['height'] ** 2)

# Save the modified DataFrame to a new CSV file
df.to_csv('bmi_with_bmi_column.csv', index=False)

print("BMI column added and file saved as 'bmi.csv'.")


