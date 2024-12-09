
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint

# Load the BMI dataset
dataset_path = r'C:\Users\azadk\OneDrive\Desktop\projects\bmi_detection\bmi.csv'
df = pd.read_csv(dataset_path)

# Directories for images
sides_folder = r'C:\Users\azadk\OneDrive\Desktop\projects\bmi_detection\illinois_doc_dataset\side\side'
front_folder = r'C:\Users\azadk\OneDrive\Desktop\projects\bmi_detection\illinois_doc_dataset\front\front'

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Data Generator
class ImageDataGeneratorBMI(Sequence):
    def __init__(self, dataframe, front_folder, side_folder, batch_size=32, shuffle=True):
        self.dataframe = dataframe
        self.front_folder = front_folder
        self.side_folder = side_folder
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        # Generate indices for the batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.dataframe.iloc[batch_indices]

        # Initialize arrays for images and labels
        front_images = []
        side_images = []
        labels = []

        for _, row in batch_df.iterrows():
            id = row['id']
            bmi = row['BMI']

            # Load and preprocess images
            front_path = os.path.join(self.front_folder, f'{id}.jpg')
            side_path = os.path.join(self.side_folder, f'{id}.jpg')

            try:
                front_img = load_img(front_path, target_size=(224, 224))
                front_img = preprocess_input(np.expand_dims(img_to_array(front_img), axis=0))

                side_img = load_img(side_path, target_size=(224, 224))
                side_img = preprocess_input(np.expand_dims(img_to_array(side_img), axis=0))

                front_images.append(front_img[0])
                side_images.append(side_img[0])
                labels.append(bmi)
            except FileNotFoundError:
                continue  # Skip missing images

        # Return as tuple of numpy arrays
        return (np.array(front_images), np.array(side_images)), np.array(labels)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indices)

# Create generators
train_generator = ImageDataGeneratorBMI(train_df, front_folder, sides_folder, batch_size=32)
val_generator = ImageDataGeneratorBMI(val_df, front_folder, sides_folder, batch_size=32, shuffle=False)

# Load VGG16 model without the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the VGG16 layers
for layer in base_model.layers:
    layer.trainable = False

# Define the input layers for the front and side images
input_front = Input(shape=(224, 224, 3))
input_side = Input(shape=(224, 224, 3))

# Get VGG16 feature maps for front and side images
vgg_front = base_model(input_front)
vgg_side = base_model(input_side)

# Flatten the outputs of VGG16
vgg_front = Flatten()(vgg_front)
vgg_side = Flatten()(vgg_side)

# Concatenate both image features
merged = Concatenate()([vgg_front, vgg_side])

# Add custom fully connected layers
fc1 = Dense(512, activation='relu')(merged)
fc2 = Dense(256, activation='relu')(fc1)
output = Dense(1, activation='linear')(fc2)

# Create the model
model = Model(inputs=[input_front, input_side], outputs=output)

# Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

# Create a directory for saving models if it doesn't exist
log_dir = 'log1'
os.makedirs(log_dir, exist_ok=True)

# Correct filepath for ModelCheckpoint with `.keras` extension
checkpoint_filepath = os.path.join(log_dir, 'model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.keras')

# Updated ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    save_best_only=False,  # Save every epoch
    save_weights_only=False,  # Save the entire model
    mode='auto',
    verbose=1
)

# Train the model with the callback
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[checkpoint_callback]  # Add the callback here
)

# Evaluate the model on the validation set
loss, mae = model.evaluate(val_generator)
print(f'Validation loss: {loss}, Validation MAE: {mae}')
