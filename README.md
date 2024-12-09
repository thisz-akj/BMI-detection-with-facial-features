Here's the updated README with the correct repository name:  

---

# BMI-Detection-with-Facial-Features

This project uses a deep learning model to predict a person's BMI (Body Mass Index) and gender based on two input images: a front-view and a side-view image. The model leverages a pre-trained VGG16 network for feature extraction and has been fine-tuned for this specific task.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Output Results](#output-results)
- [Model Details](#model-details)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project predicts:
1. **BMI (Body Mass Index)**: A numerical value indicating whether a person is underweight, normal, or overweight.
2. **Gender**: Categorized as male or female based on the input images.

The model processes two images: a front view and a side view. Predictions are based on features extracted using a convolutional neural network.

---

## Features
- **Two-Image Input**: Utilizes front and side view images for accurate prediction.
- **BMI Categorization**: Categorizes BMI as:
  - Underweight
  - Normal
  - Overweight
- **User-Friendly Interface**: Supports selecting images using a graphical file dialog.
- **Expandable**: The model can be further trained or integrated into larger systems.

---

## Installation

### Prerequisites
- Python 3.7 or later
- Required Python libraries:
  - TensorFlow
  - NumPy
  - Pillow
  - Tkinter (for GUI)
  
### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/BMI-detection-with-facial-features.git
   cd BMI-detection-with-facial-features
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the trained model and save it to the appropriate path:
   ```
   C:\Users\azadk\OneDrive\Desktop\projects\bmi_detection\log\model_epoch_07_val_loss_29.54.keras
   ```

---

## Usage

1. Run the script:
   ```bash
   python bmi_gender_prediction.py
   ```

2. A dialog will prompt you to select two images:
   - **Front View Image**
   - **Side View Image**

3. The script will display:
   - Predicted BMI
   - BMI Category (Underweight, Normal, Overweight)
   - Predicted Gender (Male or Female)

4. Example Output:

   ![A02244](https://github.com/user-attachments/assets/8f5c48cc-242c-4cfd-8709-8026cd8e863e)
   ![A02244](https://github.com/user-attachments/assets/d497b7d7-dda4-42fd-8ba5-a54f4aab8349)


   ```
   Predicted BMI: 23.45
   Predicted Gender: Male
   BMI Category: Over Weight
   ```

---

## Output Results

After running the script, the output is displayed in the terminal. Here’s a breakdown of what each output represents:

1. **Predicted BMI**:
   - A numerical value, such as `23.45`, representing the predicted Body Mass Index.

2. **BMI Category**:
   - Describes the category based on the BMI:
     - `Underweight`: BMI < 18.5
     - `Normal`: 18.5 ≤ BMI < 25
     - `Overweight`: BMI ≥ 25

## Model Details

- **Architecture**: Based on VGG16 for feature extraction.
- **Input Size**: 224x224 pixels for both front and side view images.
- **Outputs**:
  - BMI (continuous value)
  - Gender (binary classification: Male/Female)

---

## Limitations
- **Image Quality**: Low-resolution or poorly-lit images may affect predictions.
- **Bias**: The model's accuracy may vary based on the dataset it was trained on.
- **Generalization**: Predictions may not generalize well to certain populations or body types not represented in the training data.

---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to your branch and create a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

--- 

Let me know if further edits are needed!
