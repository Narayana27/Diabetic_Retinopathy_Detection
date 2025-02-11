# Diabetic Retinopathy Detection using ResNet-18  
This project aims to detect and classify Diabetic Retinopathy (DR) using deep learning techniques. A ResNet-18 model was built and trained on an image dataset with five classes representing different stages of diabetic retinopathy: **No_DR, Mild, Moderate, Severe, and Proliferate_DR**. The goal is to provide accurate predictions to assist in the early detection and treatment of diabetic retinopathy.

---

## Overview
### Objective  
- To classify images into one of the five categories of diabetic retinopathy using a ResNet-18 deep learning architecture.
- Achieve high accuracy and precision by leveraging convolutional neural networks (CNNs).

### Dataset  
- **Training Data Distribution:**  
  - Mild: 370 images  
  - Moderate: 1000 images  
  - No_DR: 1805 images  
  - Proliferate_DR: 295 images  
  - Severe: 193 images  

### Final Model Performance  
- **Accuracy:** 84%  
- **Precision, Recall, F1-score (Class-wise):**

| Class            | Precision | Recall | F1-score | Support |
|------------------|-----------|--------|----------|---------|
| Mild             | 0.83      | 0.56   | 0.67     | 85      |
| Moderate         | 0.74      | 0.86   | 0.79     | 212     |
| No_DR            | 0.95      | 0.97   | 0.96     | 337     |
| Proliferate_DR   | 0.67      | 0.65   | 0.66     | 54      |
| Severe           | 0.81      | 0.56   | 0.66     | 45      |

---

## Methodology
### Step 1: Importing Required Libraries and Dataset  
We utilized the following libraries:
- **Python Libraries**: Pandas, Numpy, Matplotlib, Seaborn, Sklearn, PIL, Keras, TensorFlow  

The dataset was loaded from Google Drive and processed using Python’s `os` module.

### Step 2: Data Exploration and Preprocessing  
- Checked the number of images and distribution across different classes.  
- Implemented data augmentation to improve model generalization.

### Step 3: Building the Model (ResNet-18 Architecture)  
- The ResNet-18 model was implemented using Keras and TensorFlow.  
- **Input shape**: (256, 256, 3)  
- **Architecture Highlights**:  
  - Zero padding  
  - Convolutional layers with Batch Normalization and ReLU activation  
  - MaxPooling and AveragePooling layers  
  - Fully connected layer with softmax activation for multi-class classification  

### Step 4: Model Training  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Metrics:** Accuracy  

The model was trained for **2 epochs** with early stopping and checkpointing.

---

## Results  
- Achieved an overall accuracy of **84%** on the validation set.  
- The **No_DR class** had the highest F1-score of 0.96, while **Mild and Severe classes** showed room for improvement due to imbalanced data.  
- The model's macro and weighted averages indicate solid performance across most classes.

---

## Conclusion  
This project demonstrates that ResNet-18 can effectively classify diabetic retinopathy stages from retinal images. Future improvements could focus on:
- Balancing the dataset to improve recall for underrepresented classes.
- Fine-tuning the model with more epochs and advanced data augmentation techniques.
- Exploring other architectures like ResNet-50 or EfficientNet for enhanced accuracy.

