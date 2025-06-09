# 🔢 CNN-Based Digit Classification on MNIST (Keras & TensorFlow)

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) from the MNIST dataset using Keras and TensorFlow. The model achieves high training and validation accuracy and is evaluated using both performance metrics and a confusion matrix.

---

## 📌 Overview

- 🔍 Loaded and preprocessed the MNIST dataset (28×28 grayscale images)
- 🔄 Normalized pixel values and one-hot encoded labels
- 🧠 Designed and trained a CNN with Conv2D, Dropout, BatchNormalization, and MaxPooling layers
- 📉 Tracked training/validation accuracy and loss over 10 epochs
- 🔎 Evaluated predictions using a confusion matrix
- 💾 Saved the trained model (`.h5`) for later use

---

## 🧰 Tools & Libraries

- Python 3  
- TensorFlow & Keras  
- NumPy & Pandas  
- Matplotlib & Seaborn  
- scikit-learn

---

## 📊 Dataset

**MNIST**: 70,000 grayscale images of handwritten digits  
- 60,000 for training  
- 10,000 for testing  
- Each image: 28×28 pixels, 1 color channel (grayscale)

---

## 🧠 Model Architecture

Input (28x28x1)
↓
Conv2D(32) + ReLU + Dropout + BatchNorm
↓
Conv2D(64) + ReLU + MaxPooling + Dropout + BatchNorm
↓
Conv2D(64) + ReLU + MaxPooling + Dropout + BatchNorm
↓
Conv2D(128) + ReLU + MaxPooling + Dropout + BatchNorm
↓
Flatten
↓
Dense(32) + ReLU + Dropout + BatchNorm
↓
Dense(10) + Softmax

---

## 📈 Evaluation

- ✅ Training Accuracy: ~99%  
- ✅ Validation Accuracy: ~98%  
- 🔍 Confusion Matrix to assess per-class performance  
- 📊 Plotted training history for accuracy and loss

---

## 📁 Files

- `mnist_cnn.ipynb`: Complete training and evaluation notebook  
- `CNN_mnist.h5`: Trained CNN model file  
- *(Optional)*: `accuracy_loss_plot.png`, `confusion_matrix.png`

---

## ☁️ How to Run (in Google Colab)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `mnist_cnn.ipynb`
3. Run all cells in sequence
4. (Optional) Save the trained model using `model.save('CNN_mnist.h5')`

---

## 👩‍💻 Author

**Parmida Mohammadzadeh**  
M.Sc. in Data Science | Computer Vision Learner  
📍 Based in Iran
