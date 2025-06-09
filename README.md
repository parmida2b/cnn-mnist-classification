# 🔢 CNN-Based Digit Classification on MNIST (Keras & TensorFlow)

This project uses a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using Keras and TensorFlow. The model is trained to recognize digits 0 through 9 with high accuracy and is evaluated using both metrics and a confusion matrix.

---

## 📌 Overview

- 🔍 Loaded and preprocessed the MNIST dataset (grayscale, 28x28)
- 🔄 Normalized pixel values and one-hot encoded labels
- 🧠 Designed and trained a CNN with multiple Conv2D, Dropout, and BatchNormalization layers
- 📉 Tracked training & validation accuracy/loss
- 🔎 Evaluated model performance with a confusion matrix
- 💾 Saved the trained model for future inference

---

## 🧰 Tools & Libraries

- Python 3  
- TensorFlow & Keras  
- NumPy & Pandas  
- Matplotlib & Seaborn  
- scikit-learn

---

## 📊 Dataset

**MNIST**: 70,000 images of handwritten digits  
- 60,000 training samples  
- 10,000 test samples  
- Each image is 28x28 grayscale

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

- Training Accuracy: ~99%  
- Validation Accuracy: ~98%  
- Confusion Matrix to evaluate per-class performance  
- Plotted training history for accuracy/loss

---

## 📁 Files

- `mnist_cnn.ipynb`: Complete notebook (Google Colab compatible)  
- `CNN_mnist.h5`: Trained model file  
- *(Optional)* `accuracy_loss_plot.png`, `confusion_matrix.png`

---

## ☁️ How to Run

1. Open [Google Colab](https://colab.research.google.com/)  
2. Upload `mnist_cnn.ipynb`  
3. Run all cells  
4. (Optional) Save the trained model using `model.save()`

---

## 👩‍💻 Author

**Parmida Mohammadzadeh**  
M.Sc. in Data Science | Computer Vision Learner  
📍 Based in Iran
