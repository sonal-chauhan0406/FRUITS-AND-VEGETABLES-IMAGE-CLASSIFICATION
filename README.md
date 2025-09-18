# 🥦🍎 Fruits & Vegetables Image Classification

A Convolutional Neural Network (CNN) based deep learning project to classify images of fruits and vegetables into 36 categories. The model is trained, validated, and tested on a well-structured dataset split into training, validation, and testing sets.<br>

# 🚀 Project Overview

The goal of this project is to build an image classification model that can accurately identify different fruits and vegetables from images. The model is implemented using TensorFlow & Keras, trained on labeled datasets, and tested for performance on unseen images.<br>

✅ Built a CNN model from scratch<br>
✅ Supports 36 fruit & vegetable classes<br>
✅ Achieves high accuracy on test data<br>
✅ Works with single image predictions as well as batch testing<br>
✅ Visualizes predictions with Matplotlib<br>

# 📂 Dataset

The dataset was manually organized into three subsets:<br>

Training set → Used to train the model<br>

Validation set → Used to tune hyperparameters and avoid overfitting<br>

Test set → Used for evaluating final model performance<br>

Number of classes: 36<br>
Examples: apple, banana, beetroot, carrot, tomato, watermelon, etc.<br>

# 🧑‍💻 Tech Stack

Python 3.9+

TensorFlow / Keras (Deep Learning Framework)

NumPy (Numerical Computation)

Matplotlib (Data Visualization)

OpenCV (Image Processing)

# 🏗️ Model Architecture

The model is a Convolutional Neural Network (CNN) built using Keras’ Sequential API.

Input Layer → (64x64 RGB images)

Convolution + Pooling Layers

Fully Connected Dense Layers

Softmax Output Layer (36 neurons for 36 classes)

# 📊 Results

Model tested on 359 images across 36 classes.

Example prediction:<br>

It's a apple
It's a beetroot
