Breast Cancer Prediction using Deep Learning

## Project Description

This project implements a Deep Learning model to classify breast tumors as Malignant or Benign using the Breast Cancer Wisconsin dataset. The goal is to demonstrate how machine learning techniques can assist in early detection and classification of cancer based on diagnostic features.


## Objectives

* Analyze breast cancer dataset
* Build a neural network for classification
* Evaluate model performance
* Predict tumor type using new input data


## Dataset Information

* Dataset: Breast Cancer Wisconsin Dataset
* Source: `sklearn.datasets.load_breast_cancer()`
* Total Samples: 569
* Features: 30 numerical features
* Target Classes:

  * 0 → Malignant
  * 1 → Benign


## Data Analysis

The dataset was explored using the following steps:

* Checked shape and structure of the data
* Verified absence of missing values
* Generated statistical summary using `.describe()`
* Checked class distribution using `value_counts()`
* Compared feature means using `groupby()`


## Data Preprocessing

* Separated features (X) and labels (Y)
* Split dataset into training (80%) and testing (20%) sets
* Standardized features using `StandardScaler`

Standardization ensures that all features contribute equally to model training.


## Model Architecture

A Sequential Neural Network model was built using TensorFlow/Keras:

* Input Layer: 30 features
* Flatten Layer
* Dense Layer: 20 neurons with ReLU activation
* Output Layer: 2 neurons with Sigmoid activation


## Model Compilation

* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Metrics: Accuracy


## Model Training

* Epochs: 10
* Validation Split: 0.1

The model learns patterns from training data and validates performance during training.


## Visualization

The following graphs were plotted:

* Training vs Validation Accuracy
* Training vs Validation Loss

These graphs help in understanding model performance and detecting overfitting or underfitting.


## Model Evaluation

The trained model was evaluated on test data:

```python
loss, accuracy = model.evaluate(X_test_std, Y_test)
```

Accuracy represents how well the model performs on unseen data.

---

## Prediction

The model outputs probability values for both classes. Example:

```python
[0.25, 0.75]
```

The final class is determined using:

```python
np.argmax()
```

* 0 → Malignant
* 1 → Benign

## Custom Input Prediction

Steps for predicting new data:

1. Input feature values
2. Convert to NumPy array
3. Reshape input
4. Apply standardization
5. Predict using trained model

Output:

* "The tumor is Malignant"
* "The tumor is Benign"

## Results

* The model achieves high accuracy on test data
* Successfully classifies tumors based on input features
* Demonstrates effectiveness of neural networks for classification tasks

## Limitations

* Small dataset size
* Basic neural network architecture
* Not suitable for real-world medical diagnosis without validation


## Conclusion

This project demonstrates how deep learning can be applied to healthcare data for classification problems. Proper preprocessing and model design lead to accurate predictions even with a simple neural network.

 Future Improvements

* Increase training epochs
* Use deeper neural networks
* Apply regularization techniques
* Try different algorithms
* Deploy as a web application using Flask or Streamlit

