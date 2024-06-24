# Breast Cancer Prediction using Machine Learning

Welcome to the Breast Cancer Prediction project! This repository contains code and documentation for predicting breast cancer using machine learning techniques. Our goal is to build a model that can accurately classify whether a tumor is benign or malignant based on various features.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Breast cancer is one of the most common cancers affecting women worldwide. Early detection and accurate diagnosis are crucial for effective treatment. This project utilizes machine learning algorithms to predict the likelihood of a breast cancer diagnosis based on patient data.

## Dataset
We use the Breast Cancer Wisconsin (Diagnostic) Dataset, which contains features computed from digitized images of fine needle aspirate (FNA) of breast mass. The dataset is publicly available and can be downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).

### Features
- **Radius**: Mean of distances from center to points on the perimeter
- **Texture**: Standard deviation of gray-scale values
- **Perimeter**
- **Area**
- **Smoothness**: Local variation in radius lengths
- **Compactness**: Perimeter² / Area - 1.0
- **Concavity**: Severity of concave portions of the contour
- **Concave points**: Number of concave portions of the contour
- **Symmetry**
- **Fractal dimension**: "Coastline approximation" - 1

### Target
- **Diagnosis**: Malignant (M) or Benign (B)

## Project Structure
<br>
<br>breast-cancer-prediction/
<br>├── data/
<br>│ └── breast_cancer_data.csv
<br>├── notebooks/
<br>│ └── breast_cancer_prediction.ipynb
<br>├── src/
<br>│ ├── data_preprocessing.py
<br>│ ├── feature_engineering.py
<br>│ ├── model_training.py
<br>│ └── model_evaluation.py
<br>├── results/
<br>│ └── model_performance.png
<br>├── README.md
<br>└── requirements.txt

## Installation
To run this project, you need Python 3.6 or above. Follow these steps to set up the environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/breast-cancer-prediction.git
    cd breast-cancer-prediction
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
You can explore the project through the provided Jupyter Notebook `breast_cancer_prediction.ipynb` in the `notebooks` directory. The notebook covers data preprocessing, feature engineering, model training, and evaluation.

Alternatively, you can run the scripts directly:

1. Preprocess the data:
    ```bash
    python src/data_preprocessing.py
    ```

2. Perform feature engineering:
    ```bash
    python src/feature_engineering.py
    ```

3. Train the model:
    ```bash
    python src/model_training.py
    ```

4. Evaluate the model:
    ```bash
    python src/model_evaluation.py
    ```

## Model
We explore various machine learning algorithms including:
- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- k-Nearest Neighbors (k-NN)

We utilize techniques like cross-validation, hyperparameter tuning, and feature selection to optimize the models.

## Results
Our best-performing model achieved an accuracy of XX% on the test set. Below is the performance of various models:

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | XX%      | XX%       | XX%    | XX%      |
| Decision Tree       | XX%      | XX%       | XX%    | XX%      |
| Random Forest       | XX%      | XX%       | XX%    | XX%      |
| SVM                 | XX%      | XX%       | XX%    | XX%      |
| k-NN                | XX%      | XX%       | XX%    | XX%      |

## Contributing
We welcome contributions! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
