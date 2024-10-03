# Machine Learning Models Repository

This repository contains the development of machine learning models that accomplish three distinct tasks. Each task explores a unique area of machine learning, including natural language processing, classification, and generative modeling.

## Table of Contents
- [Task 1: Movie Genre Classification](#task-1-movie-genre-classification)
- [Task 2: Credit Card Fraud Detection](#task-2-credit-card-fraud-detection)
- [Task 3: Handwritten Text Generation](#task-3-handwritten-text-generation)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Task 1: Movie Genre Classification

**Objective:** Create a machine learning model that predicts the genre of a movie based on its plot summary or other textual data.

**Approach:**
- Utilize **TF-IDF** or **word embeddings** for feature extraction.
- Experiment with classifiers such as **Naive Bayes**, **Logistic Regression**, or **Support Vector Machines (SVMs)**.
  
The goal is to correctly classify a movie into one or more genres using its textual description.

---

## Task 2: Credit Card Fraud Detection

**Objective:** Build a model that detects fraudulent credit card transactions based on transaction data.

**Approach:**
- Use features from a credit card transaction dataset.
- Test algorithms such as **Logistic Regression**, **Decision Trees**, and **Random Forests** to classify transactions as fraudulent or legitimate.
  
The aim is to minimize false positives while ensuring accurate fraud detection.

---

## Task 3: Handwritten Text Generation

**Objective:** Implement a **character-level Recurrent Neural Network (RNN)** to generate text that mimics handwritten characters.

**Approach:**
- Train the model on a dataset of handwritten text examples.
- The model learns patterns in the handwritten text and generates new text based on those patterns.
  
This task explores generative modeling and sequence-to-sequence learning.

---

## Requirements

To run these models, ensure the following libraries and dependencies are installed:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- TensorFlow or PyTorch (for Task 3)
- Matplotlib (for visualizations)
- Jupyter Notebook (optional, for experimentation)

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Salma-Benaroub/ML-Models.git
   ```
2. Navigate to the project directory:
   ```bash
   cd ML-Models
   ```
3. Install the required dependencies as mentioned in the `requirements.txt` file.

---

## Usage

1. **Task 1: Movie Genre Classification**
   - Navigate to the `movie_genre_classification/` directory and run the classification script of the .ipynb file `model.ipynb`
   
2. **Task 2: Credit Card Fraud Detection**
   - Go to the `fraud_detection/` directory and use the provided dataset to run the model.
   
3. **Task 3: Handwritten Text Generation**
   - Move to the `handwritten_text_generation/` directory and run the RNN model for text generation.

---

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your improvements or bug fixes.

---
