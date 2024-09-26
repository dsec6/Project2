# SMS Spam Detection Project

## Project Overview
This project implements a machine learning model to detect spam SMS messages using the SMS Spam Collection Dataset. It demonstrates the process of data preprocessing, model training, and evaluation using various classification algorithms.

## Table of Contents
1. Installation
2. Project Structure
3. Usage
4. Data Model Implementation
5. Data Model Optimization
6. Results
7. Future Improvements

## Installation

1. Clone this repository:
   git clone https://github.com/dsec6/Project2.git
   cd Project2

2. Create a virtual environment and activate it:
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the required packages:
   pip install -r requirements.txt

## Project Structure

Project2/
│
├── data_processing.ipynb
├── model_training.ipynb
├── spam.csv
├── cleaned_sms_data.csv
├── train_data.csv
├── test_data.csv
├── best_model.joblib
├── tfidf_vectorizer.joblib
├── model_results.csv
├── requirements.txt
└── README.md

## Usage

1. Data Processing:
   - Open and run all cells in `data_processing.ipynb` to process the raw data and generate the cleaned datasets.

2. Model Training and Evaluation:
   - Open and run all cells in `model_training.ipynb` to train models, evaluate their performance, and visualize the results.

## Data Model Implementation

- The `data_processing.ipynb` notebook thoroughly describes the data extraction, cleaning, and transformation process.
- Cleaned data is exported as CSV files for the machine learning model.
- The `model_training.ipynb` notebook initializes, trains, and evaluates multiple models.
- The best-performing model demonstrates an accuracy of over 97%, exceeding the 75% classification accuracy requirement.

## Data Model Optimization

- The model optimization process is documented in the `model_training.ipynb` notebook.
- It shows iterative changes made to different models (Naive Bayes, Logistic Regression, Random Forest, SVM) and the resulting changes in model performance.
- Overall model performance is displayed at the end of the notebook, including accuracy scores and confusion matrices for each model.

## Results

The project achieves the following results:

- Data preprocessing successfully cleans and prepares the SMS data for model training.
- Multiple classification models are trained and evaluated:
  - Naive Bayes
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
- The best-performing model (typically Random Forest or SVM) achieves an accuracy of over 97% on the test set.
- Detailed performance metrics, including precision, recall, and F1-score, are provided for each model.
- Confusion matrices are visualized to show the models' performance in classifying spam and ham messages.

## Future Improvements

- Experiment with more advanced text preprocessing techniques, such as lemmatization or named entity recognition.
- Try ensemble methods to potentially improve model performance further.
- Implement cross-validation for more robust evaluation of model performance.
- Explore deep learning models like LSTM or BERT for potentially higher accuracy.
- Develop a simple web application to demonstrate the spam detection in real-time.
