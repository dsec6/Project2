# SMS Spam Detection Project

## Project Overview
This project implements a comprehensive approach to detect spam and phishing attempts across various messaging platforms. It combines machine learning models for SMS spam detection with a Tines workflow for processing images of messages from different sources. The project utilizes the SMS Spam Collection Dataset to train and evaluate multiple classification algorithms.

## Table of Contents
1. Installation
2. Project Structure
3. Usage
4. Data Model Implementation
5. Data Model Optimization
6. Results
7. Tines Workflow Integration
8. Key Findings
9. Future Improvements

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
   - This notebook thoroughly describes the data extraction, cleaning, and transformation process.
   - Implements tokenization and lemmatization for text normalization.
   - Conducts exploratory data analysis, including message length analysis and time series visualization.

2. Model Training and Evaluation:
   - Open and run all cells in `model_training.ipynb` to train models, evaluate their performance, and visualize the results.
   - This notebook initializes, trains, and evaluates multiple models: Naive Bayes, Logistic Regression, Random Forest, and Support Vector Machine (SVM).
   - Utilizes TF-IDF vectorization for feature extraction.
   - Employs a 75/25 train/test split for model evaluation.

## Data Model Implementation

- The project uses the SMS Spam Collection Dataset, available at: https://www.kaggle.com/uciml/sms-spam-collection-dataset
- Cleaned data is exported as CSV files for the machine learning model.
- The best-performing model demonstrates an accuracy of over 97%, exceeding the 75% classification accuracy requirement.

## Data Model Optimization

- The model optimization process is documented in the `model_training.ipynb` notebook.
- It shows iterative changes made to different models and the resulting changes in model performance.
- Implements hyperparameter tuning using GridSearchCV.
- Analyzes feature importance to identify key spam indicators.
- Conducts cross-validation to ensure model robustness.
- Explores threshold adjustment for optimizing precision-recall trade-off.

## Results

The project achieves the following results:
- Data preprocessing successfully cleans and prepares the SMS data for model training.
- Multiple classification models are trained and evaluated.
- The best-performing model (typically Random Forest or SVM) achieves an accuracy of over 97% on the test set.
- Detailed performance metrics, including precision, recall, and F1-score, are provided for each model.
- Confusion matrices are visualized to show the models' performance in classifying spam and ham messages.
- Random Forest model emerges as the top performer with high precision and recall for both spam and ham categories.

## Tines Workflow Integration

- A Tines workflow has been created to extend the project's capabilities:
  - Accepts image inputs of messages from various platforms (SMS, Instagram DM, Facebook Messenger, etc.)
  - Utilizes OCR API for text extraction from images
  - Integrates with LLM for advanced phishing, spam, and scam analysis
  - Implements URL defanging for safe analysis and reporting
- The workflow is accessible at: https://blue-nest-7031.tines.com/pages/sms/
- Provides a user-friendly interface for message analysis

## Key Findings

1. Machine learning models, particularly Random Forest, show exceptional accuracy in SMS spam detection.
2. The integration of OCR and LLM in the Tines workflow extends the project's capabilities to various messaging platforms.
3. Feature importance analysis reveals critical indicators of spam messages.
4. Hyperparameter tuning and threshold adjustment offer marginal improvements, indicating the robustness of the initial model.

## Future Improvements

- Experiment with more advanced text preprocessing techniques, such as named entity recognition.
- Try ensemble methods to potentially improve model performance further.
- Implement cross-validation for more robust evaluation of model performance.
- Explore deep learning models like LSTM or BERT for potentially higher accuracy.
- Develop a system for periodic model retraining to adapt to evolving spam patterns.
- Expand the Tines workflow to handle a wider range of input formats and messaging platforms.
- Develop a simple web application to demonstrate the spam detection in real-time.

This project demonstrates the practical application of machine learning in cybersecurity, combining traditional spam detection techniques with modern workflow automation for comprehensive message analysis.