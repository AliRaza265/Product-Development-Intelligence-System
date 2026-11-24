# Product Research & Intelligence System

An AI-powered product review analysis system developed during my PEL internship. The application analyzes customer reviews and predicts their **sentiment** and **product aspects** using Machine Learning and Natural Language Processing.

## Project Overview

Customer reviews contain valuable information about product quality, performance, and customer satisfaction. Manually analyzing large numbers of reviews can be time-consuming.

This project helps simplify the process by providing automated sentiment and aspect prediction for individual comments as well as uploaded CSV and Excel files.

## Key Features

- **Sentiment Prediction:** Classifies customer comments into sentiment categories.
- **Aspect Prediction:** Identifies the product aspect discussed in a customer comment.
- **Single Comment Analysis:** Predicts sentiment and aspect for an individual customer review.
- **Bulk Review Analysis:** Upload CSV or Excel files to analyze multiple comments.
- **Data Visualization:** Displays sentiment and aspect distributions through interactive charts.
- **Interactive Dashboard:** Provides an easy-to-use interface built with Streamlit.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Linear Support Vector Classification (Linear SVC)
- Streamlit
- Plotly
- Pickle

## Machine Learning Models

The system uses two separate prediction models:

### 1. Sentiment Prediction Model

- Algorithm: Linear SVC
- Task: Sentiment Classification
- Accuracy: **70.95%**

### 2. Aspect Prediction Model

- Task: Product Aspect Classification
- Number of Classes: 14
- Accuracy: **63.41%**

### Model Performance Summary

| Model | Accuracy |
|---|---:|
| Sentiment Prediction (Linear SVC) | 70.95% |
| Aspect Prediction | 63.41% |

Both models were evaluated using accuracy, confusion matrices, precision, recall, and F1-score.

## Project Structure

```text
Product-Research-Intelligence-System/
│
├── Models/
│   ├── Aspect_encoder.pkl
│   ├── Aspect_model.pkl
│   ├── Aspect_vectorizor.pkl
│   ├── Sentiment_encoder.pkl
│   ├── Sentiment_model.pkl
│   └── Sentiment_vectorizor.pkl
│
├── data_set/
│
├── app.py
├── reviews_preprocessing.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Directory

```bash
cd Product-Research-Intelligence-System
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Application Workflow

1. Enter an individual customer comment or upload a CSV/Excel file.
2. The application processes the customer comments.
3. The sentiment model predicts the sentiment.
4. The aspect model predicts the relevant product aspect.
5. Results are displayed in a structured table.
6. Interactive charts provide an overview of customer feedback.

## Model Evaluation

The models were evaluated using a test dataset of 451 samples for each reported classification evaluation.

The Sentiment Prediction Model achieved 70.95% accuracy, while the Aspect Prediction Model achieved 63.41% accuracy.

Performance can vary across individual classes, so precision, recall, and F1-score are also considered.

## Version Control

Git and GitHub are used for version control and project management.

- Git is used to track source-code changes.
- GitHub is used to maintain the project repository.
- Separate branches are used during development and model updates.
- Commits document changes and improvements throughout development.

## Project Purpose

This project demonstrates the practical application of Machine Learning, Natural Language Processing, data analysis, and web application development to automate product review analysis and extract useful insights from customer feedback.

## Author

**Ali Raza**

Software Engineering Student  
University of Sahiwal

**Project:** PEL Internship Project  
**GitHub:** [AliRaza265](https://github.com/AliRaza265)

---

*Developed as part of a practical Machine Learning and Product Research project.*
