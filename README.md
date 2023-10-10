# Loan Prediction

This repository contains the code and resources for a data science project that predicts whether a person will be a loan defaulter or not based on certain criteria. This project is done under CodeClause Internship

## Dataset

The dataset used in this project, named [Loan Prediction.csv](/Loan%20Prediction.csv). You can also access the dataset from the [Kaggle dataset page](https://www.kaggle.com/datasets/kmldas/loan-default-prediction).

## Dependencies
-   Python (>=3.6)
-   Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, Imbalanced-learn

## Notebook

- [Loan Prediction Notebook](/Loan%20Prediction%20Notebook.ipynb): This Jupyter Notebook contains the entire project workflow, including data cleaning, exploratory data analysis (EDA), and model training.

## Machine Learning Models

We employed several machine learning models to predict loan defaulters:

1. **Random Forest Classifier**
   - Accuracy: 93.3%
   - [Trained Model](models/model_RF_pickle)

2. **Decision Tree Classifier**
   - Accuracy: 91.4%
   - [Trained Model](/models/model_DT_pickle)

3. **Logistic Regression**
   - Accuracy: 76.8%
   - [Trained Model](/models/model_LR_pickle)

You can use these trained models for making predictions on new data.

## Instructions

To reproduce the project and use the trained models, follow these steps:

1. Clone the repository to your local machine:

        git clone https://github.com/S-T-R-A-N-G-E-R/CodeClauseInternship_LoanPrediction.git

2. Install the required libraries mentioned in the [requirements.txt](/requirements.txt) file:

        pip install -r requirements.txt

3. Open and run the [Loan Prediction Notebook](/Loan%20Prediction%20Notebook.ipynb) to explore the project, perform data analysis, and use the trained models.

## Acknowledgments

*   This project was completed under the CodeClause Internship program.
*   The dataset used in this project was sourced from Kaggle.
*   Special thanks to the contributors and open-source libraries used in this project.

## Author

Swapnil Roy

-   LinkedIn: https://www.linkedin.com/in/swapnilroy001/
-   Email: swapnilroydata@outlook.com
-   Kaggle: https://www.kaggle.com/royswapnil

