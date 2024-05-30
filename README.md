# Credit Card Fraud Detection

## Project Overview
This project aims to develop a robust machine learning model to predict fraudulent credit card transactions. Accurately detecting fraud can significantly reduce financial losses for the company and improve the overall efficiency of transaction monitoring. The dataset used for this project contains various features related to credit card transactions, and the target variable indicates whether a transaction is fraudulent.

## Business Case
Credit card fraud is a critical issue in the financial industry, leading to substantial financial losses each year. By leveraging machine learning techniques, we aim to identify potentially fraudulent transactions early in the process, thereby saving costs and improving customer satisfaction by minimizing false accusations.

## Dataset
The dataset can be downloaded from this [link](https://drive.google.com/file/d/1b0fZcCBZaka5Pu8TsWnHi3kClA-4wdrn/view?usp=sharing).

## Tools and Libraries
The following tools and libraries were used in this project:

- **Python Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`
- **Data Processing:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`, `imbalanced-learn`

## Approach
### Data Preprocessing
- **Loading Data:** The dataset was loaded using `pandas`.
- **Exploratory Data Analysis (EDA):** Visualizations were created using `seaborn` and `matplotlib` to understand the distribution of features and the target variable.
- **Handling Missing Values:** Missing values were appropriately handled to ensure data quality.
- **Encoding Categorical Variables:** Categorical variables were encoded using one-hot encoding.
- **Feature Scaling:** Numerical features were scaled to ensure uniformity.

### Data Balancing Techniques
Given the imbalanced nature of the dataset, various data balancing techniques were applied:

- **Random Oversampling:** Increase the number of minority class samples by randomly duplicating them.
- **SMOTE (Synthetic Minority Over-sampling Technique):** Generate synthetic samples for the minority class.
- **ADASYN (Adaptive Synthetic Sampling):** Similar to SMOTE but generates synthetic samples considering the density distribution.
- **Random Undersampling:** Reduce the number of majority class samples by randomly removing them.
- **Tomek Links:** Remove overlapping samples between the majority and minority classes.

### Model Building
Several machine learning pipelines with random forest classifier were tested to identify the best-performing model:

- **Random Forest Classifier:**
  - A Random Forest model was built with 100 estimators.
  - Grid search Cross-validation was used to evaluate the model's performance by performing hyperparameter tuning.

### Model Evaluation
The models were evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score

### Recommended Model
After careful evaluation, I would choose class weights was chosen as the recommended model to balance cumulative positive and negative business impact across multiple metrics. But ultimate decision rests with business leadership.

## Conclusion
The class weights enabled random forest model was selected as the final model for predicting fraudulent credit card transactions. This model not only demonstrated the best performance during evaluation but also provides flexibility for further tuning and improvement. Implementing this model can help financial institutions effectively identify and mitigate fraudulent transactions, leading to significant cost savings and operational efficiency.
