# Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models to predict whether an individual's annual income exceeds $50K based on demographic and employment-related attributes.

The project demonstrates an end-to-end machine learning workflow including:

Data preprocessing
Model training
Performance evaluation
Web app deployment using Streamlit

# Dataset Description

The dataset used in this project is the Adult Income Dataset, which contains demographic and socioeconomic information such as:

Age
Education
Occupation
Marital Status
Hours per week
Capital gain/loss
### Target Variable:
<=50K → Income less than or equal to $50,000
>50K → Income greater than $50,000

The dataset satisfies the assignment constraints with more than 500 instances and 12+ features, making it suitable for classification tasks.

# Machine Learning Models Implemented

The following classification algorithms were trained and evaluated on the same dataset:

Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors
Naive Bayes (Gaussian)
Random Forest (Ensemble)
XGBoost (Ensemble)

# Evaluation Metrics

To ensure a comprehensive evaluation, multiple metrics were used instead of relying solely on accuracy:

Accuracy
Precision
Recall
F1 Score
AUC (Area Under ROC Curve)
Matthews Correlation Coefficient (MCC)

# Model Performance Comparison


| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|--------|------------|------|------------|---------|------------|------|
| Logistic Regression | 0.857 | 0.911 | 0.736 | 0.607 | 0.666 | 0.580 |
| Decision Tree | 0.820 | 0.758 | 0.611 | 0.640 | 0.625 | 0.507 |
| K-Nearest Neighbors | 0.839 | 0.865 | 0.669 | 0.617 | 0.642 | 0.539 |
| Naive Bayes | 0.623 | 0.846 | 0.377 | 0.930 | 0.536 | 0.394 |
| Random Forest | 0.861 | 0.907 | 0.732 | 0.644 | 0.685 | 0.599 |
| XGBoost | 0.878 | 0.933 | 0.777 | 0.674 | 0.722 | 0.647 |

| ML Model | Observation about model performance |
|--------|--------------------------------------|
| Logistic Regression | Logistic Regression demonstrated strong baseline performance, suggesting that several features exhibit near-linear separability. The model balances interpretability and predictive power, making it a reliable reference for comparing more complex algorithms. |
| Decision Tree | The Decision Tree captured non-linear relationships within the dataset but showed slightly lower generalization compared to ensemble methods. This is likely due to its tendency to overfit when trained without pruning constraints. |
| K-Nearest Neighbors | KNN achieved moderate performance, indicating that local feature similarity contributes to classification. However, its sensitivity to feature scaling and computational overhead make it less suitable for large-scale deployment. |
| Naive Bayes | Naive Bayes produced very high recall but low precision, meaning it classified many samples as high income. This behavior stems from its strong independence assumption between features, which rarely holds in real-world socioeconomic datasets. |
| Random Forest | Random Forest improved upon the Decision Tree by aggregating multiple trees, reducing variance and enhancing robustness. Its strong performance highlights the effectiveness of ensemble learning for structured data. |
| XGBoost | XGBoost achieved the best overall results across evaluation metrics. Its gradient boosting mechanism sequentially corrects prior errors, enabling the model to capture complex feature interactions while maintaining strong generalization. |