 Customer Churn Prediction using Machine Learning

 Project Information
- Project Title: Customer Churn Prediction using Machine Learning Techniques
- Student Name: Manasi Renuse
- Degree: MTech in Computer Science Engineering(Artificial Intelligence)
- Institution: Pimpri Chinchwad University
- Guide: Dr.Sagar Pande
- Year: 2025-2027

Project Overview
This project implements customer churn prediction using multiple machine learning algorithms including clustering (K-Means, DBSCAN, Hierarchical) and classification (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting).

Objectives
- Segment customers using clustering algorithms
- Predict customer churn using classification models
- Identify high-risk customers
- Provide retention recommendations

 Project Structure

Technologies Used
-Programming Language: Python 3.9+
-Libraries:
  - pandas (Data manipulation)
  - numpy (Numerical computing)
  - scikit-learn (Machine learning)
  - matplotlib & seaborn (Visualization)

Key Results
- Total Customers Analyzed: 1,000
- Churn Rate: 24.4%
- High-Risk Customers Identified:** 253 (25.3%)
- Best Model: Decision Tree (F1-Score: 0.243)
- Customer Segments: 4 clusters identified

 Clustering Results
- Cluster 0: 142 customers (22.5% churn, short tenure)
- Cluster 1: 277 customers (28.5% churn, long tenure)
- Cluster 2: 291 customers (22.3% churn, medium tenure)
- Cluster 3: 290 customers (23.4% churn, high charges)

 Classification Model Performance
| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 75.5% | 0.00 | 0.570 |
| Decision Tree | 59.5% | 0.243 | 0.484 |
| Random Forest | 76.5% | 0.113 | 0.498 |
| Gradient Boosting | 74.5% | 0.136 | 0.533 |

 Top 5 Important Features
1. Monthly Charges (21.18%)
2. Charge Per Month (15.35%)
3. Tenure (14.05%)
4. Age (12.39%)
5. Total Charges (11.55%)

Key Insights
- Customers with 4+ complaints have highest churn rate (>25%)
- Short tenure customers (<12 months) more likely to churn
- High monthly charges correlate with increased churn
- Month-to-month contracts have 28.5% churn vs 22% for annual

References
1. Scikit-learn Documentation: https://scikit-learn.org
2. Pandas Documentation: https://pandas.pydata.org