# 🦆 Customer Churn Prediction

A machine learning project to predict customer churn using the [Customer Churn Dataset](https://www.kaggle.com/datasets/anandshaw2001/customer-churn-dataset/data) from Kaggle. This project includes a complete end-to-end pipeline with data analysis, model training, API deployment, and an interactive Streamlit web application.

## 🎯 Overview

Customer churn prediction is crucial for businesses to identify customers at risk of leaving and implement retention strategies. This project leverages machine learning to predict whether a customer will churn based on their profile and behavior patterns.


## 📊 Dataset

**Source**: [Kaggle - Customer Churn Dataset](https://www.kaggle.com/datasets/anandshaw2001/customer-churn-dataset/data)

The dataset contains customer information from a bank and includes various demographic, account, and behavioral features.

## Dataset Composition

### Total Records
- **Training samples**: ~10,000 customer records
- **Class distribution**: Imbalanced dataset
  - Active customers (0): ~79.6% (7,963 records)
  - Churned customers (1): ~20.4% (2,037 records)

### Features Overview
- **Total features**: 14 independent variables
- **Unique Identifiers**: 2
- **Numerical features**: 5
- **Categorical features**: 7
- **Target variable**: 1 (Exited)

---

## Feature Descriptions

### 1. **RowNumber**&**CustomerId**
- **Type**: Integer (Identifier)
- **Description**: Unique identifier for each customer record
- **Usage**: Should be excluded from modeling as it has no predictive power


### 2. **Surname**
- **Type**: String
- **Description**: Customer's last name
- **Usage**: Typically excluded from modeling due to high cardinality and no predictive value
- **Note**: Should be dropped during preprocessing

---

## Numerical Features

### 3. **CreditScore** (creditScore)
- **Type**: Continuous (Integer)
- **Range**: 350 - 850 (standard credit score range)
- **Description**: Measures the customer's creditworthiness based on credit history
- **Distribution**: Approximately normal distribution


### 4. **Age**
- **Type**: Continuous (Float/Integer)
- **Range**: 18 - 92 years
- **Description**: Customer's age in years
- **Distribution**: Slightly right-skewed
- **Key Insight**: Shows relatively strong relationship with churn

### 5. **Tenure**
- **Type**: Integer
- **Range**: 0 - 10 years
- **Description**: Number of years the customer has been with the bank
- **Distribution**: Relatively uniform across years


### 6. **Balance**
- **Type**: Continuous (Float)
- **Range**: $0 - $250,898
- **Description**: Current account balance
- **Business Significance**:
  - Zero balances are particularly notable (36.2% of customers)



### 7. **NumOfProducts** (numofProducts)
- **Type**: Integer
- **Range**: 1 - 4
- **Description**: Number of bank products/services the customer uses
- **Distribution**: Most customers have 1-2 products
- **Key Insight**:  Shows relatively strong relationship with churn


### 8. **EstimatedSalary** (estimatedSalary)
- **Type**: Continuous (Float)
- **Range**: $11.58 - $199,992
- **Description**: Customer's estimated annual salary
- **Distribution**: Approximately uniform distribution

---

## Categorical Features

### 9. **Geography** (geography)
- **Type**: Categorical (String)
- **Unique Values**: 3 countries
  - France
  - Germany  
  - Spain
- **Description**: Customer's country of residence
- **Distribution**: 
  - France: 50.1% (majority)
  - Germany: ~25%
  - Spain: ~25%
- **Note**: Imbalanced distribution favoring France

### 10. **Gender** (gender)
- **Type**: Binary Categorical (String)
- **Unique Values**: 
  - Male
  - Female

### 11. **HasCrCard** (hasCrCard)
- **Type**: Binary (0/1)
- **Values**:
  - 0: Customer does not have a credit card
  - 1: Customer has a credit card with the bank
- **Description**: Indicates whether the customer holds a credit card


### 12. **IsActiveMember** (isActiveMember)
- **Type**: Binary (0/1)
- **Values**:
  - 0: Inactive member
  - 1: Active member
- **Description**: Indicates whether the customer actively uses bank services

### 13. **IsZeroBalance** (isZeroBalance)
- **Type**: Binary (0/1)
- **Values**:
  - 0: Customer has a balance > $0
  - 1: Customer has exactly $0 balance
- **Description**: Flags accounts with zero balance
- **Business Significance**:
  - 36.2% of customers have zero balance
  - May indicate dormant accounts
  - High correlation with potential churn
  - Could signal customers who have moved funds elsewhere
- **Note**: This is a derived feature from the Balance column

---

## Target Variable

### 14. **Exited** (exited)
- **Type**: Binary (0/1)
- **Values**:
  - 0: Customer retained (stayed with the bank)
  - 1: Customer churned (closed account/left the bank)
- **Description**: Whether the customer has left the bank
