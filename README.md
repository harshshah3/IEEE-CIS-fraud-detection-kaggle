# IEEE-CIS-fraud-detection-kaggle
## CP610 - Data Analysis - End term project

### Abstract

Customer Fraud detection is an important sector in the finance industries and public as well as private banks. Over the years, many techniques have been proposed to save consumers from these frauds. For this project, I have used machine learning based techniques to find unusual patterns of consumer’s transactions which can help in blocking these kinds of likely frauds. The dataset is from IEEE-CIS Fraud
detection competition on Kaggle. In this project, I have done an extensive data analysis and basic feature engineering to find some interesting insights. Model selection and performance comparison is also covered. From this project work, it can be observed that LightGBM has outperformed the other models which are Linear Regression, Decision tree and Random forest classifier with an roc auc score of 84%.

### About competition

IEEE-CIS works across a variety of AI and machine learning areas, including deep neural networks, fuzzy systems, evolutionary computation, and swarm intelligence. Today they’re partnering with the world’s leading payment service company, Vesta Corporation, seeking the best solutions for fraud prevention industry, and now you are invited to join the challenge.

In this competition, you’ll benchmark machine learning models on a challenging large-scale dataset. The data comes from Vesta's real-world e-commerce transactions and contains a wide range of features from device type to product features. You also have the opportunity to create new features to improve your results.

### Objective

- In [this competition](https://www.kaggle.com/c/ieee-fraud-detection), task is to predict the probability that an online transaction is fraudulent, as denoted by the binary target **isFraud**.

### Data

- Name: IEEE-CIS-Fraud-Detection
- Source: https://www.kaggle.com/c/ieee-fraud-detection/data
- Number of Class Labels: 2 [0,1]
- Number of Features: 394 (transaction table) + 41 (identity table)
- The data is broken into two files identity and transaction, which are joined by TransactionID. Not all transactions have corresponding identity information.
- Class Distribution: Imbalance data (96.5% - 3.5%)
- Categorical Features - _Transaction_
	* ProductCD
	* card1 - card6
	* addr1, addr2
	* P_emaildomain
	* R_emaildomain
	* M1 - M9
- Categorical Features - _Identity_
	* DeviceType
	* DeviceInfo
	* id_12 - id_38
	* The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).

### Performance Analysis

|Model Used|ROC AUC Score|Training time (in seconds)|
|---|---|---|
|Linear Regression|54%|30|
|Decision Tree|77%|77|
|Random Forest|74%|308|
|Light GBM|84%|948|
|Light GBM (K-fold cross validation)|96%|16200|

**Private score**

ROC_AUC_SCORE: 84%

#### **[Report](CP610_Project_Report_shah5610.pdf) for more detais.**
