# Sampling Techniques for Imbalanced Credit Card Dataset

**Course:** UCS654 – Predictive Analytics using Statistics  
**Assignment:** Assignment 2 (Sampling Techniques)  
**Author:** Pritish Dutta  
**Roll Number:** 102303473  

---

## Objective
The objective of this assignment is to understand the significance of sampling techniques in handling highly imbalanced datasets and to analyze how different sampling strategies influence the performance of various machine learning models.


---

## Dataset
The dataset used in this project is a credit card fraud detection dataset containing transactions labeled as fraudulent or non-fraudulent.

Dataset Link:  
https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv

The dataset was extremely imbalanced:

| Class         | Count |
|---------------|-------|
| 0 (Non-Fraud) | 763   |
| 1 (Fraud)     | 9     |

Such severe imbalance can bias machine learning models toward the majority class, making them less effective at detecting fraudulent transactions. Therefore, applying appropriate sampling techniques becomes essential for building reliable predictive models.

---

## Methodology

The following steps were performed to complete the analysis:

### 1. Data Preprocessing
- Loaded the dataset using Pandas  
- Separated features and target variable  
- Applied StandardScaler to normalize feature values and improve model performance  
- Performed a stratified train-test split to preserve class distribution  

---

### 2. Sampling Techniques Used

Five sampling techniques were applied to balance the dataset:

1. **Random Undersampling**  
   Reduces the number of majority class samples to match the minority class, resulting in a balanced dataset but with potential information loss.

2. **Random Oversampling**  
   Duplicates minority class samples to achieve balance without discarding valuable majority class data.

3. **SMOTE (Synthetic Minority Oversampling Technique)**  
   Generates synthetic samples based on feature similarity rather than duplication, helping models generalize better.

4. **ADASYN (Adaptive Synthetic Sampling)**  
   An advanced oversampling technique that focuses on generating synthetic data for harder-to-learn observations.

5. **Stratified Sampling**  
   Maintains the original class distribution while splitting the dataset, ensuring fair model evaluation.

---

### 3. Machine Learning Models

The following classification models were trained and evaluated on each sampled dataset:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)

---

## Results

After training the models on all sampling techniques, the accuracies were recorded and compared.

| Model               | Sampling Technique with Highest Accuracy |
|---------------------|------------------------------------------|
| Logistic Regression | 100%                                     |
| Decision Tree       | 98.71%                                   |
| Random Forest       | 98.71%                                   |
| KNN                 | 100%                                     |
| SVM                 | 98.71%                                   |

---

## Discussion

The experimental results demonstrate that sampling techniques significantly impact model performance when dealing with imbalanced datasets.

Oversampling methods, particularly SMOTE and ADASYN, produced strong results because they generated synthetic minority samples instead of simply duplicating existing ones. This allowed the models to learn better decision boundaries and improved their ability to classify fraudulent transactions.

Random undersampling helped achieve class balance but may have removed useful majority-class information. Stratified sampling ensured fair evaluation but did not fully address the imbalance problem.

Overall, the results highlight the importance of selecting an appropriate sampling strategy before training machine learning models on skewed datasets.

---

## Conclusion

This project demonstrated that handling class imbalance is a critical step in building effective fraud detection systems. Applying sampling techniques improved model reliability and predictive performance.

Among the evaluated approaches, synthetic oversampling techniques proved to be highly effective for this dataset. Proper preprocessing combined with suitable sampling methods can substantially enhance classification outcomes in real-world machine learning applications.

---

## Technologies Used
- Python  
- Scikit-learn  
- Imbalanced-learn  
- Pandas  
- NumPy  

---

## How to Run

Install dependencies:

```bash
pip install pandas scikit-learn imbalanced-learn tabulate
