# Credit_Risk_Analysis

# Overview

The purpose of the current analysis was to use machine learning statistical algorithms to make predictions based on data patterns provided. The supervised learning models utilized a free dataset from Lending Club, a P2P lending service company, in order to evaluate and predict credit risk. 

Machine learning techniques were used to train and evaluate the data with unbalanced classes. The dataset from the Lending Club has an unbalanced classification problem due to the number of good loans outweighing the amount of risky loans. In order to balance the classifications to facilitate more meaningful predictions and improve the accuracy score, machine learning algorithms were employed to resample the data.  The algorithms used include: RandomOverSampler, SMOTE, ClusterCentroids, SMOTEENN, BalancedRandomForestClassifier, and EasyEnsembleClassifier.

# Results

Machine learning was used to resample the dataset using Python libraries: `scikit-learn` and `imbalanced-learn`.  These tools help evaluate the results and provide a comparison amongst the analyses. 

The original dataset contained 115,675 loan applications in Q1 of 2019. “Loan status" was used to determine whether the application was considered low or high risk. Applications that had "current" as the "loan status" were classified as "low risk" and the remaining as "high risk". This reduced the dataset to 68,817 total applications with 99% classified as "low risk". 

![INSERT DATA_COUNT_TOTAL.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/1c63764e1b37426267ab49b76c77e61cc49c09f6/Resources/Data_Count_Total.png)

Using the 75/25% method to split the data for training vs. testing, 51,366 "low risk" and 246 "high risk" applications were categorized into the training set.   

![INSERT TRAINING_DATA.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/Training_Data.png)

## Deliverable 1: Use Resampling Models to Predict Credit Risk

### Oversampling

The RandomOverSampler Model randomly selects from the minority class and adds it to the training set until both classifications are equal. The results classified 51,366 records each as “High Risk” and “Low Risk”.

![INSERT OVERSAMPLE_COUNT.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/Oversample_Count.png)

  * Balanced accuracy score: 64%.

![INSERT OVERSAMPLE_ACCURACY.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/Oversample_Accuracy.png)

  * The "High Risk" precision rate was only 1% with the recall at 66% giving this model an F1 score of 2%.
  * "Low Risk" had a precision rate of 100% and recall at 62%.  
  
![INSERT OVERSAMPLE_CM.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/Oversample_CM.png)
  
![INSERT OVERSAMPLE_CLASS.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/da8cc00ae221d90af29a228d38d566976d47ec74/Resources/Oversample_Class.png)

SMOTE (Synthetic Minority Oversampling Technique) Model, like RandomOverSampler, increases the size of the minority class by creating new values based on the value of the closest neighbours to the minority class instead of random selection. 

  * The balanced accuracy score improved slightly to 65.1%.

![INSERT SMOTE_ACCURACY.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/Smote_Accuracy.png)

  * Like RandomOverSampler, the "High Risk" precision rate again was only 1% with the recall degraded to 61% giving this model an F1 score of 2%.
  * "Low Risk" had a precision rate of 100% and an improved recall at 69%.  

![INSERT SMOTE_CM.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/SMOTEENN_CM.png)
  
![INSERT SMOTE_CLASS.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/Smote_Class.png)

### Undersampling

ClusterCentroids Model is an algorithm that identifies clusters of the majority class to generate synthetic data points that are representative of the clusters. The model classified 246 records each as “High Risk” and “Low Risk”.

![INSERT UNDERSAMPLE_COUNT.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/Undersample_Count.png)

  * Balanced accuracy score was lower than the oversampling models at 54.5%.

![INSERT UNDERSAMPLE_ACCURACY.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/Undersample_Accuracy.png)

  * The "High Risk" precision rate again was only at 1% with the recall at 69% giving this model an F1 score of 1%.
  * "Low Risk" had a precision rate of 100% and with a lower recall at 40% compared to the oversampling models.  

![INSERT UNDERSAMPLE_CM.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/UNDERSAMPLE_CM.png)
  
![INSERT UNDERSAMPLE_CLASS.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/Undersample_Class.png)

## Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk

### Combination Sampling

SMOTEENN (Synthetic Minority Oversampling Technique + Edited NearestNeighbors) Model combines aspects of both oversampling and undersampling. The model classified 68,460 records as “High Risk” and 62,011 as “Low Risk”.

![INSERT SMOTEENN_COUNT.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/SMOTEENN_Count.png)

  * The balanced accuracy score improved to 64.5% when using a combined sampling model.

![INSERT SMOTEENN_ACCURACY.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/SMOTEENN_Accuracy.png)

  * The "High Risk" precision rate did not improve was only 1%, however the recall increased to 72% giving this model an F1 score of 2%.
  * "Low Risk" still showed a precision rate of 100% with the recall at 57%.  
  
![INSERT SMOTEENN_CM.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/SMOTEENN_CM.png)

![INSERT SMOTEENN_Class.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/SMOTEENN_Class.png)

## Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk

Compare two new machine learning models that reduce bias to predict credit risk. The models classified 51,366 as “High Risk” and 246 as “Low Risk”.

![INSERT BALACNCED_COUNT.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/Balanced_Count.png)

BalancedRandomForestClassifier Model constructs two trees of the same and equal size to the minority class to represent one for the majority class and one for the minority class. 

  * The balanced accuracy score increased to 78.9% for this model.

![INSERT BALANCED_ACCURACY.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/Balanced_Accuracy.png)

  * The "High Risk precision rate increased to 3% with the recall at 70% giving this model an F1 score of 6%.
  * "Low Risk" still had a precision rate of 100% with the recall at 87%.  
  * The top feature by importance was "total_rec_prncp" at 7.9% of the total.

![INSERT BALANCED_CM.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/Balanced_CM.png)
  
![INSERT BALANCED_CLASS.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/Balanced_Class.png)

![INSERT BALANCED_FEATURE.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/BalancedFeature.png)

EasyEnsembleClassifier Model utilizes a set of classifiers where individual decisions are combined to classify new examples.

  * The balanced accuracy score increased to 93.2% with this model.

![INSERT EASY_ACCURACY.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/Easy_Accuracy.png)

  * The "High Risk precision rate increased to 9% with the recall at 92% giving this model an F1 score of 16%.
  * "Low Risk" still had a precision rate of 100% with the recall now at 94%.  

![INSERT EASY_CM.PNG](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/Easy_CM.png)
  
![INSERT EASY_CLASS.PNg](https://github.com/jbowman86/Credit_Risk_Analysis/blob/3d32b407d2a1c0e3bdcc602bafc788a94e0d164d/Resources/Easy_Class.png)

# Summary

In reviewing all six models, the EasyEnsembleClassifer model yielded the best results with an accuracy rate of 93.2% and a 9% precision rate when predicting "High Risk" candidates. The sensitivity rate was also the highest at 92% compared to the other models. The result for predicting "Low Risk" was also the highest with the sensitivity rate at 94% and an F1 score of 97%. Therefore, if a model needed to be recommended to perform this type of analysis, the EasyEnsembleClassifier model would be the best choice.

**Ranking of models in descending order based on "High Risk" results:**
* EasyEnsembleClassifer: 93.2% accuracy, 9% precision, 92% recall, and 16% F1 Score
* BalancedRandomForestClassifer: 78.9% accuracy, 3% precision, 70% recall and 6% F1 Score
* SMOTE: 65.2% accuracy, 1% precision, 61% recall and 2% F1 Score
* SMOTEENN: 64.5% accuracy, 1% precision, 72% recall and 2% F1 Score
* RandomOverSampler: 64.0% accuracy, 1% precision, 66% recall and 2% F1 Score
* ClusterCentroids: 54.5% accuracy, 1% precision, 69% recall and 1% F1 Score

Additionally, it should be understood that original dataset had 99% of the applications classified as "Low Risk" with only 1% of the data classified in the "High Risk" category. This may skew results as there is a risk that the machine learning algorithms are creating clusters drawing from a very small dataset of actual "High Risk" applications. The margin of risk created may be greater than some banks would be willing to accept.
