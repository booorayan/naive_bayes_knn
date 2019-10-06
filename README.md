# Classifying Mail as Spam or Not and Predicting Whether One Survives An Accident
#### This project uses machine learning classification models to predict whether passengers will survive a ship accident
#### and also classify mails as spam or not.
#### It achieves these two objectives separately using different ML classification models. 

#### The link to the dataset used for predicting whether a passenger will survive an accident or not is provided here: https://www.kaggle.com/c/titanic/download/train.csv

#### The link to the dataset used for classifying whether an email is spam or not is provided here:
https://archive.ics.uci.edu/ml/datasets/Spambase 

#### 07/10/2019
#### By **Booorayan**
## Description
The objectives of the project were:
  * Predict whether passengers will survive or not using knn classifier
  * Classify mails as either spam or not using naives bayes classifier 



## Experiment Design
This project employed the CRISP-DM methodology. The methodology entailed the following phases:
  * Problem Understanding
  * Data Understanding
  * Data Preparation/Cleaning:
  * Modelling:

      Models used included naives bayes classifier, random forest, knn, xgboost classifier and logistic regression
  * Evaluation of Model:

      The metrics used for evaluating the performance of the models included classification report, confusion matrix, 
      cross val score and accuracy score
      
         print('\n' + '===='*20)
         np.round(metrics.accuracy_score(ttargg_test, bern_pred) * 100, 2)
         print('\n' + '===='*20)
         print('Classification Report:\n', metrics.classification_report(ttargg_test, bern_pred))
         print('\n' + '===='*20)
         print('Confusion Matrix:\n', metrics.confusion_matrix(ttargg_test, bern_pred))
         print('\n' + '===='*20)

      
      Various classification models were compared with the knn classifier to determine the best models for predicting
      whether one survived or not.
      
         results = pd.DataFrame({'Model': ['Naive Bayes Gaussian', 'Random Forest', 'XGBoost', 'Logistic Regression'],
                         'Test Accuracy Score': [gacc, rfacc, xgbacc, logacc]})
         results.sort_values('Test Accuracy Score', ascending=False, inplace=True)

         results

## Libraries Used
The following python libraries were used for the project:
  * pandas 
  * numpy
  * scikit 
  * seaborn
  * matplotlib

The ML models used for the project include:
  * KNN classifier
  * Logistic regression
  * XGBoost Classifier
  * Random forest 
  * Naives Bayes Classifier


### License
*MIT*
Copyright (c) 2019 **Booorayan**
