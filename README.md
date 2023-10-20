# Practical Application III

In this Practical Application 3 (PA3), I will be going through what I did in the 10 Problems assigned to me.

The [data](https://archive.ics.uci.edu/dataset/222/bank+marketing) being represented in PA3 is from a Portugese banking institution and is a collection of the results of multiple marketing campaigns.

## Problem 1: Understanding the Data
I just explained my examination of the **Materials and Methods** section in the [CRISP-DM-BANK.pdf](https://github.com/dwho0937wei-dotcom/Module17_Project/blob/main/CRISP-DM-BANK.pdf) along with how many marketing campaigns that the data represents.
This is my following response:

According to the "Materials and Methods" section, the data represent 17 marketing campaigns from May 2008 to November 2010. Each row represents each call made during the entire series of campaigns, and there are 79354 calls in total.
During the series of campaigns, there was also an attractive long-term deposit application, with good interest, being offered to every customer and hence the target variable 'y' represents if the customer accepts it.
Each call has the following attributes related to the customer:
- age (age)
- type of job (job)
- marital status (marital) - highest educational level (education)
- has credit in default? (default)
- has housing loan? (housing)
- has personal loan? (loan)
- type of contact communication (contact)
- last contact day of the week (day_of_week)
- last contact month of the year (month)
- duration of their last contact measured in seconds (duration)
- number of contacts made during this campaign towards the customer (campaign)
- number of days passed by after the customer was last contacted from a previous campaign (pdays)
- number of contacts made before this campaign and for this customer (previous)
- outcome of the previous marketing campaign (poutcome)
- has the client subscribed a term deposit (y)

## Problem 2: Read in the Data
I just assigned the variable "campaign_calls_df" as the DataFrame of the dataset "bank-additional-full.csv".

## Problem 3: Understanding the Features
When I ran the following code:
```python
campaign_calls_df.isnull().sum()
```
there are no null values in any of the input (features) and output (y) variables.

However, that is when reading the data description given to me in this problem becomes significant.
When reading the description carefully, I've realized that features job, marital, education, default, housing, and loan have the value "unknown" interpreted as their missing value.
I decided on a whim to count the value 999 in feature pdays as a missing value too even though it means that the customer hasn't been previously contacted according to the data description.

When creating a dictionary of the specified features with their corresponding frequency of missing values, I have:

Missing Values

{'job': 330,
 'marital': 80,
 'education': 1731,
 'default': 8597,
 'housing': 990,
 'loan': 990,
 'pdays': 39673}

 Knowing that there are 79354 calls in total in the dataset, I believe that the number of missing values are too significant for me to just simply delete away.
 Hence, I decided to use the SimpleImputer to replace all "unknown" values with frequent non-missing values of their corresponding features.
 I used the IterativeImputer to replace all "999" values in feature pdays.

 After that, I've confirmed that there are no more missing values by recreating the dictionary shown below:

 Missing Values

{'job': 0,
 'marital': 0,
 'education': 0,
 'default': 0,
 'housing': 0,
 'loan': 0,
 'pdays': 0}

 ## Problem 4: Understanding the Task
 Here, I simply stated the *Business Objective*. This is my following response:

 The *Business Objective* is to find the best classification model for predicting whether the customer will subscribe to a term deposit offer or not based on the available data description of the customer. That way, the Portuguese banking can most accurately predict which customer will likely accept the offer or not and respond accordingly.

 ## Problem 5: Engineering Features
As given, I'll only be preparing the bank information features (columns 1-7) shown below for the training/testing for the upcoming models in **Problem 8-10**.
That means I'll disregard other features in the dataset from the upcoming training/testing in **Problem 8-10**, i.e. they won't be used for the specified training/testing.

![alt-text](https://github.com/dwho0937wei-dotcom/Module17_Project/blob/main/README_images/Problem_5_DataFrame.PNG)

I've decided to scale the 'age' column using the StandardScaler(), OneHotEncode the nominal categorical columns job, marital, and education using the pd.get_dummies(), and finally, binarize the binary columns default, housing, and loan using the LabelEncoder(). This is the transformed DataFrame in its partial form:

![alt-text](https://github.com/dwho0937wei-dotcom/Module17_Project/blob/main/README_images/Problem_5_DataFrame_Transformed.PNG)

The transformed DataFrame will be assigned to variable "X_encoded" and the output variable (y) will be assigned to variable "y".

Just to let you know beforehand, I'll be encoding the DataFrame differently in **Problem 11**.

## Problem 6: Train/Test Split
This is how I train/test split it.
```python
from sklearn.model_selection import train_test_split
```
```python
# Train/Test ratio shall be 70:30
X_encoded_train, X_encoded_test, y_train, y_test = train_test_split(X_encoded, y, test_size = 0.3, random_state = 42)
```

## Problem 7: Baseline Model
I decided to have the baseline model assume that all customer will accept the term deposit offer and thus, all output variable y will be "yes".
When calculating the model's training and testing accuracy, they are extremely low as shown below:
Baseline Training Accuracy:  0.11276057021955534
Baseline Testing Accuracy:  0.11240592376790483

## Problem 8: A Simple Model
Here, I just built a simple logistic regression model with its default hyperparameters except having "random_state = 42".
I also calculated the time it took to train the model which will be later useful in **Problem 11**.

## Problem 9: Score the Model
When scoring the logistic regression from **Problem 8** by its training accuracy using variables "X_encoded_train" and "y_train" and its testing accuracy using variables "X_encoded_test" and "y_test", I have the following high accuracy:

Logistic Regression
Training Accuracy:  0.8872394297804447
Testing Accuracy:  0.8875940762320952

## Problem 10: Model Comparisons
Following the same steps I've did in **Problem 8 & 9** on models KNN, Decision Tree, and Support Vector Machine including setting "random_state = 42" if possible, I then create a DataFrame that shows the Train Time, Train Accuracy, and Test Accuracy of each of the four models Logistic Regression, KNN, Decision Tree, and Support Vector Machine.
Here's the resulting DataFrame below:

![alt-text](https://github.com/dwho0937wei-dotcom/Module17_Project/blob/main/README_images/Problem_10_Performance_DataFrame.PNG)

By the looks of it, I'd say that the Training/Testing Accuracy are similar in all four models so I would prefer the fastest model of the four which would be the KNN.
However, note that KNN is only chosen when all four models are in their default parameters, some in "random_state = 42". 
You will see which of these four models I've chosen in the end of **Problem 11**.

## Problem 11: Improving the Model
I hope you're ready for **Problem 11** because there're a lot of things that I'll be doing including some backtracking.

To start off, I've decided to revert back to the DataFrame in its imputed form in **Problem 3** because there were some features that I believe I should've encoded with the OrdinalEncoder() rather than pd.get_dummies() in **Problem 5**. Also, instead of relying on just the bank information features (columns 1-7) for the models' training/testing as specified in **Problem 5**, I will this time rely on ALL the features in the ENTIRE dataset. 

Hence, starting from the DataFrame in its original form, passing through what I did in **Problem 3**, skipping past (i.e. NOT doing) **Problem 5**, and the numerous new steps I've taken to encode the DataFrame, here's the entire transformation process in ordered steps:

1. Filling in the missing values "unknown" in columns job, marital, education, default, housing, and loan.
2. Filling in the missing values "999" in column pdays
