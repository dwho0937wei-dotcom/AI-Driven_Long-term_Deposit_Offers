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

 The *Business Objective* is to find the best classification model for predicting whether the customer will subscribe to a term deposit offer or not (output variable y) based on the available data description of the customer (features). That way, the Portuguese banking can most accurately predict which customer will likely accept the offer or not and respond accordingly.

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

### DataFrame Transformation
Hence, starting from the DataFrame in its original form, passing through what I did in **Problem 3**, skipping past (i.e. NOT doing) **Problem 5**, and the numerous new steps I've taken to encode the DataFrame, here's the entire transformation process in ordered steps:

1. Filling in the missing values "unknown" in columns job, marital, education, default, housing, and loan.
2. Filling in the missing values "999" in column pdays.
3. Changing column pdays from object-type to numeric-type (float64).
4. OneHotEncoding "pd.get_dummies()" the nominal categorical columns job, marital, contact, and poutcome.
5. OrdinalEncoding the ordinal categorical columns education, month, and day_of_week.
6. LabelEncoding (Binarizing) the binary categorical columns default, housing, and loan.
7. Normalizing the non-encoded numeric features age, duration, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, and nr.employed using the StandardScaler().

As I can't show the entire DataFrame since it is too long, this is the partial DataFrame before the transformation:

![alt-text](https://github.com/dwho0937wei-dotcom/Module17_Project/blob/main/README_images/Problem_11_Original_DataFrame.PNG)

and this is the partial DataFrame after the transformation:

![alt-text](https://github.com/dwho0937wei-dotcom/Module17_Project/blob/main/README_images/Problem_11_Transformed_DataFrame.PNG)

### Train/Test Split
After that, I've assigned the features to variable "X_transformed" and the output variable to variable "y_transformed".
I then train/test split it into "X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed" in a train/test ratio of 70:30.

### Deciding A Performance Metric (Recall)
When calculating the class no:yes ratio in the output variable y, I get:                                                                                                                      
Ratio no:yes is 88.73458288821988 : 11.265417111780131

As you can see, the class is incredibly imbalanced.
Not only that, there are way more 'no' negatives than 'yes' positives.
Having limited 'yes' positives implies that we should prioritize on finding these true positives since they are valuably rare.
Hence, in response, I shall choose **recall** as the models' performance metric.

### GridSearching The Four Models
Since I am using a new performance metric recall and I have new training & testing sets in **Train/Test Split** section, different from **Problem 5**, I decided for every model of the four, I will reuse its simple form from **Problem 8-10** to train and test on the new sets and instead of calculating its accuracy, I'll be calculating its recall this time. I will also be gridsearching with the hyperparameters of my own choice.
For every model, I'll show its simple form, what I gridsearched from it, and its adjusted hyperparameters form, as well as the two forms' training time, and training/testing recalls.
#### Logistic Regression (LR)
For this model, its default max_iter is 100 but due to the large training set it's training on, I had to bump the max_iter up to 1000.

Simple Form:
```python
LogisticRegression(max_iter=1000)
```

Simple's Result:                                                                                                                                                        
Logistic Regression                                                                                                                                                                             
Training Time:  0.5116322040557861                                                                                                                                                              
Training Recall:  0.39118457300275483                                                                                                                                                            
Testing Recall:  0.4246176256372906                                                                                                                                                              

Params I've GridSearched:
```python
lr_params = {'penalty': ['l1', 'l2', 'elasticnet'],
             'tol': [1e-7 * 10**i for i in range(7)], # [1e-7, 1e-6, ..., 1e-1]
             'C': np.linspace(1e-10, 1, 10)}
```

GridSearched Form:
```python
lr_best = LogisticRegression(max_iter=1000, 
                             random_state=42, 
                             C=0.3333333334,
                             penalty='l2',
                             tol=1e-07)
```

GridSearched's Result:                                                                                                                                                                           
Best Recall LR                                                                                                                                                                                   
Training Time:  0.5236012935638428                                                                                                                                                              
Training Recall:  0.3899602081420263                                                                                                                                                             
Testing Recall:  0.42534595775673706                                                                                                                                                             

#### K-Nearest Neighbor (KNN)

Simple Form:
```python
KNeighborsClassifier()
```

Simple's Result:                                                                                                                                                        
KNN                                                                                            
Training Time:  0.041918039321899414                                                                       
Training Recall:  0.5485154576063667                                                                      
Testing Recall:  0.4282592862345229    

Params I've GridSearched:
```python
knn_params = {'n_neighbors': np.arange(3, 8, 1),
              'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
```

GridSearched Form:                                                                                       
```python
knn_best = KNeighborsClassifier(n_neighbors = 3,
                                weights = 'distance',
                                algorithm = 'auto'
                            )
```

GridSearched's Result:                                                                                                 
Best Recall KNN                                                                                                      
Training Time:  0.04787302017211914                                                                                  
Training Recall:  1.0                                                                                                       
Testing Recall:  0.44646758922068464                                                                                                  

#### Decision Tree (DT)

Simple Form:
```python
DecisionTreeClassifier(random_state = 42)
```

Simple's Results:                                                            
DT                                                                                                              
Training Time:  0.2433474063873291                                                                     
Training Recall:  1.0                                                                            
Testing Recall:  0.5069191551347414                                                                             

Params I've GridSearched:                                                                                                               
```python
dt_params = {'criterion': ['gini', 'entropy', 'log_loss'],
             'splitter': ['best', 'random'],
             'max_features': [None, 'sqrt', 'log2'],
             'max_leaf_nodes': [5, 10, 15],
             'max_depth': [5, 10, 15]}
```

GridSearched Form:                                                                                              
```python
dt_best = DecisionTreeClassifier(
                criterion='gini',
                max_features=None,
                splitter='best',
                max_leaf_nodes=10,
                max_depth=5
            )
```

GridSearched's Result:                                                                        
Best Recall DT                                                                                            
Training Time:  0.08676648139953613                                                                                    
Training Recall:  0.6636057545148454                                                                                  
Testing Recall:  0.6722505462490895                                                                                            

#### Support Vector Machine (SVM)

Simple's Form:
```python
SVC(random_state = 42)
```

Simple's Result:                                                                   
SVM                                                                                              
Training Time:  13.517419815063477                                                                     
Training Recall:  0.35751453933272115                                                          
Testing Recall:  0.3758193736343773                                                                        

Params I've GridSearched:
```python
svc_params = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
             'degree': [1, 3, 5, 7, 10]}
```

GridSearched Form:
```python
svc_best = SVC(
                kernel='rbf',
                degree=1
            )
```

GridSearched's Result:                                                                                                  
Best Recall SVM                                                                                                      
Training Time:  16.14379382133484                                                                                                     
Training Recall:  0.35751453933272115                                                                                                  
Testing Recall:  0.3758193736343773                                                                                    

### Visualizations
To compare the models' performances, I've decided to create three barplots: first for Training Time, second for Training Recall, and third for Testing Recall.                                  
Here're the resulting barplots:

![alt-text](https://github.com/dwho0937wei-dotcom/Module17_Project/blob/main/README_images/Problem_11_Training_Time_Barplot.PNG)

![alt-text](https://github.com/dwho0937wei-dotcom/Module17_Project/blob/main/README_images/Problem_11_Training_Recall_Barplot.PNG)

![alt-text](https://github.com/dwho0937wei-dotcom/Module17_Project/blob/main/README_images/Problem_11_Testing_Recall_Barplot.PNG)

According to the barplots, I'd say the Decision Tree takes the win for the best testing recall. On top of that, it's the second fastest model, just has a slightly longer performance time than the almost instantaneous KNN.

### Conclusion
In conclusion, we have Decision Tree model:
```python
dt_best = DecisionTreeClassifier(
                criterion='gini',
                max_features=None,
                splitter='best',
                max_leaf_nodes=10,
                max_depth=5
            )
```
as the best classification model. With this model, the Portuguese banking will have a greater chance of finding these precious customers who are willing to accept their offer (true positives).
While prioritizing recall means that they will more likely stumble upon customers that will unexpectedly reject their offer (false positives),
I personally perceive it as at least the only thing they are wasting is time.
