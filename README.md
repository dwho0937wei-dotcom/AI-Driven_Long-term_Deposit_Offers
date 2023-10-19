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
I decided on a whim to count the value 999 in feature pdays as a missing value too even though it means that the customer hasn't been previously contacted.

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

 ## Problem 4: Understanding the Task
