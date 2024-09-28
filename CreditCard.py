# -*- coding: utf-8 -*-
"""
Created on Sat May 11 12:55:29 2024

@author: ssarkar4
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df= pd.read_csv("CreditCard.csv")

print(df.head())
print("Number of rows:", len(df))

#Data Cleaning and transformation
#Dropping unnecessary column from my main frame
df.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis=1, inplace=True)

# Checking for duplicated values
duplicated_count = df.duplicated().sum()
print("Number of duplicated rows:", duplicated_count)

#DataCleaning1
print(df.columns)
# Define the new column names for easy interpretation
new_column_names = {
    "CLIENTNUM": "Client_ID",
    "Attrition_Flag": "Churn_Status",
    "Customer_Age": "Age",
    "Gender": "Gender",
    "Dependent_count": "Dependents",
    "Education_Level": "Education_Level",
    "Marital_Status": "Marital_Status",
    "Income_Category": "Income_Range",
    "Card_Category": "Card_Type",
    "Months_on_book": "Tenure_Months",
    "Total_Relationship_Count": "Product_Count",
    "Months_Inactive_12_mon": "Months_Inactive_12",
    "Contacts_Count_12_mon": "Contacts_Count_12",
    "Credit_Limit": "Credit_Limit",
    "Total_Revolving_Bal": "Revolving_Balance",
    "Avg_Open_To_Buy": "Available_Credit",
    "Total_Amt_Chng_Q4_Q1": "Amt_Change_Q4Q1",
    "Total_Trans_Amt": "Total_Trans_Amount",
    "Total_Trans_Ct": "Total_Trans_Count",
    "Total_Ct_Chng_Q4_Q1": "Trans_Count_Change_Q4Q1",
    "Avg_Utilization_Ratio": "Avg_Utilization_Ratio",
}
# Rename the columns
df.rename(columns=new_column_names, inplace=True)
print(df.columns)

#DataCleaning2
# Create a copy of the original dataframe
df_before_normalization = df.copy()

# Function to normalize column names and text values to lowercase
def normalize_text_and_column_names(df):
    # Normalize column names
    df.columns = [col.lower() for col in df.columns]
    # Normalize text values in the DataFrame
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].str.lower()
    return df

#calling the function
df = normalize_text_and_column_names(df)


#Data Cleaning 3
# Creating DataFrame with Dtype, Unique, and Null information
#only to check before modification
df_info = pd.DataFrame(df.dtypes, columns=['Dtype'])
df_info['Unique'] = df.nunique().values
df_info['Null'] = df.isnull().sum().values
# Displaying the DataFrame
print(df_info)

# Convert object columns to category data type
df['churn_status'] = df['churn_status'].astype('category')
df['gender'] = df['gender'].astype('category')
df['education_level'] = df['education_level'].astype('category')
df['marital_status'] = df['marital_status'].astype('category')
df['income_range'] = df['income_range'].astype('category')
df['card_type'] = df['card_type'].astype('category')

# Creating DataFrame with Dtype, Unique, and Null information
#only to check after modification
df_info = pd.DataFrame(df.dtypes, columns=['Dtype'])
df_info['Unique'] = df.nunique().values
df_info['Null'] = df.isnull().sum().values
# Displaying the DataFrame
print(df_info)


# Data Cleaning 4
# Print the counts of unique values in the 'Income_Range' column before imputation
income_range_counts_before = df['income_range'].value_counts()
print("Income Range Counts before imputation:")
print(income_range_counts_before)

# Print the counts of unique values in the 'Education_Level' column before imputation
education_level_counts_before = df['education_level'].value_counts()
print("Education Level Counts before imputation:")
print(education_level_counts_before)

# Calculate the distribution of income categories (excluding 'Unknown' values)
income_range_distribution = df[df['income_range'] != 'unknown']['income_range'].value_counts(normalize=True)

# Calculate the distribution of education levels (excluding 'Unknown' values)
education_level_distribution = df[df['education_level'] != 'unknown']['education_level'].value_counts(normalize=True)

# Impute missing values based on the distribution
def impute_missing_categories(row, distribution, column):
    if row[column] == 'unknown':
        return np.random.choice(distribution.index, p=distribution.values)
    else:
        return row[column]

# Apply the imputation function to the 'Income_Range' column
df['income_range'] = df.apply(lambda row: impute_missing_categories(row, income_range_distribution, 'income_range'), axis=1)

# Apply the imputation function to the 'Education_Level' column
df['education_level'] = df.apply(lambda row: impute_missing_categories(row, education_level_distribution, 'education_level'), axis=1)

# Count the occurrences of each unique value in the 'Income_Range' column after imputation
income_range_counts_after = df['income_range'].value_counts()

# Print the counts of unique values in the 'Income_Range' column after imputation
print("Income Range Counts after imputation:")
print(income_range_counts_after)

# Count the occurrences of each unique value in the 'Education_Level' column after imputation
education_level_counts_after = df['education_level'].value_counts()

# Print the counts of unique values in the 'Education_Level' column after imputation
print("Education Level Counts after imputation:")
print(education_level_counts_after)



#Summary Statistics
# 1. Inactive Months
months_inactive_12_stats = df['months_inactive_12'].describe()

# 2. Total Transaction Count
total_trans_count_stats = df['total_trans_count'].describe()

# 3. Revolving Balance
revolving_balance_stats = df['revolving_balance'].describe()

# 4. Available Credit
available_credit_stats = df['available_credit'].describe()

# 5. Amount Change Q4Q1
amount_change_q4q1_stats = df['amt_change_q4q1'].describe()

# Print the summary statistics for each variable
print("1. Months Inactive:")
print(months_inactive_12_stats)
print("\n2. Total Transaction Count:")
print(total_trans_count_stats)
print("\n3. Revolving Balance:")
print(revolving_balance_stats)
print("\n4. Available Credit:")
print(available_credit_stats)
print("\n5. Amount Change Q4Q1:")
print(amount_change_q4q1_stats)


# Visualization 1
# Q1: What is the distribution of ages among the customers?
plt.figure(figsize=(10, 6))
# Calculate percentages
age_counts = df['age'].value_counts(normalize=True) * 100

# Plot histogram with percentages
plt.hist(df['age'], bins=[25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80],
         weights=np.ones(len(df['age'])) / len(df['age']) * 100, color='mediumorchid', edgecolor='black', alpha=0.7)

plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Percentage')
plt.xticks([25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#Visualization2
#Q2) How does churn status vary across different demographic segments, such as income range and gender?
#Part-1 Plotting churn with income range
plt.figure(figsize=(10, 6))
sns.countplot(x='income_range', hue='churn_status', data=df, palette='Set2')

# Adding values on top of each bar
for p in plt.gca().patches:
    height = p.get_height()
    plt.gca().text(p.get_x() + p.get_width() / 2, height + 3, int(height), ha='center', va='bottom')

plt.title('Churn Status by Income Range')
plt.xlabel('Income Range')
plt.ylabel('Count')
plt.legend(title='Churn Status')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Part2 Plotting churn with gender
# Group data by gender and churn status, and calculate churn counts
churn_counts = df.groupby(['gender', 'churn_status']).size().unstack()

# Plotting the grouped bar chart
plt.figure(figsize=(10, 6))
ax = churn_counts.plot(kind='bar', stacked=True, color=['blue', 'lightblue'])
plt.title('Churn Status by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Churn Status')

# Adding text annotations for counts on top of each bar
for p in ax.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height}', (x + width/2, y + height*1.02), ha='center')

plt.tight_layout()
plt.show()


#Visulization3
#Q3) How does the distribution of credit limits vary across different education levels?
# Plotting the box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='education_level', y='credit_limit', data=df, palette='Set2')
plt.title('Credit Limit by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Credit Limit')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Visulization4
#Q4)What is the distribution of card types among customers?
# Calculate the distribution of card types
card_type_counts = df['card_type'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(card_type_counts, labels=card_type_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Card Types')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()

#Visulization5
#Q5)What is the relationship between credit limit and average utilization ratio?
# Plotting scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['credit_limit'], df['avg_utilization_ratio'], alpha=0.5, color='green')
plt.title('Credit Limit vs. Avg Utilization Ratio')
plt.xlabel('Credit Limit')
plt.ylabel('Average Utilization Ratio')
plt.grid(True)
plt.show()

#Visulization6
#Q6)How does the total transaction amount vary with tenure months?
# Plotting scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['tenure_months'], df['total_trans_amount'], alpha=0.5, color='green')
plt.title('Total Transaction Amount vs. Tenure Months')
plt.xlabel('Tenure Months')
plt.ylabel('Total Transaction Amount')
plt.grid(True)
plt.show()

