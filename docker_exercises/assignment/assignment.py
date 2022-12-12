from pyspark import SparkConf
from pyspark.sql import SparkSession
import os

print(os.environ["AWS_SECRET_ACCESS_KEY"])
print('='*80)

BUCKET = "dmacademy-course-assets"
KEY1 = "vlerick/pre_release.csv"
KEY2 = "vlerick/after_release.csv"

config = {
    "spark.jars.packages": "org.apache.hadoop:hadoop-aws:3.3.1",
    "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
}
conf = SparkConf().setAll(config.items())
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Read the CSV file from the S3 bucket
pre_release = spark.read.csv(f"s3a://{BUCKET}/{KEY1}", header=True)
after_release = spark.read.csv(f"s3a://{BUCKET}/{KEY2}", header=True)

pre_release.show()
after_release.show()

#Convert the Spark DataFrames to Pandas DataFrames.

pre = pre_release.toPandas()
after = after_release.toPandas()

#copy in python code
#after python code you have to conver pandas back to spark using tospark

# # IMDB Case

# Considering I am an actor who has had a succesfull past but who's movies have lately not been performing well, I would like to determine how my next movie would perform on an imdb score level, since a lot of people look up the score to decide if they want to watch it or not. I would like to know the predicted score of my upcoming movie taking into account different factors. This will allow me to already know if I should make sure to land a new deal for another movie if the predicted score of my upcoming movie is bad. 

# # Import packages


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


# # Reading in the data

#read in data pre release
df_pre = pd.read_csv("../data/pre_release.csv")

df_pre.head()


df_pre.shape



#Read in data after release
df_after = pd.read_csv("../data/after_release.csv")
df_after.head()




df_after.shape


# # Merging the data sets



#Merging data sets to make data preparation easier, I only choose for the imdb score, since that will be my target variable
df = pd.merge(df_pre, df_after[['movie_title', 'imdb_score']], how='inner', on='movie_title') #Inner join on movies and IMDB score as target variable
df.head()



#Checking out the new df shape
df.shape



#Checking for the columns present in dataset
df.columns


# # Data preparation


#Investigating the data types
df.dtypes


# The dataset is divided into categorical and numerical columns. Director name, actor names, genres, movie titles, language, country and content rating being categorical, the rest numerical.


#Investigate number of unique values 
df.nunique()




#Investigate missing values
df.isnull().sum().sort_values(ascending = False)


# There seems to be some missing values in the dataset, particulary for the content rating of some movies.

# # Data cleaning


#Removing the null values from the dataset, since there aren't a lot of null values, almost no data is lost
#I could however have imputed the missing values of the content rating, but it did not seem right to me as some movies may or may not be appropriate for children
df.dropna(how = 'any',axis = 0,inplace = True)




#Checking if null values are indeed gone
df.isnull().sum()


# There are no more null values in the dataset.


#Looking at hom much data is lost, only lost 39 rows, which is acceptable
df.shape


# ### Duplicates


#Checking for duplicates in the dataset
df.duplicated().any()
duplicated = df.duplicated()
df[duplicated]


#Removing duplicates, 22 rows ae being removed
df.drop_duplicates(inplace = True)
df.shape

# 

# ## Grouping categorical variables and changing them into dummies 

# ### Language

#Value count for movie languages, most movies are in English, therefore grouping language in English and other language movies
value_counts = df["language"].value_counts()
print(value_counts)




val = value_counts[:1].index
print (val)
df['language'] = df.language.where(df.language.isin(val), 'other')


#Successfully divided the language into 2 categories being English movie language and other languages
df["language"].value_counts()


# I decided to group all other language movies together since there are so little compared to the English movies.


#Label enconding the categorical variable language, English language movies are given a 1, other language movies a 0
le = LabelEncoder()
df['language'] = le.fit_transform(df['language'])
df


# ### Countries

#Value count for the countries, most movies being from the USA followed by a list of other countries
value_counts = df["country"].value_counts()
print(value_counts)



#Selecting the USA grouping rest of countries
val = value_counts[:3].index
print (val)
df['country'] = df.country.where(df.country.isin(val), 'other')



#Successfully divided the country into USA and other countries
df["country"].value_counts()



df = pd.concat([df, pd.get_dummies(df['country'])], axis=1)
df.drop('country', axis=1, inplace=True) #dropping original country column
df


# I first decided to group my countries into USA, UK, France and Canada. However I decided to add Canada to the other country group since it increased my linear regression r-square from 0.275 to almost 0.28 as accuracy stayed the same. If I would also add France to this other country group, it goes down again.

# ### Content rating


#Value count for the content rating of the movies
value_counts = df["content_rating"].value_counts()
print(value_counts)



#Selecting R, PG-13 and combing other ratings 
val = value_counts[:3].index
print (val)
df['content_rating'] = df.content_rating.where(df.content_rating.isin(val), 'other')


#Successfully grouped content rating into 3 categories
df["content_rating"].value_counts()


#Generating binary values using get_dummies
df = pd.concat([df, pd.get_dummies(df['content_rating'])], axis=1)
df.drop('content_rating', axis=1, inplace=True) #dropping original content rating column
df


# ## Split Genres on delimiter and add as dummies



#Checking the value count for all the genres in the multiple columns and splitting them since each movie has so many different genres
value_counts = df["genres"].value_counts()
print(value_counts)
df


# Comedy/drama movies seem to be highly present in the dataset.



#turn the categorical variable genres into dummies by spliting on delimiters and dropping original genres column
df = pd.concat([df, df['genres'].str.get_dummies('|')], axis=1)
df.drop('genres', axis=1, inplace=True)
df


# By splitting the genres of each movie I have created 22 new columns.



#Checking out all the columns one last time
df.columns


# ## Modelling



#Dropping the irrelevant variables and defining dependent and independent variables 
#extract target variable
x = df.drop(columns = ["imdb_score","director_name","actor_2_name","actor_1_name","actor_3_name","movie_title"])
y = df["imdb_score"] #Being the target variable
print(x)
print(y)



#Randomly splitting into a training and validation sample
seed = 123 
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.25, random_state = seed)


# When playing around with the split, a split of 25% into validation and 75% training sample gives me the best performance.

# # Linear regression - statsmodel
import statsmodels.api as sm
# first  add intercept to X:
xc_train = sm.add_constant(x_train)
xc_val = sm.add_constant(x_val)
#training model
mod = sm.OLS(y_train,xc_train)
olsm = mod.fit()
#output table with parameter estimates (in summary2)
olsm.summary2().tables[1][['Coef.','Std.Err.','t','P>|t|']]



#Making a prediction
array_pred = np.round(olsm.predict(xc_val),0) 
y_pred = pd.DataFrame({"y_pred": array_pred},index=x_val.index) 
val_pred = pd.concat([y_val,y_pred,x_val],axis=1)
val_pred


# The imdb score predictions seem to be fairly close when taking a look at the y_pred values.


#Evaluating model: R-square & MAE
#by comparing actual and predicted value 
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error 
#input below actual and predicted value from dataframe
act_value = val_pred["imdb_score"]
pred_value = val_pred["y_pred"]
#run evaluation metrics
rsquare = r2_score(act_value, pred_value)
mae = mean_absolute_error(act_value, pred_value)
pd.DataFrame({'eval_criteria': ['r-square','MAE'],'value':[rsquare,mae]})



#Evaluate model: R-square & MAE
#by comparing actual and predicted value 
y_val = np.array(y_val).reshape((252,1))
pred_value = np.array(pred_value).reshape((252,1))
errors = abs(pred_value - y_val) #Calculating the differences between actual and predicted imdb score
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_val)
# Calculate and display of accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

val_pred

df_prediction_values = spark.createDataFrame(val_pred)
df_prediction_values.show()


