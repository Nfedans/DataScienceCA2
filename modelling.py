import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

# for feature engineering date variable
from datetime import datetime

# for turning categorical variable into numerical
from collections import Counter

#
#                         IMPORTANT CHANGES
# Video RANK is no longer my response variable, I am instead predicting VIEWS
# This doesn't change much however as my points about outside businesses looking for
# Influencers to advertise their products, would want their advertisement to have many
# Views, and a video ranking highly, pretty much means it is getting a lot of views currently.
#
# Rumble creator's goal to rank highly / get many views is also similar in nature
# Therefore, the business understanding still applies regradless if i am trying to
# predict views or video rank.
#

# DATA IMPORT########################################################

import os

cwd = os.getcwd()
print(cwd)
os.chdir("C:/Users/nfeda/OneDrive - Dundalk Institute of Technology/Y3/Data Science/CA2")

# Read the data set of "Marketing Analysis" in data.
data = pd.read_csv("dataForModelling.csv")

# Little cleanup pre-feature-engineering############################

# Drop the Duration column because the "Duration Seconds" column replaces it and makes the values a useable format for calculations
data.drop('Duration', axis=1, inplace=True)
# Drop an index column which was automatically added from reading data from csv
data.drop('Unnamed: 0', axis=1, inplace=True)

# Feature Engineering###############################################

# In this section I have two main goals: 1) turn "channels" variable from categorical -> numerical
# And 2) create a variable derived from video upload dates and the scrape date (4th novemeber) which
# will show at the time of the scrape process, how many days a video has been uploaded for.
# I am fairly certain a this new variable will correlate well with views

#########Feature Engineering Step 1: Identify Variables

data.info()
#  #   Column                          Non-Null Count  Dtype
# ---  ------                          --------------  -----
#  0   Rank                            1221 non-null   int64 (Numerical, Explanatory)
#  1   Channel                         1221 non-null   object (Categorical) IGNORE
#  2   Views                           1221 non-null   int64 (Numerical, Response)
#  3   Likes                           1221 non-null   int64 (Numerical, Explanatory)
#  4   Uploaded                        1221 non-null   object (Numerical, but needs to be developed) IGNORW
#  4   Subscribers                     1221 non-null   int64 (Numerical, Explanatory)
#  5   Duration Seconds                1221 non-null   int64 (Numerical, Explanatory)


#########Feature Engineering Step 3: Drop certain variables

# Create days uploaded
# this is a rough approximation of the time I scraped my data
scrape_time_str = "2022-11-04T17:00:00-04:00"


# This function cleans up scraped dates so i can format them as objects rather than strings
def format_date(date_str):
    date_str = date_str.translate({ord('T'): " "})
    date_str = date_str[:-6]
    return date_str


# This function calculates number of days between video upload date and data scrape date
def days_between(d1, d2):
    format_str = '%Y-%m-%d %H:%M:%S'
    d1 = format_date(d1)
    d2 = format_date(d2)
    d1 = datetime.strptime(d1, format_str)
    d2 = datetime.strptime(d2, format_str)
    return abs((d2 - d1).days)


# The below code iterates through all upload dates, and for each date gives a number of days since upload variable
days_uploaded = []

for index, row in data.iterrows():
    days_uploaded.append(days_between(row['Uploaded'], scrape_time_str))

# Add column to pandas dataframe
data['days uploaded'] = days_uploaded

# Turn categorical variable "Channel" into Numerical variable
input_list = data.Channel.values.tolist()
# This gives the unique values in my list of channels, as well as frequency
value_frequency_list = Counter(input_list)


def get_channel_name(index):
    return value_frequency_list.most_common()[index][0]


# This function makes 6 new numerical variables to dataframe
def add_top_5_channels_as_columns_plus_catchall():
    for x in range(5):
        channel_name = value_frequency_list.most_common()[x][0]
        data["Chnl " + channel_name] = np.where(data.Channel == channel_name, 1, 0)

    data["Other Chnl"] = np.where((
            (data.Channel != get_channel_name(0)) &
            (data.Channel != get_channel_name(1)) &
            (data.Channel != get_channel_name(2)) &
            (data.Channel != get_channel_name(3)) &
            (data.Channel != get_channel_name(4))), 1, 0)


add_top_5_channels_as_columns_plus_catchall()

#########Feature Engineering Step 2: Drop certain variables
# Get rid of the uploaded variable since it is of no use now
data.drop('Uploaded', axis=1, inplace=True)

# Get rid of the channel variable as it has been transformed into 6 numerical, useful columns
data.drop('Channel', axis=1, inplace=True)

# Check data clean
data.isnull().sum()  # Clean data with no missing values

data.info()

#  #   Column                          Non-Null Count  Dtype
# ---  ------                          --------------  -----
#  0   Rank                            1221 non-null   int64 (Numerical, Explanatory)
#  1   Views                           1221 non-null   int64 (Numerical, Response)
#  2   Likes                           1221 non-null   int64 (Numerical, Explanatory)
#  3   Subscribers                     1221 non-null   int64 (Numerical, Explanatory)
#  4   Duration Seconds                1221 non-null   int64 (Numerical, Explanatory)
#  5   days uploaded                   1221 non-null   int64 (Numerical, Explanatory)
#  6   Chnl BonginoReport              1221 non-null   int32 (Numerical, Explanatory)
#  7   Chnl The Gateway Pundit         1221 non-null   int32 (Numerical, Explanatory)
#  8   Chnl sonsoflibertyradiolive     1221 non-null   int32 (Numerical, Explanatory)
#  9  Chnl The Post Millennial Clips  1221 non-null   int32 (Numerical, Explanatory)
#  10  Chnl X22 Report                 1221 non-null   int32 (Numerical, Explanatory)
#  11  Other Chnl                      1221 non-null   int32 (Numerical, Explanatory)

#########Feature Engineering Step 4: Scale Data
# not needed here

corrVals = data.corr()

# Plot of relationships between variables
figure(num=None, figsize=(11, 11), dpi=80, facecolor='w', edgecolor='k')
sns.pairplot(data)
# plt.savefig("foo.pdf")
plt.show()

# Correlations
data_for_heatmap = data[
    ['Subscribers', 'Rank', 'Likes', 'Views', 'Duration Seconds', 'days uploaded', 'Chnl BonginoReport',
     'Chnl The Gateway Pundit', 'Chnl sonsoflibertyradiolive', 'Chnl The Post Millennial Clips',
     'Chnl X22 Report', 'Other Chnl']].corr()

# plot the correlation matrix
fig, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(data_for_heatmap, annot=True, cmap='Reds')
plt.show()

#########Feature Engineering Step 5: Multicolinearity
# THERE IS NONE


###############REGRESSION MODELLING######################################


# Strength of variables correlating to views
# 1) Rank (-0.58)
#    Explanation: if videos were ranked solely by view count on Rumble, a mega viral video that came out 2 years ago might top the ranking
#    Even though a video released one week ago might be gaining views at a much greater rate than this old viral video, the new video
#    may not even appear in the rankings in this scenario. Rumble may rank a video with 10000 views released 10 minutes ago higher than
#    a video with 1 million views released 3 weeks ago. This explains the substantial negative correlation.
#
# 2) Likes (0.33)
#    Explanation: The more views a video has, the more likes it may have.
#
# 3) Duration seconds (0.11)
#    Explanation: Longer videos get more views.
#
# 4) Bongino Report (Channel) (-0.1)
#
# 5) X22 report (Channel) (0.083)
#
# 6) Subscribers (0.081)
#    Explanation: The more subscribers a channel has the more views, but the correlation isn't very strong.
#
# 7) Other channel (0.076)
#
# 8) the gateway pundit (Channel) (-0.063)


#########Regression Modelling - Step 1: Split Data into Train and Test

# Set the Response and the predictor variables

x = data[['Subscribers', 'Rank', 'Likes', 'Duration Seconds', 'days uploaded', 'Chnl BonginoReport',
          'Chnl The Gateway Pundit', 'Chnl sonsoflibertyradiolive', 'Chnl The Post Millennial Clips',
          'Chnl X22 Report', 'Other Chnl']]  # pandas dataframe
y = data['Views']  # Pandas series

# Splitting the Data Set into Training Data and Test Data
from sklearn.model_selection import train_test_split

# split train 66.7%, test 33.3%. Note that if run this more than once will get different selection which can lead to different model particulalry for small datasets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.333)

y_train  # Pandas series
x_train  # Pandas dataframe

#########Regression Modelling - Step 2: Model Selection

from sklearn.linear_model import LinearRegression

model1 = LinearRegression()
model2 = LinearRegression()
model3 = LinearRegression()
model4 = LinearRegression()
model5 = LinearRegression()
model6 = LinearRegression()
model7 = LinearRegression()
model8 = LinearRegression()

# Fit the variables in order of strongest correlation with Price and calculate adjusted R squared at each step.

# Model 1 - First add Rank to model
model1.fit(x_train[['Rank']], y_train)
# Show the model parameters
print(model1.coef_)
print(model1.intercept_)
# So Views = 303227.4442156891 - 293.10853521 * Rank

# Generate predictions for the train data
predictions_train = model1.predict(x_train[['Rank']])

raw_sum_sq_errors = sum((y_train.mean() - y_train) ** 2)
raw_sum_sq_errors
prediction_sum_sq_errors = sum((predictions_train - y_train) ** 2)

Rsquared1 = 1 - prediction_sum_sq_errors / raw_sum_sq_errors
Rsquared1

N = 814  # 16 data rows
p = 1  # one predictor used
Rsquared_adj1 = 1 - (1 - Rsquared1) * (N - 1) / (N - p - 1)
print("Rsquared Regression Model with Rank: " + str(Rsquared1))
print("Rsquared Adjusted Regression Model with Rank: " + str(Rsquared_adj1))
Rsquared_adj1

######Model 2 - Next add the Pop variable
model2.fit(x_train[['Rank', 'Likes']], y_train)
# Show the model parameters
print(model2.coef_)
print(model2.intercept_)
# So Price = ??

# A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
Output = pd.DataFrame(model2.coef_, ['Rank', 'Likes'], columns=['Coeff'])
print(Output)

# Generate predictions for the train data
predictions_train = model2.predict(x_train[['Rank', 'Likes']])

# Raw sum of squares of errors is based on the mean of the y values without having any predictors to help.
raw_sum_sq_errors = sum((y_train.mean() - y_train) ** 2)

# Calculate sum of squares for prediction errors.
prediction_sum_sq_errors = sum((predictions_train - y_train) ** 2)

Rsquared2 = 1 - prediction_sum_sq_errors / raw_sum_sq_errors
Rsquared2

N = 814
p = 2  # Two predictors used
Rsquared_adj2 = 1 - (1 - Rsquared2) * (N - 1) / (N - p - 1)
print("Rsquared Regression Model with Rank and Likes: " + str(Rsquared2))
print("Rsquared Adjusted Regression Model with Rank and Likes: " + str(Rsquared_adj2))

####Model 3 - Next Add the AvgAreaHouseAge
model3.fit(x_train[['Rank', 'Likes', 'Duration Seconds']], y_train)
# Show the model parameters
print(model3.coef_)
print(model3.intercept_)
# So Price = ??

# A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
Output = pd.DataFrame(model3.coef_, ['Rank', 'Likes', 'Duration Seconds'], columns=['Coeff'])
print(Output)

# Generate predictions for the train data
predictions_train = model3.predict(x_train[['Rank', 'Likes', 'Duration Seconds']])

# Raw sum of squares of errors is based on the mean of the y values without having any predictors to help.
raw_sum_sq_errors = sum((y_train.mean() - y_train) ** 2)

# Calculate sum of squares for prediction errors.
prediction_sum_sq_errors = sum((predictions_train - y_train) ** 2)

Rsquared3 = 1 - prediction_sum_sq_errors / raw_sum_sq_errors

N = 814  # 16 data rows
p = 3  # Two predictors used
Rsquared_adj3 = 1 - (1 - Rsquared3) * (N - 1) / (N - p - 1)
print("Rsquared Regression Model with Rank and Likes and duration seconds: " + str(Rsquared3))
print("Rsquared Adjusted Regression Model with Rank and Likes and duration seconds: " + str(Rsquared_adj3))

# Model 4 - Next add the AvgAreaNumberRooms
model4.fit(x_train[['Rank', 'Likes', 'Duration Seconds', 'Chnl BonginoReport']], y_train)
# Show the model parameters
print(model4.coef_)
print(model4.intercept_)
# So Price = ??

# A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
Output = pd.DataFrame(model4.coef_, ['Rank', 'Likes', 'Duration Seconds', 'Chnl BonginoReport'], columns=['Coeff'])
print(Output)

# Generate predictions for the training data
predictions_train = model4.predict(x_train[['Rank', 'Likes', 'Duration Seconds', 'Chnl BonginoReport']])

prediction_sum_sq_errors = sum((predictions_train - y_train) ** 2)

Rsquared4 = 1 - prediction_sum_sq_errors / raw_sum_sq_errors

N = 814
p = 4  # Four predictors used
Rsquared_adj4 = 1 - (1 - Rsquared4) * (N - 1) / (N - p - 1)
print("Rsquared Regression Model with Chnl BonginoReport: " + str(Rsquared4))
print("Rsquared Adjusted Regression Model with Chnl BonginoReport: " + str(Rsquared_adj4))

# Model 5 - Next add the AvgAreaNumberBedrooms
model5.fit(x_train[['Rank', 'Likes', 'Duration Seconds', 'Chnl BonginoReport', 'Chnl X22 Report']], y_train)
# Show the model parameters
print(model5.coef_)
print(model5.intercept_)
# So Price = ??

# A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
Output = pd.DataFrame(model5.coef_, ['Rank', 'Likes', 'Duration Seconds', 'Chnl BonginoReport', 'Chnl X22 Report'],
                      columns=['Coeff'])
print(Output)

# Generate predictions for the training data
predictions_train = model5.predict(
    x_train[['Rank', 'Likes', 'Duration Seconds', 'Chnl BonginoReport', 'Chnl X22 Report']])

prediction_sum_sq_errors = sum((predictions_train - y_train) ** 2)

Rsquared5 = 1 - prediction_sum_sq_errors / raw_sum_sq_errors

N = 814
p = 5  # Five predictors used
Rsquared_adj5 = 1 - (1 - Rsquared5) * (N - 1) / (N - p - 1)
print("Rsquared Regression Model with Chnl X22 Report: " + str(Rsquared5))
print("Rsquared Adjusted Regression Model with Chnl X22 Report: " + str(Rsquared_adj5))

# So based on the Adjusted R Squared value my bext model is Model 5 which includes everything bar the channel variables
# Price = ??


# Model 6 - Next add the Subscribers
model6.fit(x_train[['Rank', 'Likes', 'Duration Seconds', 'Chnl BonginoReport', 'Chnl X22 Report', 'Subscribers']],
           y_train)
# Show the model parameters
print(model6.coef_)
print(model6.intercept_)
# So Price = ??

# A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
Output = pd.DataFrame(model6.coef_,
                      ['Rank', 'Likes', 'Duration Seconds', 'Chnl BonginoReport', 'Chnl X22 Report', 'Subscribers'],
                      columns=['Coeff'])
print(Output)

# Generate predictions for the training data
predictions_train = model6.predict(
    x_train[['Rank', 'Likes', 'Duration Seconds', 'Chnl BonginoReport', 'Chnl X22 Report', 'Subscribers']])

prediction_sum_sq_errors = sum((predictions_train - y_train) ** 2)

Rsquared6 = 1 - prediction_sum_sq_errors / raw_sum_sq_errors

N = 814
p = 6  # Five predictors used
Rsquared_adj6 = 1 - (1 - Rsquared6) * (N - 1) / (N - p - 1)
print("Rsquared Regression Model with Subscribers: " + str(Rsquared6))
print("Rsquared Adjusted Regression Model with Subscribers: " + str(Rsquared_adj6))

# Model 7 - Next add the Other Chnl
model7.fit(x_train[['Rank', 'Likes', 'Duration Seconds', 'Chnl BonginoReport', 'Chnl X22 Report', 'Subscribers',
                    'Other Chnl']], y_train)
# Show the model parameters
print(model7.coef_)
print(model7.intercept_)
# So Price = ??

# A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
Output = pd.DataFrame(model7.coef_,
                      ['Rank', 'Likes', 'Duration Seconds', 'Chnl BonginoReport', 'Chnl X22 Report', 'Subscribers',
                       'Other Chnl'], columns=['Coeff'])
print(Output)

# Generate predictions for the training data
predictions_train = model7.predict(x_train[
                                       ['Rank', 'Likes', 'Duration Seconds', 'Chnl BonginoReport', 'Chnl X22 Report',
                                        'Subscribers', 'Other Chnl']])

prediction_sum_sq_errors = sum((predictions_train - y_train) ** 2)

Rsquared7 = 1 - prediction_sum_sq_errors / raw_sum_sq_errors

N = 814
p = 7  # Five predictors used
Rsquared_adj7 = 1 - (1 - Rsquared7) * (N - 1) / (N - p - 1)
print("Rsquared Regression Model with Other Chnl: " + str(Rsquared7))
print("Rsquared Adjusted Regression Model with Other Chnl: " + str(Rsquared_adj7))

# Model 8 - Next add the Chnl The Gateway Pundit
model8.fit(x_train[['Rank', 'Likes', 'Duration Seconds', 'Chnl BonginoReport', 'Chnl X22 Report', 'Subscribers',
                    'Other Chnl', 'Chnl The Gateway Pundit']], y_train)
# Show the model parameters
print(model8.coef_)
print(model8.intercept_)
# So Price = ??

# A nicer way to view the coefficients is by placing them in a DataFrame. This can be done with the following statement:
Output = pd.DataFrame(model8.coef_,
                      ['Rank', 'Likes', 'Duration Seconds', 'Chnl BonginoReport', 'Chnl X22 Report', 'Subscribers',
                       'Other Chnl', 'Chnl The Gateway Pundit'], columns=['Coeff'])
print(Output)

# Generate predictions for the training data
predictions_train = model8.predict(x_train[
                                       ['Rank', 'Likes', 'Duration Seconds', 'Chnl BonginoReport', 'Chnl X22 Report',
                                        'Subscribers', 'Other Chnl', 'Chnl The Gateway Pundit']])

prediction_sum_sq_errors = sum((predictions_train - y_train) ** 2)

Rsquared8 = 1 - prediction_sum_sq_errors / raw_sum_sq_errors

N = 814
p = 8  # Five predictors used
Rsquared_adj8 = 1 - (1 - Rsquared8) * (N - 1) / (N - p - 1)
print("Rsquared Regression Model with Chnl The Gateway Pundit: " + str(Rsquared8))
print("Rsquared Adjusted Regression Model with Chnl The Gateway Pundit: " + str(Rsquared_adj8))

# Interesting to plot the errors for the actual values
plt.scatter(y_train, predictions_train)
plt.show()  # Should be close to a straight line

plt.scatter(y_train, predictions_train - y_train)
plt.show()

#########Regression Modelling - Step 3: Model Evaluation

# Calculate the MAE (Mean Absolute Error), MAPE(Mean Absolute Percentage Error) and the RMSE (Root Mean Square Error) for the model based on the TEST set.
# These give a measure of the

predictions_test = model5.predict(x_test[['Rank', 'Likes', 'Duration Seconds', 'Subscribers', 'days uploaded']])

Prediction_test_MAE = sum(abs(predictions_test - y_test)) / len(y_test)
Prediction_test_MAPE = sum(abs(predictions_test - y_test) / y_test) / len(y_test)
Prediction_test_RMSE = (sum((predictions_test - y_test) ** 2) / len(y_test)) ** 0.5

print(Prediction_test_MAE)
print(Prediction_test_MAPE)
print(Prediction_test_RMSE)

# Can other models beat these values?????

###Plot prediction results
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, predictions_test)
plt.show()  # Should be close to a straight line

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, predictions_test - y_test)
plt.show()

# Feedback from Kevin
# Answer: Run it a number of times, maybe 1000, and pick out the best model as the median value, cant do mean
# Since you need to run with an actual generated model

# The reason rsquared is jumping around is because at the end i add a weakly correlated variable which sometimes improves the model
# but other times does not

# If i leave a detailed note as to why i ran this script 1000 times and got a median best, since the rsquared jumps around
# i can get some extra marks

# Include the channel variable if they improve the model

# Explain why rank has such a negative correlation with views for some extra marks in a note

# put down final equation with ccoefficients and intercept in comment for extra marks

# Explain correlations for extra points

# include error vs valuable plots and see if random for extra marks