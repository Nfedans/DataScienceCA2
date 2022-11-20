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
# Therefore, the business understanding still applies regardless if i am trying to
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
# I am fairly certain a this new variable will correlate well with views (this turn out not to be the case however)

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


# Splitting the Data Set into Training Data and Test Data
from sklearn.model_selection import train_test_split

#########Regression Modelling - Model Selection

from sklearn.linear_model import LinearRegression

model1 = LinearRegression()
model2 = LinearRegression()
model3 = LinearRegression()
model4 = LinearRegression()
model5 = LinearRegression()
model6 = LinearRegression()
model7 = LinearRegression()
model8 = LinearRegression()

ordered_by_cor_vars = ['Rank', 'Likes', 'Duration Seconds', 'Chnl BonginoReport', 'Chnl X22 Report',
                       'Subscribers', 'Other Chnl', 'Chnl The Gateway Pundit']

useful_ds = [{"model": model1, 'dlist': ordered_by_cor_vars[0:1]},
             {"model": model2, 'dlist': ordered_by_cor_vars[0:2]},
             {"model": model3, 'dlist': ordered_by_cor_vars[0:3]},
             {"model": model4, 'dlist': ordered_by_cor_vars[0:4]},
             {"model": model5, 'dlist': ordered_by_cor_vars[0:5]},
             {"model": model6, 'dlist': ordered_by_cor_vars[0:6]},
             {"model": model7, 'dlist': ordered_by_cor_vars[0:7]},
             {"model": model8, 'dlist': ordered_by_cor_vars}]


def getRSquaredAdj(model, dlist=[], *args):
    model.fit(x_train[dlist], y_train)
    # print(model1.coef_)
    # print(model1.intercept_)

    # Generate predictions for the train data
    predictions_train = model.predict(x_train[dlist])

    raw_sum_sq_errors = sum((y_train.mean() - y_train) ** 2)
    raw_sum_sq_errors
    prediction_sum_sq_errors = sum((predictions_train - y_train) ** 2)

    Rsquared = 1 - prediction_sum_sq_errors / raw_sum_sq_errors
    Rsquared

    N = 814
    p = len(dlist)
    Rsquared_adj = 1 - (1 - Rsquared) * (N - 1) / (N - p - 1)
    return Rsquared_adj


model_win_freq_list = []

# This loop splits up the data into train and test each time it runs, an inside loop tests 8 models
# Then returns the RSquared values, from which the max value is picked, and the corresponding model
# that generated the maximum value is put in a list.
# This list ends up having 1000 values representing the best model from each run, and finally the best
# Model is chosen by getting the mode value of the list, usually model 8.

# The reason to do this, is that not every single time model 8 is the best based on RSquared,
# Sometimes it gets beaten out by model 7
for j in range(1000):

    #########Regression Modelling - Step 1: Split Data into Train and Test

    # Set the Response and the predictor variables

    x = data[['Subscribers', 'Rank', 'Likes', 'Duration Seconds', 'days uploaded', 'Chnl BonginoReport',
              'Chnl The Gateway Pundit', 'Chnl sonsoflibertyradiolive', 'Chnl The Post Millennial Clips',
              'Chnl X22 Report', 'Other Chnl']]  # pandas dataframe
    y = data['Views']  # Pandas series

    # split train 66.7%, test 33.3%. Note that if run this more than once will get different selection which can lead to different model particulalry for small datasets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.333)

    r_squared_values = []
    for x in range(len(useful_ds)):
        r_squared_values.append(getRSquaredAdj(useful_ds[x]['model'], useful_ds[x]['dlist']))

    # If you want to see the RSquared values gotten from each run, uncomment the line below
    # print(r_squared_values)
    max_value = max(r_squared_values)
    max_index = r_squared_values.index(max_value)
    best_model = max_index + 1
    model_win_freq_list.append(best_model)

# Get the mode from the list
fnctn = max(set(model_win_freq_list), key=model_win_freq_list.count)

# Usually model 8 is the best, so analysisafter this will use model8
print("THE BEST MODEL IS MODEL", str(fnctn))

#############################################


model8.fit(x_train[ordered_by_cor_vars], y_train)

# Generate predictions for the train data
predictions_train = model8.predict(x_train[ordered_by_cor_vars])

raw_sum_sq_errors = sum((y_train.mean() - y_train) ** 2)
raw_sum_sq_errors
prediction_sum_sq_errors = sum((predictions_train - y_train) ** 2)

Rsquared8 = 1 - prediction_sum_sq_errors / raw_sum_sq_errors
Rsquared8

N = 814
p = 8
Rsquared_adj8 = 1 - (1 - Rsquared8) * (N - 1) / (N - p - 1)

print("Coefficient: ", str(model8.coef_))
print("Intercept: ", str(model8.intercept_))
print("Rsquare adjusted", str(Rsquared_adj8))

Output = pd.DataFrame(model8.coef_, ordered_by_cor_vars, columns=['Coeff'])
print(Output)

# 'Coefficient:  [-2.57037479e+02  7.49245382e+00 -1.05209602e+00 -1.31924903e+04
#  -5.39199926e+04 -4.11924327e-02  1.01699208e+04 -4.80635156e+04]
# Intercept:  267961.5110754328
# Rsquare adjusted 0.34924717372056424
#                                 Coeff
# Rank                      -257.037479
# Likes                        7.492454
# Duration Seconds            -1.052096
# Chnl BonginoReport      -13192.490251
# Chnl X22 Report         -53919.992552
# Subscribers                 -0.041192
# Other Chnl               10169.920769
# Chnl The Gateway Pundit -48063.515643'

# FORMULA: 267961.5110754328 - 257.037479*Rank + 7.492454*Likes - 1.052096*Duration Seconds - 13192.490251*Chnl BonginoReport
#           - 53919.992552*Chnl X22 Report - 0.041192*Subscribers + 10169.920769*Other Chnl - 48063.515643*Chnl The Gateway Pundit


# Interesting to plot the errors for the actual values
plt.scatter(y_train, predictions_train)
plt.show()  # Should be close to a straight line

plt.scatter(y_train, predictions_train - y_train)
plt.show()

#########Regression Modelling - Step 3: Model Evaluation

# Calculate the MAE (Mean Absolute Error), MAPE(Mean Absolute Percentage Error) and the RMSE (Root Mean Square Error) for the model based on the TEST set.
# These give a measure of the

predictions_test = model8.predict(x_test[ordered_by_cor_vars])

Prediction_test_MAE = sum(abs(predictions_test - y_test)) / len(y_test)
Prediction_test_MAPE = sum(abs(predictions_test - y_test) / y_test) / len(y_test)
Prediction_test_RMSE = (sum((predictions_test - y_test) ** 2) / len(y_test)) ** 0.5

print(Prediction_test_MAE)  # 67690.44179310325
print(Prediction_test_MAPE)  # 0.7499661708232815
print(Prediction_test_RMSE)  # 142832.02758723873

# Can other models beat these values?????

###Plot prediction results
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, predictions_test)
plt.title("Predictions v actual test values")
plt.ylabel("Predicted Values")
plt.show()  # Should be close to a straight line

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, predictions_test - y_test)
plt.title("Errors v Actual Test Values")
plt.xlabel("Actual values")
plt.ylabel("Error Values")
plt.show()

# For low values of Y, you mostly get above prediction, and for low values of y you get below prediction


figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(x_test['Likes'], predictions_test - y_test)
plt.title("Errors v Likes Test Values")
plt.xlabel("Likes values")
plt.ylabel("Error Values")
plt.show()

# The lower like values get estimated better than higher like values, whether over or under estimated, lower values are closer to 0.0