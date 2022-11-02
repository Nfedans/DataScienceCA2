# Data gives details of all videos which made it to the top 50 pages of the month on rumble.com
# Variables: index, title, channel, duration, $ earned, views, Likes, upload date, and channel subscriber count.

#Response (Dependent) variable is "index"

#We are interested in the key variables of Duration, Views, Likes, upload date, Subscribers

# STEP 1 : BUSINESS UNDERSTANDING
#
# I am trying to predict based on the variables on line 6, and perhaps variables added in the future
# as part of feature engineering, how a video might rank in the top 50 pages of videos, on Rumble
#
# If I can develop a sufficiently accurate model, and find how variables correlate, this will be 
# useful information for companies that want to advertise their products through rumble creators.
# 
# Some questions i intend to answer:
# 1) What channels consistently make it to the top 50
# 2) is there a relationship between the amount of likes on the video and amount of subscribers of the publishing channel?
# 3) What does the distribution of videos that make it to the top 50 by video duration look like?
# 4) Do longer videos get more views than short videos?
# 5) is there a relationship between the views and the amount of subscribers of the publishing channel?
# 
# THINK OF SOME BETTER STUFF WHEN I GET TO DOING IT
# 
# Who Benefits?
# 1) Companies looking for advertisement deals with content creators(They will see what they should pay attention to when evaluating a channel
# , subscriber count? video length? video upload frequency? how many likes the videos get?... or some combination of the variables listed.)
#
# 2) Content creators (They will see what they may change about their channel to rank better on rumble, such as
# upload more/less frequently? gain more subscribers? ask their subscribers to like the video?... or a combination of those actions.)
#
# 3) Rumble shareholders (If there are statistics available which show how a channel may rank better, and how advertisers may make deals
# with the optimal creators to advertise the products, this will make rumble a more active, competitive platform, and this is conducive to
# growth. If the creators were uninterested in succeeding and advertisers avoided rumble creators, rumble would be dead in the water.)



import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

#To set figure size
from matplotlib.pyplot import figure

# For colorful output
from colorama import Fore, Style
import re

#DATA IMPORT########################################################

import os
cwd = os.getcwd()
print(cwd)
os.chdir("C:/Users/nfeda/OneDrive - Dundalk Institute of Technology/Y3/Data Science/CA2")

# Read the data set of "Marketing Analysis" in data.
data= pd.read_csv("DataSciCA2Data8Columns.csv")


#1. DATA TYPES    ######################################################## 


data.info()

# Column       Non-Null Count  Dtype  
#---  ------       --------------  -----  
#Unnamed: 0    1226 non-null   int64  (IGNORE, Numerical)
# Rank         1226 non-null   int64  (Response Variable, Numerical)
# Title        1226 non-null   object (IGNORE, Categorical)
# Channel      1226 non-null   object  (Categorical)
# Duration     1226 non-null   object (Explanatory)
# Earned       1226 non-null   float64 (IGNORE, Numerical)
# Views        1226 non-null   object (Explanatory, Numerical)
# Likes        1226 non-null   object (Explanatory, Numerical)
# Uploaded     1226 non-null   object (Explanatory, Numerical)
# Subscribers  1226 non-null   object (Explanatory, Numerical)


data.head()
summaryNumerical = data.describe()


#2. DATA CLEANING ########################################################

#passed_num

def format_numbers(passed_num):
    passed_num = passed_num.casefold()
    passed_num = passed_num.translate({ord(','): None})
    if passed_num.__contains__('k'):
            passed_num = passed_num.translate({ord('k'): None})
            passed_num = 1000 * float(passed_num)
    elif passed_num.__contains__('m'):
            passed_num = passed_num.translate({ord('m'): None})
            passed_num = 1000000 * float(passed_num)
    return int(passed_num)

def get_sec(time_str):
    #print(time_str)
    contains_min_sec_time = re.search("^(([0]?[0-5][0-9]|[0-9]):([0-5][0-9]))$", time_str)
    #contains_hh_mm_ss_time = re.search("^(?:(?:([01]?\d|2[0-3]):)?([0-5]?\d):)?([0-5]?\d)$", time_str)
    if(contains_min_sec_time):
        time_str = '00:' + time_str
    elif(time_str == 'LIVE'):
        return -1
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

# Drop the video title as it will not be used for analysis.
data.drop('Title', axis = 1, inplace = True)

# Drop the earnings column title as most channels do not include this, it is useless.
data.drop('Earned', axis = 1, inplace = True)

# Drop redundant index
data.drop('Unnamed: 0', axis = 1, inplace = True)

# Make DURATION into a manipulatible format
data['Duration Seconds'] = data['Duration'].apply(lambda x: get_sec(x))

# # Delete all videos with duration LIVE (-1 in Duration Seconds Column)
data.drop(data[data['Duration Seconds'] == -1].index, inplace = True)

# reset index & delete old index
data = data.reset_index()
data.drop('index', axis = 1, inplace = True)

# Get rid of commas in VIEWS and cast as ints
data['Views'] = data['Views'].apply(lambda x: int(x.translate({ord(','): None})))

# Format thousand and millions properly for LIKES and for SUBSCRIBERS and cast as ints
data['Likes'] = data['Likes'].apply(lambda like_count: format_numbers(like_count))
data['Subscribers'] = data['Subscribers'].apply(lambda subs: format_numbers(subs))



# Check / Make UPLOADED a manipulatable format

# MAYBE DO THIS DURING FEATURE ENGINEERING
data.info()


#3. MISSING VALUES ######################################################


# Checking the missing values (Trust me, they are all there...)
numberMissing=data.isnull().sum()


#4. OUTLIERS ############################################################


#Use seaborn plot
#To set figure sizes 8x6cm
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.boxplot(x=data.Subscribers)
plt.title("Boxplot of Subscribers")
plt.ylabel("Subscribers")
plt.show()

data.Subscribers.describe()

# seems reasonable enough to me, some accounts simply have a disproportionately large audience


figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.boxplot(x=data.Likes)
plt.title("Boxplot of Likes")
plt.ylabel("Likes")
plt.show()

data.Likes.describe()

# there certainly are outliers, however after checking out the top 3 cases
# they seem reasonable, the top most liked video was a giveaway contest
# the second most liked was made by a scam artist with a cult following






#5. EXPLORATORY DATA ANALYSIS - UNIVARIATE ANALYSIS ############################################