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
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter

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
# the second most liked was made by a scam artist with a big cult following
# I end up removing the top most liked video though, as there was a big incentive to
# watch it and participate in the giveaway, so people who generally do not use
# Rumble may have heard about it and watched and liked just to participate

data = data.drop(data[data.Likes > 50000].index)

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.boxplot(x=data.Likes)
plt.title("Boxplot of Likes")
plt.ylabel("Likes")
plt.show()

data.Likes.describe()


#5. EXPLORATORY DATA ANALYSIS - UNIVARIATE ANALYSIS ############################################


#######Univariate for Categorical variable such as employment

# This calculates for each channel that made it to the top 50, how many of that channel's videos
# are in the top 50. A histogram is used to represent this data, because a bar chart or similar
# would be too big (233 channels), therefore splitting into bins, and showing that way is better.
channel_details = data.Channel.unique()
print(channel_details)

number_channel_vids = data.Channel.value_counts()

fig, ax = plt.subplots(figsize=(3, 4.5),  dpi=163)
ax.set_ylim([0, 233])
rng = np.arange(0, 240, 11.65)
ax.set_yticks(rng)
ax.hist(number_channel_vids, bins=range(0, 90, 5))
ax.set_title("Histogram of Channels")
ax.set_ylabel("amount of channels")
ax.set_xlabel("amount of videos")

secax = ax.twinx()
secax.hist(number_channel_vids, bins=range(0, 90, 5))
secax.set_ylim([0, 233])
secrng = np.arange(0, 240, 10)
secax.set_yticks(secrng)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=len(number_channel_vids)))
plt.show()

# SUMMARY STATEMENT:
# 3 quarters of channels that make it to the top 50 pages on rumble have between 1 and 4 of their videos in the top 50,
# almost 1 in 10 channels have between 5 and 9 of their videos in the top 50, and if you were to get 10 or more of your videos
# into the top 50, you would be in the top 15%. Gunning for the top spot would be difficult however as you'd need minimum 84
# of your videos to make it to the top 50.





# # This calculates for each channel that made it to the top 50, how many of that channel's videos
# # are in the top 50. A histogram is used to represent this data, because a bar chart or similar
# # would be too big (233 channels), therefore splitting into bins, and showing that way is better.
# like_details = data.Likes.unique()
# print(like_details)

# number_likes = data.Likes.value_counts()

# fig, ax = plt.subplots(figsize=(15, 10),  dpi=163)
# #ax.set_xlim([0, 45])
# rng = np.arange(0, 44, 1)
# #ax.set_xticks(rng)
# #ax.set_ylim([0, 45])
# ax.hist(number_likes)
# ax.set_title("Histogram of Likes")
# ax.set_ylabel("1000's of likes")
# ax.set_xlabel("amount of videos with the likes")
# plt.show()

# figure(num=None, figsize=(12, 100), dpi=80, facecolor='w', edgecolor='k')
# number_likes.plot.barh()

figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(data.index, data['Likes'])
plt.title("Scatterplot of likes")
plt.ylabel("amount of likes")
plt.xlabel("index")

data.Likes.describe()

# count     1221.000000
# mean      2322.754300
# std       3925.052502
# min          2.000000
# 25%        102.000000
# 50%        682.000000
# 75%       2250.000000
# max      42000.000000

# SUMMARY STATEMENT:
# 3 in 4 videos that make it to the top 50 have fewer that 2250 likes, but the average is 2322
# as some videos have enormous amounts of likes, with the most like video having over 42000 likes.



figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(data.index, data['Views'])
plt.title("Scatterplot of views")
plt.ylabel("amount of views in millions")
plt.xlabel("index")

data.Views.describe()


# count    1.221000e+03
# mean     1.184477e+05
# std      1.740035e+05
# min      2.354200e+04
# 25%      3.825400e+04
# 50%      6.144300e+04
# 75%      1.303260e+05
# max      2.226945e+06


# SUMMARY STATEMENT:
# The average amount of views in the top 50 pages worth of videos is just under 120,000, If you want a chance to land
# in the top 50 pages, your video needs over 24,000 views minimum. The most viewed video has close to 2.5 million views.
# subscriptions_pegged_to_channels = data.Subscribers.value_counts()
# df = pd.DataFrame (subscriptions_pegged_to_channels, columns = ['Subs'])

unique_subs = data.Subscribers.unique()
subdf = pd.DataFrame(unique_subs, columns = ['Subs'])

figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(range(0, len(data['Subscribers'].unique()), 1), data['Subscribers'].unique())
plt.title("Scatterplot of Subscribers")
plt.ylabel("amount of Subscribers")
plt.xlabel("index")
plt.show()

subdf.Subs.describe()

# count    2.220000e+02
# mean     1.283180e+05
# std      2.996943e+05
# min      3.000000e+00
# 25%      1.680000e+03
# 50%      2.415000e+04
# 75%      1.006000e+05
# max      2.470000e+06

subdf.median()
# Subs    24150.0

# SUMMARY STATEMENT:
# When it comes to the top 50 pages, there are accounts with 3 subscribers, and accounts with close to 2.5 million
# the average account in the top 50, when we look at unique accounts, has just over 125,000 subscribers, But that isn't entirely
# an accurate figure, as the median number of subscribers is 24,000.





# number_channel_vids = data.Channel.value_counts()

# fig, ax = plt.subplots(figsize=(3, 4.5),  dpi=163)
# ax.set_ylim([0, 233])
# rng = np.arange(0, 240, 11.65)
# ax.set_yticks(rng)
# ax.hist(number_channel_vids, bins=range(0, 90, 5))
# ax.set_title("Histogram of Channels")
# ax.set_ylabel("amount of channels")
# ax.set_xlabel("amount of videos")

# secax = ax.twinx()
# secax.hist(number_channel_vids, bins=range(0, 90, 5))
# secax.set_ylim([0, 233])
# secrng = np.arange(0, 240, 10)
# secax.set_yticks(secrng)
# ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=len(number_channel_vids)))
# plt.show()

fig, ax = plt.subplots(figsize=(12, 8),  dpi=163)
ax.hist(data['Duration Seconds'], bins=range(0, 54000, 300))
ax.set_ylim([0, 430])
# ax.set_xlim([0, 12])
ax.set_title("Histogram of Video lengths")
ax.set_ylabel("percentage of all videos")
ax.set_xlabel("hours (each bar = 5 minutes)")

secax = ax.twinx()
secax.hist(data['Duration Seconds'], bins=range(0, 54000, 300))
secax.set_ylim([0, 430])
secax.set_ylabel("Amount of videos")


ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=len(data['Duration Seconds'])))
rng = np.arange(0, 430, 30.525)
ax.set_yticks(rng)
xrng = np.arange(0, 50000, 3600)
ax.set_xticks(xrng)


def numfmt(x, pos):
    s = '{}'.format(x / 3600).rstrip('.0')
    # if s > 0:
    #     return int(s)
    # else:
    #     return "0"
    return s

yfmt = FuncFormatter(numfmt)

ax.xaxis.set_major_formatter(yfmt)
#ax.savefig("Output.pdf", dpi=400, bbox_inches='tight', pad_inches=1)

plt.show()


data['Duration Seconds'].describe()

# count     1221.000000
# mean      2728.826372
# std       4820.469359
# min          6.000000
# 25%        162.000000
# 50%       1042.000000
# 75%       3666.000000
# max      51818.000000


# SUMMARY STATEMENT:
# 1/3 of videos in the top 50 ranked pages are 5 minutes long or less, 10% of videos are between 5 and 10 minutes, 7% of videos are
# between 55 minutes and an hour.