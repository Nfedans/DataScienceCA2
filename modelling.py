import numpy as np
import pandas as pd

# for feature engineering date variable
from datetime import datetime

# for turning categorical variable into numerical
from collections import Counter

#
#                         IMPORTANT CHANGES
# Video RANK is no longer my response variable, I am instead predicting VIEWS
#
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

# Get rid of the uploaded variable since it is of no use now
data.drop('Uploaded', axis=1, inplace=True)







