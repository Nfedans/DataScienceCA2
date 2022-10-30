from bs4 import BeautifulSoup
import requests
import pandas as pd
import time

# 2 urls needed because the first page of videos does not include "&page=1"
# However all other pages specify their number in the url
urlStringFirstPage = 'https://rumble.com/videos?sort=views&date=this-month'
urlStringNumberedPage = 'https://rumble.com/videos?sort=views&date=this-month&page='

video_details = []
# Rumble's first page is index 1 and the max amount of pages it shows is 100
number_of_pages = 100


# Some videos show how much $ they earned while other videos do not
# This function is used to append video data correctly for the videos
# Which DO show the viewer how much $ it made
def append_payment_transparent_video():
    video_details.append([
        str(item('h3')[0].contents[0]),
        str(item('div')[0].contents[0]),
        str(item('span')[0]['data-value']),
        str(item('span')[1]['data-value']),
        str(item('span')[2]['data-value']),
        str(item('span')[3]['data-value']),
        str(item('time')[0]['datetime']),
    ])


# This function appends data for videos which DO NOT show amount of $ earned
def append_video():
    video_details.append([
        str(item('h3')[0].contents[0]),
        str(item('div')[0].contents[0]),
        str(item('span')[0]['data-value']),
        '0',
        str(item('span')[1]['data-value']),
        str(item('span')[2]['data-value']),
        str(item('time')[0]['datetime']),
    ])


# Since videos are not all available on a single pages,
# we use a loop to scrape data from all pages
for page in range(1, (number_of_pages + 1)):
    if page == 1:
        urlString = urlStringFirstPage
    else:
        urlString = urlStringNumberedPage + str(page)

    html = requests.get(urlString).text
    soup = BeautifulSoup(html, 'html.parser')

    for item in soup('article'):
        try:
            if item['class'][0] == "video-item":
                if item('span')[1]['class'][1] == 'video-item--earned':
                    append_payment_transparent_video()
                else:
                    append_video()
        except:
            pass
        time.sleep(0.5)

videopd = pd.DataFrame(video_details, columns=['Title', 'Channel', 'Duration', 'Earned', 'Views', 'Likes', 'Uploaded'])

# Write out to csv file
videopd.to_csv("DataSciCA2Data.csv")
