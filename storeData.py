from bs4 import BeautifulSoup
import requests
import pandas as pd
import time

urlStringFirstPage = 'https://rumble.com/videos?sort=views&date=this-month'
urlStringNumberedPage = 'https://rumble.com/videos?sort=views&date=this-month&page='

video_details = []
counter = 0
number_of_pages = 100


def appendPaymentTransparentVideo():
    video_details.append([
        str(item('h3')[0].contents[0])
        , str(item('div')[0].contents[0])
        , str(item('span')[0]['data-value'])
        , str(item('span')[1]['data-value'])
        , str(item('span')[2]['data-value'])
        , str(item('span')[3]['data-value'])
        , str(item('time')[0]['datetime'])
    ])


def appendVideo():
    video_details.append([
        str(item('h3')[0].contents[0])
        , str(item('div')[0].contents[0])
        , str(item('span')[0]['data-value'])
        , 'NA'
        , str(item('span')[1]['data-value'])
        , str(item('span')[2]['data-value'])
        , str(item('time')[0]['datetime'])
    ])


for page in range(1, (number_of_pages + 1)):
    if (page == 1):
        urlString = urlStringFirstPage
    else:
        urlString = urlStringNumberedPage + str(page)

    html = requests.get(urlString).text
    soup = BeautifulSoup(html, 'html.parser')

    for item in soup('article'):
        try:
            if (item['class'][0] == "video-item"):
                # print(item('span')[1]['class'][1])
                if (item('span')[1]['class'][1] == 'video-item--earned'):
                    appendPaymentTransparentVideo()
                else:
                    appendVideo()
                counter += 1
        except:
            pass
        time.sleep(1)

videopd = pd.DataFrame(video_details, columns=['Title', 'Channel', 'Duration', 'Earned', 'Views', 'Likes', 'Uploaded'])


