# This project will scrape data from a video streaming website called Rumble
# Rumble's robots.text file as of 14:42 on the 30th October 2022 says:
# User-agent: *
# Disallow: /l/
# Disallow: /api/
# Therefore, scraping video/channel lists is perfectly legal

from bs4 import BeautifulSoup
import requests
import pandas as pd
import time

urlStringFirstPage = 'https://rumble.com/videos?sort=views&date=this-month'
urlStringNumberedPage = 'https://rumble.com/videos?sort=views&date=this-month&page='

video_details = []
counter = 0

for page in range(1, 2):
    if (page == 1):
        urlString = urlStringFirstPage
    else:
        urlString = urlStringNumberedPage + str(page)
    html = requests.get(urlString).text
    soup = BeautifulSoup(html, 'html.parser')

    for item in soup('article'):
        try:
            if (item['class'][0] == "video-item"):
                video_details.append(str(item('h3')[0].contents[0]))

                counter += 1
        except:
            pass
        time.sleep(1)

videopd = pd.DataFrame(video_details, columns=['Title'])


