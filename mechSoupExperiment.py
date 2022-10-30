import requests
import pandas as pd
import time
import mechanicalsoup
from bs4 import BeautifulSoup

browser = mechanicalsoup.StatefulBrowser(soup_config={'features': 'lxml'})

url = "https://rumble.com/videos?sort=views&date=this-month"

browser.open(url)

print(browser.get_url())

html = requests.get(url).text
soup = BeautifulSoup(html, 'html.parser')
item = soup('article')[0]

our_link = str(item('a')[1]['href'])

browser.follow_link(our_link)
print(browser.get_url())

channel = {}
whats_thaa = "notgay"

browser_link = 'https://rumble.com/' + our_link
browser_html = requests.get(browser_link).text
browser_soup = BeautifulSoup(browser_html, 'html.parser')

for browserItem in browser_soup('div'):
    try:
        if browserItem['class'][0] == "listing-header--buttons":
            channel[browserItem('button')[0]['data-title']] = str(browserItem('span')[1].contents[0])
    except:
        pass