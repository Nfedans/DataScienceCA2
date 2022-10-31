import requests
import pandas as pd
import time
import mechanicalsoup
from bs4 import BeautifulSoup

# browser = mechanicalsoup.StatefulBrowser(soup_config={'features': 'lxml'})

# url = "https://rumble.com/videos?sort=views&date=this-month"

# browser.open(url)

# print(browser.get_url())

# html = requests.get(url).text
# soup = BeautifulSoup(html,	'html.parser')
# item = soup('article')[0]

# our_link = str(item('a')[1]['href'])

# browser.follow_link(our_link)
# print(browser.get_url())


# channel={}
# whats_thaa = "notgay"

# browser_link = 'https://rumble.com/' + our_link
# browser_html = requests.get(browser_link).text
# browser_soup = BeautifulSoup(browser_html,	'html.parser')

# for browserItem in browser_soup('div'):
#     try:
#         if browserItem['class'][0]=="listing-header--buttons":
#             channel[browserItem('button')[0]['data-title']] = str(browserItem('span')[1].contents[0])
#     except:
#         pass

# browser.open(url)
# print(browser.get_url())


# #channelpd = pd.DataFrame.from_dict(channel)
# print("END...")


browser = mechanicalsoup.StatefulBrowser(soup_config={'features': 'lxml'})
url = "https://rumble.com/videos?sort=views&date=this-month"
browser.open(url)
print(browser.get_url())

html = requests.get(url).text
soup = BeautifulSoup(html, 'html.parser')
item = soup('article')[0]

channel = {}

# in loop


for item in soup('article'):
    our_link = str(item('a')[1]['href'])

    browser.follow_link(our_link)
    print(browser.get_url())

    browser_link = 'https://rumble.com' + our_link

    browser.open(browser_link)
    browser_html = requests.get(browser_link).text
    browser_soup = BeautifulSoup(browser_html, 'html.parser')

    if (our_link.__contains__('c/') == False):
        # btns = browser_soup.find('body')
        # btns = browser_soup.find_all('button', class_ = 'round-button media-subscribe bg-green')
        # spans = browser_soup.find_all('span', class_ = 'subscribe-button-count')
        btns = browser_soup.find('main').find_all('button', class_='round-button media-subscribe bg-green')
        spans = browser_soup.find('body').find_all('span', class_='subscribe-button-count')
        # print(btns)
        # print(spans)
        # if spans[0]['class'] ==  'listing-header--letter':
        channel[btns[0]['data-title']] = str(spans[0].contents[0])
    else:
        # btns = browser_soup.find('body')
        # btns = browser_soup.find_all('button')
        # spans = browser_soup.find_all('span')
        # btns = browser_soup.find('div', class_ = "constrained").find('button', class_ = 'round-button media-subscribe bg-green')

        # browser_link = 'https://rumble.com/c/RSBN'

        # browser.open(browser_link)
        # browser_html = requests.get(browser_link).text
        # browser_soup = BeautifulSoup(browser_html,	'html.parser')

        btns = browser_soup.find_all('div', class_="constrained")[0].find('div', class_='listing-header--buttons')

        channel[btns('button')[0]['data-title']] = str(btns('span')[1].contents[0])

        # print(btns('button')[0]['data-title'])     WORKS

        # print(str(btns('span')[1].contents[0]))WORKS

        # spans = browser_soup.find('body')
        # print(btns)
        # print(spans)

    # else:
    # channel[btns[0]['data-title']] = str(spans[1].contents[0])
    # for browserItem in browser_soup('div')['class'][0] == "listing-header--buttons":
    #     try:
    #         #if browserItem['class'][0]=="constrained":
    #             #print("INSIDE IF 1")
    #         print("Above if", end ='')
    #         if browserItem('button')[0]['data-action'][0]=="subscribe":
    #                     print("INSIDE IF", end='')
    #     except:
    #         pass

    browser.open(url)
    print(browser.get_url())




