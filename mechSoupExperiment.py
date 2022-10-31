import requests
import pandas as pd
import mechanicalsoup
from bs4 import BeautifulSoup
from colorama import Fore, Style


def redirect_scrape_return():
    # This grabs the link endpoint which we will redirect to
    our_link = str(item('a')[1]['href'])

    # This redirects through the link
    browser.follow_link(our_link)

    # Grabs the html we need
    browser_link = 'https://rumble.com' + our_link
    browser_html = requests.get(browser_link).text
    browser_soup = BeautifulSoup(browser_html, 'html.parser')

    # some channels are /user/ in the url and some are /c/
    # The 2 types of channel pages are organised slightly differently
    # However, the 2 elegant lines of code below work on both 😎
    btns = browser_soup.find_all('div', class_="constrained")[0].find('div', class_='listing-header--buttons')
    channel[btns('button')[0]['data-title']] = [str(btns('span')[1].contents[0])]

    browser.open(url)


def check_key_channel(dic, key):
    if key in dic:
        print(Style.BRIGHT + Fore.YELLOW + key + " : Already Exists")
    else:
        print(Style.BRIGHT + Fore.GREEN + "Adding " + key + "...")
        redirect_scrape_return()


# browser comes from mechanicalsoup, it's a headless browser
# it is needed for following links to channel pages, scraping subscriber count
# from those pages, and going back to the url on line 11
browser = mechanicalsoup.StatefulBrowser(soup_config={'features': 'lxml'})
url = "https://rumble.com/videos?sort=views&date=this-month"

browser.open(url)
print(browser.get_url())

html = requests.get(url).text
soup = BeautifulSoup(html, 'html.parser')
item = soup('article')[0]
channel = {}

for item in soup('article'):
    # This grabs the channel name
    channel_name = str(item('div')[0].contents[0])
    check_key_channel(channel, channel_name)

channelpd = pd.DataFrame.from_dict(channel)
