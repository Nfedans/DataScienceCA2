from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import mechanicalsoup
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

    chname = btns('button')[0]['data-title']
    if (chname[-1] == " "):
        chname = chname.rstrip()

    channel[chname] = [str(btns('span')[1].contents[0])]

    browser.open(urlString)


def check_key_channel(dic, key):
    if key in dic:
        print(Style.BRIGHT + Fore.YELLOW + key + " : Already Exists")
    else:
        print(Style.BRIGHT + Fore.GREEN + "Adding " + key + "...")
        redirect_scrape_return()


# Some videos show how much $ they earned while other videos do not
# This function is used to append video data correctly for the videos
# Which DO show the viewer how much $ it made
def append_payment_transparent_video():
    channelString = str(item('div')[0].contents[0])
    if (channelString[-1] == " "):
        channelString.rstrip(" ")

    video_details.append([
        str(item('h3')[0].contents[0]),  # Title
        channelString,  # Channel
        str(item('span')[0]['data-value']),  # Duration
        str(item('span')[1]['data-value']),  # Earned
        str(item('span')[2]['data-value']),  # Views
        str(item('span')[3]['data-value']),  # Likes
        str(item('time')[0]['datetime']),  # Uploaded
    ])


# This function appends data for videos which DO NOT show amount of $ earned
def append_video():
    channelString = str(item('div')[0].contents[0])
    if (channelString[-1] == " "):
        channelString.rstrip()

    print(channelString)

    video_details.append([
        str(item('h3')[0].contents[0]),  # Title
        channelString,  # Channel
        str(item('span')[0]['data-value']),  # Duration
        '0',  # Earned
        str(item('span')[1]['data-value']),  # Views
        str(item('span')[2]['data-value']),  # Likes
        str(item('time')[0]['datetime']),  # Uploaded
    ])


# 2 urls needed because the first page of videos does not include "&page=1"
# However all other pages specify their number in the url
urlStringFirstPage = 'https://rumble.com/videos?sort=views&date=this-month'
urlStringNumberedPage = 'https://rumble.com/videos?sort=views&date=this-month&page='

video_details = []
# Rumble's first page is index 1 and the max amount of pages it shows is 100
number_of_pages = 100

channel = {}
browser = mechanicalsoup.StatefulBrowser(soup_config={'features': 'lxml'})

# Since videos are not all available on a single pages,
# we use a loop to scrape data from all pages
# for page in range(1,(number_of_pages + 1)):
for page in range(1, 11):
    if page == 1:
        urlString = urlStringFirstPage
    else:
        urlString = urlStringNumberedPage + str(page)

    # browser comes from mechanicalsoup, it's a headless browser
    # it is needed for following links to channel pages, scraping subscriber count
    # from those pages, and going back to the url on line 11
    browser.open(urlString)
    print(Fore.WHITE + Style.BRIGHT + browser.get_url())

    html = requests.get(urlString).text
    soup = BeautifulSoup(html, 'html.parser')

    for item in soup('article'):
        try:
            if item['class'][0] == "video-item":
                if item('span')[1]['class'][1] == 'video-item--earned':
                    append_payment_transparent_video()
                else:
                    append_video()

            channel_name = str(item('div')[0].contents[0])

            if (channel_name[-1] == " "):
                channel_name.rstrip()

            print(channel_name)

            check_key_channel(channel, channel_name)
        except:
            pass
        # time.sleep(0.25)

videopd = pd.DataFrame(video_details, columns=['Title', 'Channel', 'Duration', 'Earned', 'Views', 'Likes', 'Uploaded'])
channelpd = pd.DataFrame.from_dict(channel)

print(channel)

sub_list = []
for i, j in videopd.iterrows():
    try:
        given_channel = j['Channel']
        print(Fore.BLUE + Style.BRIGHT + given_channel)
        # print(i, j)
        # print(Fore.MAGENTA + channel[given_channel][0])
        # sub_list.append(channel['The Post Millennial Clips'])
        sub_list.append(channel[given_channel][0])
    except Exception as e:
        print(e)
        print(Fore.RED + Style.BRIGHT + "ALERT ALERT ALERT : NOT ADDED")
        print(i, j)
        pass

videopd['Subscribers'] = sub_list

print()
print()
print(Fore.GREEN + Style.BRIGHT + "PROGRAM COMPLETE...")