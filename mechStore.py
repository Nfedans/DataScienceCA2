from bs4 import BeautifulSoup
import requests
import pandas as pd
import mechanicalsoup
from colorama import Fore, Style


# This function redirects to individual pages, scrapes subscriber count,
# puts the subscriber count into the channel dictionary and
# redirects back to the main page were scraping from
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
    if chname[-1] == " ":
        chname = chname.rstrip()

    channel[chname] = [str(btns('span')[1].contents[0])]

    browser.open(urlString)


# This function checks is key:calue pair already exists,
# If not, it calls the redirector fuction
def check_key_channel(dic, key):
    if key in dic:
        print(Style.BRIGHT + Fore.YELLOW + key + " : Already Exists")
    else:
        print(Style.BRIGHT + Fore.GREEN + "Adding " + key + "...")
        redirect_scrape_return()


def make_append():
    channel_string = str(item('div')[0].contents[0])
    if channel_string[-1] == " ":
        channel_string = channel_string.rstrip()

    video_details.append([
        int(counter),  # Rank
        str(item('h3')[0].contents[0]),  # Title
        channel_string,  # Channel
        str(item('span')[0]['data-value']),  # Duration
        '0',  # Earned
        str(item('span')[1]['data-value']),  # Views
        str(item('span')[2]['data-value']),  # Likes
        str(item('time')[0]['datetime']),  # Uploaded
    ])


# Some videos show how much $ they earned while other videos do not
# This function is used to append video data correctly for the videos
# Which DO show the viewer how much $ it made
def make_payment_transparent_append():
    channel_string = str(item('div')[0].contents[0])
    if channel_string[-1] == " ":
        channel_string = channel_string.rstrip(" ")

    video_details.append([
        int(counter),  # Rank
        str(item('h3')[0].contents[0]),  # Title
        channel_string,  # Channel
        str(item('span')[0]['data-value']),  # Duration
        str(item('span')[1]['data-value']),  # Earned
        str(item('span')[2]['data-value']),  # Views
        str(item('span')[3]['data-value']),  # Likes
        str(item('time')[0]['datetime']),  # Uploaded
    ])


# 2 urls needed because the first page of videos does not include "&page=1"
# However all other pages specify their number in the url
urlStringFirstPage = 'https://rumble.com/videos?sort=views&date=this-month'
urlStringNumberedPage = 'https://rumble.com/videos?sort=views&date=this-month&page='

# video_details is the list which will later be turned into the main panda dataframe
video_details = []

# Rumble's first page is index 1 and the max amount of pages it shows is 100
# However, if we use just the top 50 pages, we get over 1000 results and mining is quicker
number_of_pages = 50

# channel will store {channel_name : channel_subscriber_count} K:V Pairs
channel = {}

browser = mechanicalsoup.StatefulBrowser(soup_config={'features': 'lxml'})

counter = 1

# Since videos are not all available on a single page,
# we use a loop to scrape data from all pages
for page in range(1, (number_of_pages + 1)):
    if page == 1:
        urlString = urlStringFirstPage
    else:
        urlString = urlStringNumberedPage + str(page)

    # browser comes from mechanicalsoup, it's a headless browser
    # it is needed for following links to channel pages, scraping subscriber count
    # from those pages, and going back to the url on line 11
    browser.open(urlString)
    print(Fore.WHITE + Style.BRIGHT + "SCRAPING:" + browser.get_url())

    html = requests.get(urlString).text
    soup = BeautifulSoup(html, 'html.parser')

    for item in soup('article'):
        try:
            if item['class'][0] == "video-item":
                if item('span')[1]['class'][1] == 'video-item--earned':
                    # Some videos show how much $ they earned
                    # They are treated differently from other videos
                    make_payment_transparent_append()
                else:
                    make_append()

            counter = counter + 1

            # This is a check for a nasty surprise whitespace at the end of the string
            # Where it doesnt belong (Caused problems before)
            channel_name = str(item('div')[0].contents[0])
            if channel_name[-1] == " ":
                channel_name = channel_name.rstrip()

            check_key_channel(channel, channel_name)

        except:
            pass

videopd = pd.DataFrame(video_details,
                       columns=['Rank', 'Title', 'Channel', 'Duration', 'Earned', 'Views', 'Likes', 'Uploaded'])
channelpd = pd.DataFrame.from_dict(channel)

# Below loop runs through video dataframe, selects channels,
# and appends the channel's subscriber count to sub_list
sub_list = []
for i, j in videopd.iterrows():
    try:
        given_channel = j['Channel']
        print(Fore.BLUE + Style.BRIGHT + given_channel)
        sub_list.append(channel[given_channel][0])
    except Exception as e:
        print(e)
        print(Fore.RED + Style.BRIGHT + "ALERT ALERT ALERT : NOT ADDED")
        print(i, j)
        pass

# We add subscribers column to video dataframe
videopd['Subscribers'] = sub_list

print()
print()
print(Fore.GREEN + Style.BRIGHT + "PROGRAM COMPLETE...")

# Write out to csv file
videopd.to_csv("DataSciCA2Data8Columns.csv")
