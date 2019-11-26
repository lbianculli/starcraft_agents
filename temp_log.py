import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

headers = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML like Gecko) Chrome/72.0.3626.81 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Referer': 'https://stats.nba.com/',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
}



def use_driver(url):
    """ starts up a ChromeDriver, gets dom, and closes """
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Chrome("/usr/bin/chromedriver", options=options)
    driver.get(url)
    dom = driver.page_source
    driver.close()

    return dom
  
  
player_id = 201935  #harden
season = "2016-17"
url = "https://stats.nba.com/stats/shotchartdetail?AheadBehind=&ClutchTime=&ContextFilter=&ContextMeasure=FGA&DateFrom=&DateTo=&EndPeriod=&EndRange=&GameID=&GameSegment=&LastNGames=0&LeagueID=00&Location=&Month=0&OpponentTeamID=0&Outcome=&Period=0&PlayerID=201935&PlayerPosition=&PointDiff=&Position=&RangeType=&RookieYear=&Season=2018-19&SeasonSegment=&SeasonType=Regular+Season&StartPeriod=&StartRange=&TeamID=0&VsConference=&VsDivision="
print(url)  # working fine on click
# resp = requests.get(url, headers=headers, timeout=20)
test = use_driver(url)
