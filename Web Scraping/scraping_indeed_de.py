import requests, bs4, time
import pandas as pd
import os.path
from datetime import datetime
import scraping_func as sf

path = os.getcwd()
parent_folder, current_folder = os.path.split(path)

# scraping code:

# decide what search term to use for finding jobs
searchTerm = "data"

# create empty data frame with column headers
ads = pd.DataFrame(columns=['company', 'title', 'salary', 'location', 'description', 'date', 'full_description', 'url'])

# loop for scraping

for i in range(0, 250):  # range(0:1000)
    company = []
    job_title = []
    description = []
    salary = []
    location = []
    date = []
    full_description = []
    text_list = []

    time.sleep(1)  # ensuring at least 1 second between page grabs
    url = "https://de.indeed.com/Jobs?q=" + searchTerm + "&filter=0&start=" + str(i)
    res = requests.get(url)
    soup = bs4.BeautifulSoup(res.content, features='html.parser')
    df = pd.DataFrame(
        columns=['company', 'title', 'salary', 'location', 'description', 'date', 'full_description', 'url'])
    df['company'] = sf.indeed_company(soup)
    df['title'] = sf.indeed_job_title(soup)
    df['salary'] = sf.indeed_salary(soup)
    df['location'] = sf.indeed_location(soup)
    df['description'] = sf.indeed_description(soup)
    df['date'] = sf.indeed_date(soup)

    sub_urls = sf.indeed_links(soup)
    text = []
    deets = []
    for j in sub_urls:
        res_sub = requests.get(j)
        soup_sub = bs4.BeautifulSoup(res_sub.content, features='html.parser')
        desc = sf.indeed_full_desc(soup_sub)
        text_list.append(desc)

    df['full_description'] = text_list
    df['url'] = sub_urls

    ads = ads.append(df, ignore_index=True)

today = datetime.now().strftime('%Y_%m_%d_%H_%M')
ads.to_csv(parent_folder + '/data/indeed_de_' + today + '.csv', index=True, sep='\t')