import requests, bs4, time
import pandas as pd
import os.path
from datetime import datetime
from datetime import date
import scraping_func as sf

path = os.getcwd()
parent_folder, current_folder = os.path.split(path)

# scraping code:

# decide what search term to use for finding jobs
searchTerm = "data"

# create empty data frame with column headers
ads = pd.DataFrame(columns=['company', 'job_title', 'salary', 'location', 'date', 'description', 'url'])

# loop for scraping

for i in range(0, 100):  # range(0:1000)
    text_list = []
    print(i)
    time.sleep(1)  # ensuring at least 1 second between page grabs
    url = 'https://www.indeed.com/jobs?q=' + searchTerm + '&l=United+States&start=' + str(i)
    res = requests.get(url)
    soup = bs4.BeautifulSoup(res.content, features='html.parser')
    df = pd.DataFrame(
        columns=['company', 'job_title', 'salary', 'location', 'date', 'description', 'url'])
    df['company'] = sf.indeed_company(soup)
    df['job_title'] = sf.indeed_job_title(soup)
    df['salary'] = sf.indeed_salary(soup)
    df['location'] = sf.indeed_location(soup)
    df['date'] = sf.indeed_date(soup)

    sub_urls = sf.indeed_us_links(soup)
    for j in sub_urls:
        res_sub = requests.get(j)
        soup_sub = bs4.BeautifulSoup(res_sub.content, features='html.parser')
        desc = sf.indeed_full_desc(soup_sub)
        text_list.append(desc)

    df['description'] = text_list
    df['url'] = sub_urls

    ads = ads.append(df, ignore_index=True)

# Data cleaning
ads['extraction_date'] = date.today()
ads.company = ads.company.str.strip()
ads.description = ads.description.str.strip()
ads.salary = ads.salary.str.strip()
ads['salary_low'] = ads['salary'].str.split('-', n=1, expand=True)[0]
ads['salary_high'] = ads['salary'].str.split('-', n=1, expand=True)[1]
ads['job_type'] = 'Not available'
ads['industry'] = 'Not available'
ads['education'] = 'Not available'
ads['career'] = 'Not available'

today = datetime.now().strftime('%Y_%m_%d_%H_%M')
ads.to_csv(parent_folder + '/DataScienceJobs/data/indeed_us_all.csv', sep='\t', header=None, mode='a', index=False)
