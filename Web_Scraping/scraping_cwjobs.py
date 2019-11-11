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
ads = pd.DataFrame(columns=['company', 'title', 'salary', 'location', 'date', 'full_description', 'jobtype', 'url'])

# loop for scraping

for i in range(0, 5):
    company = []
    job_title = []
    description = []
    salary = []
    location = []
    date = []
    full_description = []
    text_list = []
    type_list = []
    print(i)

    time.sleep(1)  # ensuring at least 1 second between page grabs
    url = 'https://www.cwjobs.co.uk/jobs/' + searchTerm + '?s=header&page=' + str(i)
    res = requests.get(url)
    soup = bs4.BeautifulSoup(res.content, features='html.parser')
    df = pd.DataFrame(columns=['company', 'title', 'salary', 'location', 'date', 'full_description', 'jobtype', 'url'])
    df['company'] = sf.cw_company(soup)
    df['title'] = sf.cw_job_title(soup)
    df['salary'] = sf.cw_salary(soup)
    df['location'] = sf.cw_location(soup)
    df['date'] = sf.cw_date(soup)

    sub_urls = sf.cw_links(soup)
    for j in sub_urls:
        res_sub = requests.get(j)
        soup_sub = bs4.BeautifulSoup(res_sub.content, features='html.parser')
        desc = sf.cw_full_desc(soup_sub)
        job_type = sf.cw_jobtype(soup_sub)
        text_list.append(desc)
        type_list.append(job_type)

    df['full_description'] = text_list
    df['jobtype'] = type_list
    df['url'] = sub_urls

    ads = ads.append(df, ignore_index=True)

today = datetime.now().strftime('%Y_%m_%d_%H_%M')
ads.to_csv(parent_folder + '/DataScienceJobs/data/cwjobs_' + today + '.csv', index=True, sep='\t')