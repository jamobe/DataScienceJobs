import requests, bs4, time
import pandas as pd
import numpy as np
import os.path
from datetime import datetime
from datetime import date
from scraping_indeed_us import indeed_job_title, indeed_salary, indeed_location, indeed_description, indeed_date, indeed_company, indeed_full_desc


def indeed_links(soup):
    links = []
    for div in soup.find_all(name='a', attrs={'class': 'jobtitle turnstileLink'}):
        links.append('https://de.indeed.com' + str(div['href']))
    return links


if __name__ == "__main__":
    path = os.getcwd()
    parent_folder, current_folder = os.path.split(path)
    # Completed Searches: 'ai+scientist', 'data+scientist', 'econometrics', 'business+intelligence', 'statistics',
    # 'data+engineer', 'machine+learning', 'data+analyst', 'data+manager'

    searchTerm = 'data'
    print('Scraping job descriptions for the search term: ' + searchTerm)

    total_jobs = 0
    for i in range(0, 100):  # range(0:1000)
        text_list = []
        print('scraping page ' + str(i) + ' of 100')
        time.sleep(1)  # ensuring at least 1 second between page grabs
        url = "https://de.indeed.com/Jobs?q=" + searchTerm + "&filter=0&start=" + str(i)
        res = requests.get(url)
        soup = bs4.BeautifulSoup(res.content, features='html.parser')
        df = pd.DataFrame(columns=['company', 'job_title', 'salary', 'location', 'duration', 'description', 'url'])
        df['company'] = indeed_company(soup)
        df['job_title'] = indeed_job_title(soup)
        df['salary'] = indeed_salary(soup)
        df['location'] = indeed_location(soup)
        df['duration'] = indeed_date(soup)

        sub_urls = indeed_links(soup)
        text = []
        for j in sub_urls:
            res_sub = requests.get(j)
            soup_sub = bs4.BeautifulSoup(res_sub.content, features='html.parser')
            desc = indeed_full_desc(soup_sub)
            text_list.append(desc)
        df['description'] = text_list
        df['url'] = sub_urls

        # Data cleaning
        df['extraction_date'] = date.today()
        df.company = df.company.str.strip()
        df.salary = df.salary.str.strip()
        df['salary_low'] = np.NaN
        df['salary_high'] = np.NaN
        df['jobtype'] = np.NaN
        df['industry'] = np.NaN
        df['education'] = np.NaN
        df['career'] = np.NaN
        df['ref_code'] = np.NaN

        cols = ['company', 'job_title', 'salary', 'location', 'duration', 'description', 'url',
                'extraction_date', 'salary_low', 'salary_high', 'jobtype', 'industry', 'education', 'career',
                'ref_code']

        df_final = df[cols]
        total_jobs = total_jobs + df_final.shape[0]

        if os.path.isfile(path + '/data/indeed_de_all.csv'):
            df_final.to_csv(path + '/data/indeed_de_all.csv', sep='\t', header=None, mode='a',
                            index=False)
        else:
            df_final.to_csv(path + '/data/indeed_de_all.csv', sep='\t', index=False)

    print('scraped ' + str(total_jobs) + ' job postings.')
