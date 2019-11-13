import requests, bs4, time
import pandas as pd
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

    searchTerm = "data"

    ads = pd.DataFrame(columns=['company', 'job_title', 'salary', 'location', 'date', 'description', 'url'])

    for i in range(0, 60):  # range(0:1000)
        text_list = []
        print(i)
        time.sleep(1)  # ensuring at least 1 second between page grabs
        url = "https://de.indeed.com/Jobs?q=" + searchTerm + "&filter=0&start=" + str(i)
        res = requests.get(url)
        soup = bs4.BeautifulSoup(res.content, features='html.parser')
        df = pd.DataFrame(columns=['company', 'job_title', 'salary', 'location', 'date', 'description', 'url'])
        df['company'] = indeed_company(soup)
        df['job_title'] = indeed_job_title(soup)
        df['salary'] = indeed_salary(soup)
        df['location'] = indeed_location(soup)
        df['date'] = indeed_date(soup)

        sub_urls = indeed_links(soup)
        text = []
        for j in sub_urls:
            res_sub = requests.get(j)
            soup_sub = bs4.BeautifulSoup(res_sub.content, features='html.parser')
            desc = indeed_full_desc(soup_sub)
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

    ads.to_csv(parent_folder + '/DataScienceJobs/data/indeed_de_all.csv', sep='\t', header=None, mode='a', index=False)