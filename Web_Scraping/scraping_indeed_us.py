import requests
import bs4
import time
import pandas as pd
import numpy as np
import os.path
from datetime import date


def indeed_job_title(bsoup):
    """
    Extracting job title from Indeed.com
    :param bsoup:
    :return: job title
    """
    jobs = []
    for div in bsoup.find_all(name="div", attrs={"class": "row"}):
        for a in div.find_all(name="a", attrs={"data-tn-element": "jobTitle"}):
            jobs.append(a["title"])
    return jobs


def indeed_salary(bsoup):
    """
    Extracting the salary from Indeed.com
    :param bsoup:
    :return: salary
    """
    salaries = []
    for div in bsoup.find_all(name="div", attrs={"class": "row"}):
        try:
            salaries.append(div.find(name="span", attrs={"class": "salaryText"}).text)
        except:
            salaries.append("Nothing_found")
    return salaries


def indeed_location(bsoup):
    """
    Extracting the location from Indeed.com
    :param bsoup:
    :return: location
    """
    locations = []
    for div in bsoup.find_all(name="div", attrs={"class": "row"}):
        try:
            locations.append(div.find("span", attrs={"class": "location accessible-contrast-color-location"}).text)
        except:
            locations.append("Nothing_found")
    return locations


def indeed_description(bsoup):
    """
    Extracting the basic description from Indeed.com
    :param bsoup:
    :return: basic description
    """
    description = []
    for div in bsoup.find_all(name="div", attrs={"class": "row"}):
        try:
            description.append(div.find("div", attrs={"class": "summary"}).text)
        except:
            description.append("Nothing_found")
    return description


def indeed_date(bsoup):
    """
    Extracting the publication date of the job advertisement on Indeed.com
    :param bsoup:
    :return: date
    """
    find_date = []
    for div in bsoup.find_all(name="div", attrs={"class": "row"}):
        try:
            find_date.append(div.find("span", attrs={"class": "date"}).text)
        except:
            find_date.append("Nothing_found")
    return find_date


def indeed_company(bsoup):
    """
    Extracting the company of the job advertisement on Indeed.com
    :param bsoup:
    :return: company
    """
    company = []
    for div in bsoup.find_all(name="div", attrs={"class": "row"}):
        try:
            company.append(div.find("span", attrs={"class": "company"}).text)
        except:
            company.append("Nothing_found")
    return company


def indeed_us_links(bsoup):
    """
        Extracting the link of the job advertisement on Indeed.com
        :param bsoup:
        :return: link
        """
    links = []
    for div in bsoup.find_all(name='a', attrs={'class': 'jobtitle turnstileLink'}):
        links.append('https://www.indeed.com' + str(div['href']))
    return links


def indeed_full_desc(bsoup):
    """
    Extracting the full description from Indeed.com
    :param bsoup:
    :return: full description
    """
    text = [x.text for x in bsoup.find_all(name="div", attrs={"id": "jobDescriptionText"})]
    return text


if __name__ == "__main__":
    path = os.getcwd()
    parent_folder, current_folder = os.path.split(path)

    searchTerm = 'data+scientist'
    # completed Search Terms: 'ai+scientist', 'data', 'econometrics', 'business+intelligence', 'statistics',
    # 'data+engineer', 'machine+learning', 'data+analyst', 'data+manager'
    print('Scraping job descriptions for the search term: ' + searchTerm)

    total_jobs = 0
    for i in range(0, 100):  # range(0:1000)
        text_list = []
        print('scraping page ' + str(i) + ' of 100')
        time.sleep(1)  # ensuring at least 1 second between page grabs
        url = 'https://www.indeed.com/jobs?q=' + searchTerm + '&l=United+States&start=' + str(i)
        res = requests.get(url)
        soup = bs4.BeautifulSoup(res.content, features='html.parser')
        df = pd.DataFrame(columns=['company', 'job_title', 'salary', 'location', 'duration', 'description', 'url'])
        df['company'] = indeed_company(soup)
        df['job_title'] = indeed_job_title(soup)
        df['salary'] = indeed_salary(soup)
        df['location'] = indeed_location(soup)
        df['duration'] = indeed_date(soup)

        sub_urls = indeed_us_links(soup)
        for j in sub_urls:
            res_sub = requests.get(j)
            soup_sub = bs4.BeautifulSoup(res_sub.content, features='html.parser')
            desc = indeed_full_desc(soup_sub)
            text_list.append(desc)
        df['description'] = text_list
        df['url'] = sub_urls

        # basic data cleaning
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

        if os.path.isfile(path + '/data/indeed_us_all.csv'):
            df_final.to_csv(path + '/data/indeed_us_all.csv', sep='\t', header=None, mode='a',
                            index=False)
        else:
            df_final.to_csv(path + '/data/indeed_us_all.csv', sep='\t', index=False)

    print('scraped ' + str(total_jobs) + ' job postings.')
