import requests, bs4, time
import pandas as pd
import numpy as np
import os.path
from datetime import date


def indeed_job_title(soup):
    """
    Extracting job title from Indeed.com
    :param soup:
    :return: job title
    """
    jobs = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        for a in div.find_all(name="a", attrs={"data-tn-element": "jobTitle"}):
            jobs.append(a["title"])
    return jobs


def indeed_salary(soup):
    """
    Extracting the salary from Indeed.com
    :param soup:
    :return: salary
    """
    salaries = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        try:
            salaries.append(div.find(name="span", attrs={"class": "salaryText"}).text)
        except:
            salaries.append("Nothing_found")
    return salaries


def indeed_location(soup):
    """
    Extracting the location from Indeed.com
    :param soup:
    :return: location
    """
    locations = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        try:
            locations.append(div.find("span", attrs={"class": "location accessible-contrast-color-location"}).text)
        except:
            locations.append("Nothing_found")
    return locations


def indeed_description(soup):
    """
    Extracting the basic description from Indeed.com
    :param soup:
    :return: basic description
    """
    description = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        try:
            description.append(div.find("div", attrs={"class": "summary"}).text)
        except:
            description.append("Nothing_found")
    return description


def indeed_date(soup):
    """
    Extracting the publication date of the job advertisement on Indeed.com
    :param soup:
    :return: date
    """
    date = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        try:
            date.append(div.find("span", attrs={"class": "date"}).text)
        except:
            date.append("Nothing_found")
    return date


def indeed_company(soup):
    """
    Extracting the company of the job advertisement on Indeed.com
    :param soup:
    :return: company
    """
    company = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        try:
            company.append(div.find("span", attrs={"class": "company"}).text)
        except:
            company.append("Nothing_found")
    return company


def indeed_us_links(soup):
    """
        Extracting the link of the job advertisement on Indeed.com
        :param soup:
        :return: link
        """
    links = []
    for div in soup.find_all(name='a', attrs={'class': 'jobtitle turnstileLink'}):
        links.append('https://www.indeed.com' + str(div['href']))
    return links


def indeed_full_desc(soup):
    """
    Extracting the full description from Indeed.com
    :param soup:
    :return: full description
    """
    text = [x.text for x in soup.find_all(name="div", attrs={"id": "jobDescriptionText"})]
    return text


if __name__ == "__main__":
    path = os.getcwd()
    parent_folder, current_folder = os.path.split(path)

    # 'data', 'econometrics'
    searchTerm = 'business+intelligence'
    # 'data+analyst','data+scientist','machine+learning', 'data+engineer','data+manager', ,'statistics', 'data+analyst'

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
        #if i == 0:
        #    df_final.to_csv(parent_folder + '/DataScienceJobs/data/indeed_us_all_2.csv', sep='\t', index=False)
        #else:
        df_final.to_csv(parent_folder + '/DataScienceJobs/data/indeed_us_all_2.csv', sep='\t', header=None, mode='a', index=False)
    print('scraped ' + str(total_jobs) + ' job postings.')
