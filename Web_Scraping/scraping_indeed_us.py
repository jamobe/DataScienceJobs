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

    searchTerm = "machine+learning"
    #searchTerm = "data"

    # create empty data frame with column headers
    ads = pd.DataFrame(columns=['company', 'job_title', 'salary', 'location', 'duration', 'description', 'url'])

    for i in range(0, 1000):  # range(0:1000)
        text_list = []
        print(i)
        time.sleep(1)  # ensuring at least 1 second between page grabs
        url = 'https://www.indeed.com/jobs?q=' + searchTerm + '&l=United+States&start=' + str(i)
        #url = 'https://www.indeed.com/jobs?q=' + searchTerm+ '&l='
        res = requests.get(url)
        soup = bs4.BeautifulSoup(res.content, features='html.parser')
        df = pd.DataFrame(
            columns=['company', 'job_title', 'salary', 'location', 'duration', 'description', 'url'])
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

        ads = ads.append(df, ignore_index=True)

        # basic data cleaning
        ads['extraction_date'] = date.today()
        ads.company = ads.company.str.strip()
        ads.description = ads.description.str.strip()
        ads.salary = ads.salary.str.strip()
        ads['salary_low'] = np.NaN
        ads['salary_high'] = np.NaN
        ads['jobtype'] = 'Nothing_found'
        ads['industry'] = 'Nothing_found'
        ads['education'] = 'Nothing_found'
        ads['career'] = 'Nothing_found'
        ads['ref_code'] = 'Nothing_found'
        ads = ads.replace('Nothing_found',np.NaN)

        #today = datetime.now().strftime('%Y_%m_%d_%H_%M')
        ads.to_csv(parent_folder + '/DataScienceJobs/data/indeed_us_all.csv', sep='\t', header=None, mode='a', index=False)
