import requests, bs4, time
import pandas as pd
import os.path
from datetime import datetime


def cw_jobtype(soup):
    """

    :param soup:
    :return:
    """
    jobtype = [x.text for x in soup.find_all('li', {'class': 'job-type'})]
    return jobtype


def cw_full_desc(soup):
    """

    :param soup:
    :return:
    """
    text = [x.text for x in soup.find_all('div', {'class': 'job-description'})]
    return text


def cw_links(soup):
    """

    :param soup:
    :return:
    """
    links = []
    for div in soup.find_all(name='div', attrs={'class': 'job-title'}):
        for a in div.find_all('a'):
            links.append(a['href'])
    return links


def cw_company(soup):
    """

    :param soup:
    :return:
    """
    company = []
    for div in soup.find_all(name="li", attrs={"class": "company"}):
        company.append(div.text.strip())

    return (company)


def cw_date(soup):
    """

    :param soup:
    :return:
    """
    date = []
    for div in soup.find_all('li', {'class': 'date-posted'}):
        date.append(div.text.strip())

    return (date)


def cw_location(soup):
    """

    :param soup:
    :return:
    """
    location = []
    for div in soup.find_all('li', {'class': 'location'}):
        for a in div.find('a'):
            location.append(a)

    return (location)


def cw_salary(soup):
    """

    :param soup:
    :return:
    """
    salaries = []
    for div in soup.find_all(name="li", attrs={"class": "salary"}):
        try:
            salaries.append(div.text)
        except:
            salaries.append("Nothing_found")
    return (salaries)


def cw_job_title(soup):
    """

    :param soup:
    :return:
    """
    jobs = []
    for div in soup.find_all(name="div", attrs={"class": "job-title"}):
        for a in div.find_all(name="h2"):
            jobs.append(a)
    return (jobs)


if __name__ == "__main__":
    path = os.getcwd()
    parent_folder, current_folder = os.path.split(path)

    searchTerm = "data"

    ads = pd.DataFrame(columns=['company', 'title', 'salary', 'location', 'date', 'full_description', 'jobtype', 'url'])

    for i in range(0, 5):
        text_list = []
        type_list = []
        print(i)
        time.sleep(1)  # ensuring at least 1 second between page grabs
        url = 'https://www.cwjobs.co.uk/jobs/' + searchTerm + '?s=header&page=' + str(i)
        res = requests.get(url)
        soup = bs4.BeautifulSoup(res.content, features='html.parser')
        df = pd.DataFrame(columns=['company', 'title', 'salary', 'location', 'date', 'full_description', 'jobtype', 'url'])
        df['company'] = cw_company(soup)
        df['title'] = cw_job_title(soup)
        df['salary'] = cw_salary(soup)
        df['location'] = cw_location(soup)
        df['date'] = cw_date(soup)
        sub_urls = cw_links(soup)

        for j in sub_urls:
            res_sub = requests.get(j)
            soup_sub = bs4.BeautifulSoup(res_sub.content, features='html.parser')
            desc = cw_full_desc(soup_sub)
            job_type = cw_jobtype(soup_sub)
            text_list.append(desc)
            type_list.append(job_type)
        df['full_description'] = text_list
        df['jobtype'] = type_list
        df['url'] = sub_urls
        ads = ads.append(df, ignore_index=True)

        today = datetime.now().strftime('%Y_%m_%d_%H_%M')
        ads.to_csv(parent_folder + '/DataScienceJobs/data/cwjobs_' + today + '.csv', index=True, sep='\t')
