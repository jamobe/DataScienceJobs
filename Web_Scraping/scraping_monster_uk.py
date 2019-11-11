import requests, bs4, time
import pandas as pd
import os.path
from datetime import datetime
from datetime import date
from collections import defaultdict


def monster_links(soup):
    """
    Retrieves Links from Monster.uk website
    :param soup:
    :return: link
    """
    links = []
    for div in soup.find_all(name='a', attrs={'data-bypass': 'true'}):
        links.append(div['href'])
    return links


def monster_jobtitle(soup):
    """Extracting titles from job descriptions from Monster.uk
    :param soup:
    :return: title
    """
    title = 'no_title'
    for div in soup.find_all(name='h1', attrs={'class': 'title'}):
        title = div.text
    return title


def monster_descr(soup):
    """
    Extracting detailed job description from Monster.uk
    :param soup:
    :return: description
    """
    jobdesc = []
    for div in soup.find_all(name='span', attrs={'id': 'TrackingJobBody'}):
        jobdesc = div.text
    return jobdesc


def monster_salary(soup):
    """
    Extracting salary from Monster.uk
    :param soup:
    :return: salary
    """
    salary = []
    for div in soup.find_all('div', {'class': 'col-xs-12 cell'}):
        salary = div.text
        salary = salary.replace('Salary', '').strip()
    return salary


# Function for extracting meta-data from the website:
def monster_summary(soup):
    """
    Extracting meta-data from Monster.uk
    :param soup:
    :return: location, job type, posted, industries, education level, career level, reference code
    """
    output = ['nan', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan']
    separation = ['Location', 'Job type', 'Posted', 'Industries', 'Education level', 'Career level', 'Reference code']
    for div in soup.find_all('dl', {'class': 'header'}):
        summary = div.text
        summary = summary.replace('\n', '')
        for idx, item in enumerate(separation):
            if item in summary:
                output[idx] = summary.replace(item, '')
    return output


if __name__ == "__main__":
    path = os.getcwd()
    parent_folder, current_folder = os.path.split(path)

    url = 'https://www.monster.co.uk/jobs/search/?q=data&saltyp=1&cy=uk&stpage=1&page=10'

    #  GET request
    res = requests.get(url)
    soup = bs4.BeautifulSoup(res.content, features='html.parser')

    all_links = monster_links(soup)
    links = [s for s in all_links if "job-openings" in s]

    # today = datetime.now().strftime('%Y_%m_%d_%H_%M')
    data = defaultdict(list)

    for i in range(0, len(links)):
        print(i)
        url = links[i]
        res = requests.get(url)
        time.sleep(1)  # ensuring at least 1 second between page grabs
        soup = bs4.BeautifulSoup(res.content, features='html.parser')
        data['title'].append(monster_jobtitle(soup))
        data['description'].append(monster_descr(soup))
        data['salary'].append(monster_salary(soup))
        location_new, jobtype_new, duration_new, industry_new, education_new, career_new, ref_code_new = monster_summary(soup)
        data['location'].append(location_new)
        data['jobtype'].append(jobtype_new)
        data['duration'].append(duration_new)
        data['industry'].append(industry_new)
        data['education'].append(education_new)
        data['career'].append(career_new)
        data['ref_code'].append(ref_code_new)
        data['url'].append(url)

    df = pd.DataFrame(data)

    # Data cleaning
    df['extraction_date'] = date.today()
    df['job_title'] = df['title'].str.split('-', n=1, expand=True)[0]
    df['company'] = df['title'].str.split('-', n=1, expand=True)[1]
    df.drop(['title'], axis=1, inplace=True)

    df['salary_low'] = df['salary'].str.split('-', n=1, expand=True)[0]
    df['salary_high'] = df['salary'].str.split('-', n=1, expand=True)[1]

    df.to_csv(parent_folder + '/DataScienceJobs/data/monster_all.csv', sep='\t', header=None, mode='a', index=False)
    # df.to_csv(parent_folder + '/data/monster_all.csv', sep='\t', header=None, mode='a')
