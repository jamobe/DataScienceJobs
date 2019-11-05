import requests, bs4, time
import pandas as pd
import os.path
from datetime import datetime
from datetime import date
import scraping_func as sf

path = os.getcwd()
parent_folder, current_folder = os.path.split(path)

url = 'https://www.monster.co.uk/jobs/search/?q=data&saltyp=1&cy=uk&stpage=1&page=10'

#  GET request
res = requests.get(url)
soup = bs4.BeautifulSoup(res.content, features='html.parser')

all_links = sf.monster_links(soup)
links = [s for s in all_links if "job-openings" in s]

today = datetime.now().strftime('%Y_%m_%d_%H_%M')

job_title_company = []
description = []
salary = []
location = []
jobtype = []
industry = []
education = []
career = []
duration = []
ref_code = []

for i in range(0, len(links)): #
    url = links[i]
    res = requests.get(url)
    time.sleep(1)  # ensuring at least 1 second between page grabs
    soup = bs4.BeautifulSoup(res.content, features='html.parser')
    job_title_company.append(sf.monster_jobtitle(soup))
    description.append(sf.monster_descr(soup))
    salary.append(sf.monster_salary(soup))
    location_new, jobtype_new, duration_new, industry_new, education_new, career_new, ref_code_new = sf.monster_summary(
        soup)
    location.append(location_new)
    jobtype.append(jobtype_new)
    duration.append(duration_new)
    industry.append(industry_new)
    education.append(education_new)
    career.append(career_new)
    ref_code.append(ref_code_new)

d = {'title': job_title_company, 'description': description, 'salary': salary, 'location': location, 'jobtype': jobtype,
     'duration': duration, 'industry': industry, 'education': education, 'career': career, 'ref_code': ref_code,
     'url': links}

df = pd.DataFrame(d)

# Data cleaning
df['extraction_date'] = date.today()
df['job_title'] = df['title'].str.split('-', n=1, expand=True)[0]
df['company'] = df['title'].str.split('-', n=1, expand=True)[1]
df.drop(['title'], axis=1, inplace=True)

df['salary_low'] = df['salary'].str.split('-', n=1, expand=True)[0]
df['salary_high'] = df['salary'].str.split('-', n=1, expand=True)[1]

df.to_csv(parent_folder + '/DataScienceJobs/data/monster_all.csv', sep='\t', header=None, mode='a', index=False)
#df.to_csv(parent_folder + '/data/monster_all.csv', sep='\t', header=None, mode='a')