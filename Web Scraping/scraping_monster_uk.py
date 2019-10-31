import requests, bs4, time
import pandas as pd
import os.path
from datetime import datetime
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
with open(parent_folder+'/DataScienceJobs/data/monster_links_'+today+'.txt', 'w') as f:
    for item in links:
        f.write("%s\n" % item)

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

for i in range(0, len(links)):
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
df.to_csv(parent_folder + '/DataScienceJobs/data/monster_' + today + '.csv', index=True, sep='\t')