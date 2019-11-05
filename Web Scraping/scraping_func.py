############################################
#     Monster.uk scraping Functions        #
############################################

def monster_links(soup):
    links =[]
    for div in soup.find_all(name='a', attrs={'data-bypass': 'true'}):
        links.append(div['href'])
    return links

# Function for extracting the job title from the website:
def monster_jobtitle(soup):
    title = 'no_title'
    for div in soup.find_all(name='h1', attrs={'class': 'title'}):
        title = div.text
    return title

# Function for extracting the job description from the website:
def monster_descr(soup):
    jobdesc = []
    for div in soup.find_all(name='span', attrs={'id': 'TrackingJobBody'}):
        jobdesc = div.text
    return jobdesc

# Function for extracting salary from the website:
def monster_salary(soup):
    salary = []
    for div in soup.find_all('div',{'class': 'col-xs-12 cell'}):
        salary = div.text
        salary = salary.replace('Salary','').strip()
    return salary

# Function for extracting meta-data from the website:
def monster_summary(soup):
    output = ['nan','nan', 'nan', 'nan', 'nan', 'nan', 'nan']
    separation =['Location', 'Job type', 'Posted', 'Industries', 'Education level', 'Career level', 'Reference code']
    for div in soup.find_all('dl', {'class': 'header'}):
        summary = div.text
        summary = summary.replace('\n', '')
        for idx, item in enumerate(separation):
            if item in summary:
                output[idx] = summary.replace(item,'')
    return output

#########################################
#         cwjobs website functions      #
#########################################

def cw_jobtype(soup):
    jobtype = [x.text for x in soup.find_all('li', {'class': 'job-type'})]
    return jobtype


def cw_full_desc(soup):
    text = [x.text for x in soup.find_all('div', {'class': 'job-description'})]
    return text


def cw_links(soup):
    links = []
    for div in soup.find_all(name='div', attrs={'class': 'job-title'}):
        for a in div.find_all('a'):
            links.append(a['href'])
    return links


def cw_company(soup):
    company = []
    for div in soup.find_all(name="li", attrs={"class": "company"}):
        company.append(div.text.strip())

    return (company)


def cw_date(soup):
    date = []
    for div in soup.find_all('li', {'class': 'date-posted'}):
        date.append(div.text.strip())

    return (date)


def cw_location(soup):
    location = []
    for div in soup.find_all('li', {'class': 'location'}):
        for a in div.find('a'):
            location.append(a)

    return (location)


def cw_salary(soup):
    salaries = []
    for div in soup.find_all(name="li", attrs={"class": "salary"}):
        try:
            salaries.append(div.text)
        except:
            salaries.append("Nothing_found")
    return (salaries)


def cw_job_title(soup):
    jobs = []
    for div in soup.find_all(name="div", attrs={"class": "job-title"}):
        for a in div.find_all(name="h2"):
            jobs.append(a)
    return (jobs)

##########################################################
#      indeed.de and Indeed.us website functions         #
##########################################################

def indeed_job_title(soup):
    jobs = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        for a in div.find_all(name="a", attrs={"data-tn-element": "jobTitle"}):
            jobs.append(a["title"])
    return jobs


def indeed_salary(soup):
    salaries = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        try:
            salaries.append(div.find(name="span", attrs={"class": "salaryText"}).text)
        except:
            salaries.append("Nothing_found")
    return salaries


def indeed_location(soup):
    locations = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        try:
            locations.append(div.find("span", attrs={"class": "location accessible-contrast-color-location"}).text)
        except:
            locations.append("Nothing_found")

    return locations


def indeed_description(soup):
    description = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        try:
            description.append(div.find("div", attrs={"class": "summary"}).text)
        except:
            description.append("Nothing_found")

    return description


def indeed_date(soup):
    date = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        try:
            date.append(div.find("span", attrs={"class": "date"}).text)
        except:
            date.append("Nothing_found")

    return (date)


def indeed_company(soup):
    company = []
    for div in soup.find_all(name="div", attrs={"class": "row"}):
        try:
            company.append(div.find("span", attrs={"class": "company"}).text)
        except:
            company.append("Nothing_found")

    return company


def indeed_links(soup):
    links = []
    for div in soup.find_all(name='a', attrs={'class': 'jobtitle turnstileLink'}):
        links.append('https://de.indeed.com' + str(div['href']))
    return links

def indeed_us_links(soup):
    links = []
    for div in soup.find_all(name='a', attrs={'class': 'jobtitle turnstileLink'}):
        links.append('https://www.indeed.com' + str(div['href']))
    return links


def indeed_full_desc(soup):
    text = [x.text for x in soup.find_all(name="div", attrs={"id": "jobDescriptionText"})]
    return text

