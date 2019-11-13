#define functions for parsing HTML

def extract_job_title_from_result(soup): 
    import pandas as pd
    jobs = []
    for div in soup.find_all(name="div", attrs={"class":"row"}):
        try:
            for a in div.find_all(name="a", attrs={"data-tn-element":"jobTitle"}):
                jobs.append(a["title"])
        except:
                jobs.append("Nothing_found")
    return pd.DataFrame(jobs)

def extract_salary_from_result(soup): 
    salaries = []
    for div in soup.find_all(name="div", attrs={"class":"row"}):
        try:
            salaries.append(div.find(name="span",attrs={"class":"salaryText"}).text)
        except:
            salaries.append("Nothing_found")
    return salaries

def extract_location_from_result(soup):
    locations = []
    for div in soup.find_all(name="div", attrs={"class":"row"}):
        try:
            locations.append(div.find("span", attrs={"class": "location accessible-contrast-color-location"}).text)
        except:
            locations.append("Nothing_found")
   
    return locations

def extract_description_from_result(soup):
    description = []
    for div in soup.find_all(name="div", attrs={"class":"row"}):
        try:
            description.append(div.find("div", attrs={"class": "summary"}).text)
        except:
            description.append("Nothing_found")
   
    return description

def extract_date_from_result(soup): 
    import pandas as pd
    date = []
    for div in soup.find_all(name="div", attrs={"class":"row"}):
        try:
            date.append(div.find("span", attrs={"class": "date"}).text)
        except:
            date.append("Nothing_found")
    return date

def extract_company_from_result(soup): 
    company = []
    for div in soup.find_all(name="div", attrs={"class":"row"}):
        try:
            company.append(div.find("span", attrs={"class": "company"}).text)
        except:
            company.append("Nothing_found")
   
    return company

def extract_links(soup):
    links =[]
    for div in soup.find_all(name='a', attrs={'class':'jobtitle turnstileLink'}):
        links.append('https://www.indeed.co.uk'+str(div['href']))
    return links

def extract_full_desc(soup):
    text=[x.text for x in soup.find_all(name="div",attrs={"id":"jobDescriptionText"})]
    return text


def extract_headlines_from_result(soup): 
    import pandas as pd
    headlines =[]
    try:
        vals=[x.text for x in soup.find_all(name="span",attrs={"class":"jobsearch-JobMetadataHeader-iconLabel"})]
        a ='_'.join(vals)
        headlines.append(a)
    except:
        headlines.append("Nothing_found")
    return headlines
