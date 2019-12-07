### code to scrape indeed uk website and dump a pickle of the raw data into a file on the local drive


import requests, bs4, time
import pandas as pd
import os
from itertools import cycle
import traceback
from datetime import date
import pickle



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




if __name__ == "__main__":
    #Define variables
    
    PATH = "data/indeed_raw_pickles/"

    #create empty data frame with column headers
    ads=pd.DataFrame(columns=['company','job_title','salary','location','description','date','full_description','other_deets','extraction_date'])

    # loop for scraping
    #proxy_pool = cycle(proxies)
    jobs =  ['data+science','data+analyst','business+intelligence','data+engineer','AI Scientist','Machine+Learning+Engineer','database+developer','solutions+architect','insight+analytics','database+administrator','data+analytics','data+security','CRM+manager','data+manager','econometrics','statistics']
    for searchTerm in jobs:
        print(searchTerm)
        for i in range(0,1000,10):
            #proxy = next(proxy_pool)
            time.sleep(1) #ensuring at least 1 second between page grabs
            url = "https://www.indeed.co.uk/jobs?q="+searchTerm+"&sort=date&filter=0&l="+"&start="+str(i)
            res = requests.get(url)      
            soup = bs4.BeautifulSoup(res.content, features="lxml")
            df=pd.DataFrame()
            #print(sf.extract_company_from_result(soup))
            df['company'] = extract_company_from_result(soup)
            df['job_title'] = extract_job_title_from_result(soup)
            df['salary'] = extract_salary_from_result(soup)
            df['location'] = extract_location_from_result(soup)
            df['description'] = extract_description_from_result(soup)
            df['date'] = extract_date_from_result(soup)

            sub_urls=extract_links(soup)
            urls_list =[]
            text_list=[]
            deets_list=[]
            for j in sub_urls:
                res_sub = requests.get(j)
                soup_sub = bs4.BeautifulSoup(res_sub.content, features="lxml")
                desc= extract_full_desc(soup_sub)
                text_list.append(desc)
                other_deets=extract_headlines_from_result(soup_sub)
                deets_list.append(other_deets)
                urls_list.append(sub_urls)  
            df['full_description']=pd.DataFrame(text_list)
            df['other_deets']=pd.DataFrame(deets_list)
            df['extraction_date'] = date.today()
            ads=ads.append(df, ignore_index=True)
            print(i)



        ads.to_pickle(PATH+"RAW_indeeduk_"+searchTerm+str(date.today())+".pkl")
