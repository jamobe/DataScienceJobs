### code to scrape indeed uk website, and dump a pickle of the raw data into a file on the local drive


import requests, bs4, time
import pandas as pd
import os
import scraping_funcs_indeeduk as sf
from itertools import cycle
import traceback
from datetime import date
import pickle

#Define variables
searchTerm="machine learning"
PATH = "C:/Users/lundr/DataScienceJobs/data/pickles/"

#create empty data frame with column headers
ads=pd.DataFrame(columns=['company','job_title','salary','location','description','date','full_description','other_deets','extraction_date'])

# loop for scraping
#proxy_pool = cycle(proxies)

for i in range(0,1000,10):
    #proxy = next(proxy_pool)
    time.sleep(1) #ensuring at least 1 second between page grabs
    url = "https://www.indeed.co.uk/jobs?q="+searchTerm+"&filter=0&l="+"&start="+str(i)
    res = requests.get(url)      
    soup = bs4.BeautifulSoup(res.content)
    df=pd.DataFrame()
    #print(sf.extract_company_from_result(soup))
    df['company'] = sf.extract_company_from_result(soup)
    df['job_title'] = sf.extract_job_title_from_result(soup)
    df['salary'] = sf.extract_salary_from_result(soup)
    df['location'] = sf.extract_location_from_result(soup)
    df['description'] = sf.extract_description_from_result(soup)
    df['date'] = sf.extract_date_from_result(soup)

    sub_urls=sf.extract_links(soup)
    urls_list =[]
    text_list=[]
    deets_list=[]
    for j in sub_urls:
        res_sub = requests.get(j)
        soup_sub = bs4.BeautifulSoup(res_sub.content)
        desc=sf.extract_full_desc(soup_sub)
        text_list.append(desc)
        other_deets=sf.extract_headlines_from_result(soup_sub)
        deets_list.append(other_deets)
        urls_list.append(sub_urls)  
    df['full_description']=pd.DataFrame(text_list)
    df['other_deets']=pd.DataFrame(deets_list)
    df['extraction_date'] = date.today()
    ads=ads.append(df, ignore_index=True)
    print(i)



ads.to_pickle(PATH+"RAW_indeeduk_"+searchTerm+str(date.today()))  
