import requests, bs4, time
import pandas as pd
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup
from datetime import date
import os
import re
import scraping_funcs_harnham as sf
import pickle

#urls for German data science jobs
links=[]
for i in range(1,10):
    url="https://www.harnham.com/jobs?options=1111,606&page="+str(i)+"&size=60"
    req=Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage=urlopen(req).read()
    page_soup = soup(webpage, "html.parser")
    a=sf.extract_links(page_soup)
    for element in a:
        links.append(element)

#urls for UK marketing/insight jobs
for i in range(1,10):
    url="https://www.harnham.com/jobs?options=973,606&page="+str(i)+"&size=60"
    req=Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage=urlopen(req).read()
    page_soup = soup(webpage, "html.parser")
    a=sf.extract_links(page_soup)
    for element in a:
        links.append(element)
            
#urls digital analytics
for i in range(1,10):
    url="https://www.harnham.com/jobs?options=1035,606&page="+str(i)+"&size=60"
    req=Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage=urlopen(req).read()
    page_soup = soup(webpage, "html.parser")
    a=sf.extract_links(page_soup)
    for element in a:
        links.append(element)

#urls for data and technology jobs
for i in range(1,10):
    url="https://www.harnham.com/jobs?options=972,606&page="+str(i)+"&size=60"
    req=Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage=urlopen(req).read()
    page_soup = soup(webpage, "html.parser")
    a=sf.extract_links(page_soup)
    for element in a:
        links.append(element)
    


len(links)
    

# scrape all child pages    
info=pd.DataFrame(columns=("job_ref","job_title","location","salary","description","type","extraction_date","country"))
for urls in links:
        time.sleep(1) #ensuring at least 1 second between page grabs
        url="https://www.harnham.com"+urls
        req=Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage=urlopen(req).read()
        page_soup = soup(webpage, "html.parser")
        df=pd.DataFrame()
        df['job_ref']=[sf.extract_jobref(page_soup)]
        df['job_title']=[sf.extract_job_title(page_soup)]
        df['location']=[sf.extract_location(page_soup)]
        df['salary']=[sf.extract_salary(page_soup)]
        df['description']=[sf.extract_description(page_soup)]
        df['type']=[sf.extract_type(page_soup)]
        df['extraction_date'] = date.today()
        df['country'] = 'GER'
        df['url'] = url
        info=info.append(df,ignore_index=True)

        

info.to_pickle(PATH+"RAW_harnham_ger_"+str(date.today())+".pkl")