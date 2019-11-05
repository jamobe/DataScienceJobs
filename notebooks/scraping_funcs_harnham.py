def extract_links(soup):
    links =[]
    for div in soup.find_all(name='a', attrs={'class':'job-block__title-link'}):
        links.append(div['href'])
    return links

def extract_salary(page_soup):
    for div in page_soup.find_all("div",{"class","cop-widget dynamic-widget salary-widget"}):
         a=[x.text for x in div.find_all("span")]
    return a[0].lstrip('Salary:\n').rstrip(' \n')


def extract_location(page_soup):
    a=[]
    for div in page_soup.find_all("div",{"class","cop-widget dynamic-widget text-widget"}):
         a.append([x.text for x in div.find_all("div",{"class",""})])
    return a[0]

def extract_jobref(page_soup):
    a=[]
    for div in page_soup.find_all("div",{"class","cop-widget dynamic-widget text-widget"}):
         a.append([x.text for x in div.find_all("div",{"class",""})])
    return a[1]

def extract_type(page_soup):
    a=[x.text for x in page_soup.find_all("li",{"class","JobType-wrapper"})]
    type=a[0].lstrip('Job type\n                    \n').rstrip(' \n')
    return type

def extract_description(page_soup):
    b=page_soup.find_all("div",{"class":"cop-widget dynamic-widget description-widget"})
    for div  in b:
        c=str([x.text for x in div.find_all("p")]) +" "+ str([x.text for x in div.find_all("li")])
        desc=''.join(c)
    return desc

def extract_job_title(page_soup):
    return [x.text for x in page_soup.find_all("strong")][0]
    