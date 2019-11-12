
def clean_column(pandas_df_col):
    import re

    a=pandas_df_col
    for i in range(len(a)):
        rep = {"[": " ",
               "]": " ",
               "|": " ",
               "''": " ",
               "'": " ",
               ",": " ",
               "+": " ",
               "/": " ",
               "€": " ",
               "£": " ",
               "$": " ",
               "benefits":" ",
               "THE COMPANY": " ",
               "THE ROLE": " ",
               "THE BENEFITS": " ",
               "HOW TO APPLY": " ",
               "KEYWORDS": " ",
               "YOUR SKILLS AND EXPERIENCE": " ",
               "YOUR SKILLS AND EXPERTISE": " ",
               "Please register your interest by sending your CV via the Apply link on this page":" ",
               "BENEFITS":" ",
               "CONTACT":" ",
               "OVERVIEW":" ",
               "SALARY":" ",
               "For further details":" ",
               "to enquire about other roles please contact":" ",
               "Nick Mandella":" ",
               "Harnham":" ",
               "On a daily basis":" ",
               "you will be:":" ",
               "you will join:":" ",
               "!" :" ",
               "." : " ",
               "0" :" ",
              "1" : " ",
              "2" : " ",
              "3" :" ",
              "4" : " ",
              "5" : " ",
              "6" : " ",
              "7" :" ",
              "8" : " ",
              "9" : " ",
              "\n":" "}
        rep = dict((re.escape(k), v) for k, v in rep.items()) 
        pattern = re.compile("|".join(rep.keys()))
        if type(a[i])!="str":a[i]=str(a[i])
        #a[i]=str(pandas_df_col[i])
        a[i]=(pattern.sub(lambda m: rep[re.escape(m.group(0))],a[i]))
    return a

def create_split_salary_range(pandas_df_col):
    import pandas as pd
    a = pandas_df_col.str.split(" - ", n = 1, expand = True)[0]
    b = pandas_df_col.str.split(" - ", n = 1, expand = True)[1]
    c = pandas_df_col.str.split("a", n = 1, expand = True)[1]
    
    for x in range(len(b)):
        if pd.isnull(b[x]):
            b[x]=a[x]
        if pd.isnull(c[x]):
            c[x] = 'None'
    return a,b,c

def clean_salary(pandas_df_col,currency_symbol):
    import pandas as pd
    a = pandas_df_col 
    a = [a[x].split("£",1)[1] if (a[x] not in ['Nothing_found', 'None']) else 'None' for x in (range(len(a)))]
    a = [(a[x].split("a",1)[0]) if a[x] is not 'None' else (a[x]) for x in range(len(a))]
    a = [a[x].replace(",","") if a[x] is not 'None' else (a[x]) for x in (range(len(a))) ]
    a = [pd.to_numeric(a[x]) if a[x] is not 'None' else (a[x]) for x in (range(len(a))) ]

    return a    

import numpy as np

def clean_other_deets(pandas_df_col):
    location_2 = pandas_df_col.str.split('_',n = 2, expand = True)[0]
    salary_2 = list(np.zeros(len(pandas_df_col)))
    type_2 = list(np.zeros(len(pandas_df_col)))
    

    for i in range(len(pandas_df_col)):
        pandas_df_col[i] = str(pandas_df_col[i]).strip('[]')
        if pandas_df_col.str.split('_',n = 2, expand = True)[2][i] is not None:
            type_2[i] = pandas_df_col.str.split('_',n = 2, expand = True)[1][i]
            salary_2[i] = pandas_df_col.str.split('_',n = 2, expand = True)[2][i]
        elif  pandas_df_col.str.split('_',n = 2, expand = True)[1][i] is None:
            type_2[i] = 'None'
            salary_2[i] = 'None'   
        elif '£' in pandas_df_col.str.split('_',n = 2, expand = True)[1][i]:
            type_2[i] = 'None'
            salary_2[i] = pandas_df_col.str.split('_',n = 2, expand = True)[1][i]
        else:
            type_2[i] = pandas_df_col.str.split('_',n = 2, expand = True)[1][i]
            salary_2[i] = 'None'

    return location_2, salary_2, type_2 

def create_days_from_post(date_posted_col):

    a  = 'None'

for i in range(ads.shape[0]):
    if (ads['date'][i] == 'Today')|(ads['date'][i] == 'Just posted') :
        ads['days_from_post'] = 0
    elif ads['date'][i] == 'Nothing_found':
        pass
    else:
        a = ads['days_from_post'][i] = ads['date'][i].split("d",1)[0]
        ads['days_from_post'][i] = a.split("+",1)[0]
