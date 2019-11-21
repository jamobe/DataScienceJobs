
def clean_column(pandas_df_col):
    """
    cleaning text columns, removing various regular expressions
    :param pandas_df_col:
    :return: column text in string format
    """
    import re
    a=pandas_df_col
    for i in range(len(a)):
        rep = {"[": " ",
               "]": " ",
             "\n":" "}
        rep = dict((re.escape(k), v) for k, v in rep.items()) 
        pattern = re.compile("|".join(rep.keys()))
        if type(a[i])!="str":a[i]=str(a[i])
        #a[i]=str(pandas_df_col[i])
        a[i]=(pattern.sub(lambda m: rep[re.escape(m.group(0))],a[i]))
    return a

def create_split_salary_range(pandas_df_col):
    """
    splits the salary column into the upper and lower ranges and whether its per, day, week, year etc.
    where there is only one salary value, upper and lower values are allocated as the same.
    :param pandas_df_col:
    :return: column text in string format
    """
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
    """
     Cleans the salary columns, removing all text around numbers
    :param pandas_df_col:
    :return: salary as numeric values
    """
    import pandas as pd
    a = pandas_df_col 
    a = [a[x].split("£",1)[1] if (a[x] not in ['Nothing_found', 'None']) else 'None' for x in (range(len(a)))]
    a = [(a[x].split("a",1)[0]) if a[x] is not 'None' else (a[x]) for x in range(len(a))]
    a = [a[x].replace(",","") if a[x] is not 'None' else (a[x]) for x in (range(len(a))) ]
    a = [pd.to_numeric(a[x]) if a[x] is not 'None' else (a[x]) for x in (range(len(a))) ]

    return a    



def clean_other_deets(pandas_df_col):
    """
    Extracts info on salary, location and job type from other deets column, where they are given
    :param pandas_df_col:
    :return: text for three columns
    """
    import numpy as np
    
    location_2 = pandas_df_col.str.split('_',n = 2, expand = True)[0]
    salary_2 = list(np.zeros(len(pandas_df_col)))
    type_2 = list(np.zeros(len(pandas_df_col)))
    

    for i in range(len(pandas_df_col)):
        pandas_df_col[i] = str(pandas_df_col[i]).strip('[]')
        if pandas_df_col.str.split('_',n = 2, expand = True)[0][i] is None:
            location_2 = 'None'
            type_2[i] = 'None'
            salary_2[i] = 'None'
        elif pandas_df_col.str.split('_',n = 2, expand = True)[2][i] is not None:
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
    """
    Extracts the number of days ago the job was posted, removing surrounding text
    :param pandas_df_col:
    :return: days as string
    """
    a  = []
    for i in range(len(date_posted_col)):
        if (date_posted_col[i] == 'Today')|(date_posted_col[i] == 'Just posted') :
            a.append(0)
        elif date_posted_col[i] == 'Nothing_found':
            a.append('None')
        else:
            b = date_posted_col[i].split("d",1)[0]
            a.append(b.split("+",1)[0])
    return a



def create_posted_date(days_from_post_col, extraction_date_col):
    """
    Calculates the estimated day on which the job was posted. "30+" is treated as 30 days
    :param days_from_post_col,extraction_date_col :
    :return: date
    """
    import datetime
    from datetime import date
    import pandas as pd
    a = []
    for x in range(len(days_from_post_col)):
        if days_from_post_col[x] is not 'None':
             a. append(pd.to_datetime(extraction_date_col[x]) - datetime.timedelta(days=int((days_from_post_col[x]))))
        else:
            a.append('None')
    return a

def salary_average(low, high):
    """
    Calculates the average of the salary range as a integer
    :param days_from_post_col,extraction_date_col :
    :return: a new variable salary average as an integer
    """
    import numpy as np
    a=list(np.zeros(len(low)))
    for i in range(len(low)):
        if low[i] is not 'None':
            a[i] = int((low[i] +high[i])/2)
        else:
            a[i] = 'None'           
    return a

def check_locations(pandas_df_col, country_col):
    import os
    import pandas as pd
    path = os.getcwd()
    parent_folder, current_folder = os.path.split(path)
    loc_UK = pd.read_csv(parent_folder+'/data/uk_location_lookup.csv')
    loc_GER = pd.read_csv(parent_folder+'/data/locations.csv')
    loc_USA = pd.read_csv(parent_folder+'/data/us-states.csv')[['region']]
    
    lookup_UK = loc_UK.set_index('location').T.to_dict('list')
    lookup_GER = loc_GER.set_index('location').T.to_dict('list')
    zipbObj = zip(loc_USA['region'], loc_USA['region'])
    lookup_USA= dict(zipbObj)

    
    a=[]
    for i in range(len(country_col)):
        string = str(pandas_df_col[i])
        if country_col[i] == 'UK':
            location = [val[0] for key,val in lookup_UK.items() if key in string]
        elif country_col[i] == 'GER':
            location = [val[0] for key,val in lookup_GER.items() if key in string]
        elif country_col[i] == 'USA':
            
            location = [val[0] for key,val in lookup_USA.items() if key in string]
        else:
            location = np.nan        
        
        a.append(location)
    a = [ str(a[x]).replace("[","") for x in range(len(a))]
    a = [a[x].replace("]","") for x in range(len(a))]
    a = [a[x].replace("'","") for x in range(len(a))]
    
    return a

def clean_salary_type(pandas_df_col):
    times = {'year': 'yearly',
             'month':'monthly',
             'week' : 'weekly',
             'hour' : 'hourly',
             'day' : 'daily'
    }

    return pandas_df_col.map(times)   
                        
