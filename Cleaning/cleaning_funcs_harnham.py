#function to clean the location column in the Harnham data set


def remove_duped_info(pandas_df_col,list_of_cols_to_compare):
    a=pandas_df_col
    for i in list_of_cols_to_compare:
        for j in range(len(a)):     
            l=a[j].lower()
            l_2 =str(i[j]).lower()
            a[j] = l.replace(l_2," ")
    return a


def clean_column(pandas_df_col):
    import re

    a=pandas_df_col
    for i in range(len(a)):
        rep = {"[": " ",
               "]": " ",
            }
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
    c = pandas_df_col.str.split(" per ", n = 1, expand = True)[1]
    c = c.str.split("+",n = 1, expand = True)[0]

    for x in range(len(b)):
        if pd.isnull(b[x]):
            b[x]=a[x]
    return a,b,c

def clean_salary(pandas_df_col,currency_symbol):
    import pandas as pd     
    a = [pandas_df_col[x].split(currency_symbol,1)[1] for x in range(len(pandas_df_col))]        
    a = [a[x].split("per",1)[0] for x in range(len(a))]
    a = pd.to_numeric(a)
    return a

def clean_jobref(pandas_df_col):
    a = [ str(pandas_df_col[x]).replace("[","") for x in range(len(pandas_df_col))]
    a = [a[x].replace("]","") for x in range(len(pandas_df_col))]
    a = [a[x].replace("'","") for x in range(len(pandas_df_col))]
    return a

def check_locations(pandas_df_col, country_col):
    import os
    import pandas as pd
    path = os.getcwd()
    parent_folder, current_folder = os.path.split(path)
    loc_UK = pd.read_csv(parent_folder+'/data/uk_location_lookup.csv')
    loc_GER = pd.read_csv(parent_folder+'/data/locations.csv')
    loc_USA = pd.read_csv(parent_folder+'/data/us-states.csv')
    
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
            
            location = [val for key,val in lookup_USA.items() if key in string]
        else:
            location = np.nan        
        
        a.append(location)
    a = [ str(a[x]).replace("[","") for x in range(len(a))]
    a = [a[x].replace("]","") for x in range(len(a))]
    a = [a[x].replace("'","") for x in range(len(a))]
    
    return a

def clean_salary_type(pandas_df_col):
    a = list(pandas_df_col.copy())
    for i in range(len(a)):
        if 'day' in a[i]:
            a[i] = 'daily'
        elif 'year' or 'annum' in a[i]:
            a[i] = 'yearly'
    return a