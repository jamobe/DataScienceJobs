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
              "9" : " "}
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

    for x in range(len(b)):
        if pd.isnull(b[x]):
            b[x]=a[x]
    return a,b

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

def check_locations(string):
    path = os.getcwd()
    parent_folder, current_folder = os.path.split(path)
    parent_folder
    loc = pd.read_csv(parent_folder+'/data/uk_location_lookup.csv')
    loc2 = pd.read_csv(parent_folder + '/data/locations.csv')
    UK = loc.set_index('location').T.to_dict('list')
    Other = loc2.set_index('location').T.to_dict('list')

    location = [key for key,val in UK.items() if key in string]
    if not location:
        location = [key for key,val in UK.items() if string in key]    
    if not location:
        location = [key for key, val in Other.items() if key in string]
    if not location:
        location = [key for key, val in Other.items() if string in key]
    if not location:
        location = ["not_found"]
    return ','.join(location)