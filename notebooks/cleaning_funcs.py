#function to clean the location column in the Harnham data set

def clean_location(pandas_df_col):
    import re
    a=pandas_df_col
    for i in range(len(a)):
        rep = {"[": "",
               "]": "",
               "''": "",
               "'": ""}
        a[i]=str(pandas_df_col[i])
        rep = dict((re.escape(k), v) for k, v in rep.items()) 
        pattern = re.compile("|".join(rep.keys()))
        return pattern.sub(lambda m: rep[re.escape(m.group(0))], a[i])

    

def remove_duped_info(pandas_df_col,list_of_cols_to_compare):
    for i in list_of_cols_to_compare:
        a =[x.replace(y,"") for x,y in zip(pandas_df_col,i)]
    return a


def clean_column(pandas_df_col):
    import re
    series=[]
    a=pandas_df_col
    for i in range(len(a)):
        rep = {"[": "",
               "]": "",
               "|": "",
               "''": "",
               "'": "",
               ",": "",
               "+": "",
               "/": "",
               "benefits":"",
               "THE COMPANY": "",
               "THE ROLE": "",
               "THE BENEFITS": "",
               "HOW TO APPLY": "",
               "KEYWORDS": "",
               "YOUR SKILLS AND EXPERIENCE": "",
               "YOUR SKILLS AND EXPERTISE": "",
               "Please register your interest by sending your CV via the Apply link on this page":"",
               "BENEFITS":"",
               "CONTACT":"",
               "OVERVIEW":"",
               "SALARY":"",
               "For further details":"",
               "to enquire about other roles please contact":"",
               "Nick Mandella":"",
               "Harnham":"",
               "On a daily basis":"",
               "you will be:":"",
               "you will join:":"",
               "!":"",
               ".": ""}
        rep = dict((re.escape(k), v) for k, v in rep.items()) 
        pattern = re.compile("|".join(rep.keys()))
        if type(a[i])!="str":a[i]=str(a[i])
        a[i]=str(pandas_df_col[i])
        clean_column=(pattern.sub(lambda m: rep[re.escape(m.group(0))],a[i]))
        return clean_column

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
    a =[x.split(currency_symbol,1)[1] for x in pandas_df_col]
    a =[x.split("per",1)[0] for x in pandas_df_col]
    a =pd.to_numeric(pandas_df_col)
    return a    
