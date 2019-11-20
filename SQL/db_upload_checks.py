import psycopg2
from sqlalchemy import create_engine
import pandas as pd

#connect to the database
PASSWORD = pd.read_pickle('C:/Users/lundr/DataScienceJobs/data/SQL_password.pkl')
engine = create_engine('postgresql://postgres:'+PASSWORD.iloc[0,0]+'@dsj-1.c9mo6xd9bf9d.us-west-2.rds.amazonaws.com:5432/')

# 1. Check regions and countries are complete 
regions_complete = pd.read_sql(''' SELECT count(region) FROM landing ''', engine).iloc[0,0]
country_complete = pd.read_sql(''' SELECT count(country) FROM landing ''', engine).iloc[0,0]
total_length = pd.read_sql(''' SELECT count(*) FROM landing ''', engine).iloc[0,0]

if regions_complete == total_length:
    print('region column complete: pass')
else:
    print('region column complete: fail')
    
if country_complete == total_length:
    print('country column complete: pass')
else:
    print('country column complete: fail')  
    
    
    
# 2. Check regions and countries match
country_regions = pd.read_sql(''' SELECT country, region FROM landing ''', engine)

loc_UK = pd.read_csv('~/DataScienceJobs/data/uk_location_lookup.csv')['region'].unique()
loc_GER = pd.read_csv('~/DataScienceJobs/data/locations.csv')['region'].unique()
loc_USA = pd.read_csv('~/DataScienceJobs/data/us-states.csv')['location'].unique()

country_region_dict={'UK':list(loc_UK),
                    'Germany': list(loc_GER),
                    'USA': list(loc_USA)   
}

count = 0
for i in range(len(country_regions)):
    if country_regions['region'][i] in country_region_dict[country_regions['country'][i]]:
        count +=1
    else:
        pass


length = pd.read_sql(''' SELECT count(*) FROM landing ''', engine).iloc[0,0]
if count == length:
    print("All regions match their country: check passed")
else:
    print(str(length - count)+" regions are not in the dictionary or do not match their country")

    
# 4. check that there are no salary_low < 10000 and yearly
salary_low_check = pd.read_sql(''' SELECT count(*) FROM landing WHERE salary_type = 'yearly' AND salary_low > 0 AND salary_low <10000 ''', engine).iloc[0,0]

if salary_low_check == 0:
    print(" All yearly salaries greater than 10,000: check passed")
    
    
    
# 5. check the salary types
import numpy as np

types = ['yearly','daily','monthly','weekly','hourly',None]

type_check = pd.read_sql(''' SELECT salary_type FROM landing''', engine)['salary_type'].unique()

count = 0
for i in type_check:
    if i in types:
        pass
    else:
        count +=1
        
if count == 0:
    print("check that all salary types in dictionary: passed")
else:
    print(str(count)+" salary types not in dictionary")
