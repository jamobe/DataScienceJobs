import os.path
from sqlalchemy import create_engine
import pandas as pd

if __name__ == "__main__":
    path = os.getcwd()

    #connect to the database
    PASSWORD = pd.read_pickle(path + '/data/SQL_access.pkl')
    engine = create_engine('postgresql://postgres:'+PASSWORD+'@dsj-1.c9mo6xd9bf9d.us-west-2.rds.amazonaws.com:5432/')

    # 1. Check regions and countries are complete
    regions_complete = pd.read_sql(''' SELECT count(region) FROM all_data ''', engine).iloc[0,0]
    country_complete = pd.read_sql(''' SELECT count(country) FROM all_data ''', engine).iloc[0,0]
    total_length = pd.read_sql(''' SELECT count(*) FROM all_data ''', engine).iloc[0,0]

    if regions_complete == total_length:
        print('region column complete: pass')
    else:
        print('region column complete: fail')

    if country_complete == total_length:
        print('country column complete: pass')
    else:
        print('country column complete: fail')



    # 2. Check regions and countries match
    country_regions = pd.read_sql(''' SELECT country, region FROM all_data ''', engine).reset_index()

    loc_UK = pd.read_csv(path + '/data/uk_location_lookup.csv')['region'].unique()
    loc_GER = pd.read_csv(path + '/data/locations.csv')['region'].unique()
    loc_USA = pd.read_csv(path + '/data/us-states.csv')['region'].unique()

    country_region_dict={'UK':list(loc_UK),
                        'Germany': list(loc_GER),
                        'USA': list(loc_USA),
                            None: None,
                          'Netherlands': 'North Holland',
                          'Australia': 'Tasmania',
                          'Sweden': None,
                          'Belgium': 'Brussels',
                          'Ireland': 'Leinster',
                          'Switzerland': 'Zurich',
                          'Lithuania': 'Vilnius',
                          'France': 'Ile-de-France',
                          'Spain': 'Barcelona',
                          'Poland': 'Masovia',
                          'Canada': 'Canada'}


    count = 0
    for i in range(len(country_regions)):
        if country_regions['region'][i]is not None:
            if country_regions['region'][i] in country_region_dict[country_regions['country'][i]]:
                count +=1
        else:
            pass



    length = pd.read_sql(''' SELECT count(*) FROM all_data ''', engine).iloc[0,0]
    if count == length:
        print("All regions match their country: check passed")
    else:
        print(str(length - count)+" regions are not in the dictionary or do not match their country")


    # 4. check that there are no salary_low < 10000 and yearly
    salary_low_check = pd.read_sql(''' SELECT count(*) FROM all_data WHERE salary_type = 'yearly' AND salary_low > 0 AND salary_low <10000 ''', engine).iloc[0,0]

    if salary_low_check == 0:
        print(" All yearly salaries greater than 10,000: check passed")



    # 5. check the salary types
    import numpy as np

    types = ['yearly','daily','monthly','weekly','hourly',None]

    type_check = pd.read_sql(''' SELECT salary_type FROM all_data''', engine)['salary_type'].unique()

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

        
            # 4. check for yearly salaries with upper range more than three times the lower range
    salary_ratio_check = pd.read_sql(''' SELECT count(*) FROM all_data WHERE salary_type = 'yearly' AND salary_high_euros/salary_low_euros > 3 ''', engine).iloc[0,0]

    if salary_ratio_check == 0:
        print(" salary highs no more than 3 times salarly lows: check passed")
    else:
        print ("salaries violating 1/3 ratio check:")
        print("salary_ratio_check")