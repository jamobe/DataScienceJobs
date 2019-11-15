import pandas as pd
import re
import numpy as np
import os.path
from collections import defaultdict


def extract_salary(string):
    try:
        result = re.findall(r'(?:[\£\$\€].{1}[,\d]+.?\d*)',string)
        result = ' - '.join(result)
    except:
        result = 'NaN'
    return result


def extract_salary_after(string):
    try:
        result = re.findall(r'([.|,\d]+,?\d*.[\£\$\€])',string)
        result = ' - '.join(result)
    except:
        result = 'NaN'
    return result


def find_salary(string):
    """
    Extracting salary from the job descriptions
    :param string:
    :return: salary
    """
    string = str(string)
    currencies = ['£', '€', 'EUR', '$']
    found = [i for i in string if i in currencies]
    if found:
        position = string.find(found[0])
        start = [0 if position < 20 else position - 20]
        salary = string[start[0]:position + 40]
    else:
        salary = np.NaN
    return salary


def check_currency(string):
    """
    Extract currency from salary
    :param string:
    :return: currency
    """
    currencies = ['£', '€', 'EUR', '$']
    found = [i for i in string if i in currencies]
    if not found:
        found = np.NaN
    else:
        found = found[0]
    return found


def clean_salary(df, column_names):
    """
    cleaning salary and transforming to floats
    :param df:
    :param column_names:
    :return: salaries in float format
    """
    for column in column_names:
        if df[column] is not None:
            df[column] = df[column].str.strip()
            #df[column] = df[column].str.split(' ', n=1, expand=True)[0]
            df[column] = df[column].str.split('.', n=1, expand=True)[0]
            df[column].replace(regex=True, inplace=True, to_replace=r'\D', value=r'')
            df[column] = pd.to_numeric(df[column])
    return df


def check_locations(string):
    """
    Extract locations from job description
    :param string:
    :return: location
    """
    path = os.getcwd()
    loc = pd.read_csv(path +'/data/locations_UK.csv')
    loc2 = pd.read_csv(path + '/data/locations.csv')
    UK_cities = loc.set_index('location').T.to_dict('list')
    Cities = loc2.set_index('location').T.to_dict('list')
    location = [key for key,val in UK_cities.items() if key in string]
    if not location:
        location = [key for key, val in Cities.items() if key in string]
    return ','.join(location)


if __name__ == "__main__":
    website = 'indeed_de' #374
    #website = 'monster' #776
    #website = 'indeed_us' #235

    path = os.getcwd()
    parent_folder, current_folder = os.path.split(path)
    csv_path = '/data/'+website+'_all.csv'

    df = pd.read_csv(path + csv_path, sep='\t')
    length0 = df.shape[0]
    print('Uploaded: ' + csv_path + '\n')
    print('Containing ' + str(length0) + ' entries... \n')

    df = df.drop_duplicates(
        subset=['description', 'salary', 'location', 'jobtype', 'industry', 'education', 'career', 'ref_code', 'url',
                'job_title', 'company'], keep='last', inplace=False)
    length1 = df.shape[0]
    print('Removed '+ str(length0-length1) + ' duplicate entries: ' + str(length1) + '...\n')

    df = df[df.astype(str)['description'] != '[]']
    length2 = df.shape[0]
    print('Removed '+ str(length1-length2) +' empty job descriptions: ' + str(length2) + '...\n')

    df = df.reset_index(drop=True)

    # back calculate the posted date of job advertisement
    df.duration.replace(regex=True, inplace=True, to_replace=r'Today', value=r'0')
    df.duration.replace(regex=True, inplace=True, to_replace=r'\D', value=r'')
    df.duration = pd.to_numeric(df['duration'])
    df.extraction_date = pd.to_datetime(df.extraction_date)
    df['posted_date'] = df['extraction_date'] - pd.to_timedelta(df['duration'], unit='D')
    df.drop(['extraction_date', 'duration'], axis=1, inplace=True)
    print('Back calculated the date of the job posting...\n')

    # Extracting salary from description, identifying currency, splitting between salary range (low, high)
    mask1 = ~df.salary.isnull()
    df['currency'] = df.loc[mask1, 'salary'].astype(str).apply(check_currency)
    print('Extracted currency of salary...\n')
    mask2 = df.currency.isnull()
    df.loc[mask2, 'salary'] = df.loc[mask2, 'description'].apply(find_salary)
    mask3 = ~df.salary.isnull()
    df['currency'] = df.loc[mask3, 'salary'].apply(check_currency)
    print('Extracted salary from job description if not available...\n')

    # conversion german . to , only for indeed_de
    if website =='indeed_de':
        #df.loc[df.currency == '€', 'salary'] = df.loc[df.currency == '€', 'salary'].str.replace('.', ',000.')
        df['salary_extract'] = df['salary'].apply(extract_salary_after)
        df.loc[df.currency == '€', 'salary_extract'] = df.loc[df.currency == '€', 'salary_extract'].str.replace('.', ',')
    else:
        df['salary_extract'] = df['salary'].apply(extract_salary)

    df['salary_low'] = df['salary_extract'].str.split('-', n=1, expand=True)[0]
    df['salary_high'] = df['salary_extract'].str.split('-', n=1, expand=True)[1]
    df = clean_salary(df, ['salary_high', 'salary_low'])
    df = df.drop(columns='salary_extract', axis=1)
    df = df.dropna(subset=['salary'])

    names = defaultdict(list)
    names['yearly'] = ['year', 'annum', 'p.a', 'Jahr']
    names['montly'] = ['month', 'Monat']
    names['hourly'] = ['hour', 'Stunde']
    names['daily'] = ['day', 'Tag']
    for key, values in names.items():
        for synonyms in values:
            df.loc[df['salary'].str.contains(synonyms), 'per'] = key
    length3 = df.shape[0]
    print('Removed '+ str(length2-length3) + ' NaN from Salary column: ' + str(length3) + '...\n')

    # get location from description if not available
    mask4 = df.location.isnull()
    df.loc[mask4, 'location'] = df[mask4]['description'].apply(check_locations)
    print('Extracted locations from job description if not available...\n')

    # transforming scraped locations
    if website == 'indeed_us':
        df['state'] = df.location.str.split(',', n=1, expand=True)[1]
        df['state'] = df['state'].str.strip()
        df['state'] = df['state'].str.split(' ', n=1, expand=True)[0]
        df['state'] = df['state'].str.upper()
        df.loc[df.location.str.contains('Wisconsin') == True, 'state'] = 'WI'
        df.loc[df.location.str.contains('North Carolina') == True, 'state'] = 'NC'
        df.loc[df.location.str.contains('Texas') == True, 'state'] = 'TX'
        df.loc[df.location.str.contains('Indiana') == True, 'state'] = 'IN'
        df.loc[df.location.str.contains('New Jersey') == True, 'state'] = 'NJ'
        locations_us = pd.read_csv(path + '/data/us-states.csv')
        df2 = pd.merge(df, locations_us, on='state', how='left')
        df.location = df.location.str.split(',', n=1, expand=True)[0]
        df.location = df.location.str.title()
        df.location = df.location.str.strip()
        df.loc[df.location.str.contains('United States') == True, 'country'] = 'USA'
        df2 = df2.reset_index(drop=True)
    else:
        df.location = df.location.str.split(',', n=1, expand=True)[0]
        df.location = df.location.str.title()
        df.location = df.location.str.strip()
        df.loc[df.location.str.contains('London') == True, 'location'] = 'London'
        df.loc[df.location.str.contains('Swindon') == True, 'location'] = 'Swindon'
        df.loc[df.location.str.contains('Franfurt') == True, 'location'] = 'Frankfurt'
        df.loc[df.location.str.contains('B6 7Eu') == True, 'location'] = 'Birmingham'
        df.loc[df.location.str.contains('B63 3Bl') == True, 'location'] = 'Halesowen'
        df.loc[df.location.str.contains('Chandler') == True, 'location'] = 'Chandlers Ford'
        df.loc[df.location.str.contains('Docklands') == True, 'location'] = 'London'
        df.loc[df.location.str.contains('Berlin') == True, 'location'] = 'Berlin'
        df.loc[df.location.str.contains('Frankfurt') == True, 'location'] = 'Frankfurt'
        df.loc[df.location.str.contains('Stuttgart') == True, 'location'] = 'Stuttgart'
        df.loc[df.location.str.contains('Zuffenhausen') == True, 'location'] = 'Stuttgart'
        df.loc[df.location.str.contains('München') == True, 'location'] = 'München'
        df.loc[df.location.str.contains('Main') == True, 'location'] = 'Frankfurt'
        location_UK = pd.read_csv(path + '/data/locations_UK.csv')
        location_EU= pd.read_csv(path + '/data/locations.csv')
        location = location_UK.append(location_EU, ignore_index=True)
        df2 = pd.merge(df, location, on='location', how='left')  # .drop('Unnamed: 0', axis=1)
        df2 = df2.reset_index(drop=True)
    print('Cleaned location data...\n')

    df2.loc[df2.location.str.contains('Uk Wide'), 'location'] = 'NaN'
    df2.loc[df2.location.str.contains('England'), 'location'] = 'NaN'
    df2.loc[df2.location.str.contains('Deutschland'), 'location'] = 'NaN'
    print('Identified region and country for each location...\n')

    output = '/data/cleaned_'+ website + '.csv'

    cols = ['description', 'salary', 'location', 'jobtype', 'industry', 'education',
            'career', 'ref_code', 'url', 'job_title', 'company', 'salary_low',
            'salary_high', 'posted_date', 'currency', 'per', 'region', 'country']
    df3 = df2[cols]

    length4 = df3.shape[0]
    print('Save '+ str(length4) +' results in ' + output + '...\n')
    df3.to_csv(path + output, index=False)
    print('Done!')

