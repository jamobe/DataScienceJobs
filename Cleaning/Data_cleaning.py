import pandas as pd
import datetime
import numpy as np
import os.path
from collections import defaultdict


def find_salary(string):
    """
    Extracting salary from the job descriptions
    :param string:
    :return: salary
    """
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
        df[column] = df[column].str.strip()
        df[column] = df[column].str.split(' ', n=1, expand=True)[0]
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
    #website = 'indeed_de' #231
    #website = 'monster' #609
    website = 'indeed_us' #93

    path = os.getcwd()
    parent_folder, current_folder = os.path.split(path)
    csv_path = '/data/'+website+'_all.csv'

    df = pd.read_csv(path + csv_path, sep='\t')
    print('Uploaded: ' + csv_path + '\n')

    df = df.drop_duplicates(
        subset=['description', 'salary', 'location', 'jobtype', 'industry', 'education', 'career', 'ref_code', 'url',
                'job_title', 'company'], keep='last', inplace=False)
    print('Removed duplicate entries ...\n')

    df = df[df.astype(str)['description'] != '[]']
    print('Removed empty job descriptions ...\n')

    df = df.reset_index(drop=True)

    # back calculate the posted date of job advertisement
    df.duration.replace(regex=True, inplace=True, to_replace=r'Today', value=r'0')
    df.duration.replace(regex=True, inplace=True, to_replace=r'\D', value=r'')
    df.duration = pd.to_numeric(df['duration'])
    df.extraction_date = pd.to_datetime(df.extraction_date)
    df['posted_date'] = df['extraction_date'] - pd.to_timedelta(df['duration'], unit='D')
    df.drop(['extraction_date', 'duration'], axis=1, inplace=True)
    print('Back calculated the date of the job posting ...\n')

    # Extracting salary from description, identifying currency, splitting between salary range (low, high)
    mask1 = ~df.salary.isnull()
    df['currency'] = df.loc[mask1, 'salary'].astype(str).apply(check_currency)
    print('Extracted currency of salary ...\n')
    mask2 = df.currency.isnull()
    df.loc[mask2, 'salary'] = df.loc[mask2, 'description'].apply(find_salary)
    mask3 = ~df.salary.isnull()
    df['currency'] = df.loc[mask3, 'salary'].apply(check_currency)
    print('Extracted salary from job description if not available ...\n')

    # conversion german . to , only for indeed_de
    if website =='indeed_de':
        df.loc[df.currency == '€', 'salary'] = df.loc[df.currency == '€', 'salary'].str.replace('.', '')
        df.loc[df.currency == '€', 'salary'] = df.loc[df.currency == '€', 'salary'].str.replace(',', '.')

    df['salary_low'] = df['salary'].str.split('-', n=1, expand=True)[0]
    df['salary_high'] = df['salary'].str.split('-', n=1, expand=True)[1]
    df = clean_salary(df, ['salary_high', 'salary_low'])
    df = df.dropna(subset=['salary'])

    names = defaultdict(list)
    names['yearly'] = ['year', 'annum', 'p.a', 'Jahr']
    names['montly'] = ['month', 'Monat']
    names['hourly'] = ['hour', 'Stunde']
    names['daily'] = ['day', 'Tag']
    for key, values in names.items():
        for synonyms in values:
            df.loc[df['salary'].str.contains(synonyms), 'per'] = key
    print('Cleaned the salary data (split salary range between low and high) ...\n')

    # transforming scraped locations
    df.location = df.location.str.split(',', n=1, expand=True)[0]
    df.location = df.location.str.title()
    df.location = df.location.str.strip()
    df.loc[df.location.str.contains('London') == True, 'location'] = 'London'
    df.loc[df.location.str.contains('Swindon') == True, 'location'] = 'Swindon'
    df.loc[df.location.str.contains('Franfurt') == True, 'location'] = 'Frankfurt'
    df.loc[df.location.str.contains('B6 7Eu') == True, 'location'] = 'Birmingham'
    df.loc[df.location.str.contains('B63 3Bl') == True, 'location'] = 'Halesowen'
    df.loc[df.location.str.contains('Chandler') == True, 'location'] = 'Chandlers Ford'
    df.loc[df.location.str.contains('Docklands') == True, 'location'] = 'Melbourne'
    df.loc[df.location.str.contains('Berlin') == True, 'location'] = 'Berlin'
    df.loc[df.location.str.contains('Frankfurt') == True, 'location'] = 'Frankfurt'
    df.loc[df.location.str.contains('Stuttgart') == True, 'location'] = 'Stuttgart'
    df.loc[df.location.str.contains('Zuffenhausen') == True, 'location'] = 'Stuttgart'
    df.loc[df.location.str.contains('München') == True, 'location'] = 'München'
    df.loc[df.location.str.contains('Main') == True, 'location'] = 'Frankfurt'
    print('Cleaned location data ...\n')

    # get location from description if not available
    mask4 = df.location.isnull()
    df.loc[mask4, 'location'] = df[mask4]['description'].apply(check_locations)
    df.location = df.location.str.split(',', n=1, expand=True)[0]
    print('Extracted locations from job description if not available ...\n')

    loc = pd.read_csv(path + '/data/locations_UK.csv')
    loc2 = pd.read_csv(path + '/data/locations.csv')
    location = loc.append(loc2, ignore_index=True)
    df2 = pd.merge(df, location, on='location', how='left')
    df2 = df2.reset_index(drop=True)

    df2.loc[df2.location.str.contains('Uk Wide'), 'location'] = 'NaN'
    df2.loc[df2.location.str.contains('England'), 'location'] = 'NaN'
    df2.loc[df2.location.str.contains('Deutschland'), 'location'] = 'NaN'
    print('Identified region and country for each location ...\n')

    output = '/data/cleaned_'+ website + '.csv'

    df2.to_csv(path + output, index=False)
    print('Save results in ' + output)

