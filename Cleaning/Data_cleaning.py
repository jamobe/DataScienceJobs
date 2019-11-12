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
    idx = defaultdict(list)
    idx['£'] = string.find('£')
    idx['€'] = string.find('€')
    idx['€'] = string.find('EUR')
    idx['$'] = string.find('$')
    position = [value for name, value in idx.items() if value >= 0]
    if position:
        salary = string[position[0]:position[0]+40]
    else:
        salary = 'NA'
    return salary


def check_currency(string):
    """
    Extract currency from salary
    :param string:
    :return: currency
    """
    idx = defaultdict(list)
    idx['£'] = string.find('£')
    idx['€'] = string.find('€')
    idx['€'] = string.find('EUR')
    idx['$'] = string.find('$')
    curr = [name for name, value in idx.items() if value >= 0]
    if not curr:
        curr = 'NA'
    return ''.join(curr)


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
    return ''.join(location)


if __name__ == "__main__":
    path = os.getcwd()
    parent_folder, current_folder = os.path.split(path)
    print(parent_folder)
    print(current_folder)
    csv_path = '/data/indeed_us_all.csv'
    print(csv_path)
    df = pd.read_csv(path + csv_path, sep='\t')
    print('Uploaded: ' + csv_path + '\n')

    df = df.drop_duplicates(
        subset=['description', 'salary', 'location', 'jobtype', 'industry', 'education', 'career',
                'ref_code', 'url', 'job_title', 'company'], keep='last', inplace=False)
    print('Removed duplicate entries ...\n')

    df = df[df.astype(str)['description'] != '[]']
    print('Removed empty job descriptions ...\n')

    df = df.reset_index()

    # back calculate the posted date of job advertisement
    df.duration.replace(regex=True, inplace=True, to_replace=r'Today', value=r'0')
    df.duration.replace(regex=True, inplace=True, to_replace=r'\D', value=r'')
    df.duration = pd.to_numeric(df['duration'])
    df.extraction_date = pd.to_datetime(df.extraction_date)
    df['posted_date'] = df['extraction_date'] - pd.to_timedelta(df['duration'], unit='D')
    df.drop(['extraction_date', 'duration'], axis=1, inplace=True)
    print('Back calculated the date of the job posting ...\n')

    # Extracting salary from description, identifying currency, splitting between salary range (low, high)
    df['currency'] = df.salary.apply(check_currency)
    print('Extracted currency of salary ...\n')

    mask = df.currency == 'NA'
    df.loc[mask, 'salary'] = df.loc[mask, 'description'].apply(find_salary)
    df['currency'] = df.salary.apply(check_currency)
    print('Extracted salary from job description if not available ...\n')

    df['salary_low'] = df['salary'].str.split('-', n=1, expand=True)[0]
    df['salary_high'] = df['salary'].str.split('-', n=1, expand=True)[1]
    df = clean_salary(df, ['salary_high', 'salary_low'])
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
    print('Cleaned location data ...\n')

    # get location from description if not available
    mask2 = df.location.isnull()
    df.loc[mask2, 'location'] = df[mask2]['description'].apply(check_locations)
    print('Extracted locations from job description if not available ...\n')

    loc = pd.read_csv(path + '/data/locations_UK.csv')
    loc2 = pd.read_csv(path + '/data/locations.csv')
    location = loc.append(loc2, ignore_index=True)
    df2 = pd.merge(df, location, on='location', how='left').drop(columns='index', axis=1)
    df2.loc[df2.location.str.contains('Uk Wide'), 'location'] = ''
    df2.loc[df2.location.str.contains('England'), 'location'] = ''
    print('Identified region and country for each location ...\n')

    output = '/data/cleaned_Indeed_us.csv'
    df2.to_csv(path + output, index=False)
    print('Save results in ' + output)

