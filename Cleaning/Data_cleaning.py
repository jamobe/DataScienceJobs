import pandas as pd
import re
import numpy as np
import string
import os.path
from collections import defaultdict


def extract_salary(salary_string):
    """
    Extracts from a string the salary starting with £, $, € and followed by integers
    :param salary_string: text containing salaries
    :return: (integer) salary
    """
    try:
        result = re.findall(r'(?:[\£\$\€].{1}[,\d]+.?\d*)', salary_string)
        result = ' - '.join(result)
    except:
        result = 'NaN'
    return result


def extract_salary_after(salary_string):
    """
    Extracts from a string the salary starting with integers and followed by £, $, €
    :param salary_string: text containing salaries
    :return: (integer) salary
    """
    try:
        result = re.findall(r'([.|,\d]+,?\d*.[\£\$\€])', salary_string)
        result = ' - '.join(result)
    except:
        result = 'NaN'
    return result


def find_salary(salary_string):
    """
    Extracting salary from the job descriptions
    :param salary_string: job descriptions
    :return: salary
    """
    string = str(salary_string)
    currencies = ['£', '€', 'EUR', '$']
    found = [i for i in string if i in currencies]
    if found:
        position = string.find(found[0])
        start = [0 if position < 20 else position - 20]
        salary = string[start[0]:position + 40]
    else:
        salary = np.NaN
    return salary


def check_currency(currency_string):
    """
    Extract currency ('£', '€', 'EUR', '$') from salary
    :param string:
    :return: currency
    """
    currencies = ['£', '€', 'EUR', '$']
    found = [i for i in currency_string if i in currencies]
    if not found:
        found = np.NaN
    else:
        found = found[0]
    return found


def clean_salary(dataframe, column_names):
    """
    cleaning salary and transforming to floats
    :param dataframe:
    :param column_names:
    :return: salaries in float format
    """
    for column in column_names:
        if dataframe[column] is not None:
            dataframe[column] = dataframe[column].str.strip()
            # df[column] = df[column].str.split(' ', n=1, expand=True)[0]
            dataframe[column] = dataframe[column].str.split('.', n=1, expand=True)[0]
            dataframe[column].replace(regex=True, inplace=True, to_replace=r'\D', value=r'')
            dataframe[column] = pd.to_numeric(dataframe[column])
    return dataframe


def check_locations(location_string, path):
    """
    Extract locations from job description
    :param location_string:
    :return: location
    """
    loc = pd.read_csv(path + '/data/locations_UK.csv')
    loc2 = pd.read_csv(path + '/data/locations.csv')
    UK_cities = loc.set_index('location').T.to_dict('list')
    Cities = loc2.set_index('location').T.to_dict('list')
    location = [key for key, val in UK_cities.items() if key in location_string]
    if not location:
        location = [key for key, val in Cities.items() if key in location_string]
    return ','.join(location)


def convert_euro(value, currency):
    """
    Convert salary in $ or £ to €
    :param value: salary
    :param currency: only $, €, £ accepted
    :return: converted value in €
    """
    if currency == '$':
        euro = value * 0.9
    elif currency == '£':
        euro = value * 1.14
    elif currency == '€':
        euro = value
    else:
        euro = np.NAN
    return euro


def clean_jobtype(jobtype_string):
    """
    Extract jobtype from description
    :param jobtype_string:
    :return: currency
    """
    permanent = ['Permanent', 'unbefristet']
    temporary = ['Temporary', 'Placement', 'Seasonal']
    position = [jobtype_string.find(substring) for substring in permanent]
    if max(position) > -1:
        label = 'permanent'
    else:
        temp_position = [jobtype_string.find(substring) for substring in temporary]
        if max(temp_position) > -1:
            label = 'others'
        else:
            label = np.NaN
    return label


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Lower case of all words
    2. Remove all punctuation
    3. Returns cleaned text
    """

    # transforms all to lower case words
    mess = mess.lower()

    mess = mess.replace('\\n', ' ')
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    return nopunc


if __name__ == "__main__":
    website = 'indeed_de_all'
    # website = 'monster_all'
    # website = 'indeed_us_all'

    path = os.getcwd()
    parent_folder, current_folder = os.path.split(path)
    csv_path = '/data/' + website + '.csv'

    df = pd.read_csv(path + csv_path, sep='\t', low_memory=False)
    length0 = df.shape[0]
    print('Uploaded: ' + csv_path + '\n')
    print('Containing ' + str(length0) + ' entries... \n')
    print(df.columns)
    df.drop_duplicates(subset=['description', 'ref_code', 'url'], keep='last', inplace=True)

    df = df.reset_index(drop=True)
    length1 = df.shape[0]
    print('Removed ' + str(length0 - length1) + ' duplicate entries: ' + str(length1) + '...\n')

    df = df[df.astype(str)['description'] != '[]']
    length2 = df.shape[0]
    print('Removed ' + str(length1 - length2) + ' empty job descriptions: ' + str(length2) + '...\n')

    # back calculate the posted date of job advertisement
    df.duration.replace(regex=True, inplace=True, to_replace=r'Today', value=r'0')
    df.duration.replace(regex=True, inplace=True, to_replace=r'\D', value=r'')
    df.duration = pd.to_numeric(df['duration'])
    df.extraction_date = pd.to_datetime(df.extraction_date)
    df['posted_date'] = df['extraction_date'] - pd.to_timedelta(df['duration'], unit='D')
    df.drop(['duration'], axis=1, inplace=True)
    print('Back calculated the date of the job posting...\n')

    # Extracting salary from description, identifying currency, splitting between salary range (low, high)
    mask1 = ~df.salary.isnull()
    df['currency'] = df.loc[mask1, 'salary'].astype(str).apply(check_currency)
    print('Extracted currency of salary...\n')
    mask2 = df.currency.isnull()
    df.loc[mask2, 'salary'] = df.loc[mask2, 'description'].apply(find_salary)
    mask3 = ~df.salary.isnull()
    df['currency'] = df.loc[mask3, 'salary'].apply(check_currency)
    print('Extracted salary from job description (if not available)...\n')

    # conversion german . to , only for indeed_de
    if 'indeed_de' in website:
        df['salary_extract'] = df['salary'].apply(extract_salary_after)
        df.loc[df.currency == '€', 'salary_extract'] = df.loc[df.currency == '€', 'salary_extract'].str \
            .replace('.000,00', ',000')
        df.loc[df.currency == '€', 'salary_extract'] = df.loc[df.currency == '€', 'salary_extract'].str. \
            replace('.000', ',000')
    else:
        df['salary_extract'] = df['salary'].apply(extract_salary)

    df['salary_low'] = df['salary_extract'].str.split('-', n=1, expand=True)[0]
    df['salary_high'] = df['salary_extract'].str.split('-', n=1, expand=True)[1]
    df = clean_salary(df, ['salary_high', 'salary_low'])
    df = df.drop(columns='salary_extract', axis=1)

    # fill low or high salary with high or low salary
    no_high_salary = df['salary_high'].isnull() & df['salary_low'].notnull()
    df.loc[no_high_salary, 'salary_high'] = df.loc[no_high_salary, 'salary_low']
    no_low_salary = df['salary_low'].isnull() & df['salary_high'].notnull()
    df.loc[no_low_salary, 'salary_low'] = df.loc[no_low_salary, 'salary_high']

    existing_salary_limits = df['salary_high'].notnull() & df['salary_low'].notnull()
    # calculate salary average
    df.loc[existing_salary_limits, 'salary_average'] = (df.loc[existing_salary_limits, 'salary_high'] +
                                                        df.loc[existing_salary_limits, 'salary_low']) / 2

    # convert all to Euro
    df['salary_low_euros'] = df.apply(lambda row: convert_euro(row.salary_low, row.currency), axis=1)
    df['salary_high_euros'] = df.apply(lambda row: convert_euro(row.salary_high, row.currency), axis=1)
    df['salary_average_euros'] = df.apply(lambda row: convert_euro(row.salary_average, row.currency), axis=1)

    # determining salary type: yearly, monthly, hourly, daily
    names = defaultdict(list)
    names['hourly'] = ['hour', 'Stunde']
    names['daily'] = ['day', 'Tag']
    names['monthly'] = ['month', 'Monat']
    names['yearly'] = ['year', 'annum', 'p.a', 'Jahr']
    df['salary'].fillna('', inplace=True)
    for key, values in names.items():
        for synonyms in values:
            df.loc[df['salary'].str.contains(synonyms), 'salary_type'] = key
    df['salary'].replace('', np.NaN, inplace=True)
    length3 = df.shape[0]
    # df = df.dropna(subset=['salary'])
    # print('Removed ' + str(length2 - length3) + ' NaN from Salary column: ' + str(length3) + '...\n')

    df.loc[df.jobtype.notnull(), 'jobtype'] = df.loc[df.jobtype.notnull(), 'jobtype'].apply(clean_jobtype)

    # get location from description if not available
    df.location.replace(to_replace='Nothing_found', value=np.NaN, inplace=True)
    mask4 = df.location.isnull()
    df.loc[mask4, 'location'] = df[mask4]['description'].apply(check_locations, args=(path,))
    print('Extracted locations from job description (if not available)...\n')

    # transforming scraped locations
    if 'indeed_us' in website:
        df['state'] = df.location.str.split(',', n=1, expand=True)[1]
        df['state'] = df['state'].str.strip()
        df['state'] = df['state'].str.split(' ', n=1, expand=True)[0]
        df['state'] = df['state'].str.upper()
        df.loc[df.location.str.contains('Wisconsin') == True, 'state'] = 'WI'
        df.loc[df.location.str.contains('Nevada') == True, 'state'] = 'NV'
        df.loc[df.location.str.contains('Ohio') == True, 'state'] = 'OH'
        df.loc[df.location.str.contains('Idaho') == True, 'state'] = 'ID'
        df.loc[df.location.str.contains('New York') == True, 'state'] = 'NY'
        df.loc[df.location.str.contains('Delaware') == True, 'state'] = 'DE'
        df.loc[df.location.str.contains('North Carolina') == True, 'state'] = 'NC'
        df.loc[df.location.str.contains('South Carolina') == True, 'state'] = 'SC'
        df.loc[df.location.str.contains('Texas') == True, 'state'] = 'TX'
        df.loc[df.location.str.contains('Indiana') == True, 'state'] = 'IN'
        df.loc[df.location.str.contains('New Jersey') == True, 'state'] = 'NJ'
        df.loc[df.location.str.contains('Washington') == True, 'state'] = 'WA'
        df.loc[df.location.str.contains('California') == True, 'state'] = 'CA'
        df.loc[df.location.str.contains('Illinois') == True, 'state'] = 'IL'
        df.loc[df.location.str.contains('Hawaii') == True, 'state'] = 'HI'
        df.loc[df.location.str.contains('Georgia') == True, 'state'] = 'GA'
        df.loc[df.location.str.contains('Minnesota') == True, 'state'] = 'MN'
        df.loc[df.location.str.contains('Massachusetts') == True, 'state'] = 'MA'
        df.loc[df.location.str.contains('Florida') == True, 'state'] = 'FL'
        df.loc[df.location.str.contains('Arizona') == True, 'state'] = 'AZ'
        df.loc[df.location.str.contains('Virginia') == True, 'state'] = 'VA'
        locations_us = pd.read_csv(path + '/data/us-states.csv')
        df2 = pd.merge(df, locations_us, on='state', how='left')
        df2.loc[df2.location.str.contains('Aguadilla') == True, 'region'] = 'Offshore'
        df2.loc[df2.location.str.contains('Yigo') == True, 'region'] = 'Offshore'
        df2.loc[df2.location.str.contains('Guaynabo Municipio') == True, 'region'] = 'Offshore'
        df2.loc[df2.location.str.contains('Pago Pago') == True, 'region'] = 'Offshore'
        df2.loc[df2.region.str.contains('Offshore') == True, 'country'] = 'USA'
        df2.loc[df2.location.str.contains('Remote') == True, 'country'] = 'USA'
        df2.loc[df2.location.str.contains('United States') == True, 'country'] = 'USA'
        df2.location = df2.location.str.split(',', n=1, expand=True)[0]
        df2.location = df2.location.str.title()
        df2.location = df2.location.str.strip()
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
        df.loc[df.location.str.contains('Birmingham') == True, 'location'] = 'Birmingham'
        location_UK = pd.read_csv(path + '/data/locations_UK.csv')
        location_EU = pd.read_csv(path + '/data/locations.csv')
        location = location_UK.append(location_EU, ignore_index=True)
        df2 = pd.merge(df, location, on='location', how='left')
        df2.loc[df2.location.str.contains('Sweden') == True, 'country'] = 'Sweden'
        df2 = df2.reset_index(drop=True)
    df2.dropna(subset=['country', 'region'], inplace=True)
    print('Cleaned location data...\n')

    unknown_locations = df2.country.isnull() & df2.location.notnull()
    print(df2.loc[unknown_locations]['location'].value_counts())
    print('Identified region and country for each location...\n')

    output = '/data/cleaned_' + website + '.csv'

    cols = ['job_title', 'company', 'description', 'salary', 'salary_low', 'salary_high',
            'salary_average', 'salary_low_euros', 'salary_high_euros', 'salary_average_euros',
            'location', 'jobtype', 'posted_date', 'extraction_date', 'region', 'country', 'ref_code', 'url',
            'currency', 'salary_type']
    df3 = df2[cols]

    length4 = df3.shape[0]
    print('Save ' + str(length4) + ' results in ' + output + '...\n')
    df3.to_csv(path + output, index=False)
    print('Done!')
