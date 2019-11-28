from sqlalchemy import create_engine
import pandas as pd
import pickle
import os.path

def db_upload(PASSWORD, df):
    '''
    This function uploads the data to the landing table in the database.
    There is a prompt to ensure you want to update data if there is data in the landing table already,
    because it will overwrite whatever is there already.
    params:
    PASSWORD: your db password
    df: dataframe you wish to upload
    '''

    engine = create_engine('postgresql://postgres:'+PASSWORD+'@dsj-1.c9mo6xd9bf9d.us-west-2.rds.amazonaws.com:5432/')
    exists = pd.read_sql('''SELECT EXISTS (
       SELECT 1
       FROM   information_schema.tables 
       WHERE  table_schema = 'public'
       AND    table_name = 'landing'
       );
    ''', engine).iloc[0,0]
    # has_rows = pd.read_sql('''SELECT count(*) landing''',engine).iloc[0,0]

    if exists is True: # and has_rows > 1:
        print('The landing table already exists, are you sure you wish to continue?[yes/no]')
        myinput = input()
        if myinput == 'yes':
            print('Uploading data to landing page...')
            df.to_sql(name='landing', con=engine, if_exists='replace',  index=False)
            print('Done!')
        else:
            print('Data not uploaded!')
    else:
        print('Landing page is currently empty.\n')
        print('Uploading data to landing page...')
        df.to_sql(name='landing', con=engine, if_exists='replace',  index=False)
        print('Done!')


if __name__ == "__main__":
    path = os.getcwd()

    with open(path + '/data/SQL_access.pkl', 'rb') as file:
        PASSWORD = pickle.load(file)

    print('Enter the name of your CSV File to upload to the landing page: /DataScienceJobs/data/???')
    filename = input()
    df = pd.read_csv(path + '/data/' + filename)
    print(str(df.shape) + ' job descriptions are in the CSV file.\n')
    db_upload(PASSWORD, df)
