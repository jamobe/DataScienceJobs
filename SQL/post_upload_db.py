import psycopg2
from sqlalchemy import create_engine
import pandas as pd

#connect to the database
PASSWORD = pd.read_pickle('~/DataScienceJobs/data/SQL_password.pkl')
engine = create_engine('postgresql://postgres:'+PASSWORD.iloc[0,0]+'@dsj-1.c9mo6xd9bf9d.us-west-2.rds.amazonaws.com:5432/')


#insert data from landing table if it exists

exists = pd.read_sql('''SELECT EXISTS (
   SELECT 1
   FROM   information_schema.tables 
   WHERE  table_schema = 'public'
   AND    table_name = 'landing'
   );
''', engine).iloc[0,0]

if exists == True:
    print("Landing table will be uploaded, are you sure you wish to continue?[yes/no]")
    myinput = input()
    if myinput == 'yes':
        engine.execute('''
        INSERT INTO all_data (job_title, ref_code, company,description, salary,salary_low,salary_high,currency,salary_average,salary_low_euros,salary_high_euros,salary_average_euros,salary_type,location,jobtype,posted_date,extraction_date,country,region,url)
        SELECT job_title, ref_code, company,description, salary,salary_low,salary_high,currency,salary_average,salary_low_euros,salary_high_euros,salary_average_euros,salary_type,location,jobtype,posted_date,extraction_date,country,region,url FROM landing
        ''')
    
    else:
        print("data not uploaded")
        pass

else:
    pass

# delete duplicates

engine.execute(" DELETE FROM all_data a USING all_data b WHERE a.id < b.id AND a.description = b.description;")


# Allocate train and test

# allocate train and test columns
engine.execute(''' UPDATE all_data 
                    SET train_test_label = CASE 
                                            WHEN random() > 0.2 THEN 'train'
                                            ELSE 'test'
                                            END 
                    WHERE train_test_label IS NULL''')


# produce summary stats to ensure right mix of train and test allocations

test = pd.read_sql('''SELECT count(train_test_label) from all_data WHERE train_test_label = 'test'
''', engine)
train = pd.read_sql('''SELECT count(train_test_label) from all_data WHERE train_test_label = 'train'
''', engine)

test_sal = pd.read_sql('''SELECT count(train_test_label) from all_data WHERE train_test_label = 'test' AND salary_average > 0
''', engine)
train_sal = pd.read_sql('''SELECT count(train_test_label) from all_data WHERE train_test_label = 'train' AND salary_average > 0
''', engine)

test_UK = pd.read_sql('''SELECT count(train_test_label) from all_data WHERE train_test_label = 'test' AND salary_average > 0 AND country = 'UK'
''', engine)
train_UK = pd.read_sql('''SELECT count(train_test_label) from all_data WHERE train_test_label = 'train' AND salary_average > 0 AND country = 'UK'
''', engine)

test_GER= pd.read_sql('''SELECT count(train_test_label) from all_data WHERE train_test_label = 'test' AND salary_average > 0 AND country = 'Germany'
''', engine)
train_GER = pd.read_sql('''SELECT count(train_test_label) from all_data WHERE train_test_label = 'train' AND salary_average > 0 AND country = 'Germany'
''', engine)

test_USA = pd.read_sql('''SELECT count(train_test_label) from all_data WHERE train_test_label = 'test' AND salary_average > 0 AND country = 'USA'
''', engine)
train_USA = pd.read_sql('''SELECT count(train_test_label) from all_data WHERE train_test_label = 'train' AND salary_average > 0 AND country = 'USA'
''', engine)


print("Samples in test set : {} ".format(test.iloc[0,0]))
print("Samples in train set : {} ".format(train.iloc[0,0]))
print("Proportion of test samples : {:.1%} ".format(test.iloc[0,0]/(train.iloc[0,0]+test.iloc[0,0])))
print("")
print("Samples in test set with salary: {} ".format(test_sal.iloc[0,0]))
print("Samples in train set with salary: {} ".format(train_sal.iloc[0,0]))
print("Proportion of test samples with salary: {:.1%} ".format(test_sal.iloc[0,0]/(train_sal.iloc[0,0]+test_sal.iloc[0,0])))
print("")
print("Samples in test set with salary UK: {} ".format(test_UK.iloc[0,0]))
print("Samples in train set with salary UK: {} ".format(train_UK.iloc[0,0]))
print("Proportion of test samples with salary UK: {:.1%} ".format(test_UK.iloc[0,0]/(train_UK.iloc[0,0]+test_UK.iloc[0,0])))
print("")
print("Samples in test set with salary Germany: {} ".format(test_GER.iloc[0,0]))
print("Samples in train set with salary  Germany: {} ".format(train_GER.iloc[0,0]))
print("Proportion of test samples with salary  Germany: {:.1%} ".format(test_GER.iloc[0,0]/(train_GER.iloc[0,0]+test_GER.iloc[0,0])))
print("")
print("Samples in test set with salary USA: {} ".format(test_USA.iloc[0,0]))
print("Samples in train set with salary  USA: {} ".format(train_USA.iloc[0,0]))
print("Proportion of test samples with salary  USA: {:.1%} ".format(test_USA.iloc[0,0]/(train_USA.iloc[0,0]+test_USA.iloc[0,0])))
