def db_upload(PASSWORD, df):
    
    from sqlalchemy import create_engine
    import pandas as pd
    
    engine = create_engine('postgresql://postgres:'+PASSWORD+'@dsj-1.c9mo6xd9bf9d.us-west-2.rds.amazonaws.com:5432/')
    exists = pd.read_sql('''SELECT EXISTS (
       SELECT 1
       FROM   information_schema.tables 
       WHERE  table_schema = 'public'
       AND    table_name = 'landing'
       );
    ''', engine).iloc[0,0]

    if exists == True:
        print("The landing table already exists, are you sure you wish to continue?[yes/no]")
        myinput = input()
        if myinput == 'yes':
            df.to_sql(name = 'landing',con = engine, if_exists='replace',  index=False)
        else:
            print("data not uploaded")
    else:
        df.to_sql(name = 'landing',con = engine, if_exists='replace',  index=False)