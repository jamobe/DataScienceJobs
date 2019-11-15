# Rachel's questions / working notes


Questions:
 - Cycle through job names to allow wider search? Or specify regions to get more data?
 - Data type ->
 - Do we do stop words before inputting into the database? No
 - Do we drop entries with no salaries before putting them in the database? No
 - Do we have a train and test table in the SQL database? No -> check with Adam
 - should we add columns for currency conversions in the main table? yes
 

Thoughts on Summary stats:
 - Word counts
 - Stats on job titles
 - Regions/ countries
 - Technologies / skills
 
Thoughts on the analysis:

 - Do we treat technologies as named entities?
 
 Things to remember
 - check all file paths only refer to the repo
 - csv lookup files need to be changed in functions once we have one version
 - make sure missing values are coerced to np.nan
 - standardize day, year, month etc. -> yearly, monthly daily
 


Dictionary of words to be deleted


               "|": " ",
               "''": " ",
               "'": " ",
               ",": " ",
               "+": " ",
               "/": " ",
               "€": " ",
               "£": " ",
               "$": " ",
               "benefits":" ",
               "THE COMPANY": " ",
               "THE ROLE": " ",
               "THE BENEFITS": " ",
               "HOW TO APPLY": " ",
               "KEYWORDS": " ",
               "YOUR SKILLS AND EXPERIENCE": " ",
               "YOUR SKILLS AND EXPERTISE": " ",
               "Please register your interest by sending your CV via the Apply link on this page":" ",
               "BENEFITS":" ",
               "CONTACT":" ",
               "OVERVIEW":" ",
               "SALARY":" ",
               "For further details":" ",
               "to enquire about other roles please contact":" ",
               "Nick Mandella":" ",
               "Harnham":" ",
               "On a daily basis":" ",
               "you will be:":" ",
               "you will join:":" ",
               "!" :" ",
               "." : " ",
               "0" :" ",
              "1" : " ",
              "2" : " ",
              "3" :" ",
              "4" : " ",
              "5" : " ",
              "6" : " ",
              "7" :" ",
              "8" : " ",
              "9" : " "