# DataScienceJobs
DSR Project (by Rachel and Janina): Analysing Data-related Job descriptions

## Workflow

#### 1. Web Scraping
We scraped job descriptions from the several websites (harnham (.co.uk, .de, .com), indeed (.co.uk, .de, .com), monster 
(.co.uk)). The python files to run the scrapings are in the folder 'Web_Scraping'. We scraped the job descriptions for 
different search terms (e.g. data scientist, machine learning, data, business intelligence, data engineer, data manager,
econometrics, statistics, data analyst).

#### The following columns were scraped:
* __job title:__ title of the job (e.g. Data Scientist, Machine Learning Expert, ...)
* __company:__ name of the company
* __job description:__ text describing the position
* __location:__ city, region or country, where job is located 
* __salary:__ salary range 
* __education:__ required education (e.g. Masters, Bachler, PhD, ...)
* __industry:__ industry where job is associated to 
* __career:__ career type of job (e.g. Experienced, Entry level, ....)
* __url:__ url of the job posting
* __extraction date:__ date when job was scraped
* __duration:__ for how long has the job been published
* __job type:__ type of job (e.g. permanent, temporary, ...)
* __ref code:__ unique code for job description (given by website)



