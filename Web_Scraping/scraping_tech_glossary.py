from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup
import re
import pickle
import os.path


if __name__ == "__main__":
    path = os.getcwd()
    # setup connection to website
    url = "https://glossarytech.com/"
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    webpage = urlopen(req).read()
    page_soup = soup(webpage, "html.parser")
    categories = page_soup.find_all('a', href=re.compile('^/terms/'))
    categories_list = [str(categories[x]).split('"')[1] for x in range(len(categories))]
    categories_list = [str(categories_list[x]).split('terms/')[1] for x in range(len(categories_list))]

    # On each page there are terms in a table with defintions which contain further tech terms in tags.
    # create dictionary to capture terms organised by the broad category heading under which they are listed
    tech_dict = {}

    # now loop through all category urls
    for i in categories_list:
        tech_dict[i] = ""
        for j in ["", "/page2", "/page3", "/page4", "/page5", "/page6", "/page7", "/page8"]:
            try:
                url = "https://glossarytech.com/terms/"+i+j
                req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                webpage = urlopen(req).read()
                page_soup = soup(webpage, "html.parser")

                tags = page_soup.find_all('a', href="/terms/go_to_term/")
                b = [x.text for x in tags]
                tech_dict[i]+str(b)
                cells = page_soup.find_all('tr', {'data-term': True})
                for element in cells:
                    a = [x.text for x in element.find_all('a', {'href': re.compile('^/terms/')})]
                    tech_dict[i] += str(a)

            except:
                pass

    # some of the catgories are really subcategories, so we want to aggregate further
    agg_dict = {
        'back_end-technologies': ['javascript', 'php', 'ruby', 'java', 'c-net', 'python', 'c_plus_plus', 'c', 'scala',
                                  'golang', 'functional-programming-fp', 'other-programming_languages'],
        'mobile': ['ios', 'android', 'cross_platform-hybrid'],
        'cyber_security': ['cyber_security/arcsight', 'cyber_security/nessus', 'cyber_security/iso']}

    # add news keys and value to the dictionary
    tech_dict['back_end-technologies'] = []
    for x in agg_dict['back_end-technologies']:
        tech_dict['back_end-technologies'].append(tech_dict[x])

    tech_dict['mobile'] = []
    for x in agg_dict['mobile']:
        tech_dict['mobile'].append(tech_dict[x])

    for x in agg_dict['cyber_security']:
        [tech_dict['cyber_security']].append(tech_dict[x])

    # now delete keys which are subcategories
    for i in agg_dict.values():
        for element in i:
            del tech_dict[element]

    # clean up values in dictionary, add spaces either side so we are not capturing common langiage occurences of
    # some words.
    for key in tech_dict:
        a = str(tech_dict[key]).replace(']', ",").replace('[', ",").replace("'", "").replace("\\", "").split(",")
        b = [x.lstrip().rstrip() for x in a]
        b = [' '+x+' ' for x in b]
        c = list(dict.fromkeys(b))
        tech_dict[key] = c
        try:
            tech_dict[key].remove('')
        except:
            pass
        try:
            tech_dict[key].remove('"')
        except:
            pass

    # pickle out tech dictionary
    pickle.dump(tech_dict, open(path + '/Pickles/broad_tech_dictionary.pkl', 'wb'))
