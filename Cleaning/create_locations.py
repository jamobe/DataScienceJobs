import pandas as pd
import numpy as np
from collections import defaultdict
import os.path
    
path = os.getcwd()
    
data = defaultdict(list)
data['location'] = ['Frankfurt','Berlin','Bern','Zurich','Amsterdam','Wien','Munich','Barcelona','Paris','Aurich','Den Haag','Warsaw']
data['region'] = ['Hessen','Berlin','Bern','Zurich','North Holland','Wien','Bayern','Barcelona','Ile-de-France','Niedersachsen','South Holland','Masovia']
data['country'] = ['Germany','Germany','Switzerland','Switzerland','Netherlands','Austria','Germany','Spain','France','Germany','Netherlands','Poland']
loc2 = pd.DataFrame(data)
    
loc2 = loc2.append({'location': 'Dusseldorf' , 'region' : 'Nordrhein-Westfalen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Düsseldorf' , 'region' : 'Nordrhein-Westfalen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Bonn' , 'region' : 'Nordrhein-Westfalen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Stuttgart' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Vilnius' , 'region' : 'Vilnius', 'country':'Lithuania'} , ignore_index=True)
loc2 = loc2.append({'location': 'Dublin' , 'region' : 'Leinster', 'country':'Ireland'} , ignore_index=True)
loc2 = loc2.append({'location': 'Wynyard' , 'region' : 'Tasmania', 'country':'Australia'} , ignore_index=True)
loc2 = loc2.append({'location': 'Eindhoven' , 'region' : 'North Brabant', 'country':'Netherlands'} , ignore_index=True)
loc2 = loc2.append({'location': 'Melbourne' , 'region' : 'Victoria', 'country':'Australia'} , ignore_index=True)
loc2 = loc2.append({'location': 'Haeren' , 'region' : 'Brussels', 'country':'Belgium'} , ignore_index=True)
loc2 = loc2.append({'location': 'Hamburg' , 'region' : 'Hamburg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'München' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Köln' , 'region' : 'Nordrhein-Westfalen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Potsdam' , 'region' : 'Brandenburg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Leipzig' , 'region' : 'Sachsen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Dresden' , 'region' : 'Sachsen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Karlsruhe' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Hannover' , 'region' : 'Niedersachsen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Nürnberg' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Freiburg' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Essen' , 'region' : 'Nordrhein-Westfalen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Mannheim' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Ludwigshafen Am Rhein' , 'region' : 'Rheinland-Pfalz', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Bremen' , 'region' : 'Bremen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Friedrichshafen' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Erlangen' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Wolfsburg' , 'region' : 'Niedersachsen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Aachen' , 'region' : 'Nordrhein-Westfalen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Teterow' , 'region' : 'Mecklenburg-Vorpommern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Penzberg' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Dortmund' , 'region' : 'Nordrhein-Westfalen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Bad Homburg Vor Der Höhe' , 'region' : 'Hessen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Wiesbaden' , 'region' : 'Hessen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Rostock' , 'region' : 'Mecklenburg-Vorpommern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Teltow' , 'region' : 'Brandenburg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Biberach An Der Riß' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Ingolstadt' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Saarbrücken' , 'region' : 'Saarland', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Böblingen' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Münster' , 'region' : 'Nordrhein-Westfalen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Raunheim' , 'region' : 'Hessen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Schönefeld' , 'region' : 'Brandenburg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Ismaning' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Münchberg' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Magdeburg' , 'region' : 'Sachsen-Anhalt', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Immenstadt Im Allgäu' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Eschborn' , 'region' : 'Hessen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Landshut' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Einbeck' , 'region' : 'Niedersachsen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Herzogenaurach' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Stade' , 'region' : 'Niedersachsen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Heidelberg' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Lörrach' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Pforzheim' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Ludwigsburg' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Ulm' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Konstanz' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Baden-Württemberg' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Bühl' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Inning Am Ammersee' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Wangen Im Allgäu' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Heidenheim An Der Brenz' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Trostberg An Der Alz' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Kaiserslautern' , 'region' : 'Rheinland-Pfalz', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Wörth Am Rhein' , 'region' : 'Rheinland-Pfalz', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Waldbronn' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Limburg An Der Lahn' , 'region' : 'Hessen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Göttingen' , 'region' : 'Niedersachsen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Bochum' , 'region' : 'Nordrhein-Westfalen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Wuppertal' , 'region' : 'Nordrhein-Westfalen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Paderborn' , 'region' : 'Nordrhein-Westfalen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Meerbusch' , 'region' : 'Nordrhein-Westfalen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Leverkusen' , 'region' : 'Nordrhein-Westfalen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Nordrhein-Westfalen' , 'region' : 'Nordrhein-Westfalen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Hürth' , 'region' : 'Nordrhein-Westfalen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Maulburg' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Türkheim' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Bayreuth' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Aschheim' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Badenweiler' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Abstatt' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Calbe' , 'region' : 'Sachsen-Anhalt', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Coswig' , 'region' : 'Sachsen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Halle (Saale)' , 'region' : 'Sachsen-Anhalt', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Jena' , 'region' : 'Thüringen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Salzgitter' , 'region' : 'Niedersachsen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Bielefeld' , 'region' : 'Nordrhein-Westfalen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Brandenburg' , 'region' : 'Brandenburg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Schleswig-Holstein' , 'region' : 'Schleswig-Holstein', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Kiel' , 'region' : 'Schleswig-Holstein', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Deutschland' , 'region' : '', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Hennigsdorf' , 'region' : 'Brandenburg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Gernsbach' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Ingelheim Am Rhein' , 'region' : 'Rheinland-Pfalz', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Oldenburg' , 'region' : 'Niedersachsen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Gießen' , 'region' : 'Hessen', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Meuspath' , 'region' : 'Rheinland-Pfalz', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Scheßlitz' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Tübingen' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Walldorf' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Stolberg' , 'region' : np.NaN, 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Reutlingen' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Leonberg' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Oberkochen' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Bayern' , 'region' : 'Bayern', 'country':'Germany'} , ignore_index=True)
loc2 = loc2.append({'location': 'Kusterdingen' , 'region' : 'Baden-Württemberg', 'country':'Germany'} , ignore_index=True)

loc2.to_csv(path+'/data/locations.csv',index=False)