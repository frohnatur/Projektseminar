import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import xgboost as xgb
import sqlite3

from sqlite3 import Error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


immonet_data = pd.read_excel(r"Files/Input_Data/Immonet_Bayern.xlsx", sheet_name="Tabelle2")

#return immonet_data
immoscout_data_haeuser = pd.read_excel(r"Files/Input_Data/Immoscout_Bayern_Häuser.xlsx", sheet_name="Tabelle3")
immoscout_data_wohnungen = pd.read_excel(r"Files/Input_Data/Immoscout_Bayern_Wohnungen.xlsx", sheet_name="Tabelle2")

immoscout_data = pd.concat([immoscout_data_haeuser, immoscout_data_wohnungen], axis=0, ignore_index=True)

#return immoscout_data

geo_data = pd.read_excel(r'Files/Input_Data/PLZ_Geodaten.xlsx', sheet_name='PLZ')
#return geo_data

inhabitants = pd.read_excel(r'Files/Input_Data/PLZ_Einwohnerzahlen.xlsx', sheet_name='Tabelle2')
#return inhabitants


immonet_data['plz'] = immonet_data['plz'].astype(str)
geo_data = geo_data.astype(str)
list_plz_immonet = immonet_data['plz']

dict_breitengrad_immonet = dict(zip(geo_data['PLZ'], geo_data['Breitengrad']))
list_breitengrad_immonet = [dict_breitengrad_immonet.get(key) for key in list_plz_immonet]

dict_laengengrad_immonet = dict(zip(geo_data['PLZ'], geo_data['Längengrad']))
list_laengengrad_immonet = [dict_laengengrad_immonet.get(key) for key in list_plz_immonet]

inhabitants = inhabitants.astype(str)
dict_einwohner_immonet = dict(zip(inhabitants['plz'], inhabitants['einwohner']))
list_einwohner_immonet = [dict_einwohner_immonet.get(key) for key in list_plz_immonet]

immonet_data['breitengrad'] = list_breitengrad_immonet
immonet_data['laengengrad'] = list_laengengrad_immonet
immonet_data['einwohner'] = list_einwohner_immonet

immonet_data = immonet_data.dropna(subset=['breitengrad'])
immonet_data = immonet_data.dropna(subset=['laengengrad'])
immonet_data = immonet_data.dropna(subset=['einwohner'])


immonet_data_new = immonet_data

#return immonet_data_new

geo_data = geo_data.astype(str)
immoscout_data["plz"] = immoscout_data["PLZ und Ort"].astype(str).apply(lambda row: row[:5])
list_plz_immoscout = immoscout_data['plz']

dict_breitengrad_immoscout = dict(zip(geo_data['PLZ'], geo_data['Breitengrad']))
list_breitengrad_immoscout = [dict_breitengrad_immoscout.get(key) for key in list_plz_immoscout]

dict_laengengrad_immoscout = dict(zip(geo_data['PLZ'], geo_data['Längengrad']))
list_laengengrad_immoscout = [dict_laengengrad_immoscout.get(key) for key in list_plz_immoscout]

inhabitants = inhabitants.astype(str)
dict_einwohner_immoscout = dict(zip(inhabitants['plz'], inhabitants['einwohner']))
list_einwohner_immoscout = [dict_einwohner_immoscout.get(key) for key in list_plz_immoscout]

immoscout_data['breitengrad'] = list_breitengrad_immoscout
immoscout_data['laengengrad'] = list_laengengrad_immoscout
immoscout_data['einwohner'] = list_einwohner_immoscout

immoscout_data = immoscout_data.dropna(subset=['breitengrad'])
immoscout_data = immoscout_data.dropna(subset=['laengengrad'])
immoscout_data = immoscout_data.dropna(subset=['einwohner'])

immoscout_data_new = immoscout_data

#return immoscout_data_new

immoscout_data_new.columns = immoscout_data_new.columns.str.lower()

immoscout_data_new = immoscout_data_new.drop(columns=["plz und ort", "web-scraper-order"])

immoscout_data_new.rename(
    columns={"anzahl badezimmer": "anzahl_badezimmer", "anzahl schlafzimmer": "anzahl_schlafzimmer",
             "zimmer": "anzahl_zimmer", "einkaufspreis": "angebotspreis",
             "balkon/ terrasse": "terrasse_balkon", "wohnfläche": "wohnflaeche", "etage": "geschoss",
             "grundstück": "grundstuecksflaeche", "stufenloser zugang": "barrierefrei",
             "objektzustand": "immobilienzustand",
             "keller ja/nein": "unterkellert", "gäste-wc ja/nein": "gaeste_wc",
             "energie­effizienz­klasse": "energie_effizienzklasse",
             "wesentliche energieträger": "befeuerungsart", "end­energie­verbrauch": "energie_verbrauch",
             "typ": "immobilienart", "heizungsart": "heizung", "vermietet ja/nein": "vermietet",
             "garage/ stellplatz": "anzahl_parkplatz"}, inplace=True)

# Spalteninhalte anpassen:
# Annahme NaN ist NEIN

immonet_data_new['terrasse_balkon'] = immonet_data_new['terrasse'] + '' + immonet_data_new['balkon']
immonet_data_new['terrasse_balkon'] = immonet_data_new['terrasse_balkon'].apply(lambda row: 'JA' if 'JA' in row else 'NEIN')
immonet_data_new = immonet_data_new.drop(columns=['terrasse', 'balkon'])

immoscout_data_new["aufzug"] = immoscout_data_new["aufzug"].astype(str).apply(
    lambda row: "JA" if row == "Personenaufzug" else "NEIN")

immoscout_data_new["terrasse_balkon"] = immoscout_data_new["terrasse_balkon"].astype(str).apply(
    lambda row: "JA" if "Balkon" in row else "NEIN")

immoscout_data_new["unterkellert"] = immoscout_data_new["unterkellert"].apply(
    lambda row: "JA" if row == "keller" else "NEIN")
immoscout_data_new["gaeste_wc"] = immoscout_data_new["gaeste_wc"].apply(
    lambda row: "JA" if row == "Gäste-WC" else "NEIN")
immoscout_data_new["barrierefrei"] = immoscout_data_new["barrierefrei"].apply(
    lambda row: "JA" if row == 'Stufenloser Zugang' else "NEIN")

immoscout_data_new["baujahr"] = pd.to_numeric(immoscout_data_new["baujahr"], errors='coerce')
immoscout_data_new["grundstuecksflaeche"] = immoscout_data_new["grundstuecksflaeche"].astype(str).apply(
    lambda row: re.sub('[.m²]', '', row))
immoscout_data_new["grundstuecksflaeche"] = [x.replace('.', '') for x in immoscout_data_new["grundstuecksflaeche"]]
immoscout_data_new["grundstuecksflaeche"] = [x.replace(',', '.') for x in immoscout_data_new["grundstuecksflaeche"]]
immoscout_data_new["grundstuecksflaeche"] = immoscout_data_new['grundstuecksflaeche'].astype(float)
immoscout_data_new["wohnflaeche"] = immoscout_data_new["wohnflaeche"].astype(str).apply(
    lambda row: re.sub('[m²]', '', row))
immoscout_data_new["wohnflaeche"] = [x.replace('.', '') for x in immoscout_data_new["wohnflaeche"]]
immoscout_data_new["wohnflaeche"] = [x.replace(',', '.') for x in immoscout_data_new["wohnflaeche"]]

immoscout_data_new["wohnflaeche"] = immoscout_data_new["wohnflaeche"].astype(float)

immoscout_data_new["vermietet"] = immoscout_data_new["vermietet"].astype(str).apply(
    lambda row: "JA" if row == "Vermietet" else "NEIN")

immoscout_data_new["anzahl_parkplatz"] = immoscout_data_new["anzahl_parkplatz"].fillna(0)
immoscout_data_new["anzahl_parkplatz"] = immoscout_data_new["anzahl_parkplatz"].apply(
    lambda row: re.sub('[\\D]', '', str(row)))
immoscout_data_new["anzahl_parkplatz"] = pd.to_numeric(immoscout_data_new["anzahl_parkplatz"])
immoscout_data_new["anzahl_parkplatz"] = immoscout_data_new["anzahl_parkplatz"].fillna(1)

immoscout_data_new["energie_verbrauch"] = immoscout_data_new["energie_verbrauch"].apply(
    lambda row: re.sub('[^0-9,]', '', str(row)))
immoscout_data_new["energie_verbrauch"] = immoscout_data_new["energie_verbrauch"].apply(
    lambda row: re.sub(',', '.', str(row)))
immoscout_data_new["energie_verbrauch"] = pd.to_numeric(immoscout_data_new["energie_verbrauch"])

# Spalten alphabetisch sortieren
immonet_data_new = immonet_data_new.reindex(sorted(immonet_data_new.columns), axis=1)
immoscout_data_new = immoscout_data_new.reindex(sorted(immoscout_data_new.columns), axis=1)

# Innerjoin reicht hier aus
merged_data = pd.concat([immoscout_data_new, immonet_data_new], axis=0, ignore_index=True, join="inner")

# Duplikate
merged_data = merged_data.drop_duplicates(subset=['wohnflaeche', 'grundstuecksflaeche', 'anzahl_zimmer'])

print(merged_data['grundstuecksflaeche'])