import pandas as pd
import numpy as np
import re


#Die beiden Datensätze Laden
Immoscout24Base = pd.read_excel(r'Duplikate entfernt.xlsx', sheet_name = 'Immoscout')
ImmonetBase = pd.read_excel(r'Duplikate entfernt.xlsx', sheet_name = 'Immonet')

#Spalten aus dem Immoweltdatensatz entfernen
Immoscout24Base = Immoscout24Base.drop(columns=['web-scraper-order', 'Wohnung', 'ScoutID', 'Monatsmiete', 'Hausgeld', 'Maklerprovision', 'Grunderwerbssteuer', 'Notarkosten',
                                                'Grundbucheintrag', 'Strasse', 'Ausstattung', 'Energieausweis', 'Energie­ausweistyp', 'Modernisierung/ Sanierung', 'Bezugsfrei ab',
                                                'Einbauküche', 'Nutzfläche ca'])
#Spaltennamen an Immonetdatensatz anpassen
Immoscout24Base = Immoscout24Base.rename(columns={'Wohnung-href': 'immo_url', 'Einkaufspreis': 'angebotspreis',
                                                  'Baujahr': 'baujahr', 'PLZ': 'plz', 'Ort': 'ort', 'Gäste-WC ja/nein': 'gaeste_wc',
                                                  'Keller ja/nein': 'unterkellert', 'Vermietet ja/nein': 'vermietet',
                                                  'Typ': 'immobilienart', 'Etage': 'geschoss', 'Anzahl Schlafzimmer': 'anzahl_schlafzimmer',
                                                  'Anzahl Badezimmer': 'anzahl_badezimmer', 'Objektzustand': 'immobilienzustand',
                                                  'Heizungsart': 'heizung', 'Wesentliche Energieträger': 'befeuerungsart', 'End­energie­verbrauch': 'energie_verbrauch',
                                                  'Energie­effizienz­klasse': 'energie_effizienzklasse', 'Balkon/ Terrasse': 'balkon',
                                                  'Garage/ Stellplatz':'anzahl_parkplatz','Denkmalschutzobjekt': 'denkmalschutz',
                                                  'Aufzug': 'aufzug', 'Stufenloser Zugang': 'barrierefrei', 'Grundstück': 'grundstuecksflaeche',
                                                  'Wohnfläche': 'wohnflaeche', 'Zimmer': 'anzahl_zimmer'})

#Spalten säubern
Immoscout24Base["unterkellert"] = Immoscout24Base["unterkellert"].apply(lambda row: "JA" if row == "keller" else "NEIN")
Immoscout24Base["gaeste_wc"] = Immoscout24Base["gaeste_wc"].apply(lambda row: "JA" if row == "Gäste-WC" else "NEIN")
Immoscout24Base["barrierefrei"] = Immoscout24Base["barrierefrei"].apply(lambda row: "JA" if row == 'Stufenloser Zugang' else "NEIN")

Immoscout24Base["baujahr"] = pd.to_numeric(Immoscout24Base["baujahr"], errors='coerce')
Immoscout24Base["grundstuecksflaeche"] = pd.to_numeric(Immoscout24Base["grundstuecksflaeche"], errors="ignore")

Immoscout24Base["wohnflaeche"] = pd.to_numeric(Immoscout24Base['wohnflaeche'], errors='ignore')
Immoscout24Base["terrasse_balkon"] = Immoscout24Base["balkon"].astype(str).apply(lambda row: "JA" if "Balkon" in row else "NEIN")


Immoscout24Base["vermietet"] = Immoscout24Base["vermietet"].astype(str).apply(lambda row: "JA" if row == "Vermietet" else "NEIN")
Immoscout24Base['aufzug'] = Immoscout24Base['aufzug'].astype(str)
Immoscout24Base["aufzug"] = Immoscout24Base["aufzug"].apply(lambda row: "JA" if row == "Personenaufzug" else "NEIN")

Immoscout24Base["anzahl_parkplatz"] = Immoscout24Base["anzahl_parkplatz"].fillna(0)
Immoscout24Base["anzahl_parkplatz"] = Immoscout24Base["anzahl_parkplatz"].apply(lambda row: re.sub('[\\D]', '', str(row)))
Immoscout24Base["anzahl_parkplatz"] = Immoscout24Base["anzahl_parkplatz"].apply(lambda row: "1" if row == "" else str(row))
Immoscout24Base["anzahl_parkplatz"] = pd.to_numeric(Immoscout24Base["anzahl_parkplatz"])

Immoscout24Base["energie_verbrauch"] = Immoscout24Base["energie_verbrauch"].apply(lambda row: re.sub('[^0-9,]', '', str(row)))
Immoscout24Base["energie_verbrauch"] = Immoscout24Base["energie_verbrauch"].apply(lambda row: re.sub(',', '..', str(row)))
Immoscout24Base["energie_verbrauch"] = pd.to_numeric(Immoscout24Base["energie_verbrauch"])

Immoscout24Base['etagen'] = Immoscout24Base['geschoss'].apply(lambda row: re.sub('\d*\svon ', '', str(row)))
Immoscout24Base['geschoss'] = Immoscout24Base['geschoss'].apply(lambda row: re.sub('von\s\d*', '', str(row)))

ImmonetBase['terrasse_balkon'] = ImmonetBase['terrasse'] + '' + ImmonetBase['balkon']
ImmonetBase['terrasse_balkon'] = ImmonetBase['terrasse_balkon'].apply(lambda row: 'JA' if 'JA' in row else 'NEIN')
ImmonetBase = ImmonetBase.drop(columns=['terrasse', 'balkon'])

Immoscout24Base = Immoscout24Base.reindex(sorted(Immoscout24Base.columns), axis=1)
ImmonetBase = ImmonetBase.reindex(sorted(ImmonetBase.columns), axis=1)


#Zusammenführen
ImmobilienAll = pd.concat([Immoscout24Base, ImmonetBase], axis=0, ignore_index=True, join="inner")
ImmobilienAll.loc[ImmobilienAll["angebotspreis"] <= 10000, "angebotspreis"] = ImmobilienAll["angebotspreis"] * 1000

#Unwichtige Spalten dropen
ImmobilienAll = ImmobilienAll.drop(columns=['anzahl_schlafzimmer', 'immo_url', 'energie_verbrauch', 'etagen', 'geschoss', 'ort', 'denkmalschutz'])


#Spaltennamen anpassen
ImmobilienAll.rename(columns={"befeuerungsart": "energietyp"}, inplace=True)

#Geodaten und Einwohnerzahlen mit hinzufügen
Geodaten = pd.read_excel('PLZ_Geodaten.xlsx')
list_plz = ImmobilienAll['plz']

dict_breitengrad = dict(zip(Geodaten['PLZ'], Geodaten['Breitengrad']))
list_breitengrad = [dict_breitengrad.get(key) for key in list_plz]

dict_längengrad = dict(zip(Geodaten['PLZ'], Geodaten['Längengrad']))
list_längengrad = [dict_längengrad.get(key) for key in list_plz]


Einwohner = pd.read_excel('PLZ_Einwohnerzahlen.xlsx')
dict_einwohner = dict(zip(Einwohner['plz'], Einwohner['einwohner']))
list_einwohner = [dict_einwohner.get(key) for key in list_plz]
list_einwohner


ImmobilienAll['breitengrad'] = list_breitengrad
ImmobilienAll['laengengrad'] = list_längengrad
ImmobilienAll['einwohner'] = list_einwohner


ImmobilienAll = ImmobilienAll.dropna(subset=['angebotspreis'])
ImmobilienAll = ImmobilienAll.dropna(subset=['breitengrad'])
ImmobilienAll = ImmobilienAll.dropna(subset=['laengengrad'])
ImmobilienAll = ImmobilienAll.dropna(subset=['einwohner'])



#Anpassungen
ImmobilienAll["terrasse_balkon"] = ImmobilienAll["terrasse_balkon"].astype("category")
ImmobilienAll["barrierefrei"] = ImmobilienAll["barrierefrei"].astype("category")
ImmobilienAll["energietyp"] = ImmobilienAll["energietyp"].astype("category")
ImmobilienAll["energie_effizienzklasse"] = ImmobilienAll["energie_effizienzklasse"].astype("category")
ImmobilienAll["gaeste_wc"] = ImmobilienAll["gaeste_wc"].astype("category")
ImmobilienAll["heizung"] = ImmobilienAll["heizung"].astype("category")
ImmobilienAll["immobilienart"] = ImmobilienAll["immobilienart"].astype("category")
ImmobilienAll["immobilienzustand"] = ImmobilienAll["immobilienzustand"].astype("category")
ImmobilienAll["plz"] = ImmobilienAll["plz"].astype("category")
ImmobilienAll["unterkellert"] = ImmobilienAll["unterkellert"].astype("category")
ImmobilienAll["vermietet"] = ImmobilienAll["vermietet"].astype("category")
ImmobilienAll["aufzug"] = ImmobilienAll["aufzug"].astype("category")


ImmobilienAll["energietyp"] = ImmobilienAll["energietyp"].apply(lambda row: str(row).split(",")[0])
ImmobilienAll["energietyp"] = ImmobilienAll["energietyp"].apply(lambda row: 'Pellets' if row == "Holzpellets" else row)
ImmobilienAll["energietyp"] = ImmobilienAll["energietyp"].apply(lambda row: 'Gas' if row == "Flüssiggas" else row)
ImmobilienAll["energietyp"] = ImmobilienAll["energietyp"].apply(lambda row: 'Fernwärme' if row == "Erdwärme" else row)
ImmobilienAll["energietyp"] = ImmobilienAll["energietyp"].apply(lambda row: 'Sonstige' if row not in ["", np.nan, "Öl", "Gas", "Fernwärme","Luft-/Wasserwärme", "Holz", "Pellets", "Solar", "Strom"] else row)
ImmobilienAll["heizung"] = ImmobilienAll["heizung"].apply(lambda row: 'Sonstige' if row not in ["", np.nan, "Zentralheizung", "Etagenheizung", "Ofenheizung","Fußbodenheizung"] else row)
ImmobilienAll["immobilienart"] = ImmobilienAll["immobilienart"].apply(lambda row: 'Einfamilienhaus' if row == "Einfamilienhaus (freistehend)" else row)

ImmobilienAll["immobilienart"] = ImmobilienAll["immobilienart"].apply(lambda row: 'Sonstige' if row not in ["", np.nan, "Einfamilienhaus", "Wohngrundstück", "Wohnung", "Etagenwohnung",
                                            "Sonstiges", "Mehrfamilienhaus", "Erdgeschosswohnung",
                                            "Dachgeschosswohnung",
                                            "Zweifamilienhaus", "Doppelhaushälfte", "Villa", "Reihenmittelhaus",
                                            "Reihenendhaus", "Bungalow",
                                            "Maisonette", "Apartment", "Stadthaus", "Schloss", "Bauernhaus",
                                            "Herrenhaus", "Reiheneckhaus", "Penthouse"] else row)

ImmobilienAll["immobilienzustand"] = ImmobilienAll["immobilienzustand"].apply(lambda row: 'Teil- oder vollrenovierungsbedürftig' if row == "Renovierungsbedürftig" else row)
ImmobilienAll["immobilienzustand"] = ImmobilienAll["immobilienzustand"].apply(lambda row: 'Teil- oder vollsaniert' if row == "Saniert" else row)
ImmobilienAll["immobilienzustand"] = ImmobilienAll["immobilienzustand"].apply(lambda row: 'Teil- oder vollrenoviert' if row == "Renovierungsbedürftig" else row)
ImmobilienAll["immobilienzustand"] = ImmobilienAll["immobilienzustand"].apply(lambda row: 'Sonstige' if row not in ["", np.nan, "Unbekannt", "Erstbezug", "Projektiert", "Neubau",
                                          "Teil- oder vollrenovierungsbedürftig", "Neuwertig",
                                          "Teil- oder vollsaniert", "Teil- oder vollrenoviert", "Gepflegt",
                                          "Altbau", "Modernisiert"] else row)


ImmobilienAll.loc[ImmobilienAll["anzahl_badezimmer"] == 0, "anzahl_badezimmer"] = np.nan
ImmobilienAll["anzahl_badezimmer"] = ImmobilienAll["anzahl_badezimmer"].apply(lambda x: np.random.choice(range(1, 4), p=[0.65, 0.30, 0.05]) if np.isnan(x) else x)
ImmobilienAll["anzahl_zimmer"] = ImmobilienAll["anzahl_zimmer"].apply(lambda x: np.random.choice(ImmobilienAll["anzahl_zimmer"].dropna().values) if np.isnan(x) else x)
ImmobilienAll["baujahr"] = ImmobilienAll["baujahr"].apply(lambda x: np.random.choice(ImmobilienAll["baujahr"].dropna().values) if np.isnan(x) else x)

ImmobilienAll["energietyp"] = ImmobilienAll["energietyp"].astype("category")
ImmobilienAll["energietyp"] = ImmobilienAll["energietyp"].cat.add_categories(["Unbekannt"])
ImmobilienAll["energietyp"] = ImmobilienAll["energietyp"].fillna("Unbekannt")

ImmobilienAll["energie_effizienzklasse"] = ImmobilienAll["energie_effizienzklasse"].cat.add_categories(["Unbekannt"])
ImmobilienAll["energie_effizienzklasse"] = ImmobilienAll["energie_effizienzklasse"].fillna("Unbekannt")

ImmobilienAll["heizung"] = ImmobilienAll["heizung"].astype("category")
ImmobilienAll["heizung"] = ImmobilienAll["heizung"].cat.add_categories(["Unbekannt"])
ImmobilienAll["heizung"] = ImmobilienAll["heizung"].fillna("Unbekannt")

ImmobilienAll["immobilienart"] = ImmobilienAll["immobilienart"].astype("category")
ImmobilienAll["immobilienart"] = ImmobilienAll["immobilienart"].cat.add_categories(["Unbekannt"])
ImmobilienAll["immobilienart"] = ImmobilienAll["immobilienart"].fillna("Unbekannt")

ImmobilienAll["immobilienzustand"] = ImmobilienAll["immobilienzustand"].astype("category")
ImmobilienAll["immobilienzustand"] = ImmobilienAll["immobilienzustand"].cat.add_categories(["Unbekannt"])
ImmobilienAll["immobilienzustand"] = ImmobilienAll["immobilienzustand"].fillna("Unbekannt")


# Aufzug: Annahme, wenn nicht explizit angegeben, dann existiert kein Aufzug
ImmobilienAll.loc[ImmobilienAll["aufzug"].isna(), "aufzug"] = "NEIN"

#scientific notation rückgängig machen
ImmobilienAll['breitengrad'] = ImmobilienAll['breitengrad'].apply(lambda x: '%.0f' % x)
ImmobilienAll['laengengrad'] = ImmobilienAll['laengengrad'].apply(lambda x: '%.0f' % x)

#punkte an richtiger stelle einfügen
ImmobilienAll['breitengrad'].astype(str)
ImmobilienAll['breitengrad'] = ImmobilienAll['breitengrad'].str[:2] + '.' + ImmobilienAll['breitengrad'].str[2:]

ImmobilienAll['laengengrad'].astype(str)
if ImmobilienAll['laengengrad'][0][0] == '1':
  ImmobilienAll['laengengrad'] = ImmobilienAll['laengengrad'].str[:2] + '.' + ImmobilienAll['laengengrad'].str[2:]
else:
  ImmobilienAll['laengengrad'] = ImmobilienAll['laengengrad'].str[:1] + '.' + ImmobilienAll['laengengrad'].str[1:]



# als csv speichern
ImmobilienAll.to_csv('Zusammengefügt_mit_Koordinaten.csv')

