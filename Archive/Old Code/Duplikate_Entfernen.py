import pandas as pd

#Einfügen der beiden Tabellen
tabelle_neu = pd.read_excel('Scrape_02.xlsx')
tabelle_alt = pd.read_excel('Scrape_01.xlsx')

#Zusammenfügen der Tabellen
tabelle_gesamt = tabelle_alt.append(tabelle_neu)

#Duplikate nur nach immo_url entfernen
x = tabelle_gesamt.drop_duplicates(subset = ['immo_url'])

#Duplikate nach anderen Parametern bei übrigen Variablen entfernen
y = x.drop_duplicates(subset = ['wohnflaeche', 'grundstuecksflaeche', 'anzahl_zimmer', 'energie_verbrauch'])

#Als CSV speichern
y.to_csv('CSV_duplictate_01.csv')
