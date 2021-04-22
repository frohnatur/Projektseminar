import sqlite3

import streamlit as st
import pandas as pd
import numpy as np
import pickle
#import main
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Änderungen an csv:
# Datensatz von Lennart (Excel-Datei) kopieren, in Excel und Spalten zusammen (csv Format) dann folgende Änderungen:
# Alle ohne Grundstücksfläche und Wohnfläche raus
# Spalten raus: Längengrad, Breitengrad, PLZ

db_connection = sqlite3.connect('Projektseminar/Datenbank/ImmoDB.db')
# allgemeine Streamlit Einstellungen (Tab Name; Icon; Seitenlayout; Menü)
st.set_page_config('AWI', 'Projektseminar/Files/GUI/Logo AWI klein.jpg', 'centered', 'expanded')


#Logo einfügen
st.image('Projektseminar/Files/GUI/Logo AWI.jpg')
st.image('Projektseminar/Files/GUI/AbstandshalterAWI.jpg')


#Infotext
st.subheader('Starte jetzt deine Immobilienbewertung mit AWI')
st.write('Du möchtest den Wert deiner Immobilie exakt berechnen und benötigst weiterführende Analysemöglichkeiten rund um deine Immobilie? Dann bist du bei AWI genau richtig!')


#Button und Infotext
if st.button('mehr Informationen'):
    st.write('Was ist AWI?')
    st.write('AWI ist ein Analysetool für die Immobilienwertermittlung und bietet dir umfangreiche Analysemöglichkeiten rund um die Bewertung deiner Immobilie.')
    st.write('---')
    st.write('Wofür steht AWI?')
    st.write('AWI ist ein Akronym für: Analysetool für die Wertermittlung von Immobilien')
    st.write('---')
    st.write('Was ist an AWI besonders?')
    st.write('Immobilienbewertungstools gibt es wie Sand am Meer :palm_tree: Doch AWI ist keine gewohnliche App zur reinen Bewertung von Immobilien. Um eine fundierte Entscheidungen in Hinblick auf den Kauf bzw. Verkauf einer Immobilie zu treffen, sind tiefgreifende Informationen zur eigenen Immobilie notwendig. Im Gegensatz zu herkömmlichen Tools, bietet AWI diese Informationen. Neben einer aussagekräftigen Immobilienbewertung, die AWI mithilfe eines intelligenten Machine Learning Ansatzes ermittelt, bietet AWI weiterführende Analysen, um eine möglichst umfangreiche Immobilienbetrachtung zu ermöglichen. AWI bietet beispielsweise eine exakte Beschreibung welchen Einfluss die von dir hinterlegten Faktoren auf den Immobilienpreis nehmen. So kannst du zum Beispiel zukünftige Investitionen in deine Immobilie besser planen.')        
st.write('---')


#Überschrift und Abstandshalter
st.image('Projektseminar/Files/GUI/AbstandshalterAWI.jpg')
st.subheader('Beschreibe deine Immobilie:')


#Definition der UI Eingabemaske (Features)
def user_input_features():
    
        #Eingabefeld 1 (Wohnfläche, Grundstücksfläche, Baujahr, PLZ)
        wohnflaeche = st.slider('Wohnfläche', 0, 1000, 145)
        grundstuecksflaeche = st.slider('Grundstücksfläche', 0, 1000, 430)
        
        col1, col2 = st.beta_columns(2)
        with col1:
            baujahr = st.number_input('Baujahr', max_value=2023, value=2007)
        with col2:
            plz = st.number_input('Wie lautet deine PLZ?', min_value=63739, max_value=97909, value=97070)
        
        #Eingabefeld 2 in Expander mit zwei Spalten (Anzahl Zimmer, Anzahl Parkplatz, Immoart, Anzahl Badezimmer,...)
        weitereDetails = st.beta_expander('weitere Details')
        with weitereDetails:
            col1, col2 = st.beta_columns(2)
            
            with col1:
                anzahl_zimmer = st.slider('Anzahl Zimmer', 0, 10, 5)
                anzahl_parkplatz = st.slider('Anzahl Parkplätze', 0, 5, 1)
                immobilienart = st.selectbox('Immobilienart', (
                    'Doppelhaushälfte',
                    'Einfamilienhaus',
                    'Etagenwohnung',
                    'Sonstige',
                    'Mehrfamilienhaus',
                    'Erdgeschosswohnung',
                    'Erdgeschosswohnung',
                    'Dachgeschosswohnung',
                    'Zweifamilienhaus',
                    'Wohnung',
                    'Villa',
                    'Reihenmittelhaus',
                    'Reihenendhaus',
                    'Bungalow',
                    'Maisonette',
                    'Apartment',
                    'Stadthaus',
                    'Schloss',
                    'Bauernhaus',
                    'Herrenhaus',
                    'Reiheneckhaus',
                    'Penthouse',
                    'Unbekannt'))
                heizung = st.selectbox('Art der Heizung',(
                    'Zentralheizung',
                    'Etagenheizung',
                    'Fußbodenheizung',
                    'Ofenheizung',
                    'Sonstige',                    
                    'Unbekannt'))
                
            with col2:
                anzahl_badezimmer = st.slider('Anzahl Badezimmer', 0, 5, 2)
                terrasse_balkon = st.selectbox('Terrasse/ Balkon',('NEIN', 'JA'))
                immobilienzustand = st.selectbox('Immobilienzustand', (
                    'Neuwertig',
                    'Altbau',
                    'Erstbezug',
                    'Gepflegt',
                    'Modernisiert',
                    'Neubau',                    
                    'Projektiert',
                    'Sonstige',
                    'Teil- oder vollrenovierungsbedürftig',
                    'Teil- oder vollsaniert',
                    'Teil- oder vollrenoviert',
                    'Unbekannt'))                
                energietyp = st.selectbox('Energietyp',(
                    'Gas',
                    'Fernwärme',
                    'Holz',
                    'Luft-/Wasserwärme',
                    'Öl',
                    'Pellets',
                    'Solar',
                    'Sonstige',
                    'Strom',
                    'Unbekannt'))
        
        #Eingabefeld 3 in Expander (Gäste_WC, barrierefrei, Aufzug,...)
        weitereKonfigurationen = st.beta_expander('spezifische Angaben')
        with weitereKonfigurationen:            
            gaeste_wc = st.selectbox('Gäste WC?',('NEIN', 'JA'))
            barrierefrei = st.selectbox('barrierefrei?',('NEIN', 'JA'))
            aufzug = st.selectbox('Aufzug?',('NEIN', 'JA'))
            unterkellert = st.selectbox('unterkellert?',('NEIN', 'JA'))
            vermietet = st.selectbox('aktuell vermietet?',('NEIN', 'JA'))
            energie_effizienzklasse = st.selectbox('Energieeffizienzklasse',(
                'A',
                'A+',
                'B',
                'C',
                'D',
                'E',
                'F',
                'G',
                'H',                
                'Unbekannt'))

        #Kategorische Variablen codieren
        immobilienart_string = 'SELECT immobilienart_targetenc FROM Encoding_immobilienart WHERE immobilienart=\'' + immobilienart + '\''
        immobilienart = np.float32(pd.read_sql_query(immobilienart_string, con=db_connection).iloc[0][0])

        heizung_string = 'SELECT heizung_targetenc FROM Encoding_heizung WHERE heizung=\'' + heizung + '\''
        heizung = np.float32(pd.read_sql_query(heizung_string, con=db_connection).iloc[0][0])

        immobilienzustand_string = 'SELECT immobilienzustand_targetenc FROM Encoding_immobilienzustand WHERE immobilienzustand=\'' + immobilienzustand + '\''
        immobilienzustand = np.float32(pd.read_sql_query(immobilienzustand_string, con=db_connection).iloc[0][0])

        energietyp_string = 'SELECT energietyp_targetenc FROM Encoding_energietyp WHERE energietyp=\'' + energietyp + '\''
        energietyp = np.float32(pd.read_sql_query(energietyp_string, con=db_connection).iloc[0][0])

        energie_effizienzklasse_string = 'SELECT energie_effizienzklasse_targetenc FROM Encoding_energie_effizienzklasse WHERE energie_effizienzklasse=\'' + energie_effizienzklasse + '\''
        energie_effizienzklasse = np.float32(pd.read_sql_query(energie_effizienzklasse_string, con=db_connection).iloc[0][0])


        #Zuordnung der Eingabe-Features
        data = {'plz': plz,
                'immobilienart': immobilienart,
                'immobilienzustand': immobilienzustand,
                'barrierefrei': barrierefrei,
                'terrasse_balkon': terrasse_balkon,
                'unterkellert': unterkellert,
                'vermietet': vermietet,
                'energietyp': energietyp,
                'heizung': heizung,
                'gaeste_wc': gaeste_wc,
                'energie_effizienzklasse': energie_effizienzklasse,
                'aufzug': aufzug,
                'anzahl_badezimmer': anzahl_badezimmer,
                'anzahl_zimmer': anzahl_zimmer,
                'anzahl_parkplatz': anzahl_parkplatz,
                'baujahr': baujahr,
                'grundstuecksflaeche': grundstuecksflaeche,
                'wohnflaeche': wohnflaeche}
        features = pd.DataFrame(data, index=[0])

        features = features.assign(aufzug=(features['aufzug'] == 'JA').astype(int))
        features = features.assign(barrierefrei=(features['barrierefrei'] == 'JA').astype(int))
        features = features.assign(gaeste_wc=(features['gaeste_wc'] == 'JA').astype(int))
        features = features.assign(terrasse_balkon=(features['terrasse_balkon'] == 'JA').astype(int))
        features = features.assign(unterkellert=(features['unterkellert'] == 'JA').astype(int))
        features = features.assign(vermietet=(features['vermietet'] == 'JA').astype(int))

        #Metadaten aus Datenbank auslesen
        Metadaten = pd.read_sql_query('SELECT * FROM Meta_Data_upd WHERE plz=plz', con=db_connection, index_col="index")
        Metadaten = Metadaten.assign(
            supermarkt_im_plz_gebiet=(Metadaten['Supermarkt im PLZ Gebiet'] == 'JA').astype(int))
        Metadaten.drop(columns=['Supermarkt im PLZ Gebiet'], inplace=True)

        features = features.merge(Metadaten, how="inner", on="plz")
        features.drop(columns=['plz'], inplace=True)

        verstädterung = features['Grad_der_Verstädterung'].to_list()[0]
        verstädterung_string = 'SELECT Grad_der_Verstädterung_targetenc FROM Encoding_Grad_der_Verstädterung WHERE Grad_der_Verstädterung=\'' + verstädterung + '\''
        verstädterung = np.float32(pd.read_sql_query(verstädterung_string, con=db_connection).iloc[0][0])
        features['Grad_der_Verstädterung'] = verstädterung

        soziolage = features['sozioökonomische_Lage'].to_list()[0]
        soziolage_string = 'SELECT sozioökonomische_Lage_targetenc FROM Encoding_sozioökonmische_Lage WHERE sozioökonomische_Lage=\'' + soziolage + '\''
        soziolage = np.float32(pd.read_sql_query(soziolage_string, con=db_connection).iloc[0][0])
        features['sozioökonomische_Lage'] = soziolage

        #num_scaler = pickle.load(open('Projektseminar/num_scaler.pckl', 'rb'))

        #cat_features = features[['energietyp', 'energie_effizienzklasse',
         #                               'heizung', 'immobilienart', 'immobilienzustand', 'Grad_der_Verstädterung',
         #                               'sozioökonomische_Lage']]
        #features.drop(columns=['energietyp', 'energie_effizienzklasse',
                                     #   'heizung', 'immobilienart', 'immobilienzustand', 'Grad_der_Verstädterung',
                                      #  'sozioökonomische_Lage'], inplace=True)

        #features = pd.DataFrame(num_scaler.transform(features),
                             #  columns=features.columns, index=features.index)

        #features.to_sql(name='Features_scaler', con=db_connection, if_exists='replace')

        #features = pd.concat([features, cat_features], axis=1)

        features.rename(columns={'immobilienart': 'immobilienart_targetenc', 'immobilienzustand': 'immobilienzustand_targetenc',
                                 'energietyp': 'energietyp_targetenc', 'energie_effizienzklasse': 'energie_effizienzklasse_targetenc',
                                 'heizung': 'heizung_targetenc', 'Grad_der_Verstädterung': 'Grad_der_Verstädterung_targetenc',
                                 'sozioökonomische_Lage': 'sozioökonomische_Lage_targetenc'}, inplace=True)


        features = features.reindex(sorted(features.columns), axis=1)
        features.to_sql(name='Features', con=db_connection, if_exists='replace')
        return features

input_df = user_input_features()


# Einlesen des Models aus der Pickle-Datei
load_XGB_modell = pickle.load(open('Projektseminar/XGB_Standardmodell_20210421-2205.pckl', 'rb'))

# Abstandshalter
st.write('')
st.image('Projektseminar/Files/GUI/AbstandshalterAWI.jpg')


# Definition des Outputs
output = ''
if st.button('Wertanalyse starten'):
    output = int(load_XGB_modell.predict(input_df)[0])
    output = str(output) + '€'
    st.success('Der Wert Ihrer Immobilie liegt bei {}'.format(output))
    

# Abstandshalter
st.write('')
st.image('Projektseminar/Files/GUI/AbstandshalterAWI.jpg')
    
    
# weitere graphische Darstellungen
if st.button('Graphische Datenanalyse'):
    data = pd.read_csv('Projektseminar/Files/GUI/imputed_data_original.csv')
    st.map(data)
    

    # EDA
if st.button('Explorative Datenanalyse'):
    load_pr = pickle.load(open('pr.pkl', 'rb'))
    st_profile_report(load_pr)

if __name__ == "__main__":
    print('Hallo')