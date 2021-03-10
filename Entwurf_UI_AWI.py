import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Änderungen an csv:
# Datensatz von Lennart kopieren und in Excel kopieren und Spalten zusammen (csv Format) dann folgende Änderungen:
# Alle ohne Grundstücksfläche und Wohnfläche raus
# Spalten raus: Längengrad, Breitengrad, PLZ

# allgemeine Streamlit Einstellungen (Tab Name; Icon; Seitenlayout; Menü)
st.set_page_config('Immobilienbewertung', 'UniWueLogo.png', 'centered', 'expanded')
#Logo einfügen
st.image('Logo AWI.jpg')
st.image('AbstandshalterAWI.jpg')
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

# Daten einlesen
#@st.cache einbinden wenn laden der Daten als Def hinterlegt
imputed_data = pd.read_csv('imputed_data.csv')

# Encoding von object-Variablen in Liste
categorical_mask = (imputed_data.dtypes == "object")
categorical_columns = imputed_data.columns[categorical_mask].tolist()
category_enc = pd.get_dummies(imputed_data[categorical_columns])
imputed_data = pd.concat([imputed_data, category_enc], axis=1)
imputed_data = imputed_data.drop(columns=categorical_columns)

# Festlegung X und Y; Test und Trainingsdaten definieren
#x = imputed_data.drop('angebotspreis', axis=1).values
#y = imputed_data['angebotspreis'].values

x = imputed_data.drop(columns=["angebotspreis"]).values
y = imputed_data["angebotspreis"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# RandomForrestModel
random_forrest_reg = RandomForestRegressor()
random_forrest_reg.fit(x_train, y_train)

# Speicherung des Models als Pickle-Datei
pickle.dump(random_forrest_reg, open('random_forrest_reg.pkl', 'wb'))

st.image('AbstandshalterAWI.jpg')
st.subheader('Beschreibe deine Immobilie:')

def user_input_features():
        wohnflaeche = st.slider('Wohnfläche', 0, 1000, 200)
        grundstuecksflaeche = st.slider('Grundstücksfläche', 0, 1000, 200)
        baujahr = st.slider('Baujahr', 1980, 2023, 2000)
        
        weitereDetails = st.beta_expander('weitere Details')
        with weitereDetails:
            col1, col2 = st.beta_columns(2)
            with col1:
                anzahl_zimmer = st.slider('Anzahl Zimmer', 0, 10, 2)
                anzahl_parkplatz = st.slider('Anzahl Parkplätze', 0, 5, 0)
                immobilienart = st.selectbox('Immobilienart', ('Wohnung', 'Einfamilienhaus', 'Etagenwohnung', 'Sonstige', 'Mehrfamilienhaus', 'Erdgeschosswohnung', 'Erdgeschosswohnung', 'Dachgeschosswohnung', 'Zweifamilienhaus', 'Doppelhaushälfte', 'Villa', 'Reihenmittelhaus','Reihenendhaus', 'Bungalow', 'Maisonette', 'Apartment', 'Stadthaus', 'Schloss', 'Bauernhaus', 'Herrenhaus', 'Reiheneckhaus','Penthouse', 'Unbekannt'))
                heizung = st.selectbox('Art der Heizung',('Etagenheizung', 'Fußbodenheizung', 'Ofenheizung', 'Sonstige', 'Zentralheizung', 'Unbekannt'))
            with col2:
                anzahl_badezimmer = st.slider('Anzahl Badezimmer', 0, 5, 1)
                terrasse_balkon = st.selectbox('Terrasse/ Balkon',('NEIN', 'JA'))
                immobilienzustand = st.selectbox('Immobilienzustand', ('Altbau', 'Erstbezug', 'Gepflegt', 'Modernisiert', 'Neubau', 'Neuwertig', 'Projektiert', 'Sonstige', 'Teil- oder vollrenovierungsbedürftig', 'Teil- oder vollsaniert', 'Teil- oder vollrenoviert', 'Unbekannt'))
                energietyp = st.selectbox('Energietyp',('Gas', 'Fernwärme', 'Holz', 'Luft-/Wasserwärme', 'Öl', 'Pellets', 'Solar', 'Sonstige', 'Strom', 'Unbekannt'))
        
        weitereKonfigurationen = st.beta_expander('spezifische Angaben')
        with weitereKonfigurationen:
            gaeste_wc = st.selectbox('Gäste WC?',('NEIN', 'JA'))
            barrierefrei = st.selectbox('barrierefrei?',('NEIN', 'JA'))
            aufzug = st.selectbox('Aufzug?',('NEIN', 'JA'))
            unterkellert = st.selectbox('unterkellert?',('NEIN', 'JA'))
            vermietet = st.selectbox('aktuell vermietet?',('NEIN', 'JA'))
            energie_effizienzklasse = st.selectbox('Energieeffizienzklasse',('A', 'A+', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Unbekannt'))
            einwohner = st.slider('Einwohner', 0, 10000, 2000)
        
        data = {'immobilienart': immobilienart,
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
                'einwohner': einwohner,
                'grundstuecksflaeche': grundstuecksflaeche,
                'wohnflaeche': wohnflaeche}
        features = pd.DataFrame(data, index=[0])
        return features
input_df = user_input_features()
    
# Kombination der Input_Features mit dem Datensatz
immo_data_raw = pd.read_csv('imputed_data.csv')
immo_data = immo_data_raw.drop(columns=['angebotspreis'])
df = pd.concat([input_df,immo_data],axis=0)

# Encoding der object-Variablen
encode = ['immobilienart', 'immobilienzustand', 'barrierefrei', 'terrasse_balkon', 'unterkellert', 'vermietet', 'energietyp', 'heizung', 'gaeste_wc', 'energie_effizienzklasse', 'aufzug']
         
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1]

# Einlesen des Models aus der Pickle-Datei
load_random_forrest_reg = pickle.load(open('random_forrest_reg.pkl', 'rb'))

st.write('')
st.image('AbstandshalterAWI.jpg')

# Definition der Prediction
x_new = df.values
#prediction = random_forrest_reg.predict(x_new)
#st.subheader('Prediction')                      
#st.write(prediction)
output = ''
if st.button('Wertanalyse starten'):
    output = random_forrest_reg.predict(x_new)
    output = str(output) + '€'
    st.success('Der Wert Ihrer Immobilie liegt bei {}'.format(output))
#rmse = np.sqrt(mean_squared_error(y_test, x_test))
#st.write("RMSE: %f" % rmse)

st.write('')
st.image('AbstandshalterAWI.jpg')

#SHAP
# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
if st.button('weiterführende Analysen'):
    data = pd.read_csv('imputed_data_original.csv')
    st.map(data)
    
    #explainer = shap.TreeExplainer(random_forrest_reg)
    #shap_values = explainer.shap_values(x)

#st.header('Feature Importance')
#plt.title('Feature importance based on SHAP values')
#shap.summary_plot(shap_values, x)
#st.pyplot(bbox_inches='tight')
#st.write('---')

#plt.title('Feature importance based on SHAP values (Bar)')
#shap.summary_plot(shap_values, x, plot_type="bar")
#st.pyplot(bbox_inches='tight')


# EDA
#eda_data = pd.read_csv('imputed_data_UI.csv')
#pr = ProfileReport(eda_data, explorative=True)
#st_profile_report(pr)

#EDA hinten dran