import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
st.beta_set_page_config('Immobilienbewertung', 'UniWueLogo.png', 'centered', 'expanded')

# Headerabschnitt (Bild, Infotext, Trennlinie)
st.image('UniWueLogo.png', width=250)
st.markdown("<h1 style='text-align: center; color: #004188;'>Willkommen bei WueRate</h1>", unsafe_allow_html=True)
st.write('---')
st.markdown("<h3 style='text-align: center; color: #000000;'>Schnelle Bewertung Ihrer Immobilie in Bayern</h3>", unsafe_allow_html=True)
st.write('Sie möchten eine Immobilie kaufen oder verkaufen und benötigen Hilfe bei der Bewertung?')
st.write('Kein Problem, WueRate bietet Ihnen eine fundierte Immobilienbewertung mithilfe maschinellen Lernens. Auf Basis einer umfangreichen Datenanalyse stellt Ihnen WueRate den optimalen Angebotspreis bereit.')
st.write('---')

# Streamlit Sidebar
st.sidebar.header('Beschreiben Sie Ihre Immobilie, um Ihren optimalen Angebotspreis zu erhalten:')

# Daten einlesen
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

# Definieren der Input-Features
#uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

def user_input_features():
        immobilienart = st.sidebar.selectbox('immobilienart', ('Wohnung', 'Einfamilienhaus', 'Etagenwohnung', 'Sonstige', 'Mehrfamilienhaus', 'Erdgeschosswohnung', 'Erdgeschosswohnung', 'Dachgeschosswohnung', 'Zweifamilienhaus', 'Doppelhaushälfte', 'Villa', 'Reihenmittelhaus','Reihenendhaus', 'Bungalow', 'Maisonette', 'Apartment', 'Stadthaus', 'Schloss', 'Bauernhaus', 'Herrenhaus', 'Reiheneckhaus','Penthouse', 'Unbekannt'))
        immobilienzustand = st.sidebar.selectbox('immobilienzustand', ('Altbau', 'Erstbezug', 'Gepflegt', 'Modernisiert', 'Neubau', 'Neuwertig', 'Projektiert', 'Sonstige', 'Teil- oder vollrenovierungsbedürftig', 'Teil- oder vollsaniert', 'Teil- oder vollrenoviert', 'Unbekannt'))
        barrierefrei = st.sidebar.selectbox('barrierefrei',('NEIN', 'JA'))
        aufzug = st.sidebar.selectbox('aufzug',('NEIN', 'JA'))
        gaeste_wc = st.sidebar.selectbox('gaeste_wc',('NEIN', 'JA'))
        terrasse_balkon = st.sidebar.selectbox('terrasse_balkon',('NEIN', 'JA'))
        unterkellert = st.sidebar.selectbox('unterkellert',('NEIN', 'JA'))
        vermietet = st.sidebar.selectbox('vermietet',('NEIN', 'JA'))
        energietyp = st.sidebar.selectbox('energietyp',('Gas', 'Fernwärme', 'Holz', 'Luft-/Wasserwärme', 'Öl', 'Pellets', 'Solar', 'Sonstige', 'Strom', 'Unbekannt'))
        anzahl_parkplatz = st.sidebar.slider('anzahl_parkplatz', 0, 5, 0)
        energie_effizienzklasse = st.sidebar.selectbox('energie_effizienzklasse',('A', 'A+', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Unbekannt'))
        heizung = st.sidebar.selectbox('heizung',('Etagenheizung', 'Fußbodenheizung', 'Ofenheizung', 'Sonstige', 'Zentralheizung', 'Unbekannt'))
        anzahl_badezimmer = st.sidebar.slider('anzahl_badezimmer', 0, 5, 1)
        anzahl_zimmer = st.slider('anzahl_zimmer', 0, 10, 2)
        baujahr = st.slider('baujahr', 1980, 2023, 2000)
        einwohner = st.sidebar.slider('einwohner', 0, 10000, 2000)
        grundstuecksflaeche = st.slider('grundstuecksflaeche', 0, 1000, 200)
        wohnflaeche = st.slider('wohnflaeche', 0, 1000, 200)
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

# Definition der Prediction
x_new = df.values
prediction = random_forrest_reg.predict(x_new)
st.subheader('Prediction')                      
st.write(prediction)
#rmse = np.sqrt(mean_squared_error(y_test, x_test))
#st.write("RMSE: %f" % rmse)


#SHAP
# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
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