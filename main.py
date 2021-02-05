import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


def read_data_from_immonet():
    immonet_data = pd.read_excel(r"Files/Februar_Immonet_Bayern.xlsx", sheet_name="Tabelle2")

    return immonet_data


def read_data_from_immoscout():
    # TO-DO: Auf aktuelle Datensätze anpassen
    immoscout_data_haeuser = pd.read_excel(r"Files/Februar_Immoscout_Häuser_Bayern.xlsx",
                                           sheet_name="")
    immoscout_data_wohnungen = pd.read_excel(r"Files/Februar_Immoscout_Wohnungen_Bayern.xlsx", sheet_name="Häuser neu")

    immoscout_data = pd.concat([immoscout_data_haeuser, immoscout_data_wohnungen], axis=0, ignore_index=True)

    return immoscout_data

def read_data_from_coordinates():
    #Datensatz mit Koordinaten von Timo
    Geodaten = pd.read_excel(r'Files/PLZ mit Geodaten.xlsx', sheet_name='PLZ')
    return Geodaten

def read_data_from_inhabitants():
    #Datensatz mit Einwohnern von Yanina
    Einwohner = pd.read_excel(r'Files/PLZ_einwohner.xlsx', sheet_name='Tabelle2')
    return Einwohner

def add_coodinates_inhabitants_immonet(immonet_data, Geodaten, Einwohner):
    #Koordinaten und Einwohner auf Immonet-Daten anpassen
    immonet_data['plz'] = immonet_data['plz'].astype(str)
    Geodaten = Geodaten.astype(str)
    list_plz_immonet = immonet_data['plz']

    dict_breitengrad_immonet = dict(zip(Geodaten['PLZ'], Geodaten['Breitengrad']))
    list_breitengrad_immonet = [dict_breitengrad_immonet.get(key) for key in list_plz_immonet]

    dict_längengrad_immonet = dict(zip(Geodaten['PLZ'], Geodaten['Längengrad']))
    list_längengrad_immonet = [dict_längengrad_immonet.get(key) for key in list_plz_immonet]

    Einwohner = Einwohner.astype(str)
    dict_einwohner_immonet = dict(zip(Einwohner['plz'], Einwohner['einwohner']))
    list_einwohner_immonet = [dict_einwohner_immonet.get(key) for key in list_plz_immonet]

    immonet_data['breitengrad'] = list_breitengrad_immonet
    immonet_data['laengengrad'] = list_längengrad_immonet
    immonet_data['einwohner'] = list_einwohner_immonet

    immonet_data = immonet_data.dropna(subset=['breitengrad'])
    immonet_data = immonet_data.dropna(subset=['laengengrad'])
    immonet_data = immonet_data.dropna(subset=['einwohner'])

    immonet_data_new = immonet_data

    return immonet_data_new


def add_coordinates_inhabitants_immoscout(immoscout_data, Geodaten, Einwohner):
    #Koordinaten und Einwohner auf Immonet - Daten anpassen
    Geodaten = Geodaten.astype(str)
    immoscout_data["plz"] = immoscout_data["PLZ und Ort"].astype(str).apply(lambda row: row[:5])
    list_plz_immoscout = immoscout_data['plz']

    dict_breitengrad_immoscout = dict(zip(Geodaten['PLZ'], Geodaten['Breitengrad']))
    list_breitengrad_immoscout = [dict_breitengrad_immoscout.get(key) for key in list_plz_immoscout]

    dict_längengrad_immoscout = dict(zip(Geodaten['PLZ'], Geodaten['Längengrad']))
    list_längengrad_immoscout = [dict_längengrad_immoscout.get(key) for key in list_plz_immoscout]

    Einwohner = Einwohner.astype(str)
    dict_einwohner_immoscout = dict(zip(Einwohner['plz'], Einwohner['einwohner']))
    list_einwohner_immoscout = [dict_einwohner_immoscout.get(key) for key in list_plz_immoscout]

    immoscout_data['breitengrad'] = list_breitengrad_immoscout
    immoscout_data['laengengrad'] = list_längengrad_immoscout
    immoscout_data['einwohner'] = list_einwohner_immoscout

    immoscout_data = immoscout_data.dropna(subset=['breitengrad'])
    immoscout_data = immoscout_data.dropna(subset=['laengengrad'])
    immoscout_data = immoscout_data.dropna(subset=['einwohner'])

    immoscout_data_new = immoscout_data

    return immoscout_data_new

def merge_data(immonet_data_new, immoscout_data_new):
    # Immoscout Format an Immonet Format anpassen:

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

    immonet_data_new['terrasse_balkon'] = immonet_data_new['terrasse'] + '' + immonet_data['balkon']
    immonet_data_new['terrasse_balkon'] = immonet_data_new['terrasse_balkon'].apply(
        lambda row: 'JA' if 'JA' in row else 'NEIN')
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
    immoscout_data_new["grundstuecksflaeche"] = pd.to_numeric(immoscout_data_new["grundstuecksflaeche"],
                                                          errors="ignore")
    immoscout_data_new["wohnflaeche"] = immoscout_data_new["wohnflaeche"].astype(str).apply(lambda row: re.sub('[m²]', '', row))
    immoscout_data_new["wohnflaeche"] = pd.to_numeric(immoscout_data_new["wohnflaeche"].str.replace(",", "."),
                                                  errors="ignore")

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
    immonet_data = immonet_data_new.reindex(sorted(immonet_data_new.columns), axis=1)
    immoscout_data = immoscout_data_new.reindex(sorted(immoscout_data_new.columns), axis=1)

    #Innerjoin reicht hier aus
    merged_data = pd.concat([immoscout_data_new, immonet_data_new], axis=0, ignore_index=True, join="inner")

    return merged_data


def preprocess_data(merged_data):
    # Tausender Stellen - Scraper Fehler -> abgeschnittene Nullen korrigieren
    merged_data.loc[merged_data["angebotspreis"] <= 10000, "angebotspreis"] = merged_data["angebotspreis"] * 1000

    # Umbenennungen
    merged_data.rename(columns={"befeuerungsart": "energietyp"}, inplace=True)

    # Zeilen ohne Angebotspreis droppen
    merged_data = merged_data.dropna(subset=["angebotspreis"])

    # Nicht verwendbare Spalten droppen
    merged_data = merged_data.drop(
        columns=['anzahl_schlafzimmer', 'energie_verbrauch', 'geschoss'])

    # Spalten-Datentypen bearbeiten
    merged_data["terrasse_balkon"] = merged_data["terrasse_balkon"].astype("category")
    merged_data["barrierefrei"] = merged_data["barrierefrei"].astype("category")
    merged_data["energietyp"] = merged_data["energietyp"].astype("category")
    merged_data["energie_effizienzklasse"] = merged_data["energie_effizienzklasse"].astype("category")
    merged_data["gaeste_wc"] = merged_data["gaeste_wc"].astype("category")
    merged_data["heizung"] = merged_data["heizung"].astype("category")
    merged_data["immobilienart"] = merged_data["immobilienart"].astype("category")
    merged_data["immobilienzustand"] = merged_data["immobilienzustand"].astype("category")
    merged_data["plz"] = merged_data["plz"].astype("category")
    merged_data["unterkellert"] = merged_data["unterkellert"].astype("category")
    merged_data["vermietet"] = merged_data["vermietet"].astype("category")
    merged_data["aufzug"] = merged_data["aufzug"].astype("category")

    # Kategorische Spalten anpassen (Kategorien zusammenfassen, kleine Kategorien in Sammler "Sonstige" zusammenfassen)
    merged_data["energietyp"] = merged_data["energietyp"].apply(
        lambda row: str(row).split(",")[0])
    merged_data["energietyp"] = merged_data["energietyp"].apply(
        lambda row: 'Pellets' if row == "Holzpellets" else row)
    merged_data["energietyp"] = merged_data["energietyp"].apply(
        lambda row: 'Gas' if row == "Flüssiggas" else row)
    merged_data["energietyp"] = merged_data["energietyp"].apply(
        lambda row: 'Fernwärme' if row == "Erdwärme" else row)
    merged_data["energietyp"] = merged_data["energietyp"].apply(
        lambda row: 'Sonstige' if row not in ["", np.nan, "Öl", "Gas", "Fernwärme",
                                              "Luft-/Wasserwärme", "Holz", "Pellets", "Solar", "Strom"] else row)
    merged_data["heizung"] = merged_data["heizung"].apply(
        lambda row: 'Sonstige' if row not in ["", np.nan, "Zentralheizung", "Etagenheizung", "Ofenheizung",
                                              "Fußbodenheizung"] else row)

    merged_data["immobilienart"] = merged_data["immobilienart"].apply(
        lambda row: 'Einfamilienhaus' if row == "Einfamilienhaus (freistehend)" else row)

    merged_data["immobilienart"] = merged_data["immobilienart"].apply(
        lambda row: 'Sonstige' if row not in ["", np.nan, "Einfamilienhaus", "Wohngrundstück", "Wohnung",
                                              "Etagenwohnung",
                                              "Sonstiges", "Mehrfamilienhaus", "Erdgeschosswohnung",
                                              "Dachgeschosswohnung",
                                              "Zweifamilienhaus", "Doppelhaushälfte", "Villa", "Reihenmittelhaus",
                                              "Reihenendhaus", "Bungalow",
                                              "Maisonette", "Apartment", "Stadthaus", "Schloss", "Bauernhaus",
                                              "Herrenhaus", "Reiheneckhaus", "Penthouse"] else row)

    merged_data["immobilienzustand"] = merged_data["immobilienzustand"].apply(
        lambda row: 'Teil- oder vollrenovierungsbedürftig' if row == "Renovierungsbedürftig" else row)
    merged_data["immobilienzustand"] = merged_data["immobilienzustand"].apply(
        lambda row: 'Teil- oder vollsaniert' if row == "Saniert" else row)
    merged_data["immobilienzustand"] = merged_data["immobilienzustand"].apply(
        lambda row: 'Teil- oder vollrenoviert' if row == "Renovierungsbedürftig" else row)
    merged_data["immobilienzustand"] = merged_data["immobilienzustand"].apply(
        lambda row: 'Sonstige' if row not in ["", np.nan, "Unbekannt", "Erstbezug", "Projektiert", "Neubau",
                                              "Teil- oder vollrenovierungsbedürftig", "Neuwertig",
                                              "Teil- oder vollsaniert", "Teil- oder vollrenoviert", "Gepflegt",
                                              "Altbau", "Modernisiert"] else row)

    preprocessed_data = merged_data

    return preprocessed_data


def impute_data(preprocessed_data):
    # Zufällig mit vorhandenen Werten auffüllen
    preprocessed_data.loc[preprocessed_data["anzahl_badezimmer"] == 0, "anzahl_badezimmer"] = np.nan
    preprocessed_data["anzahl_badezimmer"] = preprocessed_data["anzahl_badezimmer"].apply(
        lambda x: np.random.choice(range(1, 4), p=[0.65, 0.30, 0.05]) if np.isnan(x) else x)
    preprocessed_data["anzahl_zimmer"] = preprocessed_data["anzahl_zimmer"].apply(
        lambda x: np.random.choice(preprocessed_data["anzahl_zimmer"].dropna().values) if np.isnan(x) else x)
    preprocessed_data["baujahr"] = preprocessed_data["baujahr"].apply(
        lambda x: np.random.choice(preprocessed_data["baujahr"].dropna().values) if np.isnan(x) else x)

    # ImmobilienMaster["energie_effizienzklasse"] = ImmobilienMaster["energie_effizienzklasse"].cat.codes
    # energie = ImmobilienMaster["energie_effizienzklasse"].value_counts(normalize=True)
    # energie.drop(labels=[-1])
    # energie_types = energie.keys()
    # energie_props = energie.values
    # ImmobilienMaster["energie_effizienzklasse"] = ImmobilienMaster["energie_effizienzklasse"].fillna(lambda x: np.random.choice(energie_types, energie_props) if x == -1 else x)

    preprocessed_data["energietyp"] = preprocessed_data["energietyp"].astype("category")
    preprocessed_data["energietyp"] = preprocessed_data["energietyp"].cat.add_categories(
        ["Unbekannt"])
    preprocessed_data["energietyp"] = preprocessed_data["energietyp"].fillna("Unbekannt")

    preprocessed_data["energie_effizienzklasse"] = preprocessed_data["energie_effizienzklasse"].cat.add_categories(
        ["Unbekannt"])
    preprocessed_data["energie_effizienzklasse"] = preprocessed_data["energie_effizienzklasse"].fillna("Unbekannt")

    preprocessed_data["heizung"] = preprocessed_data["heizung"].astype("category")
    preprocessed_data["heizung"] = preprocessed_data["heizung"].cat.add_categories(
        ["Unbekannt"])
    preprocessed_data["heizung"] = preprocessed_data["heizung"].fillna("Unbekannt")

    preprocessed_data["immobilienart"] = preprocessed_data["immobilienart"].astype("category")
    preprocessed_data["immobilienart"] = preprocessed_data["immobilienart"].cat.add_categories(
        ["Unbekannt"])
    preprocessed_data["immobilienart"] = preprocessed_data["immobilienart"].fillna("Unbekannt")

    preprocessed_data["immobilienzustand"] = preprocessed_data["immobilienzustand"].astype("category")
    preprocessed_data["immobilienzustand"] = preprocessed_data["immobilienzustand"].cat.add_categories(
        ["Unbekannt"])
    preprocessed_data["immobilienzustand"] = preprocessed_data["immobilienzustand"].fillna("Unbekannt")

    # Aufzug: Annahme, wenn nicht explizit angegeben, dann existiert kein Aufzug
    preprocessed_data.loc[preprocessed_data["aufzug"].isna(), "aufzug"] = "NEIN"

    imputed_data = preprocessed_data

    return imputed_data


def print_feature_importances(model, data):
    importances = pd.Series(data=model.feature_importances_,
                            index=data.columns)
    importances_sorted = importances.sort_values()
    importances_sorted.plot(kind='barh', color='lightgreen')
    plt.title('Features Importances')
    plt.show()


def ml_tests(imputed_data):
    # ScikitLearn Anforderung: Nur numerische Werte - Transformation der kategorischen Spalten
    categorical_mask = (imputed_data.dtypes == "category")
    categorical_columns = imputed_data.columns[categorical_mask].tolist()
    category_enc = pd.get_dummies(imputed_data[categorical_columns])
    imputed_data = pd.concat([imputed_data, category_enc], axis=1)
    imputed_data = imputed_data.drop(columns=categorical_columns)

    imputed_data = imputed_data.reset_index()

    # Ausgabe
    # print(imputed_data.info())
    # imputed_data.to_excel(excel_writer="Files/Tests/imputed_data.xlsx", sheet_name="Immobilien")

    # XGBoost Standardmodell
    print("XGBoost Standardmodell:")
    X = imputed_data.drop(columns=["angebotspreis"]).values
    y = imputed_data["angebotspreis"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    xg_reg = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=20, seed=123)
    xg_reg.fit(X_train, y_train)
    preds = xg_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % (rmse))
    print()

    print_feature_importances(model=xg_reg, data=imputed_data.drop(columns=["angebotspreis"]))

    # Grid Search parameter Tuning
    print("Grid Search Parameter Tuning:")
    gbm_param_grid = {
        'colsample_bytree': [0.3, 0.7],
        'n_estimators': [50],
        'max_depth': [2, 5]
    }
    gbm = xgb.XGBRegressor(objective="reg:squarederror")
    grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid, scoring="neg_mean_squared_error", cv=4, verbose=1)
    grid_mse.fit(X_train, y_train)
    print("Best parameters found: ", grid_mse.best_params_)
    print("Lowest RMSE Grid Search found: ", np.sqrt(np.abs(grid_mse.best_score_)))
    print()

    # Randomized Search parameter tuning
    print("Randomized Search Parameter Tuning:")
    gbm_param_grid2 = {
        'n_estimators': [25],
        'max_depth': range(2, 12)
    }

    gbm2 = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=10)
    randomized_mse = RandomizedSearchCV(estimator=gbm2, param_distributions=gbm_param_grid2,
                                        scoring="neg_mean_squared_error", n_iter=5, cv=4, verbose=1)
    randomized_mse.fit(X_train, y_train)
    print("Best parameters found: ", randomized_mse.best_params_)
    print("Lowest RMSE Randomized Search found: ", np.sqrt(np.abs(randomized_mse.best_score_)))

    DM_train = xgb.DMatrix(data=X_train, label=y_train)
    DM_test = xgb.DMatrix(data=X_test, label=y_test)
    params = {"booster": "gblinear", "objective": "reg:squarederror"}
    xg_reg2 = xgb.train(dtrain=DM_train, params=params, num_boost_round=15)
    preds2 = xg_reg2.predict(DM_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds2))
    print("RMSE: %f" % (rmse))

    reg_params = [0.1, 0.3, 0.7, 1, 10, 100]
    params1 = {"objective": "reg:squarederror", "max_depth": 3}
    rmses_l2 = []
    for reg in reg_params:
        params1["lambda"] = reg
        cv_results_rmse = xgb.cv(dtrain=DM_train, params=params1, nfold=3, num_boost_round=15, metrics="rmse",
                                 as_pandas=True)
        rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])

    print("Best rmse as a function of l2:")
    print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))
    print()

    print_feature_importances(model=xg_reg2, data=imputed_data.drop(columns=["angebotspreis"]))

    # Stochastic Gradient Boosting
    print("Stochastic Gradient Boosting:")
    sgbr = GradientBoostingRegressor(max_depth=4,
                                     subsample=0.9,
                                     max_features=0.75,
                                     n_estimators=200,
                                     random_state=2)

    sgbr.fit(X_train, y_train)
    y_pred = sgbr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE: %f" % (rmse))
    print()

    print_feature_importances(model=sgbr, data=imputed_data.drop(columns=["angebotspreis"]))

    # Random Forrest
    print("Random Forrest:")
    rf = RandomForestRegressor(n_estimators=25,
                               random_state=2)
    rf.fit(X_train, y_train)
    y_pred2 = rf.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred2))
    print("RMSE: %f" % (rmse))
    print()

    print_feature_importances(model=rf, data=imputed_data.drop(columns=["angebotspreis"]))


def main():
    immonet_data = read_data_from_immonet()
    immoscout_data = read_data_from_immoscout()
    Geodaten = read_data_from_coordinates()
    Einwohner = read_data_from_inhabitants()
    add_coodinates_inhabitants_immonet(immonet_data, Geodaten, Einwohner)
    add_coordinates_inhabitants_immoscout(immoscout_data, Geodaten, Einwohner)
    merged_data = merge_data(immonet_data_new, immoscout_data_new)
    preprocessed_data = preprocess_data(merged_data)
    imputed_data = impute_data(preprocessed_data)
    ml_tests(imputed_data)

    # Testausgaben
    # immoscout_data.to_excel(excel_writer="Files/Tests/ImmoscoutTest.xlsx", sheet_name="ImmobilienAll")
    # merged_data.to_excel(excel_writer="Files/Tests/merged_data.xlsx", sheet_name="Immobilien")

    print("fertig...")


if __name__ == "__main__":
    main()

# Last Run:

# XGBoost Standardmodell:
# RMSE: 137019.661546
#
# Grid Search Parameter Tuning:
# Fitting 4 folds for each of 4 candidates, totalling 16 fits
# Best parameters found:  {'colsample_bytree': 0.7, 'max_depth': 2, 'n_estimators': 50}
# Lowest RMSE Grid Search found:  177284.4177031987
#
# Randomized Search Parameter Tuning:
# Fitting 4 folds for each of 5 candidates, totalling 20 fits
# Best parameters found:  {'n_estimators': 25, 'max_depth': 5}
# Lowest RMSE Randomized Search found:  156688.29557870186
# RMSE: 398490.411472
# Best rmse as a function of l2:
#       l2           rmse
# 0    0.1  155939.911458
# 1    0.3  157250.669271
# 2    0.7  156574.890625
# 3    1.0  156490.554688
# 4   10.0  163362.427083
# 5  100.0  246595.109375
#
# Stochastic Gradient Boosting:
# RMSE: 148264.288115
#
# Random Forrest:
# RMSE: 127882.791338
#
# fertig...
