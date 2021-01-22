import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

if __name__ == "__main__":
    # Exceldateien der Scraper importieren
    Immoscout24Base = pd.read_excel(r"Files/20201124_Immoscout24.xlsx", sheet_name="Häuser Wü und Landkreis")
    Immoscout24Update = pd.read_excel(r"Files/20201129_Immoscout24_update.xlsx", sheet_name="Häuser neu")
    ImmonetBase = pd.read_excel(r"Files/Immobilien_Bayern.xlsx", sheet_name="Tabelle2")

    # Immoscout24 Datensätze an Immonet Format anpassen
    Immoscout24Base.columns = Immoscout24Base.columns.str.lower()
    Immoscout24Base["plz"] = Immoscout24Base["plz und ort"].apply(lambda row: row[:5])
    Immoscout24Base["ort"] = Immoscout24Base["plz und ort"].apply(lambda row: row[5:])
    Immoscout24Base = Immoscout24Base.drop(columns="plz und ort")

    Immoscout24Update.columns = Immoscout24Update.columns.str.lower()
    Immoscout24Update["plz"] = Immoscout24Update["plz und ort"].apply(lambda row: row[:5])
    Immoscout24Update["ort"] = Immoscout24Update["plz und ort"].apply(lambda row: row[5:])
    Immoscout24Update = Immoscout24Update.drop(columns=["plz und ort", "web-scraper-order"])

    Immoscout24Base = Immoscout24Base.reindex(sorted(Immoscout24Base.columns), axis=1)
    Immoscout24Update = Immoscout24Update.reindex(sorted(Immoscout24Update.columns), axis=1)
    ImmonetBase = ImmonetBase.reindex(sorted(ImmonetBase.columns), axis=1)

    # Yaninas Datensätze zusammenführen und Spalten umbenennen
    Immoscout24AllBase = pd.concat([Immoscout24Base, Immoscout24Update], axis=0, ignore_index=True)
    Immoscout24AllBase.rename(
        columns={"anzahl badezimmer": "anzahl_badezimmer", "anzahl schlafzimmer": "anzahl_schlafzimmer",
                 "zimmer": "anzahl_zimmer", "einkaufspreis": "angebotspreis",
                 "balkon/ terrasse": "balkon", "wohnfläche": "wohnflaeche", "etage": "geschoss",
                 "grundstück": "grundstuecksflaeche", "stufenloser zugang": "barrierefrei",
                 "aufzug": "fahrstuhl", "objektzustand": "immobilienzustand",
                 "keller ja/nein": "unterkellert", "gäste-wc ja/nein": "gaeste_wc",
                 "energie­effizienz­klasse": "energie_effizienzklasse",
                 "wesentliche energieträger": "befeuerungsart", "end­energie­verbrauch": "energie_verbrauch",
                 "typ": "immobilienart", "heizungsart": "heizung", "vermietet ja/nein": "vermietet",
                 "garage/ stellplatz": "anzahl_parkplatz"}, inplace=True)

    # Spalteninhalte anpassen:
    # Annahme NaN ist NEIN
    Immoscout24AllBase["unterkellert"] = Immoscout24AllBase["unterkellert"].apply(
        lambda row: "JA" if row == "keller" else "NEIN")
    Immoscout24AllBase["gaeste_wc"] = Immoscout24AllBase["gaeste_wc"].apply(
        lambda row: "JA" if row == "Gäste-WC" else "NEIN")
    Immoscout24AllBase["barrierefrei"] = Immoscout24AllBase["barrierefrei"].apply(
        lambda row: "JA" if row == 'Stufenloser Zugang' else "NEIN")

    Immoscout24AllBase["baujahr"] = pd.to_numeric(Immoscout24AllBase["baujahr"], errors='coerce')
    Immoscout24AllBase["grundstuecksflaeche"] = Immoscout24AllBase["grundstuecksflaeche"].apply(
        lambda row: re.sub('[.m²]', '', row))
    Immoscout24AllBase["grundstuecksflaeche"] = pd.to_numeric(Immoscout24AllBase["grundstuecksflaeche"],
                                                              errors="ignore")
    Immoscout24AllBase["wohnflaeche"] = Immoscout24AllBase["wohnflaeche"].apply(lambda row: re.sub('[m²]', '', row))
    Immoscout24AllBase["wohnflaeche"] = pd.to_numeric(Immoscout24AllBase["wohnflaeche"].str.replace(",", "."),
                                                      errors="ignore")
    Immoscout24AllBase["terrasse"] = Immoscout24AllBase["balkon"].astype(str).apply(
        lambda row: "JA" if "Terrasse" in row else "NEIN")
    Immoscout24AllBase["balkon"] = Immoscout24AllBase["balkon"].astype(str).apply(
        lambda row: "JA" if "Balkon" in row else "NEIN")
    Immoscout24AllBase["vermietet"] = Immoscout24AllBase["vermietet"].astype(str).apply(
        lambda row: "JA" if row == "Vermietet" else "NEIN")

    Immoscout24AllBase["anzahl_parkplatz"] = Immoscout24AllBase["anzahl_parkplatz"].fillna(0)
    Immoscout24AllBase["anzahl_parkplatz"] = Immoscout24AllBase["anzahl_parkplatz"].apply(
        lambda row: re.sub('[\\D]', '', str(row)))
    Immoscout24AllBase["anzahl_parkplatz"] = pd.to_numeric(Immoscout24AllBase["anzahl_parkplatz"])
    Immoscout24AllBase["anzahl_parkplatz"] = Immoscout24AllBase["anzahl_parkplatz"].fillna(1)

    Immoscout24AllBase["energie_verbrauch"] = Immoscout24AllBase["energie_verbrauch"].apply(
        lambda row: re.sub('[^0-9,]', '', str(row)))
    Immoscout24AllBase["energie_verbrauch"] = Immoscout24AllBase["energie_verbrauch"].apply(
        lambda row: re.sub(',', '.', str(row)))
    Immoscout24AllBase["energie_verbrauch"] = pd.to_numeric(Immoscout24AllBase["energie_verbrauch"])

    Immoscout24AllBase = Immoscout24AllBase.reindex(sorted(Immoscout24AllBase.columns), axis=1)

    ImmobilienAll = pd.concat([Immoscout24AllBase, ImmonetBase], axis=0, ignore_index=True, join="inner")

    ImmobilienAll2 = pd.concat([Immoscout24AllBase, ImmonetBase], axis=0, ignore_index=True, join="outer")

    ImmobilienAll2.to_excel(excel_writer="Files/ImmobilienAll2v3.xlsx", sheet_name="ImmobilienAll")

    # Datensatz einlesen
    ImmobilienMaster = pd.read_excel(r"Files/ImmobilienAll2v3.xlsx", index_col="Unnamed: 0")

    # Tausender Stellen - Scraper Fehler -> abgeschnittene Nullen korrigieren
    ImmobilienMaster.loc[ImmobilienMaster["angebotspreis"] <= 10000, "angebotspreis"] = ImmobilienMaster[
                                                                                            "angebotspreis"] * 1000

    # Umbenennungen
    ImmobilienMaster.rename(columns={"befeuerungsart": "energietyp"}, inplace=True)

    # Zeilen ohne Angebotspreis und nutzlose Spalten droppen
    ImmobilienMaster = ImmobilienMaster.dropna(subset=["angebotspreis"])
    ImmobilienMaster = ImmobilienMaster.drop(
        columns=[ 'anzahl_schlafzimmer', 'bezugsfrei ab', "denkmalschutzobjekt", "einbauküche", "immo_url",
                 "energieausweis", "energie­ausweistyp", 'energie_verbrauch', 'etagen', "fahrstuhl", 'geschoss',
                 "grundbucheintrag",
                 "grunderwerbssteuer", 'hausgeld', "maklerprovision",
                 "modernisierung/ sanierung", "monatsmiete", "notarkosten", "ort",
                 "scoutid",
                 "strasse", "web-scraper-start-url", "wohnung-href",
                 "denkmalschutz", "nutzfläche ca", "ausstattung",
                 "ausstattung beschreibung", "lage",
                 "objektbeschreibung", "sonstiges", "wohnung"])

    # Spalten Datentypen bearbeiten
    ImmobilienMaster["balkon"] = ImmobilienMaster["balkon"].astype("category")
    ImmobilienMaster["barrierefrei"] = ImmobilienMaster["barrierefrei"].astype("category")
    ImmobilienMaster["energietyp"] = ImmobilienMaster["energietyp"].astype("category")
    ImmobilienMaster["energie_effizienzklasse"] = ImmobilienMaster["energie_effizienzklasse"].astype("category")
    ImmobilienMaster["gaeste_wc"] = ImmobilienMaster["gaeste_wc"].astype("category")
    ImmobilienMaster["heizung"] = ImmobilienMaster["heizung"].astype("category")
    ImmobilienMaster["immobilienart"] = ImmobilienMaster["immobilienart"].astype("category")
    ImmobilienMaster["immobilienzustand"] = ImmobilienMaster["immobilienzustand"].astype("category")
    ImmobilienMaster["plz"] = ImmobilienMaster["plz"].astype("category")
    ImmobilienMaster["terrasse"] = ImmobilienMaster["terrasse"].astype("category")
    ImmobilienMaster["unterkellert"] = ImmobilienMaster["unterkellert"].astype("category")
    ImmobilienMaster["vermietet"] = ImmobilienMaster["vermietet"].astype("category")
    ImmobilienMaster["aufzug"] = ImmobilienMaster["aufzug"].astype("category")

    # Doppelkategorien rauswerfen und ähnliche Kategorien zusammenfassen
    ImmobilienMaster["energietyp"] = ImmobilienMaster["energietyp"].apply(
        lambda row: str(row).split(",")[0])
    ImmobilienMaster["energietyp"] = ImmobilienMaster["energietyp"].apply(
        lambda row: 'Pellets' if row == "Holzpellets" else row)
    ImmobilienMaster["energietyp"] = ImmobilienMaster["energietyp"].apply(
        lambda row: 'Gas' if row == "Flüssiggas" else row)
    ImmobilienMaster["energietyp"] = ImmobilienMaster["energietyp"].apply(
        lambda row: 'Fernwärme' if row == "Erdwärme" else row)
    ImmobilienMaster["energietyp"] = ImmobilienMaster["energietyp"].apply(
        lambda row: 'Sonstige' if row not in ["", np.nan, "Öl", "Gas", "Fernwärme",
                                              "Luft-/Wasserwärme", "Holz", "Pellets", "Solar", "Strom"] else row)
    ImmobilienMaster["heizung"] = ImmobilienMaster["heizung"].apply(
        lambda row: 'Sonstige' if row not in ["", np.nan, "Zentralheizung", "Etagenheizung", "Ofenheizung",
                                              "Fußbodenheizung"] else row)

    ImmobilienMaster["immobilienart"] = ImmobilienMaster["immobilienart"].apply(
        lambda row: 'Einfamilienhaus' if row == "Einfamilienhaus (freistehend)" else row)

    ImmobilienMaster["immobilienart"] = ImmobilienMaster["immobilienart"].apply(
        lambda row: 'Sonstige' if row not in ["", np.nan, "Einfamilienhaus", "Wohngrundstück", "Wohnung", "Etagenwohnung",
                                               "Sonstiges", "Mehrfamilienhaus", "Erdgeschosswohnung",
                                               "Dachgeschosswohnung",
                                               "Zweifamilienhaus", "Doppelhaushälfte", "Villa", "Reihenmittelhaus",
                                               "Reihenendhaus", "Bungalow",
                                               "Maisonette", "Apartment", "Stadthaus", "Schloss", "Bauernhaus",
                                               "Herrenhaus", "Reiheneckhaus", "Penthouse"] else row)

    ImmobilienMaster["immobilienzustand"] = ImmobilienMaster["immobilienzustand"].apply(
        lambda row: 'Teil- oder vollrenovierungsbedürftig' if row == "Renovierungsbedürftig" else row)
    ImmobilienMaster["immobilienzustand"] = ImmobilienMaster["immobilienzustand"].apply(
        lambda row: 'Teil- oder vollsaniert' if row == "Saniert" else row)
    ImmobilienMaster["immobilienzustand"] = ImmobilienMaster["immobilienzustand"].apply(
        lambda row: 'Teil- oder vollrenoviert' if row == "Renovierungsbedürftig" else row)
    ImmobilienMaster["immobilienzustand"] = ImmobilienMaster["immobilienzustand"].apply(
        lambda row: 'Sonstige' if row not in ["", np.nan, "Unbekannt", "Erstbezug", "Projektiert", "Neubau",
                                              "Teil- oder vollrenovierungsbedürftig", "Neuwertig",
                                              "Teil- oder vollsaniert", "Teil- oder vollrenoviert", "Gepflegt",
                                              "Altbau", "Modernisiert"] else row)



    # Imputation

    # Zufällig mit vorhandenen Werten auffüllen
    ImmobilienMaster.loc[ImmobilienMaster["anzahl_badezimmer"] == 0, "anzahl_badezimmer"] = np.nan
    ImmobilienMaster["anzahl_badezimmer"] = ImmobilienMaster["anzahl_badezimmer"].apply(
        lambda x: np.random.choice(range(1, 4), p=[0.65, 0.30, 0.05]) if np.isnan(x) else x)
    ImmobilienMaster["anzahl_zimmer"] = ImmobilienMaster["anzahl_zimmer"].apply(
        lambda x: np.random.choice(ImmobilienMaster["anzahl_zimmer"].dropna().values) if np.isnan(x) else x)
    ImmobilienMaster["baujahr"] = ImmobilienMaster["baujahr"].apply(
        lambda x: np.random.choice(ImmobilienMaster["baujahr"].dropna().values) if np.isnan(x) else x)

    # ImmobilienMaster["energie_effizienzklasse"] = ImmobilienMaster["energie_effizienzklasse"].cat.codes
    # energie = ImmobilienMaster["energie_effizienzklasse"].value_counts(normalize=True)
    # energie.drop(labels=[-1])
    # energie_types = energie.keys()
    # energie_props = energie.values
    # ImmobilienMaster["energie_effizienzklasse"] = ImmobilienMaster["energie_effizienzklasse"].fillna(lambda x: np.random.choice(energie_types, energie_props) if x == -1 else x)

    ImmobilienMaster["energietyp"] = ImmobilienMaster["energietyp"].astype("category")
    ImmobilienMaster["energietyp"] = ImmobilienMaster["energietyp"].cat.add_categories(
        ["Unbekannt"])
    ImmobilienMaster["energietyp"] = ImmobilienMaster["energietyp"].fillna("Unbekannt")

    ImmobilienMaster["energie_effizienzklasse"] = ImmobilienMaster["energie_effizienzklasse"].cat.add_categories(
        ["Unbekannt"])
    ImmobilienMaster["energie_effizienzklasse"] = ImmobilienMaster["energie_effizienzklasse"].fillna("Unbekannt")

    ImmobilienMaster["heizung"] = ImmobilienMaster["heizung"].astype("category")
    ImmobilienMaster["heizung"] = ImmobilienMaster["heizung"].cat.add_categories(
        ["Unbekannt"])
    ImmobilienMaster["heizung"] = ImmobilienMaster["heizung"].fillna("Unbekannt")

    ImmobilienMaster["immobilienart"] = ImmobilienMaster["immobilienart"].astype("category")
    ImmobilienMaster["immobilienart"] = ImmobilienMaster["immobilienart"].cat.add_categories(
        ["Unbekannt"])
    ImmobilienMaster["immobilienart"] = ImmobilienMaster["immobilienart"].fillna("Unbekannt")

    ImmobilienMaster["immobilienzustand"] = ImmobilienMaster["immobilienzustand"].astype("category")
    ImmobilienMaster["immobilienzustand"] = ImmobilienMaster["immobilienzustand"].cat.add_categories(
        ["Unbekannt"])
    ImmobilienMaster["immobilienzustand"] = ImmobilienMaster["immobilienzustand"].fillna("Unbekannt")


    # Aufzug: Annahme, wenn nicht explizit angegeben, dann existiert kein Aufzug
    ImmobilienMaster.loc[ImmobilienMaster["aufzug"].isna(), "aufzug"] = "NEIN"

    # ScikitLearn Anforderung: Nur numerische Werte - Transformation der kategorischen Spalten
    categorical_mask = (ImmobilienMaster.dtypes == "category")
    categorical_columns = ImmobilienMaster.columns[categorical_mask].tolist()
    category_enc = pd.get_dummies(ImmobilienMaster[categorical_columns], dummy_na=True)
    ImmobilienMaster = pd.concat([ImmobilienMaster, category_enc], axis=1)
    ImmobilienMaster = ImmobilienMaster.drop(columns=categorical_columns)

    # Ausgabe
    ImmobilienMaster = ImmobilienMaster.reset_index()
    # print(ImmobilienMaster.info())

    ImmobilienMaster.to_excel(excel_writer="Files/ImmobilienMasterV4.xlsx", sheet_name="ImmobilienAll")

    # ML Tests
    X = ImmobilienMaster.drop(columns=["angebotspreis"]).values
    y = ImmobilienMaster["angebotspreis"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    xg_reg = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=20, seed=123)
    xg_reg.fit(X_train, y_train)
    preds = xg_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % (rmse))

    #Grid Search parameter Tuning
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

    #Randomized Search parameter tuning
    gbm_param_grid2 = {
        'n_estimators': [25],
        'max_depth': range(2, 12)
    }
    gbm2 = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=10)
    randomized_mse = RandomizedSearchCV(estimator=gbm, param_distributions=gbm_param_grid,
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

    # Display Optionen für Konsole
    # with pd.option_context('display.max_rows', 5, 'display.max_columns', 17):
    #   print(ImmobilienMaster)
