import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from category_encoders import TargetEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from xgboost import plot_importance

import main

def print_feature_importances(model, data, save_string):
    importances = pd.Series(data=model.feature_importances_,
                            index=data.columns)
    importances_sorted = importances.sort_values()[:10]
    importances_sorted.plot(kind='barh', color='lightgreen')
    plt.title('Features Importances')
    fig = plt.gcf()
    fig.set_size_inches(17.5, 8)
    plt.savefig(save_string)
# Entfernen der Ausreisser
def outlier_treatment(datacolumn):
    sorted(datacolumn)
    q1, q3 = np.percentile(datacolumn, [25, 75])
    iqr = q3 - q1
    lower_range = q1 - (1.5 * iqr)
    upper_range = q3 + (1.5 * iqr)
    return lower_range, upper_range

def outlier_drop(imputed_data):
    l, u = outlier_treatment(imputed_data.angebotspreis)
    indexnames = imputed_data[imputed_data['angebotspreis'] > u].index
    imputed_data.drop(indexnames, inplace=True)
    return imputed_data

# Alle JA/NEIN Variablen in 1/0
def boolean(imputed_data):
    imputed_data = imputed_data.assign(aufzug=(imputed_data['aufzug'] == 'JA').astype(int))
    imputed_data = imputed_data.assign(barrierefrei=(imputed_data['barrierefrei'] == 'JA').astype(int))
    imputed_data = imputed_data.assign(gaeste_wc=(imputed_data['gaeste_wc'] == 'JA').astype(int))
    imputed_data = imputed_data.assign(terrasse_balkon=(imputed_data['terrasse_balkon'] == 'JA').astype(int))
    imputed_data = imputed_data.assign(unterkellert=(imputed_data['unterkellert'] == 'JA').astype(int))
    imputed_data = imputed_data.assign(vermietet=(imputed_data['vermietet'] == 'JA').astype(int))
    imputed_data = imputed_data.assign(supermarkt_im_plz_gebiet=(imputed_data['Supermarkt im PLZ Gebiet'] == 'JA').astype(int))
    imputed_data.drop(columns=['Supermarkt im PLZ Gebiet'], inplace=True)
    imputed_data['plz'] = imputed_data['plz'].astype(int)
    return imputed_data

def variables(imputed_data):
    for col in ['wohnflaeche', 'anzahl_zimmer']:
        val = imputed_data[col].mean()
        imputed_data[col] = imputed_data[col].replace(0.0, val)
    #imputed_data['zimmergr????e'] = (imputed_data['wohnflaeche'] / imputed_data['anzahl_zimmer']).round(2)

    #mean_plz = imputed_data.groupby('plz')['angebotspreis'].mean().round(2)
    #imputed_data['plz'] = imputed_data['plz'].map(mean_plz)
    return imputed_data


# Train Test Split durchf??hren
def tr_te_spl(imputed_data):
    x = imputed_data.drop(columns='angebotspreis')
    y = imputed_data['angebotspreis']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_test, x_train, y_test, y_train

# Sample mit nur nummerischen Daten erzeugen
def numeric(x_train, x_test):
    x_train_num = x_train.drop(columns=['energietyp', 'energie_effizienzklasse',
                                        'heizung', 'immobilienart', 'immobilienzustand', 'Grad_der_Verst??dterung',
                                        'sozio??konomische_Lage'])
    x_val_num = x_test.drop(columns=['energietyp', 'energie_effizienzklasse',
                                     'heizung', 'immobilienart', 'immobilienzustand', 'Grad_der_Verst??dterung',
                                     'sozio??konomische_Lage'])
    return x_train_num, x_val_num

# Normalisierung der numerischen Daten (Als Alternative zur Standardisierung)
def normalisation(x_train_num, x_val_num):
    scaler = MinMaxScaler()

    x_train_num = pd.DataFrame(scaler.fit_transform(x_train_num),
                               columns=x_train_num.columns, index=x_train_num.index)
    x_val_num = pd.DataFrame(scaler.transform(x_val_num),
                             columns=x_train_num.columns, index=x_val_num.index)
    return x_train_num, x_val_num

# Standardisierung der numerischen Daten (Als alternative zur Normalisierung)
def standardization(x_train_num, x_val_num):
    num_scaler = StandardScaler()
    num_scaler.fit(x_train_num)
    pickle.dump(num_scaler, open('num_scaler.pckl', 'wb'))

    x_train_num = pd.DataFrame(num_scaler.transform(x_train_num),
                               columns=x_train_num.columns, index=x_train_num.index)
    x_val_num = pd.DataFrame(num_scaler.transform(x_val_num),
                             columns=x_train_num.columns, index=x_val_num.index)
    return x_train_num, x_val_num

# Sample mit nur kategorischen Variablen erzeugen (Mehr als Zwei Kategorien)
def category(x_train, x_test):
    x_train_cat = x_train[['energietyp', 'energie_effizienzklasse', 'heizung', 'immobilienart', 'immobilienzustand',
                           'Grad_der_Verst??dterung', 'sozio??konomische_Lage']]
    x_val_cat = x_test[['energietyp', 'energie_effizienzklasse', 'heizung', 'immobilienart', 'immobilienzustand',
                        'Grad_der_Verst??dterung', 'sozio??konomische_Lage']]
    return x_train_cat, x_val_cat

# Kategorische Variablen Target Encoden
def target_encoding(x_train_cat, x_val_cat, y_train):
    target_encoder = TargetEncoder()
    scaler = StandardScaler()

    x_train_reference = x_train_cat
    x_train_target = target_encoder.fit_transform(x_train_cat, y_train)
    x_train_target = pd.DataFrame(scaler.fit_transform(x_train_target),
                 columns=x_train_target.columns, index=x_train_target.index)

    x_train_reference = x_train_reference.join(x_train_target.add_suffix("_targetenc"))

    #Referenztabellen mit Encodings erstellen
    energietyp = x_train_reference[['energietyp', 'energietyp_targetenc']]
    energietyp = energietyp.drop_duplicates(subset=['energietyp'])
    energietyp.to_sql(name='Encoding_energietyp', con=main.setup_database(r"Datenbank/ImmoDB.db"), if_exists='replace')

    energie_effizienzklasse = x_train_reference[['energie_effizienzklasse', 'energie_effizienzklasse_targetenc']]
    energie_effizienzklasse = energie_effizienzklasse.drop_duplicates(subset=['energie_effizienzklasse'])
    energie_effizienzklasse.to_sql(name='Encoding_energie_effizienzklasse', con=main.setup_database(r"Datenbank/ImmoDB.db"), if_exists='replace')

    heizung = x_train_reference[['heizung', 'heizung_targetenc']]
    heizung = heizung.drop_duplicates(subset=['heizung'])
    heizung.to_sql(name='Encoding_heizung', con=main.setup_database(r"Datenbank/ImmoDB.db"), if_exists='replace')

    immobilienart = x_train_reference[['immobilienart', 'immobilienart_targetenc']]
    immobilienart = immobilienart.drop_duplicates(subset=['immobilienart'])
    immobilienart.to_sql(name='Encoding_immobilienart', con=main.setup_database(r"Datenbank/ImmoDB.db"), if_exists='replace')

    immobilienzustand = x_train_reference[['immobilienzustand', 'immobilienzustand_targetenc']]
    immobilienzustand = immobilienzustand.drop_duplicates(subset=['immobilienzustand'])
    immobilienzustand.to_sql(name='Encoding_immobilienzustand', con=main.setup_database(r"Datenbank/ImmoDB.db"), if_exists='replace')

    Grad_der_Verst??dterung = x_train_reference[['Grad_der_Verst??dterung', 'Grad_der_Verst??dterung_targetenc']]
    Grad_der_Verst??dterung = Grad_der_Verst??dterung.drop_duplicates(subset=['Grad_der_Verst??dterung'])
    Grad_der_Verst??dterung.to_sql(name='Encoding_Grad_der_Verst??dterung', con=main.setup_database(r"Datenbank/ImmoDB.db"), if_exists='replace')

    sozio??konmische_Lage = x_train_reference[['sozio??konomische_Lage', 'sozio??konomische_Lage_targetenc']]
    sozio??konmische_Lage = sozio??konmische_Lage.drop_duplicates(subset=['sozio??konomische_Lage'])
    sozio??konmische_Lage.to_sql(name='Encoding_sozio??konmische_Lage', con=main.setup_database(r"Datenbank/ImmoDB.db"), if_exists='replace')

    x_val_target = target_encoder.transform(x_val_cat)
    x_val_target= pd.DataFrame(scaler.fit_transform(x_val_target),
                 columns=x_val_target.columns, index=x_val_target.index)
    return x_train_target, x_val_target

# Zusammenf??hrung kategorischer und numerischer Varibalen + Speicherung unter Standart Variablennamen
def joint(x_train_num, x_train_target, x_val_num, x_val_target):
    x_train = x_train_num.join(x_train_target.add_suffix("_targetenc"))
    x_test = x_val_num.join(x_val_target.add_suffix("_targetenc"))
    return x_train, x_test


def ml_tests(x_train, x_test, y_train, y_test, imputed_data):

    # XGBoost Standardmodell

    xg_reg = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=50, seed=123)
    xg_reg.fit(x_train, y_train)
    preds = xg_reg.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % rmse)
    print()

    datestr = time.strftime("%Y%m%d-%H%M")

    xg_reg_file = 'XGB_Standardmodell.pckl'
    with open(xg_reg_file, 'wb') as f:
        pickle.dump(xg_reg, f)

    plot_importance(xg_reg, max_num_features=10)
    fig = plt.gcf()
    fig.set_size_inches(17.5, 8 )
    plt.savefig('Files/Feature_Importances_Grafiken/xgb_feature_importances.jpg')

    # Grid Search parameter Tuning
    print("Grid Search Parameter Tuning:")
    gbm_param_grid = {
        'colsample_bytree': [0.3, 0.7],
        'n_estimators': [25, 50, 80, 100],
        'max_depth': [2, 5, 7]
    }
    gbm = xgb.XGBRegressor(objective="reg:squarederror")
    grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid, scoring="neg_mean_squared_error", cv=4, verbose=1)
    grid_mse.fit(x_train, y_train)
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
    randomized_mse.fit(x_train, y_train)
    print("Best parameters found: ", randomized_mse.best_params_)
    print("Lowest RMSE Randomized Search found: ", np.sqrt(np.abs(randomized_mse.best_score_)))

    dm_train = xgb.DMatrix(data=x_train, label=y_train)
    dm_test = xgb.DMatrix(data=x_test, label=y_test)
    params = {"booster": "gblinear", "objective": "reg:squarederror"}
    xg_reg2 = xgb.train(dtrain=dm_train, params=params, num_boost_round=15)
    preds2 = xg_reg2.predict(dm_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds2))
    print("RMSE: %f" % rmse)

    reg_params = [0.1, 0.3, 0.7, 1, 10, 100]
    params1 = {"objective": "reg:squarederror", "max_depth": 3}
    rmses_l2 = []
    for reg in reg_params:
        params1["lambda"] = reg
        cv_results_rmse = xgb.cv(dtrain=dm_train, params=params1, nfold=3, num_boost_round=15, metrics="rmse",
                                 as_pandas=True)
        rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])

    print("Best rmse as a function of l2:")
    print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))
    print()

    #print_feature_importances(model=xg_reg2, data=imputed_data.drop(columns=["angebotspreis"]))

    # Stochastic Gradient Boosting
    print("Stochastic Gradient Boosting:")
    sgbr = GradientBoostingRegressor(max_depth=4,
                                     subsample=0.9,
                                    max_features=0.75,
                                     n_estimators=200,
                                     random_state=2)

    sgbr.fit(x_train, y_train)
    y_pred = sgbr.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE: %f" % rmse)
    print()

    sgbr_file = 'sgbr_Standardmodell.pckl'
    with open(sgbr_file, 'wb') as f:
        pickle.dump(sgbr, f)

    print_feature_importances(model=sgbr, data=imputed_data.drop(columns=["angebotspreis"]), save_string='Files/Feature_Importances_Grafiken/sgbr_feature_importances.jpg')

    # Random Forrest
    print("Random Forrest:")
    rf = RandomForestRegressor(n_estimators=25,
                               random_state=2)
    rf.fit(x_train, y_train)
    y_pred2 = rf.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred2))
    print("RMSE: %f" % rmse)
    print()

    rf_file = 'rf_Standardmodell.pckl'
    with open(rf_file, 'wb') as f:
        pickle.dump(rf, f)

    print_feature_importances(model=rf, data=imputed_data.drop(columns=["angebotspreis"]), save_string='Files/Feature_Importances_Grafiken/rf_feature_importances.jpg')
