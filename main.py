import sqlite3
from sqlite3 import Error

import data_modeling as dm
# import gui
# import machine_learning as ml
import webscraper as ws


def setup_database(path):
    db_connection = None
    try:
        db_connection = sqlite3.connect(path)
        print("Connection to SQLite DB successful!")
    except Error as e:
        print(f"The error '{e}' occurred!")

    return db_connection


def main():
    # Set up database
    print("Step 1: Set up database...")

    db_connection = setup_database(r"Datenbank/ImmoDB.db")
    db_cursor = db_connection.cursor()

    # Read input data
    print("Step 2: Read in data...")

    immonet_data = dm.read_data_from_immonet()
    # immonet_data.to_sql(name='Immonet_data_raw', con=db_connection)
    # Alternative für später:
    # immonet_data.to_sql(name='Immonet_data_raw', con=db_connection, if_exists = 'append oder replace oder fail')

    immoscout_data = dm.read_data_from_immoscout()
    # immoscout_data.to_sql(name='Immoscout_data_raw', con=db_connection)

    geo_data = dm.read_geo_data()
    # geo_data.to_sql(name='Geo_data_raw', con=db_connection)

    inhabitants_data = dm.read_data_from_inhabitants()
    # inhabitants_data.to_sql(name='Inhabitants_data_raw', con=db_connection)

    # Merge input data
    print("Step 3: Merge data...")

    immonet_data_geo_inh = dm.add_geo_inhabitants_immonet(immonet_data, geo_data, inhabitants_data)
    immoscout_data_geo_inh = dm.add_geo_inhabitants_immoscout(immoscout_data, geo_data, inhabitants_data)

    merged_data = dm.merge_data(immonet_data_geo_inh, immoscout_data_geo_inh)
    merged_data.to_csv("Files/Tests/merged_data.csv")

    # Preprocessing
    print("Step 4: Preprocess data...")

    preprocessed_data = dm.preprocess_data(merged_data)
    preprocessed_data.to_csv("Files/Tests/preprocessed_data.csv")

    # EDA
    # print("Step 5: EDA...")

    # eda(preprocessed_data)

    # Imputation
    print("Step 6: Impute data...")

    imputed_data = dm.impute_data(preprocessed_data)
    imputed_data.to_csv("Files/Tests/imputed_data.csv")

    # Machine Learning
    # print("Step 7: Machine learning tests...")

    # ml_tests(imputed_data)

    # Testausgaben
    # print("Optional: Create Excel files...")

    # immonet_data.to_excel(excel_writer="Files/Tests/ImmoscoutDataTest.xlsx", sheet_name="Immobilien")
    # immoscout_data.to_excel(excel_writer="Files/Tests/ImmoscoutDataTest.xlsx", sheet_name="Immobilien")
    # geo_data.to_excel(excel_writer="Files/Tests/GeoDataTest.xlsx", sheet_name="Geodaten")
    # inhabitants_data.to_excel(excel_writer="Files/Tests/InhabitantsDataTest.xlsx", sheet_name="Einwohner")
    #
    # immonet_data_geo_inh.to_excel(excel_writer="Files/Tests/ImmoscoutDataGeoInhTest.xlsx", sheet_name="Immobilien")
    # immoscout_data_geo_inh.to_excel(excel_writer="Files/Tests/ImmoscoutDataGeoInhTest.xlsx", sheet_name="Immobilien")
    #
    # merged_data.to_excel(excel_writer="Files/Tests/merged_data.xlsx", sheet_name="Immobilien")

    print("... done.")


if __name__ == "__main__":
    main()
