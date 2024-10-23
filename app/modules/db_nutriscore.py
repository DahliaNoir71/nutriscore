import os
import pandas as pd
from sqlalchemy import create_engine, text

DB_NAME = "nutriscore.db"
DB_FULL_PATH = "app/static/" + DB_NAME
TABLE_NAME = "produits"
#CSV_NAME = "en.openfoodfacts.org.products.csv"
CSV_NAME = "openfoodfact_clean.csv"
CSV_FULL_PATH = "app/static/" + CSV_NAME
CHUNK_SIZE = 10000
VIEW_NAME = 'products_view'

def get_db_engine():
    """
    Creates and returns a SQLAlchemy engine connected to an SQLite database

    :return: SQLAlchemy engine connected to the specified SQLite database
    """
    engine = create_engine('sqlite:///' + DB_FULL_PATH)
    return engine

def create_db_from_csv():
    """
    Creates a SQLite database and populates it with data from a CSV file.

    :return: None
    """
    # Crée une connexion à la base de données SQLite
    engine = get_db_engine()
    # Insertion du CSV par lots
    count = 1
    for chunk in pd.read_csv(CSV_FULL_PATH,
                             chunksize=CHUNK_SIZE,
                             low_memory=False,
                             on_bad_lines='skip',
                             sep='\t',
                             dtype=str):
        chunk.to_sql(TABLE_NAME, con=engine, if_exists='append', index=False)
        print('\n', count, '')
        print(f"Insertion d'un chunk de {len(chunk)} lignes")
        count += 1

def set_db_nutriscore():
    """
    Checks if the database file exists; if not, creates the database from a CSV file.

    :return: None
    """
    if not os.path.isfile(DB_FULL_PATH):
        create_db_from_csv()
    else:
        print("Base de données " + DB_NAME + " déjà créée")