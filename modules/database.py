import os
import pandas as pd
from sqlalchemy import create_engine

DB_NAME = "nutriscore.db"
TABLE_NAME = "produits"
DB_FULL_PATH = "db/" + DB_NAME
CSV_NAME = "en.openfoodfacts.org.products.csv"
CSV_FULL_PATH = "csv/" + CSV_NAME
CHUNK_SIZE = 10000

def create_db_from_csv():
    # Crée une connexion à la base de données SQLite
    engine = create_engine('sqlite:///' + DB_FULL_PATH)
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

def set_database():
    if not os.path.isfile(DB_FULL_PATH):
        create_db_from_csv()