import os
import pandas as pd
from sqlalchemy import create_engine
from config import Config
from tqdm import tqdm

def get_db_engine():
    """
    Creates and returns a SQLAlchemy engine connected to an SQLite database.

    :param None: This function does not take any parameters.

    :return: SQLAlchemy engine connected to the specified SQLite database.
             The engine is used to interact with the SQLite database.
    """
    engine = create_engine('sqlite:///' + Config.DB_FULL_PATH)
    return engine

def create_db_from_csv():
    """
    Creates a SQLite database and populates it with data from a CSV file.

    :return: None

    The function reads the CSV file in chunks, specified by the `CHUNK_SIZE` in the `Config` class,
    and inserts each chunk into the SQLite database. It uses the `tqdm` library to display a progress bar
    during the CSV file reading process.

    Parameters:
    None

    Raises:
    None
    """
    print("create_db_from_csv")
    # Crée une connexion à la base de données SQLite
    engine = get_db_engine()
    # Insertion du CSV par lots
    count = 1
    chunk_iter = pd.read_csv(Config.ORIGINAL_CSV_FULL_PATH,
                             chunksize=Config.CHUNK_SIZE,
                             low_memory=False,
                             on_bad_lines='skip',
                             sep='\t',
                             dtype=str)
    with tqdm(desc="Lecture du CSV " + Config.ORIGINAL_CSV_FULL_PATH, unit='chunk') as pbar:
        for chunk in chunk_iter:
            chunk.to_sql(Config.TABLE_NAME, con=engine, if_exists='append', index=False)
            # Mise à jour de la barre de progression
            pbar.update(1)
            # (Optionnel) Afficher des informations supplémentaires dans la barre de progression
            pbar.set_postfix(rows=chunk.shape[0])



def set_db_nutriscore():
    """
    This function checks if the database file exists. If not, it creates the database from a CSV file.

    :return: None

    The function first checks if the SQLite database file specified by `Config.DB_FULL_PATH` exists.
    If the file does not exist, it calls the `create_db_from_csv()` function to create the database and populate it with data from the CSV file.
    If the file already exists, it prints a message indicating that the database has already been created.

    Parameters:
    None

    Raises:
    None
    """
    if not os.path.isfile(Config.DB_FULL_PATH):
        create_db_from_csv()
    else:
        print("Base de données " + Config.DB_NAME + " déjà créée")