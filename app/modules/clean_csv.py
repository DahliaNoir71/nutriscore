import pandas as pd
from tqdm import tqdm
from config import Config
from analyse_countries import clean_countries

def read_csv_chunks(file_path, selected_columns, chunk_size):
    """
    :param file_path: The path to the CSV file to be read.
    :param selected_columns: A list of column names to be read from the CSV file.
    :param chunk_size: The number of rows per chunk to be read from the CSV file.
    :return: A list of DataFrame chunks, each containing the selected columns from the CSV file.
    """

    # Initialisation de la liste pour stocker les morceaux sélectionnés
    selected_chunks = []

    # Lire le fichier CSV en chunks avec une barre de progression
    chunk_iter = pd.read_csv(file_path,
                             sep="\t",
                             low_memory=False,
                             header=0,
                             chunksize=chunk_size,
                             on_bad_lines="skip",
                             usecols=selected_columns)

    with tqdm(desc="Lecture de CSV", unit='chunk') as pbar:
        for chunk in chunk_iter:
            selected_chunks.append(chunk)
            # Mise à jour de la barre de progression
            pbar.update(1)
            # (Optionnel) Afficher des informations supplémentaires dans la barre de progression
            pbar.set_postfix(rows=chunk.shape[0])

    return selected_chunks

def filter_and_clean_data(dataframes, selected_columns, cols_stat, nutri_ok):
    """
    :param dataframes: A list of pandas DataFrames to be filtered and cleaned.
    :param selected_columns: A list of column names to be retained in the filtered DataFrames.
    :param cols_stat: A list of column names, for which NaN values will be filled with 0.
    :param nutri_ok: A list of acceptable nutriscore grades to filter the data.
    :return: A cleaned and filtered DataFrame with specified columns and non-NaN values.
    """
    print("\nFiltrage et nettoyage des datas")
    list_df_not_na = [
        df[df[['nutriscore_score', 'nutriscore_grade']].notna().all(axis=1)][selected_columns]
        for df in dataframes if len(df) > 0
    ]
    df_not_na = pd.concat(list_df_not_na, ignore_index=True)
    df_not_na = df_not_na[df_not_na["nutriscore_grade"].isin(nutri_ok)]
    df_not_na[cols_stat] = df_not_na[cols_stat].fillna(0)

    return df_not_na

def clean_csv():
    """
    Reads, cleans, and writes a CSV file.

    :return: None
    """
    file_path = Config.DIRECTORY_PATH + Config.ORIGINAL_CSV_NAME
    chunks = read_csv_chunks(file_path,
                             Config.SELECTED_COLS,
                             Config.CHUNK_SIZE)
    clean_data = filter_and_clean_data(chunks,
                                       Config.SELECTED_COLS,
                                       Config.COLS_STAT,
                                       Config.NUTRI_OK)
    clean_data = clean_countries(clean_data, Config.COUNTRIES_EN_COL)
    cleaned_file_path = Config.DIRECTORY_PATH + Config.CLEANED_CSV_NAME
    clean_data.to_csv(cleaned_file_path, sep='\t', index=False)

if __name__ == '__main__':
    clean_csv()
