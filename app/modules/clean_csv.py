import pandas as pd
from config import Config

def read_csv_chunks(file_path, selected_columns, chunk_size):
    """
    :param file_path: The path to the CSV file to be read.
    :param selected_columns: A list of columns to be selected from each chunk of the CSV file.
    :param chunk_size: The number of rows per chunk to be read from the CSV file.
    :return: A list of pandas DataFrame chunks, each containing only the selected columns.
    """
    print("\nLecture du fichier CSV par chunks")
    chunks = pd.read_csv(file_path,
                         sep="\t",
                         low_memory=False,
                         header=0,
                         chunksize=chunk_size,
                         on_bad_lines="skip")
    selected_chunks = [chunk[selected_columns] for chunk in chunks]
    return selected_chunks

def filter_and_clean_data(dataframes, selected_columns, cols_stat, nutri_ok):
    """
    :param dataframes: List of pandas DataFrame objects to be filtered and cleaned.
    :param selected_columns: List of column names to retain in the DataFrame after filtering.
    :param cols_stat: List of column names where missing values should be filled with 0.
    :param nutri_ok: List of acceptable 'nutriscore_grade' values to retain in the filtered DataFrame.
    :return: Concatenated and cleaned DataFrame comprising only the specified columns and filtered by acceptable 'nutriscore_grade' values.
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
    This function cleans a CSV file by reading it in chunks, filtering and cleaning the data according to specified configurations, and then outputting the cleaned data to a new CSV file.

    :return: None
    """
    print("\nDébut de script clean_csv")
    file_path = Config.DIRECTORY_PATH + Config.FILE_NAME
    chunks = read_csv_chunks(file_path,
                             Config.SELECTED_COLS,
                             Config.CHUNK_SIZE)
    clean_data = filter_and_clean_data(chunks,
                                       Config.SELECTED_COLS,
                                       Config.COLS_STAT,
                                       Config.NUTRI_OK)
    output_name = Config.DIRECTORY_PATH + Config.OUTPUT_NAME
    clean_data.to_csv(output_name, sep='\t', index=False)
    print("\nFin de script clean_csv")
