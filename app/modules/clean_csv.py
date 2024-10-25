import pandas as pd
from config import Config
from app.modules.analyse_countries import clean_countries

def read_csv_chunks(file_path, selected_columns, chunk_size):
    """
    :param file_path: The path to the CSV file to be read.
    :param selected_columns: A list of columns to be selected from each chunk of the CSV file.
    :param chunk_size: The number of rows per chunk to be read from the CSV file.
    :return: A list of pandas DataFrame chunks, each containing only the selected columns.
    """
    print("\nLecture du fichier CSV par chunks")
    # Initialisation de la liste pour stocker les morceaux sélectionnés
    selected_chunks = []

    # Initialisation du compteur
    chunk_counter = 0

    if selected_columns != []:
        # Lecture du fichier CSV par morceaux
        for chunk in pd.read_csv(file_path,
                                sep="\t",
                                low_memory=False,
                                header=0,
                                chunksize=chunk_size,
                                on_bad_lines="skip",
                                usecols=selected_columns):
            selected_chunks.append(chunk)

            # Incrémentation et affichage du compteur
            chunk_counter += 1
            print(f"Morceau {chunk_counter} traité")
    else:
        # Lecture du fichier CSV par morceaux
        for chunk in pd.read_csv(file_path,
                                sep="\t",
                                low_memory=False,
                                header=0,
                                chunksize=chunk_size,
                                on_bad_lines="skip"):
            selected_chunks.append(chunk)

            # Incrémentation et affichage du compteur
            chunk_counter += 1
            print(f"Morceau {chunk_counter} traité")

    return selected_chunks

def filter_and_clean_data(dataframes, selected_columns, cols_stat, nutri_ok):
    """
    Filters and cleans multiple DataFrames based on specified columns and criteria.

    :param dataframes: List of pandas DataFrame objects to be filtered and cleaned.
    :param selected_columns: List of column names to retain in the DataFrame after filtering.
    :param cols_stat: List of column names where missing values should be filled with 0.
    :param nutri_ok: List of acceptable 'nutriscore_grade' values to retain in the filtered DataFrame.
    :return: Concatenated and cleaned DataFrame comprising only the specified columns and filtered by acceptable 'nutriscore_grade' values.
    """
    
    print("\nFiltrage et nettoyage des datas")

    # Filter rows where 'nutriscore_score' and 'nutriscore_grade' are not missing, and select specified columns
    list_df_not_na = [
        df[df[['nutriscore_score', 'nutriscore_grade']].notna().all(axis=1)][selected_columns]
        for df in dataframes if len(df) > 0  # Only process non-empty DataFrames
    ]
    
    # Concatenate all filtered DataFrames into a single DataFrame
    df_not_na = pd.concat(list_df_not_na, ignore_index=True)

    # Keep only rows with 'nutriscore_grade' values that are in the acceptable list (nutri_ok)
    df_not_na = df_not_na[df_not_na["nutriscore_grade"].isin(nutri_ok)]

    # Replace any missing values in columns listed in cols_stat with 0
    df_not_na[cols_stat] = df_not_na[cols_stat].fillna(0)

    # Clean the country names
    df_not_na = clean_countries(df_not_na, Config.COUNTRIES_EN_COL)

    return df_not_na

def read_and_clean_csv():
    """
    Cleans a CSV file by reading it in chunks, filtering, and cleaning the data according to specified configurations,
    and then outputting the cleaned data to a new CSV file.

    :return: None
    """
    
    print("\nDébut de script clean_csv")
    
    # Define the path to the original CSV file using configuration settings
    file_path = Config.DIRECTORY_PATH + Config.ORIGINAL_CSV_NAME

    # Read the CSV file in chunks using pre-defined configurations (columns to select and chunk size)
    chunks = read_csv_chunks(file_path,
                             Config.SELECTED_COLS,
                             Config.CHUNK_SIZE)

    # Filter and clean the data chunks based on selected columns, columns to fill with 0, and acceptable 'nutriscore_grade' values
    clean_data = filter_and_clean_data(chunks,
                                       Config.SELECTED_COLS,
                                       Config.COLS_STAT,
                                       Config.NUTRI_OK)

    # Define the path to save the cleaned CSV file using configuration settings
    cleaned_file_path = Config.DIRECTORY_PATH + Config.CLEANED_CSV_NAME

    # Save the cleaned DataFrame to a new CSV file with tab-separated values and without the index column
    clean_data.to_csv(cleaned_file_path, sep='\t', index=False)

    print("\nFin de script clean_csv")

#if __name__ == '__main__':
#    read_and_clean_csv()
