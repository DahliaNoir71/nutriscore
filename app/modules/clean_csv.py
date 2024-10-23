import pandas as pd
from config import Config as cfg

def read_csv_chunks(file_path, selected_columns, chunk_size):
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
    print("\nDébut de script clean_csv")
    file_path = cfg.DIRECTORY_PATH + cfg.FILE_NAME
    chunks = read_csv_chunks(file_path,
                             cfg.SELECTED_COLS,
                             cfg.CHUNK_SIZE)
    clean_data = filter_and_clean_data(chunks,
                                       cfg.SELECTED_COLS,
                                       cfg.COLS_STAT,
                                       cfg.NUTRI_OK)
    output_name = cfg.DIRECTORY_PATH + cfg.OUTPUT_NAME
    clean_data.to_csv(output_name, sep='\t', index=False)
    print("\nFin de script clean_csv")
