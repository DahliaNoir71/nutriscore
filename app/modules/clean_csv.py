import pandas as pd

# Constantes
SELECTED_COLS = [
    "code", "product_name", "quantity", "brands", "categories",
    "ingredients_text", "nutriscore_score", "nutriscore_grade",
    "energy-kj_100g", "energy-kcal_100g", "fat_100g",
    "saturated-fat_100g", "omega-3-fat_100g", "omega-6-fat_100g",
    "sugars_100g", "added-sugars_100g", "fiber_100g",
    "proteins_100g", "salt_100g",
    "fruits-vegetables-nuts-estimate-from-ingredients_100g"
]

COLS_STAT = [
    "nutriscore_score", "nutriscore_grade", "energy-kj_100g",
    "energy-kcal_100g", "fat_100g", "saturated-fat_100g",
    "omega-3-fat_100g", "omega-6-fat_100g", "sugars_100g",
    "added-sugars_100g", "fiber_100g", "proteins_100g",
    "salt_100g", "fruits-vegetables-nuts-estimate-from-ingredients_100g"
]

DIRECTORY_PATH = "../static/"
FILE_NAME = "en.openfoodfacts.org.products.csv"
OUTPUT_NAME = 'openfoodfact_clean.csv'
CHUNK_SIZE = 10000
NUTRI_OK = ["a", "b", "c", "d", "e"]


def read_csv_chunks(file_path, selected_columns, chunk_size):
    print("\nLecture du fichier CSV par chunks")
    chunks = pd.read_csv(file_path,
                         sep="\t",
                         low_memory=False,
                         header=0,
                         chunksize=chunk_size,
                         on_bad_lines="skip")
    if selected_columns != []:
        selected_chunks = [chunk[selected_columns] for chunk in chunks]
    else:
        selected_chunks = [chunk for chunk in chunks]
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


def main():
    print("\nDébut de script clean_csv")
    file_path = DIRECTORY_PATH + FILE_NAME
    chunks = read_csv_chunks(file_path, SELECTED_COLS, CHUNK_SIZE)
    clean_data = filter_and_clean_data(chunks, SELECTED_COLS, COLS_STAT, NUTRI_OK)
    clean_data.to_csv(DIRECTORY_PATH + OUTPUT_NAME, sep='\t', index=False)


if __name__ == "__main__":
    main()
