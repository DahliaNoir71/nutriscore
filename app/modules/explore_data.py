import pandas as pd
from config import Config
from app.modules.clean_csv import read_csv_chunks

def load_dataframe():
    print("\nload_dataframe")
    products = None
    list_df = []

    # Import CSV into a DFs by chunks
    list_df = read_csv_chunks(Config.ORIGINAL_CSV_FULL_PATH, [], Config.CHUNK_SIZE)

    # Concatenate chunks in one DF
    products = pd.concat(list_df, ignore_index=True)

    # Preview results
    #print(products.head(15))

    return products

def display_graphs():
    print('Todo')