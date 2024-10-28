import pandas as pd
from config import Config
from app.modules.clean_csv import read_csv_chunks

def load_dataframe():
    """
    Load a DataFrame from a large CSV file by reading it in chunks.

    This function reads a CSV file specified by the `Config.ORIGINAL_CSV_FULL_PATH` constant,
    using the `read_csv_chunks` function from the `app.modules.clean_csv` module. The CSV file
    is read in chunks of size specified by the `Config.CHUNK_SIZE` constant. Each chunk is stored
    in a list of DataFrames, which are then concatenated into a single DataFrame.

    Parameters:
    None

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the CSV file.
    """
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
    # Display box charts for columns containing _100g in their name
    print("\ndisplay_graphs")