import pandas as pd
from config import Config
from flask import current_app
from app.modules.clean_csv import read_csv_chunks
import logging

def load_dataframe():
    print("\nload_dataframe")
    products = None
    list_df = []

    try:
        # Import CSV into a DFs by chunks
        list_df = read_csv_chunks(Config.CLEANED_CSV_FULL_PATH, [], Config.CHUNK_SIZE)

        # Concatenate chunks in one DF
        products = pd.concat(list_df, ignore_index=True)

        # Save DF in app config within the app context
        with current_app.app_context():
            current_app.config['PRODUCTS_DF'] = products
            current_app.config['loading_dataframe_status']['complete'] = True
            logging.debug("Data loading completed and status set to True.")

    except Exception as e:
        # If an error occurs, ensure the status is updated to indicate failure
        with current_app.app_context():
            current_app.config['loading_dataframe_status']['complete'] = False
            logging.error(f"Error in load_dataframe: {e}")

    return products
