import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a_default_secret_key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///nutriscore.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    DB_NAME = "nutriscore.db"
    DB_FULL_PATH = "app/static/" + DB_NAME
    TABLE_NAME = "produits"
    ORIGINAL_CSV_NAME = "en.openfoodfacts.org.products.csv"
    CLEANED_CSV_NAME = "openfoodfact_clean.csv"
    CSV_FULL_PATH = "app/static/" + CLEANED_CSV_NAME
    CHUNK_SIZE = 10000
    VIEW_NAME = 'products_view'
    SELECTED_COLS = [
        "code",
        "product_name",
        "quantity",
        "brands",
        "categories",
        "ingredients_text",
        "nutriscore_score",
        "nutriscore_grade",
        "energy-kj_100g",
        "energy-kcal_100g",
        "fat_100g",
        "saturated-fat_100g",
        "omega-3-fat_100g",
        "omega-6-fat_100g",
        "sugars_100g",
        "added-sugars_100g",
        "fiber_100g",
        "proteins_100g",
        "salt_100g",
        "fruits-vegetables-nuts-estimate-from-ingredients_100g",
        "countries",
        "countries_tags",
        "countries_en"
    ]
    COLS_STAT = [
        "nutriscore_score",
        "nutriscore_grade",
        "energy-kj_100g",
        "energy-kcal_100g",
        "fat_100g",
        "saturated-fat_100g",
        "omega-3-fat_100g",
        "omega-6-fat_100g",
        "sugars_100g",
        "added-sugars_100g",
        "fiber_100g",
        "proteins_100g",
        "salt_100g",
        "fruits-vegetables-nuts-estimate-from-ingredients_100g"
    ]
    DIRECTORY_PATH = "../static/"
    FILE_NAME = "en.openfoodfacts.org.products.csv"
    OUTPUT_NAME = 'openfoodfact_clean.csv'
    NUTRI_OK = ["a", "b", "c", "d", "e"]
