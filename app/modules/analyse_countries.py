import requests
from config import Config

def fetch_countries_en():
    """
    Fetches a list of country names in English from a specified API URL.

    :return: List of country names in English if the response is JSON. If the response is not JSON or an error occurs, appropriate error messages are printed.
    """
    try:
        response = requests.get(Config.COUNTRIES_EN_API_URL)
        response.raise_for_status()  # Vérifie si la requête a réussi (status code 200)

        # Vérifie si la réponse est en JSON
        if response.headers['Content-Type'] == 'application/json':
            countries_en = response.json()
            countries_en_names = [country['name']['common'] for country in countries_en]
            return countries_en_names
        else:
            print("La réponse n'est pas en JSON")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Error occurred: {req_err}")
    except ValueError as json_err:
        print(f"Erreur de décodage JSON: {json_err}")

def clean_countries(dataframe):
    """
    :param dataframe: The DataFrame containing country names to be cleaned.
    :return: The cleaned DataFrame with country names processed to ensure consistency and validity.
    """
    dataframe[Config.COUNTRIES_EN_COL].str.split(',').explode().drop_duplicates()
    countries_en_names = fetch_countries_en()
    dataframe[Config.COUNTRIES_EN_COL].apply(lambda x: x if x in countries_en_names else Config.UNKNOWN_STR)
    dataframe[Config.COUNTRIES_EN_COL].fillna(Config.UNKNOWN_STR)
