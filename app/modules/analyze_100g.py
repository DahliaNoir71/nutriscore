import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import Config
from clean_csv import read_csv_chunks


# Détecter les valeurs aberrantes pour chaque colonne numérique
def detect_outliers(df):
    """
    :param df: A pandas DataFrame that contains the data to check for outliers.
    :return: A DataFrame where each cell indicates whether the corresponding cell in the input DataFrame is an outlier. The DataFrame has the same column names as the input DataFrame.
    """
    outliers = pd.DataFrame(columns=df.columns)

    for col in df.select_dtypes(include=[np.number]).columns:
        lower_bound, upper_bound = get_outliers_bounds(col, df)
        outliers[col] = ((df[col] < lower_bound) | (df[col] > upper_bound))

    return outliers


def get_outliers_bounds(col, df):
    """
    :param col: Column name to calculate the outliers bounds from.
    :type col: str
    :param df: DataFrame containing the data.
    :type df: pandas.DataFrame
    :return: A tuple containing the lower bound and upper bound for outliers.
    :rtype: tuple
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return lower_bound, upper_bound


# Configure pandas pour afficher toutes les lignes
pd.set_option('display.max_rows', None)
# Configure pandas pour afficher toutes les colonnes
pd.set_option('display.max_columns', None)

cleaned_file_path = Config.DIRECTORY_PATH + Config.CLEANED_CSV_NAME

chunks_100g = read_csv_chunks(cleaned_file_path,
                              Config.COLS_100G,
                              Config.CHUNK_SIZE)
df_100g = pd.concat(chunks_100g, ignore_index=True)
print(df_100g.describe(include='all'))

# Afficher le nombre de valeurs manquantes par colonne
missing_values = df_100g.isnull().sum()
print("Valeurs manquantes par colonne:")
print(missing_values)

# Afficher un DataFrame avec les valeurs aberrantes détectées
outliers_df_100g = detect_outliers(df_100g)
print("Valeurs aberrantes détectées:")
print(outliers_df_100g.describe(include='all'))

# Tracer le boxplot
for column in df_100g.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df_100g, x=column)
    sns.despine()
    plt.xlabel(column)
    plt.ylabel(
        "Valeurs aberrantes" if column in outliers_df_100g.columns else "Valeurs non aberrantes"
    )
    plt.title("Visualisation des valeurs aberrantes avec un boxplot")
    plt.grid()
    plt.show()
