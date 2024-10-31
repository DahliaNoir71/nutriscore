import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from app.modules.clean_csv import read_csv_chunks
from config import Config


def load_data(csv_path, cols, chunk_size):
    """
    Load a large CSV file into a pandas DataFrame using chunk processing.

    Parameters:
    csv_path (str): The path to the CSV file. This file should be in a format compatible with pandas' read_csv function.
                    The file should contain the data to be loaded.
    cols (list): A list of column names to load from the CSV file. If None, all columns will be loaded.
    chunk_size (int): The size of each chunk to read from the CSV file. This parameter determines the number of rows
                      to include in each chunk. Larger chunk sizes may improve performance, but require more memory.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the loaded data. The DataFrame will have the same number of columns as
                 specified in the 'cols' parameter, and the number of rows will be equal to the total number of rows
                 in the CSV file.
    """
    chunks = read_csv_chunks(csv_path, cols, chunk_size)
    df = pd.concat(chunks, ignore_index=True)
    return df

def save_df_test(df_test):
    # Créer le répertoire s'il n'existe pas
    os.makedirs('tests', exist_ok=True)
    # Ensuite, sauvegarder le modèle et le scaler dans ce répertoire
    joblib.dump(df_test, 'tests/df_test_nutriscore.pkl')
    print("df_test sauvegardé avec succès.")


def split_data_for_test(df, test_size):
    df_test = df.sample(frac=test_size, random_state=1)  # Premier échantillon
    df = df.drop(df_test.index)  # Deuxième partie avec les lignes restantes
    save_df_test(df_test)
    return df


def preprocess_data(df, prediction_column):
    """
    Preprocess the dataset by splitting it into features (x) and target (y) variables.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the dataset. This DataFrame should contain all the columns
                        required for the prediction, including the target variable.
    predictive_columns (str): The name of the column to be used as the target variable. This column should be present
                               in the input DataFrame.

    Returns:
    tuple: A tuple containing two pandas DataFrames. The first DataFrame (x) contains the features for the prediction,
           which are all columns in the input DataFrame except the target variable. The second DataFrame (y) contains
           the target variable.
    """
    x = df.drop(prediction_column, axis=1)  # Features are all columns except the prediction column
    y = df[prediction_column]  # Target is the prediction column
    return x, y


def cross_validate_model(model, x, y, k=5, scoring='accuracy'):
    """
    Effectue une validation croisée sur le modèle spécifié.

    Paramètres :
    - model : le modèle ML à entraîner.
    - X : caractéristiques (features).
    - y : étiquettes (labels).
    - k : nombre de folds pour la cross-validation (par défaut 5).
    - scoring : métrique de scoring pour évaluer le modèle (par défaut 'accuracy').

    Retourne :
    - Un dictionnaire contenant les scores par fold, la moyenne des scores, et l’écart-type des scores.
    """

    # Effectuer la validation croisée
    scores = cross_val_score(model, x, y, cv=k, scoring=scoring)

    # Calculer les résultats
    cross_validation_results = {
        "scores": scores,
        "mean_score": np.mean(scores),
        "std_dev": np.std(scores)
    }

    # Afficher les résultats
    print("Scores de chaque fold :", cross_validation_results["scores"])
    print("Précision moyenne :", cross_validation_results["mean_score"])
    print("Écart-type des précisions :", cross_validation_results["std_dev"])

    return cross_validation_results


def train_model(x_for_train, y_for_train):
    """
    Train a logistic regression model using the provided training data.

    Parameters:
    x_train (pd.DataFrame): A pandas DataFrame containing the features for the training set.
                            The DataFrame should have the same number of columns as the features used in the model.
                            Each row represents a sample, and each column represents a feature.
    y_train (pd.Series): A pandas Series containing the target values for the training set.
                         The Series should have the same length as the number of samples in the training set.
                         Each element represents the target value for a corresponding sample.

    Returns:
    tuple: A tuple containing the trained logistic regression model and the fitted scaler.
           The model is an instance of the RandomForestClassifier class from the sklearn.ensemble module.
           The scaler is an instance of the StandardScaler class from the sklearn.preprocessing module.
           The scaler is used to standardize the features before training the model.
    """
    std_scaler = StandardScaler()
    x_for_train = std_scaler.fit_transform(x_for_train)
    model = RandomForestClassifier(verbose=1)
    cross_validate_model(model, x_for_train, y_for_train)
    model.fit(x_for_train, y_for_train)
    return model, std_scaler


def make_test_predictions(model, std_scaler, x_for_test):
    """
    Make predictions using a trained logistic regression model and a scaler.

    Parameters:
    log_reg_model (sklearn.linear_model.LogisticRegression): The trained logistic regression model.
    std_scaler (sklearn.preprocessing.StandardScaler): The scaler used to transform the input data.
    x_for_test (pd.DataFrame): A pandas DataFrame containing the features for the test set.

    Returns:
    y_prediction_test (pd.Series): A pandas Series containing the predicted target values for the test set.
    y_prediction_prob_test (np.ndarray): A numpy array containing the predicted probabilities for each class.
    """
    x_for_test = std_scaler.transform(x_for_test)
    y_prediction_test = model.predict(x_for_test)
    y_prediction_prob_test = model.predict_proba(x_for_test)
    return y_prediction_test, y_prediction_prob_test


def save_model(model, scaler):
    """
    Saves the trained model and scaler to a specified directory.

    Parameters:
    model (sklearn.ensemble.RandomForestClassifier): The trained Random Forest model to be saved.
    scaler (sklearn.preprocessing.StandardScaler): The scaler used to standardize the input data.

    Returns:
    None: This function does not return any value. It saves the model and scaler to the specified directory.

    The function creates the 'models' directory if it does not already exist.
    Then, it saves the trained model and scaler to the 'models' directory using joblib's dump function.
    Finally, it prints a success message indicating that the model and scaler have been saved.
    """
    # Créer le répertoire s'il n'existe pas
    os.makedirs('models', exist_ok=True)
    # Ensuite, sauvegarder le modèle et le scaler dans ce répertoire
    joblib.dump(model, 'models/model_nutriscore.pkl')
    joblib.dump(scaler, 'models/scaler_nutriscore.pkl')
    print("Modèle et scaler sauvegardés avec succès.")

def prepare_and_train_model(saving):
    """
    Prepares and trains a machine learning model for predicting NutriScore.

    Parameters:
    saving (bool): A flag indicating whether to save the trained model and scaler.
                    If True, the model and scaler will be saved to the 'models' directory.

    Returns:
    tuple: A tuple containing the trained model, the scaler used for standardizing the input data,
           the features for the test set, and the target values for the test set.
           The model is an instance of the RandomForestClassifier class from the sklearn.ensemble module.
           The scaler is an instance of the StandardScaler class from the sklearn.preprocessing module.
           The features and target values for the test set are obtained using the train_test_split function
           from the sklearn.model_selection module.

    The function loads a cleaned CSV file containing data for predicting NutriScore.
    It then preprocesses the data by splitting it into features and target variables.
    The data is split into training and test sets using the train_test_split function.
    The function trains a Random Forest model using the training data.
    If the 'saving' parameter is True, the trained model and scaler are saved to the 'models' directory.
    Finally, the function returns the trained model, scaler, and the features and target values for the test set.
    """
    csv_file_path = Config.DIRECTORY_PATH + Config.CLEANED_CSV_NAME
    df_prediction = load_data(csv_file_path, Config.COLS_100G, Config.CHUNK_SIZE)
    df_prediction = split_data_for_test(df_prediction,test_size=0.1)
    x, y = preprocess_data(df_prediction, Config.COL_PREDICTION)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model, scaler = train_model(x_train, y_train)
    if saving:
        save_model(model, scaler)
    return model, scaler, x_test, y_test



