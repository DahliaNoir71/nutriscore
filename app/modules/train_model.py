import os

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             confusion_matrix,
                             f1_score,
                             roc_auc_score,
                             classification_report,
                             roc_curve, auc)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize

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


def make_predictions(model, std_scaler, x_for_test):
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


def plot_probability_distribution(y_predict_prob_test):
    """
    Display the distribution of classification probabilities.

    Parameters:
    y_predict_prob_test (numpy.ndarray): Probabilities predicted by the model. This array should contain
                                         the predicted probabilities for each class for each sample in the test set.
                                         The shape of this array should be (n_samples, n_classes).

    Returns:
    None: This function does not return any value. It only displays a plot.

    The function creates a histogram plot to visualize the distribution of classification probabilities.
    The plot is titled "Distribution des probabilités de classification pour le NutriScore" and includes
    labels for the x-axis ("Probabilités de classification") and the y-axis ("Fréquence").
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(y_predict_prob_test, kde=True)
    plt.title("Distribution des probabilités de classification pour le NutriScore", fontsize=16)
    plt.xlabel("Probabilités de classification", fontsize=14)
    plt.ylabel("Fréquence", fontsize=14)
    plt.show()


def plot_roc_curve(y_for_test, y_predict_prob_test):
    """
    Calculate and display the ROC curves for each class.

    Parameters:
    y_for_test (pd.Series): A pandas Series containing the true target values for the test set.
                            The Series should have the same length as the number of samples in the test set.
                            Each element represents the target value for a corresponding sample.
    y_predict_prob_test_logreg (np.ndarray): A numpy array containing the predicted probabilities by the model.
                                            The array should have a shape of (n_samples, n_classes), where n_samples is the
                                            number of samples in the test set and n_classes is the number of classes.
                                            Each element represents the predicted probability for a corresponding class.

    Returns:
    None: This function does not return any value. It only displays a plot.

    The function calculates and displays the Receiver Operating Characteristic (ROC) curves for each class.
    The ROC curve is a plot that illustrates the diagnostic ability of a binary classifier system as its discrimination
    threshold is varied. The plot is created by plotting the true positive rate (TPR) against the false positive rate (FPR)
    at various threshold settings. The area under the curve (AUC) represents the overall performance of the classifier.
    """
    y_test_binarized = label_binarize(y_for_test, classes=['a', 'b', 'c', 'd', 'e'])  # Replace with your actual classes
    n_classes = y_test_binarized.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_predict_prob_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {["a", "b", "c", "d", "e"][i]}) - AUC = {roc_auc[i]:.2f}')

    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for chance
    plt.grid()
    plt.title("Courbes ROC pour le NutriScore", fontsize=16)
    plt.xlabel("Taux de faux positifs", fontsize=14)
    plt.ylabel("Taux de vrais positifs", fontsize=14)
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(y_for_test, y_predict_test, model):
    """
    Display the confusion matrix for a given classification model.

    Parameters:
    y_for_test (pd.Series): A pandas Series containing the true target values for the test set.
                            The Series should have the same length as the number of samples in the test set.
                            Each element represents the target value for a corresponding sample.
    y_predict_test (pd.Series): A pandas Series containing the predicted target values for the test set.
                                The Series should have the same length as the number of samples in the test set.
                                Each element represents the predicted target value for a corresponding sample.
    model (sklearn.linear_model.LogisticRegression): The trained classification model.
                                                      This model should have been trained on the training set
                                                      and used to make predictions on the test set.

    Returns:
    None: This function does not return any value. It only displays a plot.

    The function creates a confusion matrix using the true target values and predicted target values.
    The confusion matrix is visualized using a heatmap, where the true target values are displayed on the y-axis,
    and the predicted target values are displayed on the x-axis. Each cell in the heatmap represents the number of
    samples that belong to a specific combination of true and predicted target values.
    """
    conf_matrix = confusion_matrix(y_for_test, y_predict_test, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Classe prédite", fontsize=14)
    plt.ylabel("Classe réelle", fontsize=14)
    plt.title("Matrice de confusion du NutriScore", fontsize=16)
    plt.show()


def print_classification_metrics(y_for_test, y_predict_test, model, average='macro'):
    """
    Print the performance metrics of the model.

    Parameters:
    y_for_test (pd.Series): A pandas Series containing the true target values for the test set.
                            The Series should have the same length as the number of samples in the test set.
                            Each element represents the target value for a corresponding sample.
    y_predict_test (pd.Series): A pandas Series containing the predicted target values for the test set.
                                The Series should have the same length as the number of samples in the test set.
                                Each element represents the predicted target value for a corresponding sample.
    model (sklearn.linear_model.LogisticRegression): The trained logistic regression model.
                                                      This model should have been trained on the training set
                                                      and used to make predictions on the test set.
    average (str): The type of averaging to use for the precision, recall, and F1-score.
                   It can be one of the following: 'binary', 'micro', 'macro', 'weighted', 'samples'.
                   The default value is 'macro'.

    Returns:
    None: This function does not return any value. It only prints the performance metrics.
    """
    accuracy = accuracy_score(y_for_test, y_predict_test)
    print("Accuracy sur l'ensemble de test:", accuracy)

    print("\nRapport de classification détaillé :\n")
    print(classification_report(y_for_test, y_predict_test, target_names=model.classes_))

    precision = precision_score(y_for_test, y_predict_test, average=average)
    recall = recall_score(y_for_test, y_predict_test, average=average)
    f1 = f1_score(y_for_test, y_predict_test, average=average)

    print(f"\nPrécision ({average}):", precision)
    print(f"Rappel ({average}):", recall)
    print(f"Score F1 ({average}):", f1)


def evaluate_model(y_for_test, y_predict_test, y_predict_prob_test, model, average='macro'):
    """
    Evaluate the performance of a trained logistic regression model using various metrics and displays.

    Parameters:
    y_for_test (pd.Series): True target values for the test set.
    y_predict_test_logreg (pd.Series): Predicted target values for the test set.
    y_predict_prob_test_logreg (np.ndarray): Predicted probabilities by the model.
    logreg (sklearn.linear_model.LogisticRegression): The trained logistic regression model.
    average (str): The type of averaging to use for the precision, recall, and F1-score.

    Returns:
    None: This function does not return any value. It only performs evaluations and displays.
    """
    plot_probability_distribution(y_predict_prob_test)

    # Calculate the AUC ROC for the entire dataset
    roc_auc = roc_auc_score(y_for_test, y_predict_prob_test, multi_class='ovr')
    print("ROC-AUC:", roc_auc)

    plot_roc_curve(y_for_test, y_predict_prob_test)
    plot_confusion_matrix(y_for_test, y_predict_test, model)

    print_classification_metrics(y_for_test, y_predict_test, model, average)

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



