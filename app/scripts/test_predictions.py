import joblib
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

from app.modules.train_model import make_predictions

def load_data_for_test(model_path, scaler_path, test_data_path):
    """
    Load trained model, scaler, and test data from specified paths.

    Parameters:
    model_path (str): The path to the trained model file.
    scaler_path (str): The path to the scaler file.
    test_data_path (str): The path to the test data file.

    Returns:
    tuple: A tuple containing the loaded model, scaler, and test data.
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df_test = joblib.load(test_data_path)
    return model, scaler, df_test

def make_and_display_predictions(model, scaler, df_test):
    """
    This function makes predictions on the test data using a trained model and scaler,
    and then displays the actual and predicted nutriscore grades side by side.

    Parameters:
    model (sklearn.base.BaseEstimator): The trained model used for making predictions.
    scaler (sklearn.preprocessing.Scaler): The scaler used to preprocess the test data.
    df_test (pandas.DataFrame): The test data on which to make predictions. The DataFrame should contain the 'nutriscore_grade' column.

    Returns:
    pandas.DataFrame: The input DataFrame with an additional 'nutriscore_prediction' column containing the predicted grades.
    """
    df_test_scaled = df_test.drop(columns=['nutriscore_grade'])
    y_prediction_test, y_prediction_prob_test = make_predictions(model, scaler, df_test_scaled)
    df_test['nutriscore_prediction'] = y_prediction_test
    pd.set_option('display.max_rows', None)
    print(df_test[['nutriscore_grade', 'nutriscore_prediction']])
    pd.reset_option('display.max_rows')

    return df_test

def calculate_accuracy(df_test):
    """
    Calculate the accuracy of nutriscore predictions.

    This function takes a DataFrame containing actual and predicted nutriscore grades,
    and calculates the number of correct predictions, total predictions, and accuracy percentage.

    Parameters:
    df_test (pandas.DataFrame): A DataFrame containing the following columns:
        - 'nutriscore_grade': The actual nutriscore grades.
        - 'nutriscore_prediction': The predicted nutriscore grades.

    Returns:
    tuple: A tuple containing the following elements:
        - correct_predictions (int): The number of correct predictions.
        - total_predictions (int): The total number of predictions.
        - accuracy_percentage (float): The accuracy percentage of the predictions.
    """
    correct_predictions = (df_test['nutriscore_prediction'] == df_test['nutriscore_grade']).sum()
    total_predictions = len(df_test)
    accuracy_percentage = (correct_predictions / total_predictions) * 100

    return correct_predictions, total_predictions, accuracy_percentage

def get_accuracy_report(correct_predictions, total_predictions, accuracy_percentage):
    """
    Display a report containing the number of correct predictions, total predictions, and accuracy percentage.

    Parameters:
    correct_predictions (int): The number of correct predictions made by the model.
    total_predictions (int): The total number of predictions made by the model.
    accuracy_percentage (float): The accuracy percentage of the model's predictions.

    Returns:
    str: A string containing the report lines.
    """
    report = f"Nombre de bonnes prédictions : {correct_predictions}\n"
    report += f"Nombre total de prédictions : {total_predictions}\n"
    report += f"Pourcentage de bonnes prédictions : {accuracy_percentage:.2f}%"
    return report


def plot_nutriscore_heatmap(df_test, accuracy_report):
    """
    Plot a heatmap to visualize the comparison between actual and predicted nutriscore grades.

    Parameters:
    df_test (pandas.DataFrame): A DataFrame containing 'nutriscore_grade' and 'nutriscore_prediction'.

    Returns:
    None
    """
    # Créer une matrice de confusion
    confusion_matrix = pd.crosstab(df_test['nutriscore_grade'],
                                   df_test['nutriscore_prediction'],
                                   rownames=['Nutriscore Réel'],
                                   colnames=['Nutriscore Prédit'])

    # Créer une nouvelle figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Tracer la heatmap
    sns.heatmap(confusion_matrix,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                cbar=True,
                square=True,
                ax=ax)

    # Ajouter des titres et des étiquettes
    plt.title('Matrice de Confusion pour Nutriscore')
    plt.xlabel('Nutriscore Prédit')
    plt.ylabel('Nutriscore Réel')

    # Ajouter du texte en dessous de la heatmap
    # Utiliser les dimensions de la figure pour centrer le texte
    plt.text(0.5,
             -0.1,
             accuracy_report,
             ha='center',
             va='top',
             transform=ax.transAxes)

    # Ajuster l'espace en bas pour éviter le chevauchement
    plt.subplots_adjust(bottom=0.2)

    plt.show()


def main():
    """
    Main function to load data, make predictions, calculate accuracy, and display the report.
    """
    model_path = 'models/model_nutriscore.pkl'
    scaler_path = 'models/scaler_nutriscore.pkl'
    test_data_path = 'tests/df_test_nutriscore.pkl'

    model, scaler, df_test = load_data_for_test(model_path, scaler_path, test_data_path)
    df_test = make_and_display_predictions(model, scaler, df_test)
    correct_predictions, total_predictions, accuracy_percentage = calculate_accuracy(df_test)
    report = get_accuracy_report(correct_predictions, total_predictions, accuracy_percentage)
    plot_nutriscore_heatmap(df_test, report)

# Call the main function
if __name__ == "__main__":
    main()