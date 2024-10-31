import joblib


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

def make_predictions(model, scaler, df_test):
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
    y_prediction_test, y_prediction_prob_test = make_test_predictions(model, scaler, df_test_scaled)
    df_test['nutriscore_prediction'] = y_prediction_test

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