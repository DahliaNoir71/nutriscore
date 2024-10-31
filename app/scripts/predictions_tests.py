from app.modules.test_displayers import (display_predictions,
                                         plot_predictions_heatmap)
from app.modules.test_model import (load_data_for_test,
                                    make_predictions,
                                    calculate_accuracy,
                                    get_accuracy_report)


def main():
    """
    Main function to load data, make predictions, calculate accuracy, and display the report.
    """
    model_path = 'models/model_nutriscore.pkl'
    scaler_path = 'models/scaler_nutriscore.pkl'
    test_data_path = 'tests/df_test_nutriscore.pkl'

    model, scaler, df_test = load_data_for_test(model_path, scaler_path, test_data_path)
    df_test = make_predictions(model, scaler, df_test)
    display_predictions(df_test)
    correct_predictions, total_predictions, accuracy_percentage = calculate_accuracy(df_test)
    report = get_accuracy_report(correct_predictions, total_predictions, accuracy_percentage)
    plot_predictions_heatmap(df_test, report)

# Call the main function
if __name__ == "__main__":
    main()