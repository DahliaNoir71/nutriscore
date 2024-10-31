import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (roc_curve,
                             auc,
                             confusion_matrix,
                             accuracy_score,
                             classification_report,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score)
from sklearn.preprocessing import label_binarize


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

def plot_training_confusion_matrix(y_for_test, y_predict_test, model):
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

def display_model_evaluations(y_for_test, y_predict_test, y_predict_prob_test, model, average='macro'):
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
    plot_roc_curve(y_for_test, y_predict_prob_test)
    plot_training_confusion_matrix(y_for_test, y_predict_test, model)
    print_classification_metrics(y_for_test, y_predict_test, model, average)