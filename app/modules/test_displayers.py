import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def display_predictions(df_test):
    """
    Display the actual and predicted nutriscore grades from a DataFrame.

    This function sets the maximum number of rows displayed in pandas to None, prints the DataFrame
    containing 'nutriscore_grade' and 'nutriscore_prediction' columns, and then resets the maximum
    number of rows to its default value.

    Parameters:
    df_test (pandas.DataFrame): A DataFrame containing at least the columns 'nutriscore_grade' and
        'nutriscore_prediction'. The DataFrame should have a column for actual nutriscore grades and
        another column for predicted nutriscore grades.

    Returns:
    None
    """
    pd.set_option('display.max_rows', None)
    print(df_test[['nutriscore_grade', 'nutriscore_prediction']])
    pd.reset_option('display.max_rows')

def plot_predictions_heatmap(df_test, accuracy_report):
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