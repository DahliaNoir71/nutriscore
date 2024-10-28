from flask import Blueprint, render_template, redirect, url_for, flash, current_app, jsonify
from config import Config
from app.modules.forms import NutriScoreForm
from app.modules.models import db, Product
from app.modules.explore_data import load_dataframe
import threading

main = Blueprint('main', __name__)

@main.route('/loading-dataframe-status', methods=['GET', 'POST'])
def loading_dataframe_status_check():
    return jsonify(current_app.config['loading_dataframe_status'])

@main.route('/', methods=['GET', 'POST'])
def index():
    """
    Handles the main route of the application, processing both GET and POST requests to predict Nutri-Score for a given product.

    :return: Renders the index page with the NutriScoreForm or redirects to the main index page with a flash message upon successful form submission.
    """
    form = NutriScoreForm()
    if form.validate_on_submit():
        # Here, you would call your trained model to get the Nutri-Score prediction
        # Placeholder for the model call
        predicted_score = 'B'  # Replace with actual model prediction

        # Save to database
        new_product = Product(
            name=form.product_name.data,
            energy=form.energy.data,
            fat=form.fat.data,
            saturated_fat=form.saturated_fat.data,
            sugar=form.sugar.data,
            salt=form.salt.data,
            nutri_score=predicted_score
        )
        db.session.add(new_product)
        db.session.commit()

        flash(f'Nutri-Score for {form.product_name.data}: {predicted_score}', 'success')
        return redirect(url_for('main.index'))
    
    return render_template('index.html', form=form)

@main.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Handles HTTP GET and POST requests for predicting nutritional scores.

    :return: Renders the prediction form template with the form object.
    """
    form = NutriScoreForm()
    if form.validate_on_submit():
        # Handle prediction logic here
        pass
    return render_template('prediction_form.html', form=form)

@main.route('/loading_data')
def loading_data():
    # Set the status to indicate that loading has not yet completed
    current_app.config['loading_dataframe_status'] = {"complete": False}

    # Capture the current app context to pass it into the new thread
    app_context = current_app._get_current_object()

    # Function to run in a new thread
    def load_data_with_context():
        with app_context.app_context():
            load_dataframe()

    # Start the thread
    threading.Thread(target=load_data_with_context).start()

    return render_template('loading_dataframe.html')

@main.route('/training_data')
def training_data():
    #Checks if the dataframe is already loaded:
    if 'PRODUCTS_DF' not in current_app.config:
        # Send to a 'Loading dataframe' template
        return redirect(url_for('main.loading_data'))

    products = current_app.config['PRODUCTS_DF']

    # Retrieve the products DataFrame from the app config
    #products = current_app.config['PRODUCTS_DF']
    nutriscore_grades = sorted(products['nutriscore_grade'].dropna().unique())  # Extract unique values and sort alphabetically
    return render_template('training_data.html', nutriscore_grades=nutriscore_grades)

# Search Route
@main.route('/search', methods=['POST'])
def search():
    """
    Retrieves the products DataFrame from the app config and handles HTTP POST requests for searching products.

    :return: Redirects back to the training data page.
    """
    # Retrieve the products DataFrame from the app config
    products = current_app.config['PRODUCTS_DF']

    # Here, you can define the search functionality
    # For now, let's redirect back to the training data page
    return redirect(url_for('main.training_data'))