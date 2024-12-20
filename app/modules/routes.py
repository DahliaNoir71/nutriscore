from flask import Blueprint, render_template, redirect, url_for, flash, current_app, jsonify, request
from config import Config
from app.modules.forms import NutriScoreForm
from app.modules.models import db, Product
from app.modules.explore_data import load_dataframe
import threading
import math

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

    # Retrieve the products DataFrame from the app config
    products = current_app.config['PRODUCTS_DF']

    # Pagination parameters
    page = request.args.get('page', 1, type=int)  # Get the current page, default is 1
    per_page = 50  # Number of products to show per page

    # Calculate total pages
    total_products = len(products)
    total_pages = math.ceil(total_products / per_page)

    # Paginate the DataFrame
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paginated_products = products.iloc[start_index:end_index].to_dict(orient='records')

    nutriscore_grades = sorted(products['nutriscore_grade'].dropna().unique())  # Extract unique values and sort alphabetically

    cat_list = sorted(products['pnns_groups_1'].dropna().unique())  # Extract unique values and sort alphabetically

    return render_template('training_data.html',
                           nutriscore_grades=nutriscore_grades,
                           cat_list=cat_list,
                           products=paginated_products, 
                           page=page, 
                           total_pages=total_pages,
                           total_products=total_products)

# Search Route
@main.route('/search', methods=['GET', 'POST'])
def search():
    # Get the DataFrame from the app config
    products = current_app.config['PRODUCTS_DF']
    
    # Extract unique Nutriscore grades, sorted alphabetically
    nutriscore_grades = sorted(products['nutriscore_grade'].dropna().unique())

    # Retrieve all unique Categories for the sidebar
    cat_list = sorted(products['pnns_groups_1'].dropna().unique())

    return render_template('search.html',
                           nutriscore_grades=nutriscore_grades,
                           cat_list=cat_list)

@main.route('/search_results', methods=['GET'])
def search_results():
    # Get the DataFrame from the app config
    products = current_app.config['PRODUCTS_DF']

    # Check if the form was explicitly submitted
    form_submitted = request.args.get('submitted', '') == 'true'

    if form_submitted:
        # Perform a new search based on the form parameters
        search_results = products

        # Retrieve search parameters from GET request
        search_term = request.args.get('search_term', '').strip().lower()
        search_columns = request.args.getlist('search_columns')
        selected_grades = request.args.getlist('nutriscore_grades')
        pnns_groups_1 = request.args.getlist('pnns_groups_1')

        # Apply filters based on the retrieved parameters
        if selected_grades:
            search_results = search_results[search_results['nutriscore_grade'].isin(selected_grades)]
        
        if pnns_groups_1:
            search_results = search_results[search_results['pnns_groups_1'].isin(pnns_groups_1)]

        if search_term and search_columns:
            search_columns = [col for col in search_columns if col in search_results.columns]
            search_results = search_results[
                search_results[search_columns]
                .apply(lambda row: row.astype(str).str.contains(search_term, case=False, na=False).any(), axis=1)
            ]

        # Save the filtered results to the app config
        current_app.config['SEARCH_RESULTS_DF'] = search_results
    else:
        # Use the existing search results if no new search was performed
        search_results = current_app.config.get('SEARCH_RESULTS_DF', products)

    # Retrieve all unique Nutriscore grades for the sidebar
    nutriscore_grades = sorted(products['nutriscore_grade'].dropna().unique())

    # Retrieve all unique Categories for the sidebar
    cat_list = sorted(products['pnns_groups_1'].dropna().unique())

    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = 50
    total_products = len(search_results)
    total_pages = math.ceil(total_products / per_page)

    # Paginate the DataFrame
    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    paginated_products = search_results.iloc[start_index:end_index].to_dict(orient='records')

    return render_template('search_results.html', 
                           products=paginated_products,
                           cat_list=cat_list,
                           page=page, 
                           total_pages=total_pages,
                           nutriscore_grades=nutriscore_grades,
                           total_products=total_products)