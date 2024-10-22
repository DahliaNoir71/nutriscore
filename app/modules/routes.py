from flask import Blueprint, render_template, redirect, url_for, flash
from app.modules.forms import NutriScoreForm
from app.modules.models import db, Product

main = Blueprint('main', __name__)

@main.route('/', methods=['GET', 'POST'])
def index():
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
    form = NutriScoreForm()
    if form.validate_on_submit():
        # Handle prediction logic here
        pass
    return render_template('prediction_form.html', form=form)

@main.route('/browse_data')
def browse_data():
    # Logic to browse training data
    return "This page will show the training data."
