from flask_wtf import FlaskForm
from wtforms import StringField, DecimalField, SubmitField
from wtforms.validators import DataRequired, NumberRange

class NutriScoreForm(FlaskForm):
    product_name = StringField('Product Name', validators=[DataRequired()])
    energy = DecimalField('Energy (kJ)', validators=[DataRequired(), NumberRange(min=0)])
    fat = DecimalField('Fat (g)', validators=[DataRequired(), NumberRange(min=0)])
    saturated_fat = DecimalField('Saturated Fat (g)', validators=[DataRequired(), NumberRange(min=0)])
    sugar = DecimalField('Sugar (g)', validators=[DataRequired(), NumberRange(min=0)])
    salt = DecimalField('Salt (g)', validators=[DataRequired(), NumberRange(min=0)])
    submit = SubmitField('Predict')
