from flask_wtf import FlaskForm
from wtforms import StringField, DecimalField, SubmitField
from wtforms.validators import DataRequired, NumberRange

class NutriScoreForm(FlaskForm):
    """
        NutriScoreForm
        --------------

        A FlaskForm for collecting nutritional information about a product
        to predict its NutriScore.

        Attributes:
            product_name : str
                the name of the product
            energy : decimal.Decimal
                energy content of the product in kilojoules
            fat : decimal.Decimal
                fat content of the product in grams
            saturated_fat : decimal.Decimal
                saturated fat content of the product in grams
            sugar : decimal.Decimal
                sugar content of the product in grams
            salt : decimal.Decimal
                salt content of the product in grams
            submit : flask_wtf.SubmitField
                field to submit the form
    """
    product_name = StringField('Product Name', validators=[DataRequired()])
    energy = DecimalField('Energy (kJ)', validators=[DataRequired(), NumberRange(min=0)])
    fat = DecimalField('Fat (g)', validators=[DataRequired(), NumberRange(min=0)])
    saturated_fat = DecimalField('Saturated Fat (g)', validators=[DataRequired(), NumberRange(min=0)])
    sugar = DecimalField('Sugar (g)', validators=[DataRequired(), NumberRange(min=0)])
    salt = DecimalField('Salt (g)', validators=[DataRequired(), NumberRange(min=0)])
    submit = SubmitField('Predict')
