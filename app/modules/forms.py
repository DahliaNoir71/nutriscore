from flask_wtf import FlaskForm
from wtforms import StringField, DecimalField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Optional, NumberRange, ValidationError

class NutriScoreForm(FlaskForm):
    # Product Information Section
    product_name = StringField('Product Name', validators=[DataRequired()])
    quantity = StringField('Quantity', validators=[Optional()])
    brands = StringField('Brands', validators=[Optional()])
    categories = StringField('Categories', validators=[Optional()])
    ingredients_text = TextAreaField('Ingredients', validators=[Optional()])

    # Nutritional Content Section (At least one field required)
    energy_kj_100g = DecimalField('Energy (kJ)', validators=[Optional(), NumberRange(min=0)])
    energy_kcal_100g = DecimalField('Energy (kcal)', validators=[Optional(), NumberRange(min=0)])
    fat_100g = DecimalField('Fat (g)', validators=[Optional(), NumberRange(min=0)])
    saturated_fat_100g = DecimalField('Saturated Fat (g)', validators=[Optional(), NumberRange(min=0)])
    omega_3_fat_100g = DecimalField('Omega-3 Fat (g)', validators=[Optional(), NumberRange(min=0)])
    omega_6_fat_100g = DecimalField('Omega-6 Fat (g)', validators=[Optional(), NumberRange(min=0)])
    sugars_100g = DecimalField('Sugars (g)', validators=[Optional(), NumberRange(min=0)])
    added_sugars_100g = DecimalField('Added Sugars (g)', validators=[Optional(), NumberRange(min=0)])
    fiber_100g = DecimalField('Fiber (g)', validators=[Optional(), NumberRange(min=0)])
    proteins_100g = DecimalField('Proteins (g)', validators=[Optional(), NumberRange(min=0)])
    salt_100g = DecimalField('Salt (g)', validators=[Optional(), NumberRange(min=0)])

    # Submit button
    submit = SubmitField('Predict')

    def validate(self, extra_validators=None):
        """
        Custom validation method to check that at least one field in the Nutritional Content section is filled.
        """
        if not super().validate(extra_validators=extra_validators):
            return False

        # Check if at least one nutritional field has a value
        nutritional_fields = [
            self.energy_kj_100g.data, self.energy_kcal_100g.data, self.fat_100g.data, self.saturated_fat_100g.data,
            self.omega_3_fat_100g.data, self.omega_6_fat_100g.data, self.sugars_100g.data, self.added_sugars_100g.data,
            self.fiber_100g.data, self.proteins_100g.data, self.salt_100g.data
        ]

        if not any(field is not None and field != 0 for field in nutritional_fields):
            # Raise an error if all fields are empty or zero
            self.nutritional_content_error = 'At least one field in the Nutritional Content section must be filled.'
            return False

        return True
