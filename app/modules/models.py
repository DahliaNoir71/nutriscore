from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Product(db.Model):
    """
    Represents a product in the database.

    Attributes:
        id (int): Database primary key; unique identifier for the product.
        name (str): Name of the product; maximum length is 150 characters; cannot be null.
        energy (float): Energy content of the product; cannot be null.
        fat (float): Fat content of the product; cannot be null.
        saturated_fat (float): Saturated fat content of the product; cannot be null.
        sugar (float): Sugar content of the product; cannot be null.
        salt (float): Salt content of the product; cannot be null.
        nutri_score (str): Nutri-score of the product; single character; cannot be null.

    Methods:
        __repr__: Returns a string representation of the product instance.
    """
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    energy = db.Column(db.Float, nullable=False)
    fat = db.Column(db.Float, nullable=False)
    saturated_fat = db.Column(db.Float, nullable=False)
    sugar = db.Column(db.Float, nullable=False)
    salt = db.Column(db.Float, nullable=False)
    nutri_score = db.Column(db.String(1), nullable=False)

    def __repr__(self):
        return f'<Product {self.name}>'