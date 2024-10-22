from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Product(db.Model):
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
