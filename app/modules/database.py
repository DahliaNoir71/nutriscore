from flask_sqlalchemy import SQLAlchemy

# Initialize the SQLAlchemy instance
db = SQLAlchemy()

def init_db(app):
    """
    Initialize the database within the app context.
    """
    with app.app_context():
        db.create_all()
