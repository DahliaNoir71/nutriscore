from flask_sqlalchemy import SQLAlchemy

# Initialize the SQLAlchemy instance
db = SQLAlchemy()

def init_db(app):
    """
    :param app: The Flask application instance to initialize the database with.
    :return: None
    """
    with app.app_context():
        db.create_all()
