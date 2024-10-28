from flask_sqlalchemy import SQLAlchemy

# Initialize the SQLAlchemy instance
db = SQLAlchemy()

def init_db(app):
    """
    Initialize the database for the given Flask application.

    This function creates all database tables based on the models defined in the application.
    It should be called once during the application setup, typically in the application's
    main script or in an initialization function.

    :param app: The Flask application instance to initialize the database with.
    :type app: flask.Flask

    :return: None
    """
    with app.app_context():
        db.create_all()