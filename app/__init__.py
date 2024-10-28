from flask import Flask
from config import Config
import app.modules.db_nutriscore as mod_db_nutriscore

def create_app():
    """
    Initializes and configures the Flask application.

    :return: The configured Flask application instance.
    """
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize the loading status
    app.config['loading_dataframe_status'] = {"complete": False}
    
    # Initialize and bind the databases
    mod_db_nutriscore.set_db_nutriscore()

    # Import and register blueprints
    from app.modules.routes import main
    app.register_blueprint(main)

    return app
