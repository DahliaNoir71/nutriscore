from flask import Flask
from config import Config
from app.modules.database import db

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize and bind the database
    db.init_app(app)

    # Import and register blueprints
    from app.modules.routes import main
    app.register_blueprint(main)

    return app
