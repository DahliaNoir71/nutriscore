from app import create_app
from app.modules.database import db, init_db

app = create_app()

# Initialize the database
with app.app_context():
    init_db(app)

if __name__ == '__main__':
    app.run(debug=True)
