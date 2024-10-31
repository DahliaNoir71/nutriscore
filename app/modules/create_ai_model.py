import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Adjust the path to access the config file from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import Config

from app.modules.clean_csv import read_csv_chunks

def load_and_preprocess_data():
    # Load the dataset using read_csv_chunks function and concatenate all chunks into a single DataFrame
    df_chunks = read_csv_chunks(Config.CLEANED_CSV_FULL_PATH, Config.COLS_FOR_MODEL, Config.CHUNK_SIZE)
    df = pd.concat(df_chunks, ignore_index=True)

    # Encode categorical variables
    label_encoder_pnns = LabelEncoder()
    df['pnns_groups_1'] = label_encoder_pnns.fit_transform(df['pnns_groups_1'])

    # Use OrdinalEncoder for the target variable 'nutriscore_grade'
    ordinal_encoder_grade = OrdinalEncoder(categories=[['e', 'd', 'c', 'b', 'a']])
    df['nutriscore_grade'] = ordinal_encoder_grade.fit_transform(df[['nutriscore_grade']])

    return df, label_encoder_pnns, ordinal_encoder_grade

def train_model(df, label_encoder_pnns, ordinal_encoder_grade):
    # Separate features and target variable
    X = df.drop(columns="nutriscore_grade")
    y = df['nutriscore_grade'].ravel()  # Convert target to 1D array

    # Print detailed information about the training DataFrame columns
    print("Training DataFrame Info:")
    print(X.info())
    print("Training DataFrame Head:")
    print(X.head())

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a model (RandomForestClassifier in this case)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print("Model Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save the model, encoders, and scaler in 'app/ai-model'
    save_model_and_encoders(model, scaler, label_encoder_pnns, ordinal_encoder_grade)

def save_model_and_encoders(model, scaler, label_encoder_pnns, ordinal_encoder_grade):
    # Create the directory if it doesn't exist
    save_dir = os.path.join('app', 'ai-model')
    os.makedirs(save_dir, exist_ok=True)

    # Save the model and other components
    joblib.dump(model, os.path.join(save_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    joblib.dump(label_encoder_pnns, os.path.join(save_dir, 'label_encoder_pnns.pkl'))
    joblib.dump(ordinal_encoder_grade, os.path.join(save_dir, 'ordinal_encoder_grade.pkl'))

if __name__ == "__main__":
    # Load and preprocess the data
    df, label_encoder_pnns, ordinal_encoder_grade = load_and_preprocess_data()

    # Train the model
    train_model(df, label_encoder_pnns, ordinal_encoder_grade)
