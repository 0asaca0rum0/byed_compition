from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import os
import keras
from flask_cors import CORS
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

# Global variables to store model and scalers
model = None
scaler_X = None
scaler_y = None
encoder = None


def load_model_and_scalers():
    global model, scaler_X, scaler_y, encoder

    if os.path.exists('./canteen_mlp_model.keras'):
        model = keras.models.load_model('./canteen_mlp_model.keras')

        # Generate a sample dataset to fit the scalers and encoder
        df = generate_dataset()
        _, scaler_X, scaler_y, encoder = preprocess_data(df)
        return True
    else:
        return False


def generate_dataset(num_samples=50000):
    """Generate a synthetic dataset for predicting canteen presence."""
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Sunday']
    weather_types = ['good', 'bad']
    special_events = ['None', 'Festival', 'Exam']

    data = {
        'Day': np.random.choice(days, size=num_samples),
        'Weather': np.random.choice(weather_types, size=num_samples),
        'Classes': np.random.choice(['On', 'Off'], size=num_samples, p=[0.7, 0.3]),
        'Special_Events': np.random.choice(special_events, size=num_samples, p=[0.8, 0.1, 0.1]),
        'Attendance_Rate': np.random.uniform(0.2, 1.0, size=num_samples)
    }

    student_counts = []
    for i in range(num_samples):
        base_count = np.random.randint(5, 50)
        if data['Day'][i] in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Sunday']:
            base_count += np.random.randint(10, 30)
        if data['Weather'][i] == 'bad':
            base_count -= np.random.randint(5, 15)
        if data['Classes'][i] == 'On':
            base_count += np.random.randint(20, 40)
        if data['Special_Events'][i] == 'Festival':
            base_count += np.random.randint(50, 100)
        elif data['Special_Events'][i] == 'Exam':
            base_count += np.random.randint(10, 30)
        student_counts.append(max(0, base_count))

    data['Student_Count'] = student_counts

    # Add Student_Count_Last_Week feature
    data['Student_Count_Last_Week'] = [0] * 5 + student_counts[:-5]

    df = pd.DataFrame(data)
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def preprocess_data(df):
    """Preprocess the dataset by encoding categorical features and normalizing."""
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(
        df[['Day', 'Weather', 'Classes', 'Special_Events']])
    encoded_feature_names = encoder.get_feature_names_out(
        ['Day', 'Weather', 'Classes', 'Special_Events'])
    encoded_features = encoder.fit_transform(
        df[['Day', 'Weather', 'Classes', 'Special_Events']])
    # Include numerical features
    numerical_features = df[['Attendance_Rate', 'Student_Count_Last_Week']]
    X = np.hstack((encoded_features, numerical_features))
    y = df['Student_Count'].values.reshape(-1, 1)

    # Scale features and target
    scaler_X = MinMaxScaler()
    X = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y = scaler_y.fit_transform(y)

    return (X, y), scaler_X, scaler_y, encoder


@app.route('/predict', methods=['POST'])
def predict_route():
    global model, scaler_X, scaler_y, encoder

    if model is None:
        if not load_model_and_scalers():
            return jsonify({"error": "Model file not found!"})

    data = request.get_json()
    required_columns = ['Day', 'Weather', 'Classes',
                        'Special_Events', 'Attendance_Rate', 'Student_Count_Last_Week']
    for col in required_columns:
        if col not in data:
            return jsonify({"error": f"Missing column: {col}"})
    df = pd.DataFrame(data, index=[0])
    encoded_features = encoder.transform(
        df[['Day', 'Weather', 'Classes', 'Special_Events']])
    numerical_features = df[['Attendance_Rate', 'Student_Count_Last_Week']]
    X = np.hstack((encoded_features, numerical_features))
    X = scaler_X.transform(X)

    predictions = model.predict(X)
    predictions = scaler_y.inverse_transform(predictions)

    # Save predictions to a text file
    with open('predictions.txt', 'w') as f:
        for prediction in predictions.flatten():
            f.write(f"{prediction}\n")

    # Plot predictions
    plt.figure(figsize=(10, 6))
    plt.plot(predictions.flatten(), label='Predicted Student Count')
    plt.xlabel('Sample')
    plt.ylabel('Student Count')
    plt.title('Predicted Student Count Over Samples')
    plt.legend()
    plt.savefig('predictions_chart.png')

    return jsonify({"predictions": predictions.flatten().tolist()})


if __name__ == "__main__":
    app.run(debug=True)
