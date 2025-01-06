from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError
import requests
from requests.auth import HTTPBasicAuth
import os
from dotenv import load_dotenv
import json
from datetime import datetime, timedelta
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)
from functools import wraps

load_dotenv()

app = Flask(__name__)


app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('SQLALCHEMY_DATABASE_URI')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = os.getenv('SQLALCHEMY_TRACK_MODIFICATIONS', False)
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')  
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(seconds=int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES', 3600)))


db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

EXTERNAL_API_URL = os.getenv('EXTERNAL_API_URL')
EXTERNAL_API_USERNAME = os.getenv('EXTERNAL_API_USERNAME')
EXTERNAL_API_PASSWORD = os.getenv('EXTERNAL_API_PASSWORD')

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

def admin_required(fn):
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        try:
            current_user_id = get_jwt_identity()
            if not isinstance(current_user_id, (int, str)):
                return jsonify({'error': 'Invalid token subject'}), 422
            
            user = User.query.get(current_user_id)
            if not user:
                return jsonify({'error': 'User not found'}), 404
            if not user.is_admin:
                return jsonify({'error': 'Admin privileges required'}), 403
            
            return fn(*args, **kwargs)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return wrapper

# Registration route
@app.route('/auth/register', methods=['POST'])
@admin_required
def register():
    data = request.get_json()
    if not all(k in data for k in ['username', 'password']):
        return jsonify({'error': 'Missing required fields'}), 400

    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 400

    hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    new_user = User(
        username=data['username'],
        password=hashed_password,
        is_admin=data.get('is_admin', False)
    )

    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'User created successfully'}), 201

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    if not all(k in data for k in ['username', 'password']):
        return jsonify({'error': 'Missing required fields'}), 400

    user = User.query.filter_by(username=data['username']).first()
    if user and bcrypt.check_password_hash(user.password, data['password']):
        access_token = create_access_token(identity=str(user.id))
        return jsonify({
            'access_token': access_token,
            'is_admin': user.is_admin
        }), 200

    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/users', methods=['GET'])
@admin_required
def get_users():
    users = User.query.all()
    return jsonify([{
        'id': user.id,
        'username': user.username,
        'is_admin': user.is_admin,
        'created_at': user.created_at.isoformat()
    } for user in users]), 200

@app.route('/users/<int:user_id>', methods=['PUT'])
@admin_required
def update_user(user_id):
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    if 'username' in data:
        user.username = data['username']
    if 'password' in data:
        user.password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    if 'is_admin' in data:
        user.is_admin = data['is_admin']

    db.session.commit()
    return jsonify({'message': 'User updated successfully'}), 200

@app.route('/users/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': 'User deleted successfully'}), 200


model_conductivity = load_model('anomaly_detection_model_conductivity.h5', custom_objects={'mae': MeanAbsoluteError()})
model_salinity = load_model('anomaly_detection_model_salinity.h5', custom_objects={'mae': MeanAbsoluteError()})

# Fungsi untuk membuat sequences
def create_sequences(X, time_col, time_steps=30):
    Xs, times = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        times.append(time_col.iloc[i + time_steps])  # Simpan waktu dari indeks yang sesuai
    return np.array(Xs), times

# Fungsi untuk konversi format tanggal DDMMYYYY menjadi format ISO yang dibutuhkan API eksternal
def convert_date_format(date_str):
    # Mengubah DDMMYYYY menjadi format YYYY-MM-DDTHH:MM:SSZ
    return datetime.strptime(date_str, "%d%m%Y").strftime("%Y-%m-%dT%H:%M:%SZ")

# Fungsi untuk menentukan apakah nilai merupakan anomali berdasarkan aturan yang diberikan
def detect_anomaly_salinity(mae_loss, value, threshold=0.2):
    anomalies = []
    for i in range(len(mae_loss)):
        anomaly = False
        if mae_loss[i] > threshold:  # Rule 1: Jika score loss > threshold (71)
            anomaly = True
        elif value[i] < 0:  # Rule 2: Jika value < 0
            anomaly = True
        elif value[i] > 7:  # Rule 3: Jika value > 1000
            anomaly = True
        # Rule 4: Jika semua aturan di atas False, anomaly tetap False
        anomalies.append(anomaly)
    return anomalies

# Fungsi untuk menentukan apakah nilai merupakan anomali berdasarkan aturan yang diberikan
def detect_anomaly_conductivity(mae_loss, value, threshold=71):
    anomalies = []
    for i in range(len(mae_loss)):
        anomaly = False
        if mae_loss[i] > threshold:  # Rule 1: Jika score loss > threshold (71)
            anomaly = True
        elif value[i] < 0:  # Rule 2: Jika value < 0
            anomaly = True
        elif value[i] > 1000:  # Rule 3: Jika value > 1000
            anomaly = True
        # Rule 4: Jika semua aturan di atas False, anomaly tetap False
        anomalies.append(anomaly)
    return anomalies

# Endpoint untuk mengambil data dari API eksternal berdasarkan input dinamis dan melakukan prediksi untuk conductivity
@app.route('/predict_conductivity', methods=['GET'])
@jwt_required()
def predict_conductivity():
    # Ambil input tanggal dari parameter URL, contoh: 06082024 dan 08082024
    start_date_input = request.args.get('start_date')
    end_date_input = request.args.get('end_date')

    # Pastikan input tanggal valid
    if not start_date_input or not end_date_input:
        return jsonify({'error': 'start_date and end_date parameters are required'}), 400

    try:
        # Konversi input DDMMYYYY ke format ISO
        start_date_iso = convert_date_format(start_date_input)
        end_date_iso = convert_date_format(end_date_input)
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use DDMMYYYY.'}), 400

    # URL dan parameter API untuk mengambil data
    url = os.getenv('EXTERNAL_API_URL')
    params = {
        "start": start_date_iso,  # Tanggal mulai dalam format ISO
        "end": end_date_iso,  # Tanggal akhir dalam format ISO
        "device": "AI349454596D98"
    }

    # Basic authentication
    username = os.getenv('EXTERNAL_API_USERNAME')
    password = os.getenv('EXTERNAL_API_PASSWORD')

    # Mengambil data dari API eksternal
    response = requests.get(url, params=params, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        # Mengambil data JSON
        data = response.json()

        # *** Conductivity Section ***
        if 'conductivity' in data:
            conductivity_data = pd.DataFrame(data['conductivity'])

            # Membuat sequences untuk conductivity
            time_steps = 30
            X_test_conductivity, time_conductivity = create_sequences(
                conductivity_data[['value']], conductivity_data['time'], time_steps)

            # Prediksi menggunakan model conductivity
            X_pred_test_conductivity = model_conductivity.predict(X_test_conductivity)

            # Menghitung MAE loss untuk conductivity
            mae_loss_test_conductivity = pd.DataFrame(
                np.mean(np.abs(X_pred_test_conductivity - X_test_conductivity), axis=1), columns=['Error'])

            # Deteksi anomali berdasarkan aturan yang diberikan
            conductivity_anomaly = detect_anomaly_conductivity(mae_loss_test_conductivity['Error'].values,
                                                               conductivity_data['value'].values)
        else:
            return jsonify({'error': 'Conductivity data not found in the response'}), 400

        # Mengembalikan hasil prediksi dalam bentuk JSON untuk conductivity
        return jsonify({
            'conductivity_mae_loss': mae_loss_test_conductivity['Error'].tolist(),
            'conductivity_time': time_conductivity,  # Menambahkan waktu untuk conductivity
            'conductivity_value': conductivity_data['value'].tolist(),  # Menambahkan nilai asli conductivity
            'conductivity_anomaly': conductivity_anomaly  # Menambahkan status anomali (True/False)
        })
    else:
        return jsonify({'error': 'Failed to retrieve data from external API', 'status_code': response.status_code}), 400


@app.route('/predict_salinity', methods=['GET'])
@jwt_required()
def predict_salinity():
    # Ambil input tanggal dari parameter URL, contoh: 06082024 dan 08082024
    start_date_input = request.args.get('start_date')
    end_date_input = request.args.get('end_date')

    # Pastikan input tanggal valid
    if not start_date_input or not end_date_input:
        return jsonify({'error': 'start_date and end_date parameters are required'}), 400

    try:
        # Konversi input DDMMYYYY ke format ISO
        start_date_iso = convert_date_format(start_date_input)
        end_date_iso = convert_date_format(end_date_input)
    except ValueError:
        return jsonify({'error': 'Invalid date format. Use DDMMYYYY.'}), 400

    # URL dan parameter API untuk mengambil data
    url = os.getenv('EXTERNAL_API_URL')
    params = {
        "start": start_date_iso,  # Tanggal mulai dalam format ISO
        "end": end_date_iso,  # Tanggal akhir dalam format ISO
        "device": "AI349454596D98"
    }

    # Basic authentication
    username = os.getenv('EXTERNAL_API_USERNAME')
    password = os.getenv('EXTERNAL_API_PASSWORD')

    # Mengambil data dari API eksternal
    response = requests.get(url, params=params, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        # Mengambil data JSON
        data = response.json()

        # *** Salinity Section ***
        if 'salinity' in data:
            salinity_data = pd.DataFrame(data['salinity'])

            # Membuat sequences untuk salinity
            time_steps = 30
            X_test_salinity, time_salinity = create_sequences(
                salinity_data[['value']], salinity_data['time'], time_steps)

            # Prediksi menggunakan model salinity
            X_pred_test_salinity = model_salinity.predict(X_test_salinity)

            # Menghitung MAE loss untuk salinity
            mae_loss_test_salinity = pd.DataFrame(np.mean(np.abs(X_pred_test_salinity - X_test_salinity), axis=1),
                                                  columns=['Error'])

            # Deteksi anomali berdasarkan aturan yang diberikan
            salinity_anomaly = detect_anomaly_salinity(mae_loss_test_salinity['Error'].values,
                                                       salinity_data['value'].values)
        else:
            return jsonify({'error': 'Salinity data not found in the response'}), 400

        # Mengembalikan hasil prediksi dalam bentuk JSON untuk salinity
        return jsonify({
            'salinity_mae_loss': mae_loss_test_salinity['Error'].tolist(),
            'salinity_time': time_salinity,  # Menambahkan waktu untuk salinity
            'salinity_value': salinity_data['value'].tolist(),  # Menambahkan nilai asli salinity
            'salinity_anomaly': salinity_anomaly  # Menambahkan status anomali (True/False)
        })
    else:
        return jsonify({'error': 'Failed to retrieve data from external API', 'status_code': response.status_code}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running and healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True)
