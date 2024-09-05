from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError
import requests
from requests.auth import HTTPBasicAuth
import json
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load the saved model for salinity
model_salinity = load_model('anomaly_detection_model_salinity.h5',
                            custom_objects={'mae': MeanAbsoluteError()})


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
def detect_anomaly(mae_loss, value, threshold=71):
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


# Endpoint untuk debugging data salinity
@app.route('/debug_salinity', methods=['GET'])
def debug_salinity():
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
    url = "http://206.189.40.105:8010/telemetry"
    params = {
        "start": start_date_iso,  # Tanggal mulai dalam format ISO
        "end": end_date_iso,  # Tanggal akhir dalam format ISO
        "device": "AI349454596D98"
    }

    # Basic authentication
    username = "admin"
    password = "Ejs0dv-32lsdfnxz"

    # Mengambil data dari API eksternal
    response = requests.get(url, params=params, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        # Mengambil data JSON
        data = response.json()

        # *** Salinity Section ***
        if 'salinity' in data:
            salinity_data = pd.DataFrame(data['salinity'])

            # Debugging - Cek data asli yang diterima dari API eksternal
            print("Data Salinity dari API eksternal:")
            print(salinity_data.head())  # Cek beberapa record pertama

            # Membuat sequences untuk salinity
            time_steps = 30
            X_test_salinity, time_salinity = create_sequences(
                salinity_data[['value']], salinity_data['time'], time_steps)

            # Debugging - Cek input yang masuk ke model
            print("Input ke Model Salinity (5 sample pertama):")
            print(X_test_salinity[:5])

            # Prediksi menggunakan model salinity
            X_pred_test_salinity = model_salinity.predict(X_test_salinity)

            # Debugging - Cek hasil prediksi dari model
            print("Hasil Prediksi Model Salinity (5 prediksi pertama):")
            print(X_pred_test_salinity[:5])

            # Menghitung MAE loss untuk salinity
            mae_loss_test_salinity = pd.DataFrame(np.mean(np.abs(X_pred_test_salinity - X_test_salinity), axis=1),
                                                  columns=['Error'])

            # Deteksi anomali berdasarkan aturan yang diberikan
            salinity_anomaly = detect_anomaly(mae_loss_test_salinity['Error'].values, salinity_data['value'].values)

            # Debugging - Cek MAE loss dan anomali yang terdeteksi
            print("MAE Loss Salinity (5 sample pertama):")
            print(mae_loss_test_salinity.head())

            print("Anomali Salinity (5 sample pertama):")
            print(salinity_anomaly[:5])

        else:
            return jsonify({'error': 'Salinity data not found in the response'}), 400

        # Mengembalikan hasil prediksi dalam bentuk JSON untuk salinity
        return jsonify({
            'salinity_prediction': X_pred_test_salinity.tolist(),
            'salinity_mae_loss': mae_loss_test_salinity['Error'].tolist(),
            'salinity_time': time_salinity,  # Menambahkan waktu untuk salinity
            'salinity_value': salinity_data['value'].tolist(),  # Menambahkan nilai asli salinity
            'salinity_anomaly': salinity_anomaly  # Menambahkan status anomali (True/False)
        })
    else:
        return jsonify({'error': 'Failed to retrieve data from external API', 'status_code': response.status_code}), 400


# Route untuk pengecekan API status
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running and healthy'}), 200


# Menjalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)
