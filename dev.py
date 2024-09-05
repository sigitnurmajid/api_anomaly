print('test')

import os
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError

print("Current working directory:", os.getcwd())  # Cek direktori kerja saat ini
print("Model path exists:", os.path.exists('anomaly_detection_model_conductivity.h5'))  # Cek apakah file model ada

try:
    loaded_model_conductivity = load_model('anomaly_detection_model_conductivity.h5',
                                           custom_objects={'mae': MeanAbsoluteError()})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
