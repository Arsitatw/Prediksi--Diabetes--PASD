# Models/model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

class DiabetesModel:
    """
    Kelas untuk membangun, melatih, menyimpan, memuat, dan memprediksi model Random Forest untuk prediksi diabetes.
    """
    def __init__(self):
        # Inisialisasi model Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.trained = False

    def train(self, features, labels):
        """
        Melatih model Random Forest dengan data fitur dan label.
        Data dibagi menjadi 80% training dan 20% testing.
        Mengembalikan akurasi pada data testing.
        """
        features_train, features_test, labels_train, labels_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        self.model.fit(features_train, labels_train)
        predictions = self.model.predict(features_test)
        self.trained = True

        accuracy = accuracy_score(labels_test, predictions)
        print("Classification Report:\n", classification_report(labels_test, predictions))
        return accuracy

    def predict(self, input_data):
        """
        Melakukan prediksi terhadap satu data input.
        """
        if not self.trained:
            raise ValueError("Model belum dilatih. Muat atau latih model terlebih dahulu.")
        return self.model.predict([input_data])[0]

    def predict_proba(self, input_data):
        """
        Mengembalikan probabilitas prediksi.
        """
        if not self.trained:
            raise ValueError("Model belum dilatih.")
        return self.model.predict_proba([input_data])[0]

    def save_RF(self, filepath):
        """
        Menyimpan model ke file .pkl.
        """
        joblib.dump(self.model, filepath)
        print(f"Model disimpan ke {filepath}")

    def load_model(self, filepath):
        """
        Memuat model dari file .pkl.
        """
        self.model = joblib.load(filepath)
        self.trained = True
        print(f"Model dimuat dari {filepath}")
