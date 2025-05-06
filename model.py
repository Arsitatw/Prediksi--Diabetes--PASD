from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

class DiabetesModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.trained = False

    def train(self, X, y):
        """Melatih model dan mengembalikan skor akurasi."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.trained = True

        accuracy = accuracy_score(y_test, y_pred)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        return accuracy

    def predict(self, input_data):
        """Melakukan prediksi terhadap data input tunggal."""
        if not self.trained:
            raise ValueError("Model belum dilatih. Muat atau latih model terlebih dahulu.")
        return self.model.predict([input_data])[0]

    def save_model(self, filepath):
        """Simpan model ke file .pkl"""
        joblib.dump(self.model, filepath)
        print(f"Model disimpan ke {filepath}")

    def load_model(self, filepath):
        """Muat model dari file .pkl"""
        self.model = joblib.load(filepath)
        self.trained = True
        print(f"Model dimuat dari {filepath}")
