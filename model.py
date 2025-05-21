from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

class DiabetesModel:
    #inisialisai model
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.trained = False

    def train(self, features, labels):
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
        #Melakukan prediksi terhadap data yang di input dan memprediksi model
        if not self.trained:
            raise ValueError("Model belum dilatih. Muat atau latih model terlebih dahulu.")
        return self.model.predict([input_data])[0]

    def save_model(self, filepath):
        #Simpan model ke file .pkl
        joblib.dump(self.model, filepath)
        print(f"Model disimpan ke {filepath}")

    def load_model(self, filepath):
        #Muat model dari file .pkl jika model di perlukan dan berhasil dilatih
        self.model = joblib.load(filepath)
        self.trained = True
        print(f"Model dimuat dari {filepath}")
