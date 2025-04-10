import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Función para entrenar y guardar el modelo
def train_model():
    # Cargar los datos
    df = pd.read_csv('spam.csv', encoding='latin-1')

    # Renombrar y seleccionar columnas necesarias
    df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "message"})
    df = df[['label', 'message']]

    # Eliminar filas con valores nulos
    df.dropna(inplace=True)

    # Mapear etiquetas y eliminar filas que queden como NaN
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df.dropna(subset=['label'], inplace=True)

    # Separar datos
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    # Crear el modelo
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)

    # Guardar el modelo
    joblib.dump(model, 'spam_classifier.pkl')
    print("✅ Modelo entrenado y guardado exitosamente.")

# Función para predecir si un mensaje es spam
def predict_message(message):
    if not message.strip():
        return "Mensaje vacío"
    
    model = joblib.load('spam_classifier.pkl')
    truncated = message[:10]
    prediction = model.predict([truncated])[0]
    return "Spam" if prediction == 1 else "No Spam"

# Entrenar el modelo
train_model()

# Ejemplo de uso
mensaje_ejemplo = "Ganaste un premio, haz clic aquí"
print(f"Mensaje: {mensaje_ejemplo}")
print(f"Clasificación: {predict_message(mensaje_ejemplo)}")

