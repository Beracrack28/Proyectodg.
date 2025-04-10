import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from flask import Flask, request, jsonify, render_template

# Cargar y preparar el modelo solo una vez al iniciar la aplicación
model = None

# Función para cargar el modelo
def load_model():
    global model
    if model is None:
        model = joblib.load('spam_classifier.pkl')

# Cargar dataset desde la ubicación específica del archivo CSV
def train_model():
    # Cargar los datos
    df = pd.read_csv(r"C:\Users\Beracasa\Downloads\Spam\spam.csv", encoding='latin-1')
    df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "message"})
    df = df[['label', 'message']]
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    # Crear y entrenar el modelo
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)

    # Guardar el modelo entrenado
    joblib.dump(model, 'spam_classifier.pkl')

train_model()

# Iniciar la aplicación Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Cargar el modelo solo si no está cargado
        load_model()

        # Obtener el mensaje del formulario
        message = request.form['message']
        
        if not message.strip():
            return jsonify({'error': 'El mensaje no puede estar vacío'}), 400
        
        # Tomar solo las primeras 10 letras del mensaje
        truncated_message = message[:10]  # Tomamos las primeras 10 letras
        
        # Realizar la predicción con el texto truncado
        prediction = model.predict([truncated_message])[0]
        resultado = "Spam" if prediction == 1 else "No Spam"

        # Retornar la respuesta en formato JSON
        return jsonify({'mensaje': message, 'clasificacion': resultado})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Inicializa el modelo cuando la aplicación se inicie
    load_model()

    # Ejecutar la aplicación
    app.run(debug=True)
