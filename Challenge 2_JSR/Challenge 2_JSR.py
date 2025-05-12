# %% [markdown]
# # Glassdoor Job Reviews Analysis Pipelin
# **MLOps Challenge Solution**

# %%
# %%capture
# Configuración inicial de librerías
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
from pysentimiento import SentimentAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# %% [markdown]
# ## Etapa 1: Carga de Datos Locales

# %%
# Configurar ruta del archivo
file_path = r"C:\Users\jorge\OneDrive\Documentos\MCD-JORGE SANDOVAL ROSAS\2do Semestre\Programación II\Challenges\Challenge 2\glassdoor_reviews.csv"

# Verificar existencia del archivo
if not os.path.exists(file_path):
    raise FileNotFoundError(f"No se encontró el archivo en la ruta: {file_path}")

# Cargar datos
try:
    df = pd.read_csv(file_path)
    print("Dataset cargado exitosamente")
    print(f"Registros cargados: {len(df)}")
except Exception as e:
    raise ValueError(f"Error al cargar el archivo: {str(e)}")

# %% [markdown]
# ## Etapa 2: Preprocesamiento de Texto

# %%
# Descargar recursos de NLTK
nltk.download(['stopwords', 'wordnet', 'omw-1.4'])

# Verificar columnas requeridas
required_columns = ['review', 'pros', 'cons', 'language', 'rating']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Columnas faltantes en el dataset: {missing_columns}")

# Filtrar y limpiar datos
df = df[required_columns].dropna()
df = df[df['language'].isin(['en', 'es'])]

# %%
def preprocess_text(text, lang):
    """Preprocesamiento de texto para inglés y español"""
    try:
        # Limpieza básica
        text = re.sub(r'[^a-zA-ZáéíóúñüÁÉÍÓÚÑÜ\s]', '', str(text).lower())
        
        # Tokenización y limpieza
        if lang == 'en':
            stemmer = WordNetLemmatizer()
            stop_words = stopwords.words('english')
            tokens = nltk.word_tokenize(text)
        else:
            stemmer = SnowballStemmer('spanish')
            stop_words = stopwords.words('spanish')
            tokens = text.split()
            
        # Lematización/Stemming y filtrado
        processed_tokens = [
            stemmer.lemmatize(token) if lang == 'en' else stemmer.stem(token)
            for token in tokens
            if token not in stop_words and len(token) > 2
        ]
        
        return ' '.join(processed_tokens)
    except Exception as e:
        print(f"Error procesando texto: {e}")
        return ""

# %%
# Aplicar preprocesamiento
df['processed_text'] = df.apply(
    lambda x: preprocess_text(x['review'], x['language']), axis=1
)

# %% [markdown]
# ## Etapa 3: Modelado y Análisis de Sentimientos

# %%
# Configurar MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Glassdoor_Sentiment_Analysis")

# %%
with mlflow.start_run():
    # Registro de parámetros
    mlflow.log_param("dataset_version", "local_file")
    mlflow.log_param("languages", ["en", "es"])
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], 
        df['rating'], 
        test_size=0.25, 
        random_state=42
    )
    
    # Vectorización TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Entrenamiento del modelo
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_tfidf, y_train)
    
    # Evaluación
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", accuracy)
    
    # Registrar modelo
    mlflow.sklearn.log_model(model, "sentiment_classifier")
    print(classification_report(y_test, predictions))

# %%
# Análisis de Sentimientos
analyzer_es = SentimentAnalyzer(lang="es")
vader_analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text, lang):
    """Obtiene sentimiento usando librerías específicas por idioma"""
    try:
        if lang == 'es':
            result = analyzer_es.predict(text)
            return max(result.probas, key=result.probas.get)
        else:
            scores = vader_analyzer.polarity_scores(text)
            return 'pos' if scores['compound'] >= 0.05 else 'neg' if scores['compound'] <= -0.05 else 'neu'
    except:
        return 'neu'

df['sentiment'] = df.apply(
    lambda x: get_sentiment(x['review'], x['language']), axis=1
)

# %% [markdown]
# ## Etapa 4: Pipeline MLOps

# %%
with mlflow.start_run():
    # Registrar artefactos
    mlflow.log_artifact(file_path, "raw_data")
    
    # Gráfico de distribución de sentimientos
    plt.figure(figsize=(10,6))
    df['sentiment'].value_counts().plot(kind='bar', color=['green', 'blue', 'red'])
    plt.title("Distribución de Sentimientos")
    plt.savefig("sentiment_distribution.png")
    mlflow.log_artifact("sentiment_distribution.png")
    
    # Registrar parámetros del vectorizador
    mlflow.log_params(tfidf.get_params())
    
    print("Pipeline MLOps completado")

# %% [markdown]
# ## Instrucciones de Ejecución
# 1. Asegurar que el archivo esté en la ruta especificada
# 2. Iniciar MLflow:
#    ```bash
#    mlflow ui --port 5000
#    ```
# 3. Ejecutar el notebook celda por celda
# 4. Los resultados estarán disponibles en:
#    - MLflow UI: `http://localhost:5000`
#    - Archivos generados en el directorio actual

# %% [markdown]
# **Mejoras clave:**
# - Verificación de existencia del archivo
# - Validación de columnas requeridas
# - Manejo de errores mejorado
# - Ruta raw para compatibilidad con Windows
# - Eliminación de dependencias de Kaggle

# %% [markdown]
# **Notas importantes:**
# 1. Asegúrate de tener los permisos de lectura en la ruta del archivo
# 2. Verifica que la columna 'rating' contenga valores numéricos
# 3. El tiempo de ejecución variará según el tamaño del dataset