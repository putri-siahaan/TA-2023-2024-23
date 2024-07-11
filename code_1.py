from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import classification_report
from werkzeug.utils import secure_filename
import pandas as pd
import os
print(os.getcwd())


app = Flask(__name__)

# Menentukan direktori untuk menyimpan file yang diupload
UPLOAD_FOLDER = 'C:/Users/user/Downloads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained CNN model and tokenizer
cnn_model = load_model('cnn_model.h5', compile=False)
tokenizer = joblib.load('cnn_tokenizer.joblib')  
combined_slang_word = 'combined_slang_words.txt'

# Define class labels
class_labels = ['ANGER', 'DISGUST', 'FEAR', 'HAPPY', 'LOVE', 'SADNESS']

# Fungsi untuk membersihkan teks
def clean_text(text):

    # Hapus karakter khusus
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    return text

# Fungsi untuk case folding
def case_folding(text):
    return text.lower()

# Fungsi untuk tokenisasi
def tokenize(text):
    return word_tokenize(text)

# Fungsi untuk normalisasi
def normalisasi(tweet):
    kamus_slangword = eval(open(combined_slang_word).read()) # Membuka dictionary slangword
    pattern = re.compile(r'\b( ' + '|'.join (kamus_slangword.keys())+r')\b') # Search pola kata (contoh kpn -> kapan)
    content = []
    for kata in tweet:
        filteredSlang = pattern.sub(lambda x: kamus_slangword[x.group()],kata) # Replace slangword berdasarkan pola review yg telah ditentukan
        content.append(filteredSlang.lower())
    tweet = content
    return tweet

# Fungsi untuk stopword removal
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('indonesian'))  # You can change to other languages if needed
    return [token for token in tokens if token.lower() not in stop_words]

# Fungsi untuk stemming
def stemming(tokens):
    # Inisialisasi Stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(token) for token in tokens]

# Preprocessing function for CNN model
def preprocess_text_cnn(text):
    # Clean text
    cleaned_text = clean_text(text)

    # Case folding
    lower_text = case_folding(cleaned_text)

    # Tokenize text
    tokens = tokenize(lower_text)

    # Normalize text
    normalized_tokens = normalisasi(tokens)

    # Convert tokens to sequences using the tokenizer
    text_sequence = tokenizer.texts_to_sequences([normalized_tokens])

    # Pad sequences
    text_padded = pad_sequences(text_sequence, maxlen=100, padding='post')

    return text_padded


    

def classify_text(text):
    text_padded = preprocess_text_cnn(text)
    predictions = cnn_model.predict(text_padded)[0]
    max_index = np.argmax(predictions)
    highest_percentage = round(predictions[max_index] * 100, 2)
    return class_labels[max_index], highest_percentage

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    try:
        emotion, percentage = classify_text(text)
        result = {emotion: percentage}
    except Exception as e:
        result = {'Error': str(e)}
    return render_template('index.html', result=result, input_text=text)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"  # Handle missing file part in form
    file = request.files['file']
    if file.filename == '':
        return "No selected file"  # Handle no file selected
    if not file.filename.endswith('.xlsx'):
        return "Invalid file type"  # Handle unsupported file types

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        df = pd.read_excel(file_path)
        if 'Text' not in df.columns:
            return "Excel file must contain a 'Text' column"
        df['Emotion'] = df['Text'].apply(lambda x: classify_text(x)[0])
        
        output_filename = 'processed_' + filename
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        df.to_excel(output_path, index=False)
        
        return f'File uploaded and processed successfully, saved as {output_path}'
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)