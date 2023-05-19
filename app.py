
import streamlit as st
import joblib
import tensorflow as tf
import keras
import numpy as np
#lector del archivo txt
with open("Michael_Jackson.txt") as f:
    shakespeare_text = f.read()
    
"".join(sorted(set(shakespeare_text.lower())))

#Generar los tokens con el archivo de texto
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(shakespeare_text)

tokenizer.texts_to_sequences(["First"])

tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]])
max_id = len(tokenizer.word_index) 
dataset_size = tokenizer.document_count 

def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)

def next_char(text, temperature=1):
    X_new = preprocess([text])
    y_proba = model(X_new)[0, -1:, :]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]
# Es la funcion que genera la canción 
def complete_song(seed_text, num_lines=4, chars_per_line=50, temperature=1):
    song = seed_text

    for _ in range(num_lines):
        for _ in range(chars_per_line):
            song += next_char(song, temperature)
        song += '\n'  

    return song

# Cargar el modelo
model = joblib.load("MichaelJackson_model.pkl")

def main():
    st.title("Generador de letras de Michael Jackson")
    seed_text = st.text_input("Introduzca la frase con la que quiere que empieza la letra")
    song = ""
    
    if seed_text:
        tf.random.set_seed(42)
        song = complete_song(seed_text, num_lines=8, chars_per_line=80, temperature=0.5)
    st.subheader("Generador")
    st.text_area("Letra", value=song, height=400)

if __name__ == "__main__":
    main()
