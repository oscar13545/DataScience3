
import streamlit as st
import joblib
import tensorflow as tf
import keras
import numpy as np

with open("Michael_Jackson.txt") as f:
    shakespeare_text = f.read()
    
"".join(sorted(set(shakespeare_text.lower())))

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(shakespeare_text)

tokenizer.texts_to_sequences(["First"])

tokenizer.sequences_to_texts([[20, 6, 9, 8, 3]])
max_id = len(tokenizer.word_index) # number of distinct characters
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

def complete_song(seed_text, num_lines=4, chars_per_line=50, temperature=1):
    song = seed_text

    for _ in range(num_lines):
        for _ in range(chars_per_line):
            song += next_char(song, temperature)
        song += '\n'  # Add a line break after each line

    return song

# Load the model
model = joblib.load("MichaelJackson_model.pkl")

def main():
    st.title("Song Generator")

    # Get user input for the starting word or phrase
    seed_text = st.text_input("Enter the starting word or phrase")

    if seed_text:
        # Generate the song
        tf.random.set_seed(42)
        song = complete_song(seed_text, num_lines=8, chars_per_line=80, temperature=0.5)

        # Display the generated song
        st.subheader("Generated Song")
        st.text_area("Song Lyrics", value=song, height=400)

if __name__ == "__main__":
    main()
