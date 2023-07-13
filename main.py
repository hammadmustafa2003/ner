
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed
import pickle

app = FastAPI()

# Load the model from the directory
model = tf.keras.models.load_model("ner_model.model")

# Load the max sequence length from file
with open('maxlenseq', 'r') as f:
    max_seq_length = int(f.read())

def load_variables(file_path):
    with open(file_path, 'rb') as file:
        variables = pickle.load(file)
    return variables['maxlenseq'], variables['word2idx'], variables['idx2tag']

max_seq_length, word2idx, idx2tag = load_variables('file.var')

# CORS Configuration
origins = [
    "http://127.0.0.1:5500",  # Add the URL where your HTML file is hosted
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/")
async def root(model_input: str):
    print(model_input)
    # Tokenize and convert the input text into numerical sequences
    input_sequence = [word2idx[word] for word in model_input.split()]
    input_sequence = tf.keras.preprocessing.sequence.pad_sequences([input_sequence], maxlen=max_seq_length, padding="post")

    # Pass the input to the model for prediction
    prediction = model.predict(input_sequence)

    # Convert the prediction to entity labels
    predicted_labels = np.argmax(prediction, axis=-1)[0]
    predicted_entities = [idx2tag[label] for label in predicted_labels]

    # Interpret the output
    output_entities = []
    current_entity = None
    current_words = []

    for word, entity in zip(model_input.split(), predicted_entities):
        if entity != "O":
            if current_entity is None:
                current_entity = entity[2:]
                current_words.append(word)
            elif current_entity != entity[2:]:
                output_entities.append((current_entity, " ".join(current_words)))
                current_entity = entity[2:]
                current_words = [word]
            else:
                current_words.append(word)
        else:
            if current_entity is not None:
                output_entities.append((current_entity, " ".join(current_words)))
                current_entity = None
                current_words = []

    # Return the output entities as JSON response
    return {"entities": output_entities}
