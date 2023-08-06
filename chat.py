import json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate
from tensorflow.keras.models import Model
import numpy as np


class Chatbot:
    def __init__(self):
        # Load conversation data from JSON file
        with open('intents.json', 'r') as file:
            self.data = json.load(file)

        # Extract all the unique words from every pattern
        words = []
        labels = []
        docs_x = []
        docs_y = []
        for intent in self.data['intents']:
            for pattern in intent['patterns']:
                # Tokenize each pattern string and add to the list of unique words
                word_list = pattern.lower().split()
                words.extend(word_list)
                docs_x.append(word_list)
                docs_y.append(intent['tag'])
            # Add tag to the list of labels
            if intent['tag'] not in labels:
                labels.append(intent['tag'])

        # Convert words to lowercase and remove duplicates
        words = [word.lower() for word in words]
        words = list(set(words))

        # Tokenize questions and convert to matrix representation
        self.tokenizer = Tokenizer(num_words=len(words))
        self.tokenizer.fit_on_texts(docs_x)

        sequences = self.tokenizer.texts_to_sequences(docs_x)
        self.max_len = max([len(seq) for seq in sequences])
        self.padded_sequences = pad_sequences(sequences, maxlen=self.max_len)

        # Convert labels to one-hot encoded labels
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(labels)

        self.labels_matrix = self.label_encoder.transform(docs_y)
        self.labels_matrix = np.eye(len(labels))[self.labels_matrix]

        self.num_classes = len(labels)

        # Define neural network model
        input_layer = Input(shape=(self.max_len,))
        embedding_layer = Embedding(input_dim=len(words), output_dim=128)(input_layer)
        lstm_layer = LSTM(128)(embedding_layer)
        dense_layer = Dense(64, activation='relu')(lstm_layer)
        output_layer = Dense(self.num_classes, activation='softmax')(dense_layer)

        self.model = Model(input_layer, output_layer)

        # Train neural network model on the training data
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(self.padded_sequences, self.labels_matrix, epochs=50, batch_size=8)

        # Save the trained weights to a file
        self.model.save_weights('chatbot_weights.h5')

    # Load pre-trained weights
    def load(self):
        self.model.load_weights('chatbot_weights.h5')

    # Define a function to get the department prediction from the model
    def predict_intent(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        matrix = pad_sequences(sequence, maxlen=self.max_len)
        predicted_intent_matrix = self.model.predict(matrix)[0]
        predicted_intent_index = np.argmax(predicted_intent_matrix)
        predicted_intent_label = self.label_encoder.inverse_transform([predicted_intent_index])[0]

        return predicted_intent_label

    # Define a function to start the chatbot
    def start_chatbot(self, user_input):
        # Get the predicted intent from the user input
        predicted_intent = self.predict_intent(user_input)

        # Get a response based on the user input and predicted intent
        response = self.get_response(predicted_intent, self.data)

        # Output the response to the user
        return response

    # Define a function to get a response based on the predicted intent
    def get_response(self, predicted_intent, data):
        # Write your logic to get a response based on the predicted intent
        for intent in data['intents']:
            if intent['tag'] == predicted_intent:
                response = np.random.choice(intent['responses'])
                return f"{response}"