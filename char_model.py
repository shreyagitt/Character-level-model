import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

class CharRNN(Model):
    def __init__(self, vocab_size, embedding_dim=64, lstm_units=128):
        super(CharRNN, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(lstm_units, return_sequences=False)
        self.dropout = Dropout(0.2)
        self.dense = Dense(vocab_size, activation='softmax')

    def call(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.dropout(x)
        return self.dense(x)
