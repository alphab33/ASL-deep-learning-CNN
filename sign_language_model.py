import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from tensorflow.keras.models import Sequential 
import matplotlib.pyplot as plt

class SignLanguageModel:
    def __init__(self):
        self.model = None
        self.labels = ['A', 'B', 'C']

    def load_data(self):
        from datasets import sign_language
        (x_train, y_train), (x_test, y_test) = sign_language.load_data()
        self.y_train_OH = to_categorical(y_train)
        self.y_test_OH = to_categorical(y_test)
        return x_train, y_train, x_test, y_test

    def build_model(self):
        self.model = Sequential([
            Conv2D(filters=5, kernel_size=5, padding='same', activation='relu', input_shape=(50, 50, 3)),
            MaxPooling2D(pool_size=4),
            Conv2D(filters=15, kernel_size=5, padding='same', activation='relu'),
            MaxPooling2D(pool_size=4),
            Flatten(),
            Dense(3, activation='softmax')
        ])
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, x_train, y_train_OH, validation_split=0.2, epochs=2):
        hist = self.model.fit(x_train, y_train_OH, validation_split=validation_split, epochs=epochs)
        return hist

    def evaluate_model(self, x_test, y_test_OH):
        test_loss, test_acc = self.model.evaluate(x_test, y_test_OH)
        return test_loss, test_acc

    def predict(self, x_test):
        y_probs = self.model.predict(x_test)
        y_preds = np.argmax(y_probs, axis=1)
        return y_preds

    def plot_mislabeled_examples(self, x_test, y_test, y_preds):
        bad_test_idxs = np.where(y_preds != y_test)[0]
        num_mislabeled = len(bad_test_idxs)
        rows = 2
        cols = num_mislabeled // 2 + num_mislabeled % 2

        fig = plt.figure(figsize=(25, 4))
        for i, idx in enumerate(bad_test_idxs):
            ax = fig.add_subplot(rows, cols, i + 1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(x_test[idx]))
            ax.set_title(f"{self.labels[y_test[idx]]} (pred: {self.labels[y_preds[idx]]})")

        plt.show()
