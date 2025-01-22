# ___Optimized for M1 Mac___
import os
import csv
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
import pickle

#==============================
# Optimizing TensorFlow settings for M1 Mac
os.environ['TF_NUM_INTEROP_THREADS'] = '8'  # Adjust based on your hardware
os.environ['TF_NUM_INTRAOP_THREADS'] = '16'  # Adjust based on your hardware

print("Initiating CNN-R_CSV")

# Define the maximum sequence length
MAX_SEQUENCE_LENGTH = 50

#==============================
def load_csv_data(file_path):
    """
    Loads the CSV file, processes the posts and labels for each user.
    Randomly samples posts from each user to balance the data.
    """
    data = []
    user_posts = {}

    # Open the CSV file and process each row
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            user_id = row['ID']  # User identifier
            post = row['Posts']
            label = row['Label']  # Label value

            if user_id not in user_posts:
                user_posts[user_id] = []

            user_posts[user_id].append((post, label))

    # Sample data if the user has more than 2000 posts
    sampled_data = []
    for user_id, posts in user_posts.items():
        if len(posts) >= 2000:
            sampled_data.extend(random.sample(posts, 2000))
        else:
            sampled_data.extend(posts)

    for post, label in sampled_data:
        data.append({'Posts': post, 'Label': label})

    return data

#==============================
def preprocess_data(data, tokenizer=None):
    """
    Preprocesses the data: tokenizes and pads the text, and maps labels to 0 and 1.
    """
    X, y = [], []

    for record in data:
        post = record['Posts']
        label = record['Label']

        # Map labels to binary: 'depression' -> 1, others -> 0
        label = 1 if label.lower() == 'depression' else 0

        # Process each post
        for text in post.split('\n'):
            if isinstance(text, str):
                lowercase_text = text.lower()
                X.append(lowercase_text)
                y.append(label)

    if tokenizer is None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X)

    X_sequences = tokenizer.texts_to_sequences(X)
    X_padded = pad_sequences(X_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post').astype('float32')

    return X_padded, np.array(y), tokenizer

#==============================
# Load data from CSV files
train_data = load_csv_data('/Users/User/Documents/RSDD_zip/RSDD/CSV/train_45_even.csv')
validation_data = load_csv_data('/Users/User/Documents/RSDD_zip/RSDD/CSV/validation_40.csv')
test_data = load_csv_data('/Users/User/Documents/RSDD_zip/RSDD/CSV/testing_40.csv')

#==============================
# Preprocess the training, validation, and test data
X_train, y_train, tokenizer = preprocess_data(train_data)
X_validation, y_validation, _ = preprocess_data(validation_data, tokenizer)
X_test, y_test, _ = preprocess_data(test_data, tokenizer)

#==============================
# Save the tokenizer to a file
tokenizer_path = '/Users/User/Desktop/CNN-R_Train45_Even_tokenizer_1.pkl'
with open(tokenizer_path, 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

#==============================
# Define a function to create the Keras model
def create_model(activation='relu', optimizer='adam', dropout_rate=0.0, dense_units=256):
    """
    Creates and compiles a CNN model.
    """
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=MAX_SEQUENCE_LENGTH)(input_layer)
    conv_layer = Conv1D(filters=256, kernel_size=7, activation=activation)(embedding_layer)
    pooling_layer = GlobalMaxPooling1D()(conv_layer)
    dense_layer = Dense(dense_units, activation=activation)(pooling_layer)
    output_layer = Dense(1, activation='sigmoid')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model

#==============================
# Wrap the model for scikit-learn compatibility
model = KerasClassifier(build_fn=create_model, epochs=3, batch_size=32, verbose=3)

#==============================
# Initialize StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#==============================
# Train the model
model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=3, batch_size=32, verbose=3)

#==============================
# Save the trained model
model.model.save('/Users/User/Desktop/CNN-R_Train45_Even_Final_1.h5')

#==============================
# Evaluate the model on the test data
y_pred = model.predict(X_test)
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

#==============================
# Compute and display accuracy
accuracy = accuracy_score(y_test, y_pred_binary)
print(f'Accuracy: {accuracy:.2f}')

#==============================
# Generate a detailed classification report
report = classification_report(y_test, y_pred_binary, labels=[0, 1], target_names=['control', 'depression'])
print('Classification Report:\n', report)

print("Model training complete")
