import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

#==============================
# Load the pre-trained model
loaded_model = tf.keras.models.load_model('/Users/User/Desktop/Highest Ranking Model/2500/CNN-R_Train45_Even_Final_2500posts_150words.h5')  # Update the path

#==============================
# Load the tokenizer
tokenizer_path = '/Users/User/Desktop/Highest Ranking Model/2500/CNN-R_Train45_Even_tokenizer_2500posts_150words.pkl'  # Update the path
with open(tokenizer_path, 'rb') as pickle_file:
    tokenizer = pickle.load(pickle_file)

#==============================
# Function to preprocess CSV data and make predictions
def label_csv_file(input_csv_file, output_csv_file, tokenizer):
    """
    Preprocess the CSV data, make predictions using the pre-trained model, and save the results with predicted labels.
    """
    # Load the CSV file
    df = pd.read_csv(input_csv_file)  # Adjust the read_csv parameters as needed

    # Filter out rows with non-textual data in the 'posts' column
    df = df.dropna(subset=['posts'])
    df = df[df['posts'].apply(lambda x: isinstance(x, str))]

    # Preprocess the CSV data
    max_sequence_length = 150  # Ensure this matches the sequence length used during training

    # Assuming the text data is in a column named 'posts'
    X_csv_sequences = tokenizer.texts_to_sequences(df['posts'])
    X_csv_padded = pad_sequences(X_csv_sequences, maxlen=max_sequence_length, padding='post', truncating='post')
    X_csv_padded = X_csv_padded.astype('float32')  # Convert to float32 if needed

    # Make predictions on the CSV data
    y_pred = loaded_model.predict(X_csv_padded)

    # Choose an appropriate threshold based on model evaluation on a validation set
    threshold = 0.5
    y_pred_binary = [1 if pred >= threshold else 0 for pred in y_pred]

    # Add a new column 'Predicted_Label' to the DataFrame with the predicted labels
    df['Predicted_Label'] = y_pred_binary

    # Save the labeled CSV file
    df.to_csv(output_csv_file, index=False)  # Adjust the to_csv parameters as needed

    print("CSV labeling complete.")

#==============================
# Specify the input and output CSV file paths
input_csv_file = '/Users/User/Documents/reddit_anonymous/reddit_comment_research.csv'
output_csv_file = '/Users/User/Desktop/Trial_Research_Results_27.12.23.csv'

#==============================
# Label the CSV file
label_csv_file(input_csv_file, output_csv_file, tokenizer)
