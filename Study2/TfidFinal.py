import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import nltk

# Function to preprocess and split the data
def preprocess_and_split_data(excel_file_path):
    # Download NLTK stopwords
    nltk.download('stopwords')

    # Load data from Excel file
    excel_file_path = '/Users/User/Desktop/Highest Ranking Model/Research_Results_27.12.23.xlsm'

    # Read the entire dataset
    full_dataset = pd.concat(pd.read_excel(excel_file_path, sheet_name=None), ignore_index=True)

    # Drop rows with missing values in 'posts' or 'Predicted_Label' columns
    full_dataset = full_dataset.dropna(subset=['posts', 'Predicted_Label'])

    # Convert 'Predicted_Label' to numeric (handle missing values)
    full_dataset['Predicted_Label'] = pd.to_numeric(full_dataset['Predicted_Label'], errors='coerce')

    # Drop rows with missing or non-finite 'Predicted_Label'
    full_dataset = full_dataset.dropna(subset=['Predicted_Label'])
    full_dataset['Predicted_Label'] = full_dataset['Predicted_Label'].astype(int)

    # Separate into depressed and control groups
    depressed_data = full_dataset[full_dataset['Predicted_Label'] == 1]['posts'].astype(str)
    control_data = full_dataset[full_dataset['Predicted_Label'] == 0]['posts'].astype(str)

    # Display the first few rows of each dataset
    print("Depressed Data:")
    print(depressed_data.head())

    print("\nControl Data:")
    print(control_data.head())

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(full_dataset['posts'].astype(str), full_dataset['Predicted_Label'], test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# Function to get top N related words and their coefficients for each class
def get_top_words_and_coefficients(vectorizer, model, class_label, top_n=10):
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]  # Assuming binary classification

    if class_label == 0:
        top_indices = coefficients.argsort()[:top_n]
    else:
        top_indices = coefficients.argsort()[:-top_n-1:-1]

    top_words = [feature_names[idx] for idx in top_indices]
    top_coefficients = [coefficients[idx] for idx in top_indices]

    return top_words, top_coefficients

# Function to evaluate the relationship using TF-IDF and Logistic Regression
def evaluate_relationship(X_train, X_test, y_train, y_test):
    # Define custom stopwords
    custom_stopwords = ['pete']

    # Create a TF-IDF vectorizer with custom stopwords
    stop_words = list(set(stopwords.words('english')).union(custom_stopwords))
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_df=0.85)

    # Transform the training data
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Transform the testing data
    X_test_tfidf = vectorizer.transform(X_test)

    # Create a Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train_tfidf, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test_tfidf)

    # Get top words and coefficients for each class
    top_words_depressed, top_coefficients_depressed = get_top_words_and_coefficients(vectorizer, model, class_label=1, top_n=10)
    top_words_control, top_coefficients_control = get_top_words_and_coefficients(vectorizer, model, class_label=0, top_n=10)

    print("\nTop 10 words for depressed users:")
    for word, coefficient in zip(top_words_depressed, top_coefficients_depressed):
        print(f"{word}: {coefficient:.2f}")

    print("\nTop 10 words for control users:")
    for word, coefficient in zip(top_words_control, top_coefficients_control):
        print(f"{word}: {coefficient:.2f}")

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)

    print(f"\nAccuracy: {accuracy}")
    print("\nClassification Report:")
    print(classification_rep)

# Main script
excel_file_path = '/Users/User/Desktop/Highest Ranking Model/Research_Results_27.12.23.xlsm'
X_train, X_test, y_train, y_test = preprocess_and_split_data(excel_file_path)
evaluate_relationship(X_train, X_test, y_train, y_test)