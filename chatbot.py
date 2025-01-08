import random
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Text Preprocessing
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    words = word_tokenize(text)
    filtered_words = [lemmatizer.lemmatize(w.lower()) for w in words if w.isalpha()]
    return ' '.join(filtered_words) if filtered_words else 'unknown'


def get_tfidf_features(train_data, test_data):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  # Use bigrams for more context
        max_features=10000,  # Reduce complexity
        min_df=1  # Ignore very rare terms
    )
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    return X_train, X_test, vectorizer

def train_model():
    def load_data():
        print("Loading data...")
        return pd.read_json('intents.json')

    print("Training model...")
    dataset = load_data()
    text_column = 'patterns'
    label_column = 'tag'

    # Flatten the 'patterns' column and create a new DataFrame
    print("Flattening data...")
    flattened_data = []
    for index, row in dataset.iterrows():
        for pattern in row[text_column]:
            flattened_data.append((pattern, row[label_column]))

    # Create a new DataFrame with the flattened data
    flattened_df = pd.DataFrame(flattened_data, columns=[text_column, label_column])
    flattened_df['clean_text'] = flattened_df[text_column].apply(preprocess)

    # Handle classes with too few samples
    print("Filtering low-sample classes...")
    min_class_samples = 2
    class_counts = flattened_df[label_column].value_counts()
    low_count_classes = class_counts[class_counts < min_class_samples].index
    filtered_df = flattened_df[~flattened_df[label_column].isin(low_count_classes)]

    # Check Class Distribution
    print("Filtered Data Distribution:")
    print(filtered_df[label_column].value_counts())

    # Ensure test size is at least equal to the number of classes
    num_classes = filtered_df[label_column].nunique()
    test_size = max(num_classes, int(0.1 * len(filtered_df)))

    print(f"Adjusted test size to accommodate all classes: {test_size}")

    # Stratified split
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    for train_idx, test_idx in stratified_split.split(filtered_df, filtered_df[label_column]):
        train_data = filtered_df.iloc[train_idx]
        test_data = filtered_df.iloc[test_idx]

    # Get Features
    X_train, X_test, vectorizer = get_tfidf_features(train_data['clean_text'], test_data['clean_text'])

    # Handle Class Imbalance Using RandomOverSampler
    print("Handling class imbalance...")
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, train_data[label_column])

    # Train Random Forest Classifier
    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train_resampled, y_train_resampled)

    # Evaluate Model on Test Data
    y_pred = rf_model.predict(X_test)
    print("Classification Report:")
    print(classification_report(test_data[label_column], y_pred))

    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(test_data[label_column], y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=test_data[label_column].unique(), 
        yticklabels=test_data[label_column].unique()
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    print("Model training and loading complete.")
    return vectorizer, rf_model, dataset.to_dict(orient='records')


def get_response(model, vectorizer, user_input, original_dataset):
    processed_input = preprocess(user_input)
    vectorized_input = vectorizer.transform([processed_input]).toarray()
    predicted_tag = model.predict(vectorized_input)[0]

    # Logging for debugging
    print(f"Raw user input: {user_input}")
    print(f"Processed input: {processed_input}")
    print(f"Predicted tag: {predicted_tag}")
    print(f"Vectorized input shape: {vectorized_input.shape}")

    # Debug unexpected tag
    if predicted_tag not in [intent['tag'] for intent in original_dataset]:
        print(f"Unexpected tag: {predicted_tag}. Check model performance.")

    # Retrieve the response
    response_options = None
    for intent in original_dataset:
        if intent['tag'] == predicted_tag:
            response_options = intent['responses']
            break

    if response_options:
        response = random.choice(response_options)
    else:
        response = "I'm not sure how to respond to that. Please try again."

    return response





