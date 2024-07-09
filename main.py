import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
import string
import re

nltk.download('stopwords')

url = "https://raw.githubusercontent.com/jeffersonhernandezv/datasets/main/sentiment140.csv"
data = pd.read_csv(url, encoding='latin-1', header=None, usecols=[0, 5], names=['sentiment', 'message'])

def map_sentiment(value):
    if value == 0:
        return 'negative'
    elif value == 2:
        return 'neutral'
    elif value == 4:
        return 'positive'

data['sentiment'] = data['sentiment'].apply(map_sentiment)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

data['message'] = data['message'].apply(preprocess_text)

X_train, X_test, y_train, y_test = train_test_split(data['message'], data['sentiment'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

lr_classifier = LogisticRegression(max_iter=1000)
lr_classifier.fit(X_train_vectorized, y_train)
lr_predictions = lr_classifier.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, lr_predictions)
classification_rep = classification_report(y_test, lr_predictions)
conf_matrix = confusion_matrix(y_test, lr_predictions)

print(f"Accuracy: {accuracy}\n")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)

results = pd.DataFrame({'Message': X_test, 'Predicted Sentiment': lr_predictions})
sample_results = results.sample(5, random_state=42)
for index, row in sample_results.iterrows():
    print(f"Message: {row['Message']}\nPredicted Sentiment: {row['Predicted Sentiment']}\n")
