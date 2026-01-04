# sentiment_analyzer.py - Complete in one file!
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
import re
import string

# Download NLTK data (one-time only)
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
    nltk.download('stopwords')

# ========== 1. CREATE SAMPLE DATA ==========
def create_sample_data():
    """Create your own dataset quickly"""
    reviews = [
        # Positive reviews
        "I love this product! Amazing quality.",
        "Excellent service, very satisfied.",
        "Best purchase ever, highly recommend!",
        "Great value for money, works perfectly.",
        "Wonderful experience, will buy again.",
        "Perfect, exactly what I needed.",
        "Outstanding performance, very happy.",
        "Top quality, exceeded expectations.",
        
        # Negative reviews
        "Worst product ever, complete waste.",
        "Terrible quality, very disappointed.",
        "Bad service, would not recommend.",
        "Poor quality, broke immediately.",
        "Awful experience, never buying again.",
        "Horrible product, regret purchase.",
        "Useless, doesn't work at all.",
        "Disappointing, not worth the money."
    ]
    
    # 1 = Positive, 0 = Negative
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    
    return pd.DataFrame({
        'review': reviews,
        'sentiment': labels
    })

# ========== 2. TEXT CLEANING ==========
def clean_text(text):
    """Simple text cleaning"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def remove_stopwords(text):
    """Remove common stopwords"""
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# ========== 3. TRAIN MODEL ==========
def train_model(df):
    """Train a simple sentiment classifier"""
    
    # Clean the text
    print("📝 Cleaning text data...")
    df['cleaned_review'] = df['review'].apply(clean_text)
    df['cleaned_review'] = df['cleaned_review'].apply(remove_stopwords)
    
    # Split data
    X = df['cleaned_review']
    y = df['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert text to numbers (Bag of Words)
    print("🔢 Converting text to features...")
    vectorizer = CountVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train Naive Bayes (fast and simple)
    print("🤖 Training model...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_vec)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {accuracy:.2%}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    return model, vectorizer

# ========== 4. PREDICT SENTIMENT ==========
def predict_sentiment(model, vectorizer, text):
    """Predict sentiment for new text"""
    # Clean the input
    cleaned_text = clean_text(text)
    cleaned_text = remove_stopwords(cleaned_text)
    
    # Convert to features
    text_vec = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probability[prediction]
    
    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': round(confidence * 100, 2),
        'probabilities': {
            'negative': round(probability[0] * 100, 2),
            'positive': round(probability[1] * 100, 2)
        }
    }

# ========== 5. QUICK TEST WITH TEXTBLOB ==========
def quick_sentiment_with_textblob(text):
    """Even faster alternative using TextBlob"""
    from textblob import TextBlob
    
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return {
        'text': text,
        'sentiment': sentiment,
        'polarity': round(polarity, 3),
        'subjectivity': round(analysis.sentiment.subjectivity, 3)
    }

# ========== 6. MAIN EXECUTION ==========
def main():
    print("🚀 FAST SENTIMENT ANALYSIS - Starting...\n")
    
    # Option 1: Train your own model
    print("="*50)
    print("OPTION 1: Train Custom Model")
    print("="*50)
    
    # Create and train
    df = create_sample_data()
    model, vectorizer = train_model(df)
    
    # Test predictions
    test_texts = [
        "I really like this!",
        "This is terrible.",
        "Not bad, but could be better",
        "Amazing product, love it!"
    ]
    
    print("\n🧪 Testing Predictions:")
    print("-" * 50)
    for text in test_texts:
        result = predict_sentiment(model, vectorizer, text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']} ({result['confidence']}% confident)")
        print("-" * 30)
    
    # Option 2: Use TextBlob (even faster)
    print("\n" + "="*50)
    print("OPTION 2: Quick TextBlob Analysis")
    print("="*50)
    
    for text in test_texts:
        result = quick_sentiment_with_textblob(text)
        print(f"Text: {text}")
        print(f"Sentiment: {result['sentiment']} (Polarity: {result['polarity']})")
        print("-" * 30)
    
    # Interactive mode
    print("\n" + "="*50)
    print("🎮 INTERACTIVE MODE")
    print("="*50)
    print("Type 'quit' to exit")
    
    while True:
        user_input = input("\nEnter text to analyze: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        # Use trained model
        result = predict_sentiment(model, vectorizer, user_input)
        
        print(f"\n📊 Analysis Results:")
        print(f"Text: {user_input}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Probabilities → Negative: {result['probabilities']['negative']}%, "
              f"Positive: {result['probabilities']['positive']}%")
        
        # Show TextBlob result too
        tb_result = quick_sentiment_with_textblob(user_input)
        print(f"\n📈 TextBlob Analysis:")
        print(f"Sentiment: {tb_result['sentiment']}")
        print(f"Polarity: {tb_result['polarity']} (Range: -1 to 1)")
        print(f"Subjectivity: {tb_result['subjectivity']} (0=Objective, 1=Subjective)")

# ========== 7. SIMPLE COMMAND LINE INTERFACE ==========
def cli_interface():
    """Super simple command line interface"""
    print("🌟 Simple Sentiment Analyzer 🌟")
    print("\nCommands:")
    print("  analyze <text>  - Analyze sentiment of text")
    print("  train          - Train model on sample data")
    print("  test           - Test with example texts")
    print("  quit           - Exit")
    
    df = create_sample_data()
    model, vectorizer = train_model(df)
    
    while True:
        command = input("\n> ").strip().lower()
        
        if command.startswith('analyze '):
            text = command[8:]  # Remove 'analyze ' prefix
            result = predict_sentiment(model, vectorizer, text)
            print(f"Result: {result['sentiment']} ({result['confidence']}% confident)")
        
        elif command == 'train':
            print("✅ Model already trained on sample data!")
        
        elif command == 'test':
            test_texts = ["Great!", "Bad", "Okay"]
            for text in test_texts:
                result = predict_sentiment(model, vectorizer, text)
                print(f"{text} → {result['sentiment']}")
        
        elif command == 'quit':
            print("Goodbye!")
            break
        
        else:
            print("Unknown command. Try: analyze <text>, train, test, quit")

# ========== RUN ==========
if __name__ == "__main__":
    # Run the full program
    main()
    
    # Or run the simple CLI:
    # cli_interface()