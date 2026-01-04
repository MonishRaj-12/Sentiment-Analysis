# test_your_data.py - Test with your CSV/Excel
import pandas as pd
from textblob import TextBlob

# If you have a CSV file:
# df = pd.read_csv('your_reviews.csv')

# Or create your own:
your_reviews = [
    "The product works well",
    "Not satisfied with quality",
    "Fast delivery, good service",
    # Add your own reviews here...
]

def quick_analyze(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive", analysis.sentiment.polarity
    elif analysis.sentiment.polarity < 0:
        return "Negative", analysis.sentiment.polarity
    else:
        return "Neutral", analysis.sentiment.polarity

print("📊 Your Reviews Analysis:")
for review in your_reviews:
    sentiment, score = quick_analyze(review)
    print(f"\"{review}\"")
    print(f"→ {sentiment} (Score: {score:.2f})")
    print()