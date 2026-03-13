from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from werkzeug.utils import secure_filename
from models.preprocessor import TextPreprocessor
from models.sentiment_analyzer import SentimentAnalyzer

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

preprocessor = TextPreprocessor()
analyzer = SentimentAnalyzer()

# Possible column names
TEXT_COLUMNS = [
    "review_text", "review", "text", "comment",
    "feedback", "review_body", "review_content"
]

RATING_COLUMNS = [
    "rating", "stars", "score",
    "rating_value", "review_rating"
]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files allowed'}), 400

    try:

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(filepath)

        df = load_csv(filepath)

        review_col, rating_col = detect_columns(df)

        if review_col is None:
            return jsonify({
                'error': 'No review column found in CSV'
            }), 400

        # Rename detected columns
        df = df.rename(columns={review_col: 'review_text'})

        if rating_col:
            df = df.rename(columns={rating_col: 'rating'})

        results = process_reviews(df, rating_available=rating_col is not None)

        try:
            os.remove(filepath)
        except:
            pass

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def load_csv(filepath):

    """Load CSV with multiple encoding attempts"""

    encodings = ["utf-8", "utf-8-sig", "latin-1"]

    for enc in encodings:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except:
            continue

    raise Exception("Unable to read CSV file")


def detect_columns(df):

    """Auto detect review and rating columns"""

    review_col = None
    rating_col = None

    for col in df.columns:

        col_lower = col.lower()

        if review_col is None and col_lower in TEXT_COLUMNS:
            review_col = col

        if rating_col is None and col_lower in RATING_COLUMNS:
            rating_col = col

    return review_col, rating_col


def process_reviews(df, rating_available=True):

    print("Processing reviews...")

    # Remove empty reviews
    df = df.dropna(subset=["review_text"])

    # Preprocess text
    df["cleaned_text"] = df["review_text"].astype(str).apply(
        preprocessor.preprocess
    )

    if rating_available:

        def rating_to_sentiment(r):

            try:
                r = float(r)
            except:
                return "neutral"

            if r <= 2:
                return "negative"
            elif r == 3:
                return "neutral"
            else:
                return "positive"

        df["sentiment_actual"] = df["rating"].apply(rating_to_sentiment)

    else:

        # If no rating column → use TextBlob
        df["sentiment_actual"] = df["cleaned_text"].apply(
            analyzer.textblob_sentiment
        )

    df["sentiment_textblob"] = df["cleaned_text"].apply(
        analyzer.textblob_sentiment
    )

    metrics = analyzer.train_model(
        df["cleaned_text"],
        df["sentiment_actual"]
    )

    sentiment_counts = df["sentiment_actual"].value_counts().to_dict()

    # Aspect extraction
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=15)

    vectorizer.fit(df["cleaned_text"])

    top_aspects = vectorizer.get_feature_names_out().tolist()

    aspect_data = []

    for aspect in top_aspects:

        aspect_reviews = df[
            df["cleaned_text"].str.contains(aspect, case=False, na=False)
        ]

        if len(aspect_reviews) > 0:

            counts = aspect_reviews["sentiment_actual"].value_counts()

            aspect_data.append({
                "aspect": aspect,
                "positive": int(counts.get("positive", 0)),
                "neutral": int(counts.get("neutral", 0)),
                "negative": int(counts.get("negative", 0)),
                "total": len(aspect_reviews)
            })

    results = {
        "total_reviews": int(len(df)),
        "sentiment_distribution": sentiment_counts,
        "accuracy": round(metrics["accuracy"], 4),
        "aspects": aspect_data,
        "confusion_matrix": metrics["confusion_matrix"]
    }

    return results


if __name__ == "__main__":
    app.run(debug=True)