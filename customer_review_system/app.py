from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from werkzeug.utils import secure_filename
from models.preprocessor import TextPreprocessor
from models.sentiment_analyzer import SentimentAnalyzer

app = Flask(__name__)

# -------------------------------
# Path Setup (Railway Compatible)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "data")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CSV_PATH = os.path.join(UPLOAD_FOLDER, "customer_reviews.csv")

# Create CSV file if it doesn't exist
if not os.path.exists(CSV_PATH):
    df = pd.DataFrame(columns=["review_text", "rating"])
    df.to_csv(CSV_PATH, index=False)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# -------------------------------
# Initialize Models
# -------------------------------
preprocessor = TextPreprocessor()
analyzer = SentimentAnalyzer()


# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files allowed"}), 400

    try:

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        file.save(filepath)

        df = pd.read_csv(filepath)

        # Validate columns
        if "review_text" not in df.columns or "rating" not in df.columns:
            return jsonify({
                "error": 'CSV must contain "review_text" and "rating" columns'
            }), 400

        results = process_reviews(df)

        # Remove uploaded file after processing
        os.remove(filepath)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Review Processing
# -------------------------------
def process_reviews(df):

    print("Processing reviews...")

    # Clean text
    df["cleaned_text"] = df["review_text"].apply(preprocessor.preprocess)

    # Convert rating to sentiment
    def rating_to_sentiment(rating):

        if rating <= 2:
            return "negative"
        elif rating == 3:
            return "neutral"
        else:
            return "positive"

    df["sentiment_actual"] = df["rating"].apply(rating_to_sentiment)

    # TextBlob prediction
    df["sentiment_textblob"] = df["cleaned_text"].apply(
        analyzer.textblob_sentiment
    )

    # Train ML model
    metrics = analyzer.train_model(
        df["cleaned_text"],
        df["sentiment_actual"]
    )

    # Sentiment counts
    sentiment_counts = df["sentiment_actual"].value_counts().to_dict()

    # -------------------------------
    # Aspect Extraction (TF-IDF)
    # -------------------------------
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

            total = len(aspect_reviews)

            aspect_data.append({
                "aspect": aspect,
                "positive": int(counts.get("positive", 0)),
                "neutral": int(counts.get("neutral", 0)),
                "negative": int(counts.get("negative", 0)),
                "total": total
            })

    results = {
        "total_reviews": len(df),
        "sentiment_distribution": sentiment_counts,
        "accuracy": round(metrics["accuracy"], 4),
        "aspects": aspect_data,
        "confusion_matrix": metrics["confusion_matrix"]
    }

    return results


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)