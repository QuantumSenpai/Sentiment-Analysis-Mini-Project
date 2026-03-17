from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import io
import nltk
from werkzeug.utils import secure_filename
from models.preprocessor import TextPreprocessor
from models.sentiment_analyzer import SentimentAnalyzer

# ─── NLTK Auto-Download (Railway deploy ke liye) ─────────────────────────────
def download_nltk_data():
    packages = ['stopwords', 'wordnet', 'punkt', 'punkt_tab', 'omw-1.4']
    for pkg in packages:
        try:
            if pkg in ['stopwords', 'wordnet', 'omw-1.4']:
                nltk.data.find(f'corpora/{pkg}')
            else:
                nltk.data.find(f'tokenizers/{pkg}')
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

download_nltk_data()

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)

UPLOAD_FOLDER = 'data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

preprocessor = TextPreprocessor()
analyzer = SentimentAnalyzer()


# ─── Helper: Read CSV with any encoding ───────────────────────────────────────
def read_csv_safe(filepath):
    """Try multiple encodings. Returns DataFrame or raises ValueError."""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            return df
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            raise ValueError(f"CSV read failed: {str(e)}")
    raise ValueError(f"Could not decode file with any encoding. Use UTF-8 CSV. Detail: {last_error}")


# ─── Helper: Read CSV from bytes (no file save needed) ────────────────────────
def read_csv_from_bytes(file_bytes):
    """Try reading from raw bytes with multiple encodings."""
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            return df
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            raise ValueError(f"CSV parse failed: {str(e)}")
    raise ValueError(f"Cannot decode file. Please save your CSV as UTF-8. Detail: {last_error}")


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # 1. File present check
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Please select a CSV file.'}), 400

    file = request.files['file']

    # 2. Filename check
    if not file or file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    # 3. Extension check
    if not file.filename.lower().endswith('.csv'):
        return jsonify({'error': 'Only .csv files are allowed.'}), 400

    filepath = None
    try:
        # 4. Read file bytes first (avoids encoding issues at save stage)
        file_bytes = file.read()

        if len(file_bytes) == 0:
            return jsonify({'error': 'Uploaded file is empty.'}), 400

        # 5. Parse CSV from bytes (handles all encodings)
        try:
            df = read_csv_from_bytes(file_bytes)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

        # 6. Empty dataframe check
        if df.empty:
            return jsonify({'error': 'CSV file has no data rows.'}), 400

        # 7. Column name check (case-insensitive, strip spaces)
        df.columns = df.columns.str.strip().str.lower()
        if 'review_text' not in df.columns or 'rating' not in df.columns:
            return jsonify({
                'error': f'CSV must have "review_text" and "rating" columns. '
                         f'Found columns: {list(df.columns)}'
            }), 400

        # 8. Drop rows with missing values
        df = df.dropna(subset=['review_text', 'rating'])
        df['review_text'] = df['review_text'].astype(str).str.strip()
        df = df[df['review_text'] != '']

        # 9. Minimum rows check
        if len(df) < 5:
            return jsonify({
                'error': f'Need at least 5 valid reviews. Found only {len(df)} after cleaning.'
            }), 400

        # 10. Rating column numeric check
        try:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df = df.dropna(subset=['rating'])
            df['rating'] = df['rating'].astype(int)
        except Exception:
            return jsonify({'error': 'Rating column must contain numeric values (1-5).'}), 400

        # 11. Rating range check
        if not df['rating'].between(1, 5).all():
            df = df[df['rating'].between(1, 5)]
            if len(df) < 5:
                return jsonify({'error': 'Ratings must be between 1-5. Not enough valid rows.'}), 400

        # 12. Process reviews
        results = process_reviews(df)
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

    finally:
        # Always clean up saved file if any
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass


# ─── Core Processing ──────────────────────────────────────────────────────────
def process_reviews(df):
    # Preprocess text
    df = df.copy()
    df['cleaned_text'] = df['review_text'].apply(safe_preprocess)

    # Rating → Sentiment label
    def rating_to_sentiment(rating):
        try:
            r = int(float(rating))
            if r <= 2:
                return 'negative'
            elif r == 3:
                return 'neutral'
            else:
                return 'positive'
        except Exception:
            return 'neutral'

    df['sentiment_actual'] = df['rating'].apply(rating_to_sentiment)
    df['sentiment_textblob'] = df['cleaned_text'].apply(analyzer.textblob_sentiment)

    # Train model
    metrics = analyzer.train_model(df['cleaned_text'].tolist(), df['sentiment_actual'].tolist())

    # Sentiment distribution
    sentiment_counts = df['sentiment_actual'].value_counts().to_dict()
    # Ensure all keys present
    for key in ['positive', 'neutral', 'negative']:
        sentiment_counts.setdefault(key, 0)

    # Aspect extraction
    aspect_data = extract_aspects(df)

    return {
        'total_reviews': len(df),
        'sentiment_distribution': sentiment_counts,
        'accuracy': round(metrics.get('accuracy', 0), 4),
        'aspects': aspect_data,
        'confusion_matrix': metrics.get('confusion_matrix', [])
    }


def safe_preprocess(text):
    """Preprocess with fallback — never crashes."""
    try:
        return preprocessor.preprocess(str(text))
    except Exception:
        try:
            # Minimal fallback: just lowercase + strip
            return str(text).lower().strip()
        except Exception:
            return ''


def extract_aspects(df):
    """Extract top keyword aspects with sentiment breakdown."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = df['cleaned_text'].tolist()
        valid_texts = [t for t in texts if t.strip()]
        if len(valid_texts) < 2:
            return []

        vectorizer = TfidfVectorizer(max_features=15, min_df=1)
        vectorizer.fit(valid_texts)
        top_aspects = vectorizer.get_feature_names_out().tolist()

        aspect_data = []
        for aspect in top_aspects:
            try:
                mask = df['cleaned_text'].str.contains(aspect, case=False, na=False, regex=False)
                aspect_reviews = df[mask]
                if len(aspect_reviews) > 0:
                    counts = aspect_reviews['sentiment_actual'].value_counts()
                    aspect_data.append({
                        'aspect': aspect,
                        'positive': int(counts.get('positive', 0)),
                        'neutral': int(counts.get('neutral', 0)),
                        'negative': int(counts.get('negative', 0)),
                        'total': len(aspect_reviews)
                    })
            except Exception:
                continue

        return aspect_data
    except Exception:
        return []


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)