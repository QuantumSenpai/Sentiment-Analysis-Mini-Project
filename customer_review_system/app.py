from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from werkzeug.utils import secure_filename
from models.preprocessor import TextPreprocessor
from models.sentiment_analyzer import SentimentAnalyzer

app = Flask(__name__)

# Create data folder if not exists
UPLOAD_FOLDER = 'data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

preprocessor = TextPreprocessor()
analyzer = SentimentAnalyzer()

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
        
        df = pd.read_csv(filepath)
        
        if 'review_text' not in df.columns or 'rating' not in df.columns:
            return jsonify({'error': 'CSV must have "review_text" and "rating" columns'}), 400
        
        results = process_reviews(df)
        os.remove(filepath)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_reviews(df):
    print("Processing reviews...")
    
    df['cleaned_text'] = df['review_text'].apply(preprocessor.preprocess)
    
    def rating_to_sentiment(rating):
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'
    
    df['sentiment_actual'] = df['rating'].apply(rating_to_sentiment)
    
    df['sentiment_textblob'] = df['cleaned_text'].apply(analyzer.textblob_sentiment)
    
    metrics = analyzer.train_model(df['cleaned_text'], df['sentiment_actual'])
    
    sentiment_counts = df['sentiment_actual'].value_counts().to_dict()
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=15)
    vectorizer.fit(df['cleaned_text'])
    top_aspects = vectorizer.get_feature_names_out().tolist()
    
    aspect_data = []
    for aspect in top_aspects:
        aspect_reviews = df[df['cleaned_text'].str.contains(aspect, case=False, na=False)]
        if len(aspect_reviews) > 0:
            counts = aspect_reviews['sentiment_actual'].value_counts()
            total = len(aspect_reviews)
            aspect_data.append({
                'aspect': aspect,
                'positive': int(counts.get('positive', 0)),
                'neutral': int(counts.get('neutral', 0)),
                'negative': int(counts.get('negative', 0)),
                'total': total
            })
    
    results = {
        'total_reviews': len(df),
        'sentiment_distribution': sentiment_counts,
        'accuracy': round(metrics['accuracy'], 4),
        'aspects': aspect_data,
        'confusion_matrix': metrics['confusion_matrix']
    }
    
    return results

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)