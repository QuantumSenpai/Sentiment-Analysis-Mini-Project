from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.model = MultinomialNB()
        self.is_trained = False

    def textblob_sentiment(self, text):
        """TextBlob-based sentiment. Fallback if ML model not trained."""
        try:
            polarity = TextBlob(str(text)).sentiment.polarity
            if polarity > 0.1:
                return 'positive'
            elif polarity < -0.1:
                return 'negative'
            else:
                return 'neutral'
        except Exception:
            return 'neutral'

    def train_model(self, texts, labels):
        """Train Naive Bayes. Handles all edge cases safely."""
        try:
            # Filter out empty texts
            valid_pairs = [(t, l) for t, l in zip(texts, labels) if str(t).strip()]
            if len(valid_pairs) < 4:
                return self._fallback_metrics()

            texts_clean, labels_clean = zip(*valid_pairs)
            texts_clean = list(texts_clean)
            labels_clean = list(labels_clean)

            # Need at least 2 unique classes to train
            unique_classes = set(labels_clean)
            if len(unique_classes) < 2:
                return self._fallback_metrics()

            X = self.vectorizer.fit_transform(texts_clean)

            # Try stratified split; fallback to non-stratified
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, labels_clean,
                    test_size=0.2,
                    random_state=42,
                    stratify=labels_clean
                )
            except ValueError:
                # Happens when a class has too few samples for stratify
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, labels_clean,
                        test_size=0.2,
                        random_state=42
                    )
                except ValueError:
                    # Dataset too small to split at all — train on everything
                    self.model.fit(X, labels_clean)
                    self.is_trained = True
                    return self._fallback_metrics()

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            self.is_trained = True

            # Build confusion matrix safely
            try:
                cm = confusion_matrix(y_test, y_pred).tolist()
            except Exception:
                cm = []

            try:
                report = classification_report(y_test, y_pred)
            except Exception:
                report = "N/A"

            return {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'report': report,
                'confusion_matrix': cm
            }

        except Exception as e:
            # Last resort fallback — never crash
            return self._fallback_metrics()

    def predict(self, text):
        """Predict sentiment for a single text."""
        try:
            if not self.is_trained:
                return self.textblob_sentiment(text)
            X = self.vectorizer.transform([str(text)])
            return self.model.predict(X)[0]
        except Exception:
            return self.textblob_sentiment(text)

    def _fallback_metrics(self):
        """Return safe default metrics when training isn't possible."""
        return {
            'accuracy': 0.0,
            'report': 'Not enough data to train model.',
            'confusion_matrix': []
        }