import re
import nltk


def _safe_nltk_import():
    """Import NLTK components safely with fallbacks."""
    stopwords_set = set()
    lemmatizer = None

    try:
        from nltk.corpus import stopwords
        stopwords_set = set(stopwords.words('english'))
    except Exception:
        try:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            stopwords_set = set(stopwords.words('english'))
        except Exception:
            # Minimal English stopwords hardcoded as fallback
            stopwords_set = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
                'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who',
                'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
                'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
                'does', 'did', 'will', 'would', 'could', 'should', 'may',
                'might', 'shall', 'can', 'a', 'an', 'the', 'and', 'but',
                'if', 'or', 'because', 'as', 'of', 'at', 'by', 'for',
                'with', 'about', 'against', 'into', 'through', 'to', 'from',
                'in', 'on', 'not', 'no', 'so', 'than', 'too', 'very',
                'just', 'also', 'then', 'than', 'when', 'there', 'here'
            }

    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
    except Exception:
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            from nltk.stem import WordNetLemmatizer
            lemmatizer = WordNetLemmatizer()
        except Exception:
            lemmatizer = None

    return stopwords_set, lemmatizer


class TextPreprocessor:
    def __init__(self):
        self.stop_words, self.lemmatizer = _safe_nltk_import()

    def clean_text(self, text):
        """Lowercase, remove URLs, special chars, extra spaces."""
        try:
            text = str(text).lower()
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            text = re.sub(r'[^a-z\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception:
            return str(text).lower().strip()

    def tokenize(self, text):
        """Tokenize with NLTK; fallback to simple split."""
        try:
            from nltk.tokenize import word_tokenize
            return word_tokenize(text)
        except Exception:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
                from nltk.tokenize import word_tokenize
                return word_tokenize(text)
            except Exception:
                # Simple split fallback — always works
                return text.split()

    def remove_stopwords(self, tokens):
        """Remove stopwords from token list."""
        try:
            return [w for w in tokens if w not in self.stop_words and len(w) > 1]
        except Exception:
            return tokens

    def lemmatize_tokens(self, tokens):
        """Lemmatize tokens; fallback to original tokens."""
        if self.lemmatizer is None:
            return tokens
        result = []
        for word in tokens:
            try:
                result.append(self.lemmatizer.lemmatize(word))
            except Exception:
                result.append(word)
        return result

    def preprocess(self, text):
        """Full pipeline: clean → tokenize → remove stopwords → lemmatize."""
        try:
            text = self.clean_text(text)
            if not text.strip():
                return ''
            tokens = self.tokenize(text)
            tokens = self.remove_stopwords(tokens)
            tokens = self.lemmatize_tokens(tokens)
            result = ' '.join(tokens).strip()
            return result if result else text  # fallback to cleaned text
        except Exception:
            # Ultimate fallback — just return cleaned text
            try:
                return self.clean_text(text)
            except Exception:
                return str(text).lower().strip()