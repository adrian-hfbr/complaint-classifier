# src/preprocessing.py
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import spmatrix

class ComplaintPreprocessor(BaseEstimator, TransformerMixin):
    """
    Encapsulates all the preprocessing logic for the complaint data.
    This includes cleaning, filtering, and feature extraction.
    """
    def __init__(self, max_features: int = 5000, stop_words: str = 'english'):
        """
        Initializes the preprocessor with a TF-IDF vectorizer.
        """
        self.max_features = max_features
        self.stop_words = stop_words

        self.vectorizer = TfidfVectorizer(
            max_features=max_features, # Keep the top 5,000 most important words
            stop_words=stop_words, # Ignore common, low-value English words 
            ngram_range=(1, 2) # Bigrams to capture more context
        )
        self.target_column = 'Product'
        self.feature_column = 'Consumer complaint narrative'

    def fit(self, df: pd.DataFrame, y: pd.Series = None):
        """
        Fits the TF-IDF vectorizer on the provided DataFrame.

        Args:
            df (pd.DataFrame): The raw training DataFrame.
        """
        # calls the private helper method to get a clean version of the input
        df_clean = self._clean_and_filter(df)
        # Vectorizer reads through all the complaint narratives
        # to build vocabulary of uni- & bigrams & to calculate idf
        # -> trains vectorizer
        self.vectorizer.fit(df_clean[self.feature_column])
        return self

    def transform(self, X: pd.DataFrame) -> spmatrix:
        """Transforms the narrative column into a TF-IDF matrix."""
        # Apply same cleaning and filtering steps as the fit method
        df_clean = self._clean_and_filter(X)
        # Creates numerical tf-idc score matrix X from complaints &
        # learned/trained vocabulary & importance scores
        X = self.vectorizer.transform(df_clean[self.feature_column])
        return X
    
    def get_target(self, df: pd.DataFrame) -> pd.Series:
        """Returns the cleaned and consolidated target column."""
        df_clean = self._clean_and_filter(df)
        return df_clean[self.target_column]

    def _clean_and_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Private helper to orchestrate data cleaning and filtering steps,
        keeping the public fit and transform methods clean and readable.
        """
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df, columns=[self.feature_column])

        df_copy = df.copy()

        # Making the category consolidation conditional,
        # to differentiate between training & inference
        # 1. Drop rows with missing narratives (based on EDA finding)
        if self.feature_column in df_copy.columns:
            df_copy.dropna(subset=[self.feature_column], inplace=True)

        # 2. Consolidate product categories (based on EDA finding)
        if self.target_column in df_copy.columns:
            df_copy = self._consolidate_categories(df_copy)

        # 3. Apply text cleaning to the narrative
        if self.feature_column in df_copy.columns:
            df_copy[self.feature_column] = df_copy[self.feature_column].apply(self._clean_text)

        return df_copy

    def _consolidate_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Private helper to consolidate overlapping product categories."""
        # This mapping is derived directly from my EDA
        category_mapping = {
            'Credit reporting': 'Credit reporting, credit repair services, or other personal consumer reports',
            'Credit card': 'Credit card or prepaid card',
            'Prepaid card': 'Credit card or prepaid card',
            'Bank account or service': 'Checking or savings account',
            'Money transfers': 'Money transfer, virtual currency, or money service',
            'Virtual currency': 'Money transfer, virtual currency, or money service',
            'Payday loan': 'Payday loan, title loan, or personal loan'
        }
        # Take target_column, perform mapping and ".fillna" ensures that any categories
        # that are not in the category_mapping dictionary are left unchanged
        df[self.target_column] = df[self.target_column].map(category_mapping).fillna(df[self.target_column])
        return df

    def _clean_text(self, text: str) -> str:
        """Private helper to apply basic text cleaning:
        lowercase and removes non-alphanumeric chars."""
        # safety check for non-string types (like NaN)
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Keep alphanumeric and spaces
        text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
        return text