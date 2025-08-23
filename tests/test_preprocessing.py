# tests/test_preprocessing.py
import pandas as pd
from scipy.sparse import issparse

# Import the ComplaintPreprocessor Class from the preprocessing file
from src.preprocessing import ComplaintPreprocessor

def test_complaint_preprocessor_integration():
    """
    Integration test for the ComplaintPreprocessor class.
    It verifies that the entire pipeline (filtering, cleaning, consolidation, vectorization)
    works correctly on a sample DataFrame.
    """
    # (1) Arrange: sample df with edge cases
    data = {
        'Product': ['Credit card', 'Mortgage', 'Bank account or service', 'Debt collection'],
        'Consumer complaint narrative': [
            'My VISA card was charged!!', # requires consolidation
            'Complaint about my house mortgage.', # should remain unchanged
            None,  # row should be dropped.
            'They keep CALLING me about my DEBT.' # test cleaning
        ]
    }
    df = pd.DataFrame(data)
    

    # (2) Act: Instantiate and run the full fit-transform pipeline (train & trainsform)
    preprocessor = ComplaintPreprocessor(max_features=10) # Using a small max_features for a predictable test
    preprocessor.fit(df)

    # Call .transform() to get X and .get_target() to get y separately
    X_transformed = preprocessor.transform(df)
    y_transformed = preprocessor.get_target(df)

    # (3) Assert: Verify all expected outcomes

    # 1. Check that the row with the missing narrative was dropped.
    assert X_transformed.shape[0] == 3, "Should drop rows with missing narratives"
    assert len(y_transformed) == 3, "Should drop rows with missing narratives"

    # 2. Check that the output types are correct for scikit-learn.
    assert issparse(X_transformed), "Feature matrix X should be a sparse matrix"
    assert isinstance(y_transformed, pd.Series), "Target y should be a pandas Series"

    # 3. Check that the number of features matches max_features
    # It can be less than max_features if the total vocabulary is smaller.
    assert X_transformed.shape[1] <= 10, "Number of features should be at most max_features"

    # 4. Check that category consolidation works
    expected_labels = [
        'Credit card or prepaid card',  # Was 'Credit card'
        'Mortgage',
        'Debt collection'
    ]
    assert y_transformed.tolist() == expected_labels, "Category consolidation should be applied"

    # 5. Check that the vectorizer was actually fitted.
    assert hasattr(preprocessor.vectorizer, 'vocabulary_'), "Vectorizer should be fitted"
    assert len(preprocessor.vectorizer.vocabulary_) > 0, "Vectorizer vocabulary should not be empty"