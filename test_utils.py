## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np
import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = cosine_similarity(vector1, vector2)
    
    # Expected cosine similarity
    dot_prod = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    expected_result = dot_prod / (norm1 * norm2)
    
    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    data = np.array([[1, 2], [3, 4], [5, 6]])
    query = np.array([4, 5])
    
    result = nearest_neighbor(query, data)
    
    # Compute the expected index by finding the nearest neighbor manually
    distances = np.linalg.norm(data - query, axis=1)
    expected_index = np.argmin(distances)
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"