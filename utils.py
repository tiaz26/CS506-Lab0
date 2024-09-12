## Please fill in all the parts labeled as ### YOUR CODE HERE

import numpy as np

def dot_product(v1, v2):
    '''
    v1 and v2 are vectors of same shape.
    return the scalar dor product of the two vectors.
    # Hint: use `np.dot`.
    '''
    return np.dot(v1, v2)
    
def cosine_similarity(v1, v2):
    '''
    v1 and v2 are vectors of same shape.
    Return the cosine similarity between the two vectors.
    
    # Note: The cosine similarity is a commonly used similarity 
    metric between two vectors. It is the cosine of the angle between 
    two vectors, and always between -1 and 1.
    
    # The formula for cosine similarity is: 
    # (v1 dot v2) / (||v1|| * ||v2||)
    
    # ||v1|| is the 2-norm (Euclidean length) of the vector v1.
    
    # Hint: Use `dot_product` and `np.linalg.norm`.
    '''
    dot_prod = dot_product(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_prod / (norm1 * norm2)
    
def nearest_neighbor(target_vector, vectors):
    '''
    target_vector is a vector of shape d.
    vectors is a matrix of shape N x d.
    return the row index of the vector in vectors that is closest to 
    target_vector in terms of cosine similarity.
    
    # Hint: You should use the cosine_similarity function that you already wrote.
    # Hint: For this lab, you can use a for loop to iterate through vectors.
    '''
    max_similarity = -float('inf')
    nearest_index = -1
    
    for i, vector in enumerate(vectors):
        similarity = cosine_similarity(target_vector, vector)
        if similarity > max_similarity:
            max_similarity = similarity
            nearest_index = i
    
    return nearest_index
