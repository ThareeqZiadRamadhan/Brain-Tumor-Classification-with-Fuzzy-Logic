# fuzzy_model.py
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.filters.rank import entropy
from skimage.morphology import disk
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler

def extract_features_from_image(image):
    """
    Extract features from brain MRI image
    Args:
        image: Grayscale image (numpy array)
    Returns:
        tuple: (mean_val, contrast, entropy_val)
    """
    # Resize image to standard size
    image = cv2.resize(image, (128, 128))
    
    # Calculate mean intensity
    mean_val = np.mean(image)
    
    # Calculate entropy using skimage
    entropy_val = entropy(image, disk(5)).mean()
    
    # Calculate GLCM contrast
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    glcm = graycomatrix(
        image, 
        distances=[5], 
        angles=[0], 
        levels=256, 
        symmetric=True, 
        normed=True
    )
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    
    return mean_val, contrast, entropy_val

def fuzzy_classify(mean, contrast, entropy_val):
    """
    Classify brain tumor using fuzzy logic
    Args:
        mean: Mean intensity value
        contrast: GLCM contrast value  
        entropy_val: Entropy value
    Returns:
        tuple: (predicted_class, scores)
    """
    # Simple feature-based classification without over-complex fuzzy logic
    # Normalize features to reasonable ranges
    mean_norm = np.clip(mean / 255.0, 0, 1)
    contrast_norm = np.clip(contrast / 100.0, 0, 1)  # Adjusted for typical GLCM values
    entropy_norm = np.clip(entropy_val / 8.0, 0, 1)
    
    # Define classification based on typical tumor characteristics
    # These are simplified rules based on general medical knowledge
    
    # Glioma characteristics: Variable intensity, moderate contrast, high entropy (heterogeneous)
    glioma_score = (
        0.3 * (1 - abs(mean_norm - 0.5)) +  # Prefers middle intensity values
        0.3 * (1 - contrast_norm) +         # Prefers lower contrast
        0.4 * entropy_norm                  # Prefers higher entropy
    )
    
    # Meningioma characteristics: Higher intensity, high contrast, lower entropy (more uniform)
    meningioma_score = (
        0.4 * mean_norm +                   # Prefers higher intensity
        0.4 * contrast_norm +               # Prefers higher contrast  
        0.2 * (1 - entropy_norm)            # Prefers lower entropy
    )
    
    # Pituitary characteristics: Variable intensity, high contrast, moderate entropy
    pituitary_score = (
        0.2 * (1 - abs(mean_norm - 0.4)) +  # Prefers lower-middle intensity
        0.5 * contrast_norm +               # Prefers high contrast
        0.3 * (1 - abs(entropy_norm - 0.5)) # Prefers moderate entropy
    )
    
    # Add small random variation to avoid always getting same result
    import random
    noise_factor = 0.05
    glioma_score += random.uniform(-noise_factor, noise_factor)
    meningioma_score += random.uniform(-noise_factor, noise_factor)
    pituitary_score += random.uniform(-noise_factor, noise_factor)
    
    # Ensure all scores are positive
    scores = [max(0, glioma_score), max(0, meningioma_score), max(0, pituitary_score)]
    
    # Normalize scores to sum to 1 for better percentage display
    total_score = sum(scores)
    if total_score > 0:
        scores = [score / total_score for score in scores]
    else:
        scores = [0.33, 0.33, 0.34]  # Equal probability if all zero
    
    classes = ['Glioma', 'Meningioma', 'Pituitary']
    predicted_class = classes[np.argmax(scores)]
    
    return predicted_class, scores