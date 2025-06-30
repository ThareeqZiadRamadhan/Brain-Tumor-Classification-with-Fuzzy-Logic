# fuzzy_model.py
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.filters.rank import entropy
from skimage.morphology import disk
import skfuzzy as fuzz
from sklearn.preprocessing import MinMaxScaler

def extract_features_from_image(image):
    image = cv2.resize(image, (128, 128))
    mean_val = np.mean(image)
    entropy_val = entropy(image, disk(5)).mean()
    glcm = graycomatrix(image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    return mean_val, contrast, entropy_val

def fuzzy_classify(mean, contrast, entropy_val):
    scaler = MinMaxScaler()
    X = np.array([[mean, contrast, entropy_val]])
    X_scaled = scaler.fit_transform(X)
    m, c, e = X_scaled[0]

    x_range = np.linspace(0, 1, 100)

    def fuzzy_mf(val, name):
        if name == "mean":
            low = fuzz.trimf(x_range, [0, 0, 0.5])
            med = fuzz.trimf(x_range, [0.25, 0.5, 0.75])
            high = fuzz.trimf(x_range, [0.5, 1, 1])
        elif name == "contrast" or name == "entropy":
            low = fuzz.trimf(x_range, [0, 0, 0.5])
            high = fuzz.trimf(x_range, [0.5, 1, 1])
        return {
            'low': fuzz.interp_membership(x_range, low, val),
            'med': fuzz.interp_membership(x_range, med, val) if name == "mean" else 0,
            'high': fuzz.interp_membership(x_range, high, val)
        }

    m_f = fuzzy_mf(m, "mean")
    c_f = fuzzy_mf(c, "contrast")
    e_f = fuzzy_mf(e, "entropy")

    glioma_score = np.fmin(np.fmin(m_f['high'], c_f['low']), e_f['high'])
    meningioma_score = np.fmin(np.fmin(m_f['med'], c_f['high']), e_f['low'])
    pituitary_score = np.fmin(np.fmin(m_f['low'], c_f['high']), e_f['high'])

    scores = [glioma_score, meningioma_score, pituitary_score]
    classes = ['Glioma', 'Meningioma', 'Pituitary']
    return classes[np.argmax(scores)], scores
