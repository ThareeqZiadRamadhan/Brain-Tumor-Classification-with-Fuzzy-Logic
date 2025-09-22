# Brain Tumor Classification using Fuzzy Logic

A web application for classifying brain tumors from MRI images using fuzzy logic algorithms.

## Features

- **Real-time MRI Analysis**: Upload brain MRI images for instant classification
- **Fuzzy Logic Classification**: Uses advanced fuzzy logic for tumor type detection
- **Interactive Visualization**: Feature extraction charts and classification scores
- **Three Tumor Types**: Classifies Glioma, Meningioma, and Pituitary tumors
- **User-friendly Interface**: Built with Streamlit for easy interaction

## Tumor Types Classified

1. **Glioma**: Most common primary brain tumor
2. **Meningioma**: Tumor of brain/spinal cord membranes  
3. **Pituitary**: Tumor in pituitary gland

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain-tumor-classification.git
cd brain-tumor-classification
```

2. Create virtual environment:
```bash
python -m venv .venv
```

3. Activate virtual environment:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run brain_tumor_app.py
```

2. Open your browser and go to `http://localhost:8501`

3. Upload an MRI brain scan image (PNG, JPG, JPEG, BMP, TIFF)

4. View the classification results and feature analysis

## Project Structure

```
brain-tumor-classification/
├── brain_tumor_app.py      # Main Streamlit application
├── fuzzy_model.py          # Fuzzy logic classification functions
├── app.py                  # Alternative waste classification app
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── docs/                  # Additional documentation
└── sample_images/         # Example MRI images
```

## Algorithm Details

### Feature Extraction
- **Mean Intensity**: Average pixel intensity value
- **GLCM Contrast**: Gray Level Co-occurrence Matrix contrast measure  
- **Entropy**: Measure of randomness/disorder in image

### Fuzzy Logic Classification
The system uses weighted scoring based on tumor characteristics:

- **Glioma**: Variable intensity, moderate contrast, high entropy
- **Meningioma**: Higher intensity, high contrast, lower entropy
- **Pituitary**: Variable intensity, high contrast, moderate entropy

## Dependencies

- streamlit
- opencv-python
- scikit-image
- scikit-fuzzy
- matplotlib
- seaborn
- pillow
- numpy

## Medical Disclaimer

**Important**: This tool is for educational and research purposes only. It should not be used for actual medical diagnosis. Always consult qualified medical professionals for proper medical diagnosis and treatment.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Ziad - Initial work

## Acknowledgments

- Medical imaging community for research insights
- Streamlit team for the excellent web framework
- scikit-fuzzy contributors for fuzzy logic implementation