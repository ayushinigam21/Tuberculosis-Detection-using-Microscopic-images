# Tuberculosis-Detection-using-Microscopic-images

This project implements a U-Net model for automated detection of tuberculosis (TB) in microscopic images of sputum smears. 

## Description

Tuberculosis is a significant global health concern, and early detection is crucial for effective treatment. This project aims to assist healthcare professionals in diagnosing TB by automating the analysis of microscopic images.

The code performs the following steps:

1. **Image Preprocessing:**
    - Input: Microscopic images of sputum smears stained using Ziehl-Neelsen staining or similar techniques.
    - The images are loaded and preprocessed to enhance the visibility of TB bacilli. This might involve color adjustments, noise reduction, and contrast enhancement.
2. **Image Segmentation:** 
    - Extracts the region of interest (ROI) from the input image, focusing on areas likely to contain TB bacilli. This is achieved using color-based segmentation, leveraging the staining characteristics of the bacilli.
3. **Image Postprocessing:** 
    - Refines the segmented image by removing small artifacts and noise, improving the accuracy of the detection process. Techniques like morphological operations (e.g., removing small objects, filling holes) might be applied.
4. **U-Net Model:**
    - A U-Net model, a convolutional neural network architecture known for its effectiveness in biomedical image segmentation, is employed for pixel-wise classification.
    - The model is trained on a dataset of labeled microscopic images to learn the visual features associated with TB bacilli.
5. **Prediction:** 
    - Predicts the presence and location of TB bacilli in new microscopic images.
    - The model generates a probability map highlighting areas where TB bacilli are likely to be present.

## Usage

1. **Data:**
    - Place the input microscopic images in the `train` and `valid` folders.
    - Ensure the images are in a suitable format (e.g., PNG, JPG).
    - If necessary, provide corresponding ground truth masks for training.
2. **Dependencies:**
    - Install the required libraries using `pip install -r requirements.txt`.
3. **Training:**
    - Run the code to train the U-Net model using the provided training data.
    - Adjust hyperparameters like epochs, batch size, and learning rate for optimal performance.
4. **Prediction:** 
    - Load the trained model and use it to predict on new microscopic images.
    - Visualize the predicted probability maps to identify potential TB bacilli locations.

## Requirements

- Python 3.x
- TensorFlow 2.x
- OpenCV
- scikit-image
- pandas
- matplotlib

## Note

- The code is optimized for images with a resolution of 640x640. You might need to adjust it for different image sizes.
- The training process may take several hours depending on the dataset size, model complexity, and hardware.
- This project is intended for research and educational purposes. For clinical diagnosis, consult with qualified healthcare professionals.

## Acknowledgements

- The U-Net model architecture is based on the original paper by Ronneberger et al.
- The dataset used for training is publicly available [mention source if applicable].
- Inspiration and guidance were drawn from existing research on automated TB detection in microscopic images.
