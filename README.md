# Pneumonia_Detection

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

This project aims to develop a deep learning model to accurately detect pneumonia from chest X-ray images. Leveraging Convolutional Neural Networks (CNNs), the model is trained to classify X-ray images into 'Pneumonia' or 'Normal' categories, providing a valuable tool for aiding medical diagnosis.

## Features

* **Pneumonia Classification:** Classifies chest X-ray images as either pneumonia-affected or normal.
* **Deep Learning Model:** Utilizes a custom-built CNN for robust image analysis.
* **Web Application:** Provides a simple web interface for uploading X-ray images and getting predictions (as indicated by `app.py`, `static`, `templates`).
* **Jupyter Notebook for Training:** Includes a Jupyter notebook (`Pneumonia_Detection_CNN.ipynb`) detailing the model training and evaluation process.

## Dataset

The model is trained on a publicly available dataset of chest X-ray images. This dataset typically includes images categorized as 'Pneumonia' (further divided into bacterial and viral) and 'Normal'.

* **Source:** [Link to Dataset Source, e.g., Kaggle Chest X-Ray Images (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 

## Model Architecture

The core of this project is a Convolutional Neural Network (CNN). The `Pneumonia_Detection_CNN.ipynb` notebook outlines the detailed architecture, including layers, activation functions, and training parameters. The trained model is saved as `model.h5`.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Ahiram15/Pneumonia_Detection.git](https://github.com/Ahiram15/Pneumonia_Detection.git)
    cd Pneumonia_Detection
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

To retrain or understand the model training process:

1.  Ensure you have downloaded and organized the dataset as expected by the notebook.
2.  Open and run the `Pneumonia_Detection_CNN.ipynb` notebook in a Jupyter environment.

### Running the Web Application

To use the pneumonia detection web application:

1.  Ensure all dependencies are installed.
2.  Run the Flask application:
    ```bash
    python app.py
    ```
3.  Open your web browser and navigate to `http://127.0.0.1:5000/` (or the address displayed in your terminal).
4.  Upload a chest X-ray image to get a prediction.

## Results

The model achieves [mention accuracy, precision, recall, F1-score if available] on the test set. A `model_plot.png` file visualizes the model's training history (e.g., accuracy and loss over epochs).

## Technologies Used

* Python
* TensorFlow/Keras (for model building)
* Flask (for web application)
* Numpy
* Pandas
* Matplotlib (for plotting)
* Jupyter Notebook


