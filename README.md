# HR Lead Prediction Project

## Project Overview
This project aims to predict high-potential HR service leads based on company funding and hiring trends. The goal is to classify whether a company is a 'hot lead' (1) or 'not a hot lead' (0) using advanced deep learning models (LSTM and CNN). The evaluation metric is the **F1-score**.

## Objectives

- **Predict HR Service Leads**: Classify whether a company is a potential HR lead.
- **Handle Class Imbalance**: Use class weighting to address data imbalance.
- **Optimize Model Performance**: Improve F1-score through threshold tuning.
- **Generate Submission File**: Output predictions in the required `submission.csv` format.

## Workflow
1. **Data Preprocessing**: Feature scaling, reshaping, and handling class imbalance.
2. **Model Development**: Implement LSTM and CNN models.
3. **Model Training**: Train with class weights and tune hyperparameters.
4. **Model Evaluation**: Evaluate using accuracy and F1-score.
5. **Threshold Adjustment**: Optimize F1-score by tuning the prediction threshold.
6. **Submission Generation**: Create and export the `submission.csv` file.

## Data Preprocessing

- **Input Data**: Numerical features representing company information (e.g., funding, hiring trends).
- **Target Variable**: Binary (1 = Hot Lead, 0 = Not Hot Lead).
- **Feature Scaling**: Normalize input data for consistent model training.
- **Reshaping Data**: Prepare data for LSTM (3D tensor) and CNN (2D tensor) models.
- **Class Imbalance Handling**: Use `compute_class_weight` to balance positive and negative samples.

## Model Architectures

### LSTM Model

Long Short-Term Memory (LSTM) networks are suitable for capturing sequential patterns in the dataset.

### CNN Model

Convolutional Neural Networks (CNN) extract spatial features from the input using 1D convolution layers.

## Model Training

- **Loss Function**: `categorical_crossentropy` for multi-class outputs.
- **Optimizer**: Adam optimizer for efficient gradient descent.
- **Class Weights**: Penalize underrepresented classes for better recall.
- **Epochs and Batch Size**: Trained for 20 epochs with a batch size of 32.

## Model Evaluation

- **Accuracy**: Measures correct predictions but may be misleading for imbalanced datasets.
- **F1-Score**: The primary evaluation metric balancing precision and recall.

## Threshold Adjustment

Adjusting the prediction threshold from 0.5 to 0.3 improves sensitivity to positive cases, enhancing the F1-score.

## Submission Generation

Create the `submission.csv` file with two columns:

| Id  | Prediction |
| --- | ---------- |
| 0   | 1          |
| 1   | 0          |

## How to Run the Project

1. Ensure the required packages (`tensorflow`, `sklearn`, `pandas`, etc.) are installed.

2. Run the model training script.

3. Adjust the decision threshold if needed.

4. Generate the submission file using the provided code snippet.

## Future Improvements

- **Hyperparameter Tuning**: Fine-tune model parameters for better accuracy.
- **Ensemble Models**: Combine LSTM and CNN outputs for improved predictions.
- **Advanced Architectures**: Explore transformer-based models for better sequence learning.


