# Physiological Emotion Classification
Machine learning project predicting emotional states from wearable physiological sensor data using Python and logistic regression.


# Predicting Emotion from Physiological Signals

This project investigates whether physiological signals collected from
wearable sensors can be used to classify emotional states.

Using data from Empatica wearable devices and FaceReader emotion labels,
I cleaned and aligned multimodal physiological data and trained a
logistic regression classifier to distinguish between emotional and
emotionless states.

## Dataset

The project uses the PhysioNet dataset:

"A Multimodal Dataset for Investigating Working Memory in Presence of Music"

Physiological features include:

- Electrodermal activity (EDA)
- Skin temperature
- Heart rate
- Blood volume pulse (BVP)
- Interbeat interval (IBI)
- Accelerometer X, Y, and Z measurements

Target:
- 0 = Emotionless
- 1 = Emotional

## Project Workflow

1. Loaded raw physiological and FaceReader data
2. Aligned measurements using timestamps
3. Cleaned missing and invalid emotion labels
4. Performed exploratory data analysis
5. Examined feature correlations
6. Standardized features
7. Trained a logistic regression model
8. Evaluated model performance

## Results

The logistic regression model achieved:

- Accuracy: 68.99%
- Precision: 61.00%
- Recall: 59.80%
- F1 Score: 60.40%
- ROC-AUC: 72.49%

EDA and temperature showed some of the strongest differences between
emotional and emotionless observations.

## Technologies

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Visualizations

[put your correlation matrix image here]

[put your ROC curve here]

[put your confusion matrix here]

## Future Improvements

Possible extensions include:

- Comparing additional classification models
- Using participant-based train/test splits
- Feature engineering from physiological signals
- Hyperparameter tuning
- Testing on a larger participant sample
