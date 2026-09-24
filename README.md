# From Signals to Feelings: Predicting Emotion with Machine Learning

**DSC 101 Final Project** | Gabriel Pereira and Deng Arguer Bul | May 2026

Can signals from a wearable device tell us whether someone is feeling an emotion? This project uses physiological data from Empatica wristbands to train a logistic regression model that classifies moments as **emotional** (happy, sad, or angry) or **emotionless** (neutral).

## Project Overview

We originally planned to study how music and task difficulty affected stress during a working memory task. Aligning those behavioral variables with the physiological data proved unrealistic (and the data had many missing values), so we shifted to a different question: **can physiological signals alone predict emotional state?**

The pipeline:

1. Load Empatica sensor files and FaceReader emotion labels for each subject
2. Align everything onto a common timeline, using EDA as the base
3. Convert emotion labels to binary (0 = emotionless, 1 = emotional)
4. Explore the data with boxplots, KDE plots, a pairplot, and a correlation matrix
5. Train and evaluate a logistic regression model

## Dataset

**"A Multimodal Dataset for Investigating Working Memory in Presence of Music"** (PhysioNet, Version 1.0.0, 2025)

> Khazaei, S., Parshi, S., Alam, S., Amin, M. R., & Faghih, R. T. (2025). A Multimodal Dataset for Investigating Working Memory in Presence of Music. PhysioNet.

Dataset link: [**](https://physionet.org/content/multimodal-nback-music/1.0.0/)

The raw data is **not included** in this repo. To run the notebook, download it from PhysioNet and update `BASE_PATH` in the first code cell to point to your local copy.

### Features Used (from Empatica wearables)

| Feature | Description |
|---|---|
| `Empatica_EDA` | Electrodermal activity (µS) |
| `Empatica_TEMP` | Skin temperature (°C) |
| `Empatica_HR` | Heart rate (BPM) |
| `Empatica_BVP` | Blood volume pulse |
| `Empatica_IBI` | Interbeat interval (seconds) |
| `ACC_X`, `ACC_Y`, `ACC_Z` | Accelerometer readings on three axes |

**Target:** `Emotion` (0 = emotionless, 1 = emotional). Labels come from FaceReader facial expression analysis.

**Data:** 5 subjects (3F, 4F, 6M, 8M, 11F). After alignment and removing rows with missing values, 1,287 observations were used for modeling.

## Methods

- **Model:** Logistic regression (scikit-learn)
- **Preprocessing:** Standardized features with `StandardScaler` (fit on training data only)
- **Split:** 80% training / 20% testing (258 test observations)
- **Metrics:** Accuracy, precision, recall, F1, AUC, confusion matrix

## Results

| Metric | Score |
|---|---|
| Accuracy | 0.69 |
| Precision (emotional) | 0.61 |
| Recall (emotional) | 0.60 |
| F1 (emotional) | 0.60 |
| AUC | 0.72 |

**Confusion matrix** (rows = actual, columns = predicted):

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 117 | 39 |
| **Actual 1** | 41 | 61 |

### Key Findings

- **EDA** showed the clearest separation between classes in the visualizations: emotionless readings sat near 0, while emotional readings reached much higher values.
- In the model, **temperature, ACC_X, and EDA** had the largest positive coefficients.
- **BVP and IBI** showed little difference between classes and had the smallest effects.
- The model was better at identifying emotionless moments (F1 = 0.75) than emotional ones (F1 = 0.60).

## Limitations and Future Work

- Only 5 subjects, so results may not generalize to other people.
- Rows from the same subject can appear in both the training and test sets, which may make performance look better than it would on a brand-new person. Splitting by subject would be a stronger test.
- Grouping happy, sad, and angry into one "emotional" class hides differences between emotions.
- Future work could include more advanced models, feature engineering (such as rolling averages or heart rate variability), and more detailed emotion labels.

## Repository Contents

| File | Description |
|---|---|
| `DSC101_Final_Project.ipynb` | Full notebook with code, visualizations, and analysis |
| `DSC101_Final_Project.pdf` | PDF export of the notebook and report |

## How to Run

1. Clone this repo.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn jupyter`
3. Download the dataset from PhysioNet (see above).
4. Update `BASE_PATH` in the notebook to your local data folder.
5. Run the notebook from top to bottom.

## Acknowledgments

Thanks to Dr. Rimal for guidance in DSC 101, and to the dataset authors for making their data publicly available.
