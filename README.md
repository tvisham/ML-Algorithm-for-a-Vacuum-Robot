# Environmental Condition Classification Using Energy Consumption Patterns

## Overview

This repository contains the machine learning pipeline developed for research investigating how environmental conditions influence the energy consumption of autonomous mobile robots (AMRs), specifically consumer robot vacuums.

The objective of this project is to determine whether battery discharge and acceleration patterns can be used to classify operating environments using supervised machine learning. Experimental data collected under controlled conditions were transformed into statistical features and used to train a Random Forest classifier.

The resulting model achieved **89.21% classification accuracy**, demonstrating that environmental factors produce distinguishable energy consumption signatures.

---

## Research Motivation

As autonomous mobile robots become increasingly common in households, improving their energy efficiency has become an important sustainability challenge. While previous work has focused primarily on hardware and navigation optimization, relatively little research has examined how external environmental factors affect robot energy consumption.

This project explores whether environmental conditions—including temperature and surface friction—can be inferred solely from energy consumption behavior, providing a foundation for future adaptive energy management systems.

---

## Methodology

The machine learning pipeline follows these steps:

1. Load experimental energy consumption data.
2. Convert relevant measurements to numeric values and remove invalid entries.
3. Aggregate time-series measurements by experiment.
4. Extract statistical features describing battery discharge and acceleration behavior.
5. Encode environmental labels.
6. Split the data into training and testing sets.
7. Train a Random Forest classifier.
8. Evaluate model performance using accuracy, precision, recall, F1-score, and a confusion matrix.

### Engineered Features

The model is trained using experiment-level features derived from the raw measurements, including:

- Mean battery discharge rate
- Maximum battery discharge rate
- Standard deviation of battery discharge rate
- Mean acceleration decrease rate
- Standard deviation of acceleration decrease rate
- Initial battery percentage
- Final battery percentage
- Total battery consumption
- Experiment duration

---

## Model

The project uses a **Random Forest Classifier** from scikit-learn.

Configuration:

- 80/20 train-test split
- Fixed random seed (`random_state=42`)

The model predicts the environmental condition associated with each experiment based on engineered energy consumption features.

---

## Results

The trained model achieved an overall classification accuracy of **89.21%**.

The evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

These results indicate that environmental conditions produce distinct energy consumption patterns that can be effectively classified using supervised machine learning.

---

## Repository Structure

```
.
├── energy_classifier.py
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.9+
- pandas
- scikit-learn

Install the required packages with:

```bash
pip install -r requirements.txt
```

---

## Running the Code

Clone the repository and install the required dependencies. Then execute:

```bash
python energy_classifier.py
```

The script will:

- Preprocess the experimental data
- Generate statistical features
- Train the Random Forest classifier
- Output the model accuracy, classification report, and confusion matrix

> **Note:** The experimental dataset used in this study is not included in this repository because it was collected specifically for the associated research project and is not publicly available.

---

## Future Work

Possible extensions include:

- Hyperparameter optimization
- Cross-validation
- Comparison with additional classification algorithms
- Feature importance analysis
- Model explainability using SHAP or similar techniques
- Real-time environmental classification for adaptive robot control

---

## Citation

If you use or build upon this code, please cite the associated research publication.

---

## License

This repository is provided for academic and research purposes.
