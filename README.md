# Deep Learning Challenge — Charity Funding Predictor

## Overview
A binary classification model built with TensorFlow and Keras to predict whether 
applicants funded by Alphabet Soup, a nonprofit foundation, will use their funding 
successfully. The model was trained on metadata from more than 34,000 previously 
funded organizations.

---

## Results

### Data Preprocessing
- **Target variable:** `IS_SUCCESSFUL` — whether funding was used effectively
- **Features used:** Application type, classification, funding amount requested, 
  use case, organization type, and income classification
- **Removed:** `EIN` and `NAME` columns (identifiers with no predictive value)

### Model Architecture
The optimized model uses four hidden layers with dropout regularization:

| Layer | Neurons | Activation |
|-------|---------|------------|
| Hidden 1 | 200 | ReLU |
| Hidden 2 | 150 | Tanh |
| Hidden 3 | 100 | ReLU |
| Hidden 4 | 50 | ReLU |
| Output | 1 | Sigmoid |

Dropout rates of 0.4, 0.3, and 0.2 were applied after the first three hidden layers 
to reduce overfitting.

### Optimization Steps
- Rare categorical values in `APPLICATION_TYPE` and `CLASSIFICATION` were binned 
  into an `Other` category to reduce noise
- Neuron count and layer depth were increased iteratively
- Mixed activation functions (ReLU and Tanh) were tested across layers
- Early stopping was implemented to halt training when validation accuracy plateaued

---

## Tech Stack
- Python (Pandas, Scikit-learn, TensorFlow/Keras)
- Google Colab
- Jupyter Notebook

---

## Repository Contents
| File | Description |
|------|-------------|
| `AlphabetSoupCharity.ipynb` | Base model notebook |
| `AlphabetSoupCharity_Optimization.ipynb` | Optimized model notebook |
| `AlphabetSoupCharity.h5` | Saved base model weights |
| `AlphabetSoupCharity_Optimization.h5` | Saved optimized model weights |

---

## Future Improvements
Tree-based models such as Random Forest, Gradient Boosting, or XGBoost may offer 
comparable accuracy with greater interpretability, particularly for understanding 
which features most influence funding success.

---

## Data Source
Dataset provided by Alphabet Soup containing funding application records 
with organizational metadata and outcome labels.
