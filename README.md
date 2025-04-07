# DA-218o-Project-2-Income-Survey-Dataset

## Overview
This project analyzes an income survey dataset to understand various demographic and economic factors that influence income levels.

## Project Structure
```
└── DA-218o-Project-2-Income-Survey-Dataset/
    ├── data/              # Dataset files
    ├── output/            # Generated plots and results
    │   ├── regression/    # Regression analysis outputs
    │   ├── classification/# Classification analysis outputs
    │   └── clustering/    # Clustering analysis outputs
    ├── regression.py      # Income prediction using regression
    ├── classification.py  # Marital status classification
    ├── clustering.py      # Income group clustering
    └── LICENSE           # Project license
```

## Analysis Files
- **regression.py**: Implements Bayesian regression models using PyMC to predict total income based on various features. Includes linear, non-linear, and robust regression approaches.
- **classification.py**: Classifies marital status various classification algorithms and evaluates model performance.
- **clustering.py**: Performs unsupervised learning to identify natural income groups and patterns in the data.

## Required Libraries
```
numpy         # Numerical computations
pandas        # Data manipulation and analysis
matplotlib    # Data visualization
seaborn      # Statistical data visualization
scikit-learn # Machine learning tools
pymc         # Bayesian statistical modeling
arviz        # Bayesian analysis visualization
```

## Getting Started
1. Clone the repository
2. Install required dependencies:
   ```
   pip install numpy pandas matplotlib seaborn scikit-learn pymc arviz
   ```
3. Run the analysis scripts:
   ```
   python regression.py
   python classification.py
   python clustering.py
   ```

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.