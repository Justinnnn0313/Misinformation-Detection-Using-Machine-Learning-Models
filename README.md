# 🔍 Misinformation Detection Using Machine Learning Models

<div align="center">

**A Comprehensive Analysis of Misinformation Patterns Using Logistic Regression and CART**

[![Author](https://img.shields.io/badge/Author-Wang%20Jianning-blue)](https://github.com/Justinnnn0313)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

</div>

---

## 📋 Project Overview

This project explores and models misinformation patterns using machine learning techniques. By applying **Logistic Regression** and **CART (Classification and Regression Tree)** models, we identify the most influential indicators of misinformation and compare their predictive performance.

### 🎯 Main Objectives

1. **Understand data distribution** of misinformation vs. non-misinformation
2. **Preprocess and standardize** the dataset for model training
3. **Build and evaluate** two predictive models with different approaches
4. **Identify key indicators** that strongly predict misinformation
5. **Compare model performance** to recommend the best approach

---

## 📊 Key Findings

### Model Performance Comparison

| Metric | Logistic Regression | CART Decision Tree |
|--------|-------------------|-------------------|
| **Accuracy** | 86.0% | 86.0% |
| **False Positive Rate** | Lower | Higher |
| **False Negative Rate** | ⭐ **Lower** | Higher |
| **Interpretability** | ⭐ **High** | High |
| **Recommendation** | ⭐ **Preferred** | Alternative |

### 🌟 Most Important Features

1. **Source Domain Reliability** - Strongest predictor
2. **External Fact Checks Count** - Second strongest predictor

Both models identified these features as the dominant factors in detecting misinformation.

### 📈 Data Insights

- **Misinformation Proportion**: Higher proportion of misinformation than non-misinformation in dataset
- **Platform Distribution**: Twitter is the most common platform (25.8% of total data)
- **Geographic Focus**: Germany (country) and Los Angeles (city) have the most records
- **Source Credibility**: Non-misinformation sources have significantly higher reliability scores

---

## 📁 Project Structure

```
Misinformation-Detection-Using-Machine-Learning-Models/
├── Wang Jianning.py                                    # Main analysis script
├── misinformation2.csv                                 # Dataset (30+ features)
├── Misinformation Detection Using Machine Learning Models.pdf  # Full report
├── README.md                                           # This file
├── requirements.txt                                    # Python dependencies
│
├── Visualizations/
│   ├── Confusion Matrix.png                           # Model confusion matrices
│   ├── Important Indicators.png                       # Feature importance
│   ├── Logistics Regression vs CART result.png       # Model comparison
│   ├── Proportion of misinformation.png              # Data distribution
│   └── Standardization.png                           # Before/after preprocessing
│
└── Data/
    ├── misinformation2.csv                           # Raw dataset
    └── data_describe.csv                             # Data summary statistics
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Justinnnn0313/Misinformation-Detection-Using-Machine-Learning-Models.git
cd Misinformation-Detection-Using-Machine-Learning-Models
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Analysis

```bash
python "Wang Jianning.py"
```

The script will:
- Load and explore the misinformation dataset
- Perform data preprocessing and standardization
- Train both Logistic Regression and CART models
- Generate visualizations and comparison reports
- Output results to CSV files

---

## 🔬 Technical Details

### Dataset

**File**: `misinformation2.csv`

**Features**: 30+ including:
- `source_domain_reliability` - Credibility score of the source
- `external_fact_checks_count` - Number of external fact checks
- Categorical variables: platform, country, city, etc.
- Temporal variables: date, time, timestamp
- Other indicators: engagement metrics, source characteristics

**Target Variable**: `is_misinformation` (Binary: 0 = Not Misinformation, 1 = Misinformation)

### Data Preprocessing Pipeline

```
Raw Data (30 features)
    ↓
[Remove irrelevant columns: id, timestamp, date, time]
    ↓
[One-hot encoding for categorical variables]
    ↓
[StandardScaler normalization]
    ↓
[Train-Test Split: 70%-30% with stratified sampling]
    ↓
Ready for Model Training
```

### Models

#### 1. **Logistic Regression**
- Statistical model estimating probability of misinformation
- One-hot encoded categorical variables for enhanced interpretability
- Uses gradient descent optimization
- **Advantage**: Lower false negative rate (better at catching misinformation)
- **Interpretation**: Coefficients show feature impact direction

#### 2. **CART (Classification and Regression Tree)**
- Decision tree-based algorithm using if-else rules
- Captures non-linear feature interactions
- More complex model with multiple branches
- **Advantage**: Handles non-linear relationships naturally
- **Interpretation**: Visual decision paths

### Evaluation Metrics

- **Accuracy**: Percentage of correct predictions
- **False Positive Rate (FPR)**: Incorrectly flagged as misinformation
- **False Negative Rate (FNR)**: Missed misinformation (more critical)
- **Confusion Matrix**: Detailed TP, TN, FP, FN breakdown

---

## 📊 Key Results

### Data Exploration Phase

**Finding 1: Misinformation Distribution**
- Misinformation occurs more frequently than non-misinformation
- Suggests dataset represents realistic misinformation prevalence

**Finding 2: Platform & Geographic Analysis**
- Twitter: 25.8% of total data, 29.1% of misinformation
- Most affected regions: Germany and Los Angeles
- Platform-specific misinformation patterns exist

**Finding 3: Source Credibility**
- Non-misinformation sources have significantly higher reliability scores
- Clear distinction in source quality between categories
- Establishes source reliability as a key predictor

### Model Performance Phase

**Accuracy**: Both models achieved 86% accuracy
- Identical performance suggests models have found core patterns
- Small dataset size may limit differentiation
- Both models effectively capture the relationship

**Why Logistic Regression is Recommended**:
1. ✅ **Lower False Negative Rate**: Better at catching actual misinformation
2. ✅ **Better Interpretability**: Coefficients have clear meaning
3. ✅ **Simpler Model**: Less risk of overfitting
4. ✅ **Computational Efficiency**: Faster training and prediction

---

## 💡 Feature Importance Insights

### Top 5 Features for Misinformation Detection

**Logistic Regression Coefficients:**
1. Source Domain Reliability (strongest negative coefficient)
2. External Fact Checks Count
3. [Feature 3]
4. [Feature 4]
5. [Feature 5]

**CART Feature Importance:**
1. Source Domain Reliability (highest importance %)
2. External Fact Checks Count
3. [Feature 3]
4. [Feature 4]
5. [Feature 5]

### Interpretation

- **Source Domain Reliability**: Reliable sources are less likely to spread misinformation
- **External Fact Checks**: Posts with more fact-checks are associated with misinformation detection efforts
- Combination of these two features provides strong misinformation signal

---

## 📈 Results Visualization

### Confusion Matrix Analysis

**Logistic Regression:**
- True Negatives: [X] | False Positives: [Y]
- False Negatives: [Z] | True Positives: [W]

**CART:**
- True Negatives: [X] | False Positives: [Y]
- False Negatives: [Z] | True Positives: [W]

*See generated PNG files for visual comparison*

### Model Comparison Charts

The project generates comprehensive visualizations:
- 📊 **FPR Comparison**: False Positive Rate side-by-side
- 📊 **FNR Comparison**: False Negative Rate comparison
- 📊 **Error Rate**: Overall error rate comparison
- 📊 **Accuracy**: Accuracy metrics comparison
- 📊 **Feature Importance**: Top 5 features for each model

---

## 🛠️ Technical Stack

**Programming Language**: Python 3.11

**Libraries**:
```
pandas>=1.3.0          # Data manipulation and analysis
matplotlib>=3.4.0      # Plotting and visualization
seaborn>=0.11.0        # Statistical data visualization
numpy>=1.20.0          # Numerical computing
scikit-learn>=1.0.0    # Machine learning algorithms
```

**Machine Learning Components**:
- `LogisticRegression` - Classification model
- `DecisionTreeClassifier` - CART implementation
- `StandardScaler` - Data normalization
- `train_test_split` - Data partitioning
- `confusion_matrix` - Performance evaluation

---

## 📝 Analysis Workflow

```
Step 1: Data Loading
   └─→ Load misinformation2.csv
   
Step 2: Exploratory Data Analysis (EDA)
   ├─→ Examine data distribution
   ├─→ Analyze categorical features
   └─→ Visualize source reliability patterns
   
Step 3: Data Cleaning
   ├─→ Check for missing values (None found)
   ├─→ Check for duplicates (None found)
   └─→ Remove irrelevant columns
   
Step 4: Feature Engineering
   ├─→ One-hot encode categorical variables
   └─→ Standardize numerical features
   
Step 5: Model Training
   ├─→ Train Logistic Regression
   └─→ Train CART Decision Tree
   
Step 6: Model Evaluation
   ├─→ Calculate accuracy
   ├─→ Compute confusion matrices
   └─→ Calculate error rates
   
Step 7: Feature Importance Analysis
   ├─→ Extract LR coefficients
   ├─→ Extract CART importances
   └─→ Visualize top features
   
Step 8: Visualization & Reporting
   └─→ Generate comprehensive charts and comparisons
```

---

## 🎯 Conclusions

### Main Findings

1. **Both models achieve 86% accuracy**, indicating robust predictive patterns
2. **Logistic Regression recommended** due to lower false negative rate
3. **Source reliability is critical** - the strongest predictor of misinformation
4. **External fact-checks validate** the misinformation detection process
5. **Dataset limitations** - identical performance suggests need for larger, more diverse dataset

### Practical Applications

- **Content Moderation**: Prioritize posts from unreliable sources
- **Fact-Checking**: Focus resources on high-risk domains
- **Platform Policy**: Use source reliability as filtering criterion
- **Research**: Further investigation into source credibility mechanisms

---

## 🚀 Future Improvements

1. **Expand Dataset**
   - Include more diverse sources and platforms
   - Increase temporal coverage
   - Add more geographic regions

2. **Advanced Models**
   - Implement Random Forest ensemble
   - Test XGBoost for better accuracy
   - Try neural networks for complex patterns

3. **Feature Engineering**
   - Create interaction features
   - Extract temporal patterns
   - Incorporate sentiment analysis
   - Add network-based features

4. **Model Enhancement**
   - Hyperparameter tuning
   - Cross-validation implementation
   - Class imbalance handling (SMOTE)
   - Feature selection optimization

5. **Real-world Deployment**
   - Build API endpoint for predictions
   - Create web interface
   - Implement real-time monitoring
   - Develop alert system

---

## 📄 Project Files

| File | Description |
|------|-------------|
| `Wang Jianning.py` | Complete Python analysis script |
| `misinformation2.csv` | Dataset with 30+ features |
| `Misinformation Detection Using Machine Learning Models.pdf` | Detailed project report |
| `Confusion Matrix.png` | Model confusion matrices visualization |
| `Important Indicators.png` | Feature importance chart |
| `Logistics Regression vs CART result.png` | Model comparison visualization |
| `Proportion of misinformation.png` | Data distribution chart |
| `Standardization.png` | Before/after preprocessing comparison |

---

## 👨‍💼 Author

**Wang Jianning**

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Kaggle community for machine learning resources
- Scikit-learn documentation for model implementations
- Seaborn and Matplotlib for visualization capabilities

---

## 📞 Questions & Support

For questions or issues regarding this project:
1. Check the [project report](Misinformation%20Detection%20Using%20Machine%20Learning%20Models.pdf)
2. Review the [Python script](Wang%20Jianning.py) for implementation details
3. Open an issue on GitHub

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

Made with ❤️ by Wang Jianning

</div>
