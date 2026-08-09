# 🚀 Spaceship Titanic Prediction

<p align="center">
  <img src="images/banner.png" width="900">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue">
  <img src="https://img.shields.io/badge/Model-XGBoost-success">
  <img src="https://img.shields.io/badge/Task-Binary_Classification-orange">
  <img src="https://img.shields.io/badge/Kaggle-Spaceship_Titanic-20BEFF">
</p>

<p align="center">
  <img src="images/cli_demo.gif" width="650">
</p>

---

# 🚀 Spaceship Titanic Prediction

A machine learning project that predicts whether a passenger aboard the **Spaceship Titanic** was **transported to another dimension** after a spacetime anomaly. The project uses an **XGBoost Classifier** along with extensive preprocessing, feature engineering, and an interactive **Command Line Interface (CLI)** for real-time predictions.

## ✨ Features

- 🚀 Predicts whether a passenger was **Transported** or **Not Transported**
- 🌳 Built using **XGBoost Classifier**
- 💻 Interactive **CLI application**
- 🧩 Advanced missing value handling
- ⚙️ Extensive feature engineering
- 📊 Modular preprocessing pipeline
- 📁 Clean and organized project structure

## 🛠 Tech Stack

- Python
- NumPy
- Pandas
- Scikit-Learn
- XGBoost
- Matplotlib
- Seaborn
- SciPy

## 📂 Project Structure

```text
Spaceship-Titanic/
│
├── dataset/
├── notebook/
├── src/
├── result.csv
└── README.md
```

## 🔧 Data Preprocessing

The Spaceship Titanic dataset contains numerous missing values across multiple features. Instead of relying solely on simple statistical imputation, a preprocessing pipeline was developed to preserve meaningful relationships within the data.

The pipeline includes:

- Missing value imputation using group-based information
- Categorical encoding
- Feature scaling where required
- Data validation before inference

## 🧠 Feature Engineering

Several new features were engineered to improve predictive performance:

- 🚪 Cabin Deck extraction
- 🚪 Cabin Number extraction
- ↔️ Cabin Side extraction
- 👥 Passenger Group extraction from PassengerId
- 🧩 Group-based feature completion
- 📊 Additional engineered features derived from passenger information

## 🤖 Model

The final model uses an **XGBoost Classifier**, selected for its strong performance on structured tabular datasets and its ability to model complex feature interactions efficiently.

## 📈 Model Evaluation

Performance was evaluated using:

- Accuracy
- F1 Score

These metrics provide a balanced evaluation of the binary classification problem.

## ⚠️ Challenges

- Large number of missing values
- Complex relationships between passengers belonging to the same group
- Meaningful feature engineering from PassengerId and Cabin
- Preventing data leakage during preprocessing