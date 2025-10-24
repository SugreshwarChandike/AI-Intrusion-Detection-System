# 🧠 AI-Based Intrusion Detection System (IDS)

An Intrusion Detection System (IDS) monitors network traffic to identify suspicious activity or attacks.  
This project uses **machine learning** to detect intrusions automatically using the **NSL-KDD dataset**.

---

## 📋 Project Overview

The AI-Based IDS analyzes network traffic patterns and classifies them as **normal** or **malicious**.  
It uses a **Random Forest** model trained on preprocessed NSL-KDD data.

### 🎯 Objectives
- Preprocess network data (encoding, scaling)
- Train and evaluate an ML model
- Detect intrusions automatically
- Visualize performance metrics

---

## ⚙️ Technologies Used
- Python
- Pandas & NumPy — data handling
- Scikit-learn — ML algorithms
- Joblib — model saving
- Matplotlib & Seaborn — visualization

---

## 🧠 How It Works

1. Load dataset from `/data/KDDTrain+.csv`
2. Encode categorical features
3. Scale numerical values
4. Train a Random Forest classifier
5. Evaluate predictions and display metrics
6. Save trained model to `/models/intrusion_model.pkl`

---

## 📁 Project Structure

AI-Intrusion-Detection-System/
├── data/ ← Dataset files
├── models/ ← Trained ML models
├── notebooks/ ← Jupyter notebooks / Python scripts
├── src/ ← Optional helper scripts
├── requirements.txt ← List of Python libraries
└── README.md ← Project documentation

---

## ▶️ How to Run

1. Clone the repository or open it in **GitHub Codespaces**.
2. Install the required libraries:
3. 3. Run the notebook inside `notebooks/IDS_Model.ipynb` (or `IDS_Model.py`).
4. Check the console for classification results and saved model files.

---

## 📊 Example Outputs
After running the model, you’ll see results like:

- **Confusion Matrix**  
- **Accuracy**, **Precision**, **Recall**, **F1-Score**

You can also visualize these metrics using Seaborn and Matplotlib.

---

## 🧾 Dataset
Dataset used: [NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd)  
Includes both normal and attack traffic records.

---

## 🧰 Future Improvements
- Add deep learning models (e.g., LSTM or autoencoders)
- Build a web dashboard for real-time detection
- Deploy model using Flask or FastAPI

---

### 👨‍💻 Author
**Sugreshwar Chandike**  
*AI and Cybersecurity Enthusiast*  
[GitHub Profile](https://github.com/SugreshwarChandike)
