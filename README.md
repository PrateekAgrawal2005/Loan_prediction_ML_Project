# 🚀 Loan Prediction & Image Color Quantization App

This is a **Machine Learning + Computer Vision web app** built using **Streamlit**.
It performs:

* 🏦 **Loan Approval Prediction**
* 🎨 **Image Color Quantization (K-Means Clustering)**

---

## 🌐 Live Demo

👉 **Working URL:**
https://loanpredictionmlproject-dovahwqtcwkhzy76zadh5y.streamlit.app/

---

## 📌 Features

### 🏦 Loan Prediction

* Predicts whether a loan will be **Approved or Rejected**
* Uses trained ML model (`loan_classifier_pipeline.pkl`)
* Takes user input like:

  * Income
  * Loan Amount
  * Credit History
  * Education, Property Area, etc.
* Displays **prediction + confidence score**

---

### 🎨 Image Color Quantization

* Upload any image
* Reduce number of colors using **K-Means Clustering**
* Compare:

  * Original Image
  * Quantized Image
* Shows:

  * 📦 Original Size
  * 📦 Quantized Size
  * ⚡ Processing Time

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **Scikit-learn**
* **Pandas**
* **Matplotlib**
* **Pillow (PIL)**
* **Joblib**

---

## 📂 Project Structure

```
├── main.py                      # Main Streamlit App
├── color_quantization.py       # Image processing logic
├── loan_classifier_pipeline.pkl # Trained ML model
├── pyproject.toml              # Dependencies
├── README.md
```

---

## ⚙️ Installation (Local Setup)

```bash
git clone <your-repo-link>
cd <project-folder>

pip install -r requirements.txt   # OR use pyproject.toml

streamlit run main.py
```

---

## ☁️ Deployment

This project is deployed using **Streamlit Cloud**.

Steps:

1. Push code to GitHub
2. Go to Streamlit Cloud
3. Select repo
4. Set:

   * **Main file:** `main.py`
   * **Python version:** 3.13 (recommended)

---

## 📊 Learning Highlights

* End-to-end ML project deployment
* Feature engineering & preprocessing
* Model pipeline usage
* Streamlit UI development
* Image processing using clustering

---

## 👨‍💻 Author

**Prateek Agrawal**
B.Tech AIML Student

---

## ⭐ Future Improvements

* Add authentication system
* Improve UI/UX
* Add more ML models
* Optimize image processing speed

---

⭐ If you like this project, give it a star!
