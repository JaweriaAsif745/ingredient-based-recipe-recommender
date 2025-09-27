# 🍳 Ingredient-Based Recipe Recommendation System

This project is a **recipe recommendation system** that suggests recipes based on the ingredients you provide.
It uses **text preprocessing, TF-IDF vectorization, and K-Nearest Neighbors (KNN)** to find recipes most similar to the user’s input ingredients.

The project also performs **EDA (exploratory data analysis)** and **sentiment analysis** on user reviews to provide deeper insights.

---

## 📊 Dataset

The dataset is taken from **Kaggle**:
[Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)

I only use **two files** from the dataset:

* `RAW_interactions.csv` → contains recipe reviews and ratings
* `RAW_recipes.csv` → contains recipe details (name, ingredients, steps, time)

---

## 📂 Project Structure

```
Recipe_project/
│
├── RAW_interactions.csv/          # Dataset folder (user interactions)
│   └── RAW_interactions.csv
│
├── RAW_recipes.csv/               # Dataset folder (recipe details)
│   └── RAW_recipes.csv
│
├── notebook/                     # Jupyter notebooks
│   └── Recipes_final.ipynb        # Preprocessing, EDA, model training
│
├── models/                        # Artifacts generated after training
│   ├── recipes_combined.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── recipes_knn_model.pkl
│   └── recipe_vectors.npz
│
├── app/                           # Flask web application
│   ├── static/                    # CSS, images, and videos
│   │   ├── style.css
│   │   ├── images/
│   │   │   ├── 1.jpg
│   │   │   ├── 1_enhanced.jpg
│   │   │   └── background.jpg
│   │   └── videos/
│   │       ├── recipe1.mp4
│   │       └── recipe2.mp4
│   ├── templates/                 # HTML templates
│   │   └── index.html
│   ├── app.py                     # Flask server
│   └── model.py                   # Model loading + recommendation logic
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
```

---

## ⚙️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/JaweriaAsif745/ingredient-based-recipe-recommender.git
   cd ingredient-based-recipe-recommender
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle:
   [Food.com Recipes and Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)

   **Place the two required files in the project directory as shown above.**

---

## 🏗️ Generating Models

Since model files are too large to upload on GitHub, you need to generate them yourself:

1. Open the notebook:

   ```bash
   notebook/Recipes_final.ipynb
   ```

2. Run all cells. This will:

   * Clean and preprocess the dataset
   * Perform EDA & sentiment analysis
   * Train the KNN recommendation model
   * Save the following artifacts in the `models/` folder:

     * `recipes_combined.pkl`
     * `tfidf_vectorizer.pkl`
     * `recipes_knn_model.pkl`
     * `recipe_vectors.npz`

---

## 🚀 Running the App

Once models are generated, start the Flask app:

```bash
cd app
python model.py
```

Then open your browser at:
👉 `http://127.0.0.1:5000/`

---

## 🎥 Demo

### 🔗 Video Demo

https://github.com/user-attachments/assets/a539eed0-3731-4609-81bb-3f5174680398

### 📸 Screenshots

**Home Page**

<img width="1896" height="870" alt="image" src="https://github.com/user-attachments/assets/50502eff-75bb-45b7-b56c-f29ed54fe2ae" />

**Recommendations Display**

<img width="1828" height="838" alt="image" src="https://github.com/user-attachments/assets/2fa778da-d8b0-448c-ba89-daf03c6cbded" />


---

## 🧠 Tech Stack

* **Python** (Flask, Pandas, Scikit-learn, NLTK, TextBlob)
* **Frontend**: HTML, CSS, JavaScript
* **ML Techniques**: TF-IDF, KNN, Sentiment Analysis

---

## 👩‍💻 Author

Developed by **Jaweria Asif** ✨

