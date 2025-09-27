from flask import Flask, request, jsonify, render_template
import joblib
from scipy.sparse import load_npz
from textblob import TextBlob

app = Flask(__name__)

# Load artifacts
def load_artifacts():
    recipes_combined = joblib.load("../models/recipes_combined.pkl")
    tfidf = joblib.load("../models/tfidf_vectorizer.pkl")
    recipe_vectors = load_npz("../models/recipe_vectors.npz")
    knn = joblib.load("../models/recipes_knn_model.pkl")
    return recipes_combined, tfidf, recipe_vectors, knn


recipes_combined, tfidf, recipe_vectors, knn = load_artifacts()

# Recommendation function
def recommend_recipes(user_ingredients, tfidf, knn, recipes_combined, n_neighbors=5):
    user_input = ' '.join(user_ingredients)
    user_vector = tfidf.transform([user_input])
    distances, indices = knn.kneighbors(user_vector, n_neighbors=n_neighbors)
    recommended_recipes = recipes_combined.iloc[indices[0]].drop_duplicates(subset='name')
    
    recommendations = []
    for _, recipe in recommended_recipes.iterrows():
        review = recipe['review']
        sentiment = TextBlob(review).sentiment.polarity
        sentiment_label = (
            'positive' if sentiment > 0 else
            'negative' if sentiment < 0 else
            'neutral'
        )
        steps = recipe['steps'].split('.') if isinstance(recipe['steps'], str) else recipe['steps']
        steps = [step.strip() for step in steps if step.strip()]  # Clean empty strings
        
        recommendations.append({
            'name': recipe['name'],
            'rating': recipe['rating'],
            'minutes': recipe['minutes'],
            'ingredients': recipe['ingredients'],
            'steps': steps,
            'review': recipe['review'],
            'sentiment_label': sentiment_label
        })
    return recommendations

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_ingredients = data.get('ingredients', [])
    recommendations = recommend_recipes(user_ingredients, tfidf, knn, recipes_combined)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
