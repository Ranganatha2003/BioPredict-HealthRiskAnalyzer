from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import bcrypt
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
import requests

# Initialize App
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('SESSION_SECRET', 'dev_secret_key')

# Database Config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bio.db' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    glucose = db.Column(db.Float, nullable=False)
    bp = db.Column(db.Float, nullable=False)
    result = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# --- ML Model ---
model = None

def load_model():
    global model
    try:
        model = joblib.load('model.pkl')
    except:
        print("Model not found. Please run train_model.py first.")

# --- OpenAI Helper ---
def get_ai_advice(prediction_data, user_query=None):
    """
    Get AI-powered health suggestions and diet plans based on prediction data.
    Uses Replit AI Integrations for OpenAI access.
    """
    api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
    base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
    if not api_key:
        return "AI Advisor is temporarily unavailable. Please check back later."

    system_prompt = (
        "You are a professional health and diet advisor. "
        "Based on the user's health metrics (Age, BMI, Glucose, Blood Pressure) and their risk prediction, "
        "provide clear, actionable health suggestions and a personalized diet plan. "
        "Maintain a supportive, professional tone. If they ask a question, answer it concisely."
    )
    
    user_context = (
        f"Metrics: Age {prediction_data['age']}, BMI {prediction_data['bmi']}, "
        f"Glucose {prediction_data['glucose']} mg/dL, Blood Pressure {prediction_data['bp']} mm Hg. "
        f"Risk Level: {prediction_data['result']}."
    )
    
    if user_query:
        user_message = f"{user_context}\nUser Question: {user_query}"
    else:
        user_message = f"Please provide a health improvement plan and diet suggestions based on these metrics: {user_context}"

    try:
        response = requests.post(
            f"{base_url}/chat/completions" if base_url else "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            },
            timeout=15
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"AI Error: {e}")
        return "I couldn't generate a plan at the moment. Please consult a doctor for personalized advice."

# --- Routes ---

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            return render_template('register.html', error="Username already exists")
            
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        new_user = User(username=username, password_hash=hashed.decode('utf-8'))
        db.session.add(new_user)
        db.session.commit()
        
        session['user_id'] = new_user.id
        session['username'] = new_user.username
        return redirect(url_for('dashboard'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid credentials")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    prediction = None
    ai_advice = None
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            bmi = float(request.form['bmi'])
            glucose = float(request.form['glucose'])
            bp = float(request.form['bp'])
            
            # Predict
            if model:
                input_data = pd.DataFrame([[age, bmi, glucose, bp]], 
                                        columns=['Age', 'BMI', 'Glucose', 'BloodPressure'])
                pred_val = model.predict(input_data)[0]
                prediction = "High Risk" if pred_val == 1 else "Low Risk"
                
                # Save
                new_pred = Prediction(
                    user_id=session['user_id'],
                    age=age, bmi=bmi, glucose=glucose, bp=bp,
                    result=prediction
                )
                db.session.add(new_pred)
                db.session.commit()
                
                # Get initial AI advice
                ai_advice = get_ai_advice({'age': age, 'bmi': bmi, 'glucose': glucose, 'bp': bp, 'result': prediction})
            else:
                prediction = "Error: Model not loaded"
                
        except ValueError:
            prediction = "Invalid input values"

    # Get history
    history = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.created_at.desc()).all()
    
    return render_template('dashboard.html', 
                         username=session['username'], 
                         prediction=prediction,
                         ai_advice=ai_advice,
                         history=history)

@app.route('/api/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    last_pred = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.created_at.desc()).first()
    
    if not last_pred:
        return jsonify({"response": "Please complete a health assessment first so I can give you personalized advice."})
    
    metrics = {
        'age': last_pred.age,
        'bmi': last_pred.bmi,
        'glucose': last_pred.glucose,
        'bp': last_pred.bp,
        'result': last_pred.result
    }
    
    response = get_ai_advice(metrics, data.get('query'))
    return jsonify({"response": response})

# --- Init ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
