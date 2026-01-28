from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import bcrypt
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Initialize App
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('SESSION_SECRET', 'dev_secret_key')

# Database Config - Using SQLite for compatibility but structured for MySQL
# To use MySQL, change to: 'mysql+mysqlconnector://user:password@host/db'
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
            else:
                prediction = "Error: Model not loaded"
                
        except ValueError:
            prediction = "Invalid input values"

    # Get history
    history = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.created_at.desc()).all()
    
    return render_template('dashboard.html', 
                         username=session['username'], 
                         prediction=prediction,
                         history=history)

# --- Init ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
