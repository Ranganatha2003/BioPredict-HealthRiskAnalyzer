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
from werkzeug.utils import secure_filename
import pdfplumber
from PIL import Image
import pytesseract

# Initialize App
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get('SESSION_SECRET', 'dev_secret_key')

# Database Config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bio.db' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

# --- Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, username, password_hash):
        self.username = username
        self.password_hash = password_hash

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    age = db.Column(db.Float, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    glucose = db.Column(db.Float, nullable=False)
    bp = db.Column(db.Float, nullable=False)
    result = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, user_id, age, bmi, glucose, bp, result):
        self.user_id = user_id
        self.age = age
        self.bmi = bmi
        self.glucose = glucose
        self.bp = bp
        self.result = result

class BloodReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    analysis_json = db.Column(db.Text, nullable=False) # Stores extracted values and risks
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, user_id, filename, analysis_json):
        self.user_id = user_id
        self.filename = filename
        self.analysis_json = analysis_json

# --- ML Model ---
model = None

def load_model():
    global model
    try:
        model = joblib.load('model.pkl')
    except:
        print("Model not found. Please run train_model.py first.")

# --- Reference Ranges and Risk Mapping ---
BLOOD_PARAMETERS = {
    'Hemoglobin': {
        'aliases': ['Hb', 'Hemoglobin', 'Hgb'],
        'min': 13.5, 'max': 17.5, 'unit': 'g/dL',
        'risks': {'low': 'Anemia Risk', 'high': 'Polycythemia Risk'},
        'suggestions': {
            'low': 'Increase iron-rich foods like spinach, lentils, and red meat. Consult for iron supplements.',
            'high': 'Stay well-hydrated. Consult a doctor to rule out underlying conditions.'
        }
    },
    'RBC': {
        'aliases': ['RBC', 'Red Blood Cell', 'Erythrocytes'],
        'min': 4.5, 'max': 5.9, 'unit': 'million/mcL',
        'risks': {'low': 'Possible Anemia or nutritional deficiency', 'high': 'Possible dehydration or bone marrow disorder'},
        'suggestions': {
            'low': 'Ensure adequate intake of Vitamin B12 and Folic acid.',
            'high': 'Increase water intake and avoid tobacco products.'
        }
    },
    'WBC': {
        'aliases': ['WBC', 'White Blood Cell', 'Leukocytes'],
        'min': 4500, 'max': 11000, 'unit': 'cells/mcL',
        'risks': {'low': 'Weakened immune system', 'high': 'Infection or inflammation risk'},
        'suggestions': {
            'low': 'Focus on immune-boosting foods and maintain good hygiene.',
            'high': 'Rest and monitor for fever. Consult if symptoms persist.'
        }
    },
    'Platelet count': {
        'aliases': ['Platelets', 'Platelet count', 'PLT'],
        'min': 150000, 'max': 450000, 'unit': 'mcL',
        'risks': {'low': 'Risk of bleeding/easy bruising', 'high': 'Risk of blood clots'},
        'suggestions': {
            'low': 'Avoid activities with high injury risk. Avoid blood-thinning meds like aspirin unless prescribed.',
            'high': 'Maintain an active lifestyle and stay hydrated to improve circulation.'
        }
    },
    'Fasting Glucose': {
        'aliases': ['Fasting Glucose', 'Blood Sugar', 'Glucose', 'FBS'],
        'min': 70, 'max': 100, 'unit': 'mg/dL',
        'risks': {'low': 'Hypoglycemia risk', 'high': 'Diabetes/Prediabetes risk'},
        'suggestions': {
            'low': 'Carry a source of fast-acting sugar. Eat regular, balanced meals.',
            'high': 'Reduce sugar and refined carb intake. Increase physical activity.'
        }
    },
    'Total Cholesterol': {
        'aliases': ['Cholesterol', 'Total Cholesterol', 'TC'],
        'min': 0, 'max': 200, 'unit': 'mg/dL',
        'risks': {'low': 'Usually not a concern', 'high': 'Heart Disease Risk'},
        'suggestions': {
            'low': 'Maintain a balanced diet with healthy fats.',
            'high': 'Limit saturated fats. Increase fiber intake from vegetables and oats.'
        }
    },
    'Triglycerides': {
        'aliases': ['Triglycerides', 'TG', 'TRIG'],
        'min': 0, 'max': 150, 'unit': 'mg/dL',
        'risks': {'low': 'Usually not a concern', 'high': 'Metabolic Risk/Heart health'},
        'suggestions': {
            'low': 'Ensure adequate calorie and healthy fat intake.',
            'high': 'Limit alcohol and sugar. Engage in regular aerobic exercise.'
        }
    },
    'Creatinine': {
        'aliases': ['Creatinine', 'CRE', 'CREA'],
        'min': 0.7, 'max': 1.3, 'unit': 'mg/dL',
        'risks': {'low': 'Low muscle mass/malnutrition', 'high': 'Kidney Function Risk'},
        'suggestions': {
            'low': 'Ensure adequate protein intake and muscle-strengthening exercise.',
            'high': 'Limit high-protein intake and stay well-hydrated. Discuss kidney health with a doctor.'
        }
    }
}

# --- Analysis Logic ---
def analyze_blood_text(text):
    results = []
    has_abnormal = False
    
    import re
    # Clean text for better matching (remove multiple spaces/newlines)
    clean_text = re.sub(r'\s+', ' ', text)
    
    # Rule-based extraction for all defined parameters
    for param, ref in BLOOD_PARAMETERS.items():
        # Try primary name and all aliases
        found_val = None
        for alias in ref.get('aliases', [param]):
            # More flexible regex pattern:
            # 1. Alias (case-insensitive)
            # 2. Optional non-alphanumeric separators (spaces, colons, dashes, etc.)
            # 3. Numeric value (handles decimals with . or ,)
            # 4. Optional unit (ignored for extraction)
            escaped_alias = re.escape(alias).replace('\\ ', '\\s*')
            pattern = fr"{escaped_alias}\s*[^0-9\n]*?\s*(\d+[\.,]?\d*)"
            
            # Find all matches and take the most likely one (often the first or one with units nearby)
            # In clinical reports, the first occurrence is usually the result
            match = re.search(pattern, clean_text, re.IGNORECASE)
            
            if match:
                val_str = match.group(1).replace(',', '.')
                try:
                    found_val = float(val_str)
                    break # Stop if we found a valid number for this parameter
                except ValueError:
                    continue
        
        if found_val is not None:
            val = found_val
            status = "Normal"
            risk = "N/A"
            suggestion = "Keep up your healthy lifestyle."
            
            if val < ref['min']:
                status = "Low"
                risk = ref['risks'].get('low', 'Value below range')
                suggestion = ref['suggestions'].get('low', 'Consult a doctor.')
                has_abnormal = True
            elif val > ref['max']:
                status = "High"
                risk = ref['risks'].get('high', 'Value above range')
                suggestion = ref['suggestions'].get('high', 'Consult a doctor.')
                has_abnormal = True
            
            results.append({
                'parameter': param,
                'value': val,
                'unit': ref['unit'],
                'range': f"{ref['min']} - {ref['max']}",
                'status': status,
                'risk': risk,
                'suggestion': suggestion
            })
    
    return {'data': results, 'has_abnormal': has_abnormal}

# --- OpenAI Helper ---
def get_ai_advice(prediction_data, user_query=None):
    api_key = os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY")
    base_url = os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
    if not api_key:
        return "AI Advisor is temporarily unavailable. Please check back later."

    system_prompt = (
        "You are a professional health and diet advisor. "
        "Based on the user's health metrics and their risk prediction, "
        "provide clear, actionable health suggestions and a personalized diet plan. "
        "Maintain a supportive, professional tone. If they ask a question, answer it concisely."
    )
    
    user_context = str(prediction_data)
    
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
            if model:
                input_data = pd.DataFrame([[age, bmi, glucose, bp]], columns=['Age', 'BMI', 'Glucose', 'BloodPressure'])
                pred_val = model.predict(input_data)[0]
                prediction = "High Risk" if pred_val == 1 else "Low Risk"
                new_pred = Prediction(user_id=session['user_id'], age=age, bmi=bmi, glucose=glucose, bp=bp, result=prediction)
                db.session.add(new_pred)
                db.session.commit()
                ai_advice = get_ai_advice({'age': age, 'bmi': bmi, 'glucose': glucose, 'bp': bp, 'result': prediction})
            else:
                prediction = "Error: Model not loaded"
        except ValueError:
            prediction = "Invalid input values"
    history = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.created_at.desc()).all()
    return render_template('dashboard.html', username=session['username'], prediction=prediction, ai_advice=ai_advice, history=history)

@app.route('/blood-report', methods=['GET', 'POST'])
def blood_report():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    analysis = None
    if request.method == 'POST':
        file = request.files.get('report')
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            text = ""
            if filename.endswith('.pdf'):
                with pdfplumber.open(filepath) as pdf:
                    text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                text = pytesseract.image_to_string(Image.open(filepath))
            elif filename.endswith('.csv'):
                df = pd.read_csv(filepath)
                text = df.to_string()
            
            analysis = analyze_blood_text(text)
            new_report = BloodReport(
                user_id=session['user_id'],
                filename=filename,
                analysis_json=json.dumps(analysis)
            )
            db.session.add(new_report)
            db.session.commit()

    reports = BloodReport.query.filter_by(user_id=session['user_id']).order_by(BloodReport.created_at.desc()).all()
    return render_template('blood_report.html', username=session['username'], analysis=analysis, reports=reports)

@app.route('/api/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.json
    last_pred = Prediction.query.filter_by(user_id=session['user_id']).order_by(Prediction.created_at.desc()).first()
    metrics = {}
    if last_pred:
        metrics = {'age': last_pred.age, 'bmi': last_pred.bmi, 'glucose': last_pred.glucose, 'bp': last_pred.bp, 'result': last_pred.result}
    response = get_ai_advice(metrics, data.get('query'))
    return jsonify({"response": response})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
