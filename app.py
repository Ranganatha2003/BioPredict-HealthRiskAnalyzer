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
        'aliases': ['Hb', 'Hemoglobin', 'Hgb', 'Haemoglobin'],
        'min': 13.0, 'max': 17.0, 'unit': 'g/dL',
        'risks': {'low': 'May indicate Anemia', 'high': 'May indicate Polycythemia'},
        'suggestions': {
            'low': 'Increase iron-rich foods like spinach and lentils. Consult for iron supplements.',
            'high': 'Stay well-hydrated. Consult a doctor to rule out underlying conditions.'
        }
    },
    'RBC Count': {
        'aliases': ['RBC', 'Red Blood Cell', 'Erythrocyte Count'],
        'min': 4.5, 'max': 5.5, 'unit': 'mill/cumm',
        'risks': {'low': 'May indicate Anemia or nutritional deficiency', 'high': 'Possible dehydration or bone marrow disorder'},
        'suggestions': {
            'low': 'Ensure adequate intake of Vitamin B12 and Folic acid.',
            'high': 'Increase water intake and avoid tobacco products.'
        }
    },
    'WBC Count': {
        'aliases': ['WBC', 'White Blood Cell', 'Leukocyte Count', 'Total WBC'],
        'min': 4000, 'max': 11000, 'unit': 'cells/cumm',
        'risks': {'low': 'Possible weakened immune system', 'high': 'Possible infection or inflammation'},
        'suggestions': {
            'low': 'Focus on immune-boosting foods and maintain good hygiene.',
            'high': 'Rest and monitor for fever. Consult if symptoms persist.'
        }
    },
    'Platelet count': {
        'aliases': ['Platelets', 'Platelet count', 'PLT', 'Thrombocytes'],
        'min': 150000, 'max': 450000, 'unit': 'cells/cumm',
        'risks': {'low': 'Possible risk of easy bruising/bleeding', 'high': 'Possible risk of blood clots'},
        'suggestions': {
            'low': 'Avoid activities with high injury risk. Avoid blood-thinning meds like aspirin unless prescribed.',
            'high': 'Maintain an active lifestyle and stay hydrated to improve circulation.'
        }
    },
    'PCV': {
        'aliases': ['PCV', 'Hematocrit', 'HCT'],
        'min': 40, 'max': 50, 'unit': '%',
        'risks': {'low': 'May indicate Anemia', 'high': 'May indicate Dehydration'},
        'suggestions': {
            'low': 'Consult for iron level check and balanced diet.',
            'high': 'Drink plenty of fluids.'
        }
    },
    'MCV': {
        'aliases': ['MCV', 'Mean Corpuscular Volume'],
        'min': 80, 'max': 100, 'unit': 'fL',
        'risks': {'low': 'May indicate Microcytic Anemia', 'high': 'May indicate Macrocytic Anemia'},
        'suggestions': {
            'low': 'Often associated with iron deficiency.',
            'high': 'Often associated with Vitamin B12 or folate deficiency.'
        }
    },
    'MCH': {
        'aliases': ['MCH', 'Mean Corpuscular Hemoglobin'],
        'min': 27, 'max': 32, 'unit': 'pg',
        'risks': {'low': 'May indicate Iron deficiency', 'high': 'Possible nutritional issues'},
        'suggestions': {
            'low': 'Consider iron-rich diet.',
            'high': 'Check B12/Folate levels.'
        }
    },
    'MCHC': {
        'aliases': ['MCHC', 'Mean Corpuscular Hb Conc'],
        'min': 32, 'max': 36, 'unit': 'g/dL',
        'risks': {'low': 'May indicate Iron deficiency', 'high': 'Possible Hereditary Spherocytosis'},
        'suggestions': {
            'low': 'Increase dietary iron intake.',
            'high': 'Consult a doctor for further evaluation.'
        }
    },
    'RDW': {
        'aliases': ['RDW', 'Red Cell Distribution Width', 'RDW-CV'],
        'min': 11.5, 'max': 14.5, 'unit': '%',
        'risks': {'low': 'No clinical significance', 'high': 'May indicate nutritional deficiencies'},
        'suggestions': {
            'low': 'Normal finding.',
            'high': 'Possible variation in red cell sizes, consult for deficiency check.'
        }
    },
    'Neutrophils': {
        'aliases': ['Neutrophils', 'Polymorphs'],
        'min': 40, 'max': 80, 'unit': '%',
        'risks': {'low': 'Risk of infection', 'high': 'Possible acute infection'},
        'suggestions': {
            'low': 'Avoid exposure to sick individuals.',
            'high': 'Consult doctor if fever or pain occurs.'
        }
    },
    'Lymphocytes': {
        'aliases': ['Lymphocytes'],
        'min': 20, 'max': 40, 'unit': '%',
        'risks': {'low': 'Possible immune deficiency', 'high': 'Possible viral infection'},
        'suggestions': {
            'low': 'Consult for immune health.',
            'high': 'Rest and allow the body to recover from infection.'
        }
    },
    'Eosinophils': {
        'aliases': ['Eosinophils'],
        'min': 1, 'max': 6, 'unit': '%',
        'risks': {'low': 'Usually normal', 'high': 'Possible allergy or parasitic infection'},
        'suggestions': {
            'low': 'Normal finding.',
            'high': 'Consult for possible allergy testing.'
        }
    },
    'Monocytes': {
        'aliases': ['Monocytes'],
        'min': 2, 'max': 10, 'unit': '%',
        'risks': {'low': 'Usually normal', 'high': 'Possible chronic infection'},
        'suggestions': {
            'low': 'Normal finding.',
            'high': 'Consult for further blood investigation.'
        }
    },
    'Basophils': {
        'aliases': ['Basophils'],
        'min': 0, 'max': 2, 'unit': '%',
        'risks': {'low': 'Normal', 'high': 'Possible inflammatory condition'},
        'suggestions': {
            'low': 'Normal finding.',
            'high': 'Discuss with doctor if persistent.'
        }
    },
    'Fasting Glucose': {
        'aliases': ['Glucose', 'Blood Sugar', 'FBS', 'Sugar Fasting'],
        'min': 70, 'max': 100, 'unit': 'mg/dL',
        'risks': {'low': 'Hypoglycemia risk', 'high': 'May indicate Prediabetes/Diabetes'},
        'suggestions': {
            'low': 'Carry a fast-acting sugar source.',
            'high': 'Reduce refined carbs and stay active.'
        }
    }
}

# --- Analysis Logic ---
def analyze_blood_text(text):
    results = []
    has_abnormal = False
    
    import re
    # Line-by-line parsing for robustness
    lines = text.split('\n')
    
    for param, ref in BLOOD_PARAMETERS.items():
        found_val = None
        for alias in ref.get('aliases', [param]):
            alias_pattern = re.escape(alias).replace('\\ ', '\\s*')
            
            # Look for alias in each line
            for line in lines:
                if re.search(fr"\b{alias_pattern}\b", line, re.IGNORECASE):
                    # Look for the first valid numeric value in the same line after the alias
                    # This pattern matches numbers with optional decimals and handles commas/dots
                    val_matches = re.findall(r'(\d+[\.,]?\d*)', line)
                    
                    # We usually want the number that follows the alias
                    # But if the line contains multiple numbers (like reference range), we need careful picking
                    # Typically: [Value] [Units] [Ref Range]
                    # Strategy: Find index of alias, look for numbers after it
                    alias_match = re.search(fr"\b{alias_pattern}\b", line, re.IGNORECASE)
                    text_after_alias = line[alias_match.end():]
                    
                    # Also look for values specifically with common blood unit prefixes
                    numeric_after = re.search(r'(\d+[\.,]?\d*)', text_after_alias)
                    
                    if numeric_after:
                        val_str = numeric_after.group(1).replace(',', '.')
                        try:
                            # Basic validation: CBC values are rarely > 1,000,000 or < 0
                            # Except Platelets (150,000+) and WBC (4,000+)
                            candidate_val = float(val_str)
                            
                            # Simple logic to filter out reference range values if they appear together
                            # If the number is identical to our min or max ref, it might be the range
                            # But if it's the only number, we take it.
                            found_val = candidate_val
                            break # Found for this alias
                        except ValueError:
                            continue
                if found_val is not None: break
            if found_val is not None: break
        
        if found_val is not None:
            val = found_val
            status = "Normal"
            risk = "N/A"
            suggestion = "Your levels are within the normal range."
            
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
