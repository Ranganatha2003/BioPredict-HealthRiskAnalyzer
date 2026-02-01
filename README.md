# 🧬 BioPredict – Health Risk Analyzer

BioPredict is a Flask-based health analysis web application designed to assist in **early risk assessment of lifestyle-related health conditions**.  
The system supports **diabetes risk prediction** using machine learning and **comprehensive blood report analysis** using rule-based biomedical reference ranges.

⚠️ *This application provides health risk insights only and is **not a medical diagnosis system***.

---

## 🚀 Features

### 👤 User Management
- Secure user registration and login
- Password hashing using bcrypt
- Session-based authentication

### 🧪 Diabetes Risk Prediction
- Predicts **Type-2 Diabetes risk** (Low / High)
- Inputs:
  - Age
  - BMI
  - Blood Glucose Level
  - Blood Pressure
- Machine learning model built using **Scikit-learn**
- Prediction history stored per user

### 🩸 Blood Report Analysis (CBC)
- Upload lab reports (PDF / Image / CSV)
- Extracts major **Complete Blood Count (CBC)** parameters:
  - Hemoglobin
  - RBC, WBC
  - Platelet Count
  - PCV
  - MCV, MCH, MCHC
  - RDW
  - Neutrophils, Lymphocytes, Eosinophils, Monocytes, Basophils
- Compares values with standard reference ranges
- Identifies **possible health risks** such as:
  - Anemia risk
  - Dehydration risk
  - Infection risk
- Provides **general lifestyle & dietary suggestions**
- Stores analysis history for logged-in users

---

## 🧠 System Design Overview

