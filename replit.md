# BioPredict - Health AI Application

## Overview

BioPredict is a health prediction web application that uses machine learning to assess diabetes risk factors. Users can input health metrics (age, BMI, glucose levels, blood pressure) and receive instant risk assessments powered by a Random Forest classifier. The application includes user authentication, prediction history tracking, and an AI health assistant chat feature.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Framework
- **Flask** serves as the primary web framework, handling HTTP routes, session management, and template rendering
- SQLite database via Flask-SQLAlchemy for persistent storage of users and predictions
- Jinja2 templating for server-side HTML rendering

### Authentication System
- Session-based authentication using Flask's built-in session management
- Password hashing with bcrypt for secure credential storage
- User registration and login flows with form validation

### Machine Learning Pipeline
- **Scikit-learn Random Forest Classifier** trained on diabetes risk data
- Model serialization via joblib (`model.pkl`)
- Training script (`train_model.py`) handles data preprocessing and model training
- Features: Age, BMI, Glucose, Blood Pressure → Binary risk classification

### Frontend Architecture
- Server-rendered HTML templates extending a base layout (`base.html`)
- Custom CSS with CSS variables for theming
- Dashboard includes:
  - Health metrics input form
  - Real-time prediction results
  - AI chat assistant interface
  - Text-to-speech for health advice

### Data Models
- **User**: id, username, password_hash, created_at
- **Prediction**: id, user_id (FK), age, bmi, glucose, bp, result, created_at

### Replit Integration Files
The `.replit_integration_files` directory contains pre-built utilities for AI integrations including:
- Voice/audio streaming with WebAudio API and AudioWorklet
- Chat conversation storage with Drizzle ORM
- Image generation endpoints
- Batch processing utilities with rate limiting

## External Dependencies

### Python Packages
- **Flask** + **Flask-SQLAlchemy** + **Flask-CORS**: Web framework and database
- **bcrypt**: Password hashing
- **joblib**: ML model serialization
- **pandas** + **numpy**: Data manipulation
- **scikit-learn**: Machine learning algorithms

### AI/ML Services
- OpenAI API (via Replit AI Integrations) for:
  - Health advice chat responses
  - Text-to-speech for accessibility
  - Voice conversation capabilities

### Database
- SQLite (`bio.db`) for local development
- Drizzle ORM schemas available for PostgreSQL migration if needed

### Frontend Assets
- Google Fonts (Inter) for typography
- Custom CSS styling system