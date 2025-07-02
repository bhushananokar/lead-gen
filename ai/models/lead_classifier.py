# ai/models/lead_classifier.py - ML Models for Lead Classification and Scoring

# ============================
# IMPORTS
# ============================
import numpy as np
import pandas as pd
import pickle
import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import joblib

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Install with: pip install scikit-learn")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# ENUMS AND DATA CLASSES
# ============================

class LeadPriority(Enum):
    """Lead priority classifications"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class LeadQuality(Enum):
    """Lead quality classifications"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class ModelType(Enum):
    """Available model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"
    SVM = "svm"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    model_type: ModelType
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_columns: List[str] = field(default_factory=list)
    target_column: str = "priority"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

@dataclass
class TrainingResult:
    """Results from model training"""
    model: Any
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    feature_importance: Dict[str, float]
    training_time: float
    cv_scores: List[float]
    confusion_matrix: np.ndarray
    classification_report: str

@dataclass
class PredictionResult:
    """Results from model prediction"""
    predicted_class: str
    prediction_probability: Dict[str, float]
    confidence: float
    feature_contributions: Dict[str, float]
    model_version: str
    prediction_time: float

# ============================
# FEATURE ENGINEERING CLASS
# ============================

class LeadFeatureEngineer:
    """Feature engineering for lead classification"""
    
    def __init__(self):
        self.text_vectorizer = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False
    
    def extract_features(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from lead data for ML models
        
        Args:
            lead_data: Raw lead data dictionary
            
        Returns:
            Dictionary of engineered features
        """
        features = {}
        
        # Basic contact features
        features['has_email'] = 1 if lead_data.get('email') else 0
        features['has_phone'] = 1 if lead_data.get('phone') else 0
        features['has_linkedin'] = 1 if lead_data.get('linkedin_url') else 0
        
        # Email quality features
        email = lead_data.get('email', '')
        features['email_is_corporate'] = 1 if self._is_corporate_email(email) else 0
        features['email_quality_score'] = self._calculate_email_quality(email)
        
        # Name features
        first_name = lead_data.get('first_name', '')
        last_name = lead_data.get('last_name', '')
        features['name_completeness'] = (1 if first_name else 0) + (1 if last_name else 0)
        features['name_length'] = len(first_name) + len(last_name)
        
        # Title features
        title = lead_data.get('title', '').lower()
        features['title_seniority'] = self._calculate_title_seniority(title)
        features['title_decision_power'] = self._calculate_decision_power(title)
        features['title_length'] = len(title)
        features['title_word_count'] = len(title.split()) if title else 0
        
        # Company features
        company = lead_data.get('company', {})
        features['company_size_score'] = self._encode_company_size(company.get('size', ''))
        features['industry_relevance'] = self._calculate_industry_relevance(company.get('industry', ''))
        features['location_score'] = self._calculate_location_score(company.get('location', ''))
        features['has_company_website'] = 1 if company.get('website') else 0
        
        # Data quality features
        features['data_completeness'] = self._calculate_completeness(lead_data)
        features['data_freshness'] = self._calculate_freshness(lead_data.get('created_at'))
        
        # Source features
        source = lead_data.get('source', 'unknown').lower()
        features['source_quality'] = self._evaluate_source_quality(source)
        features['source_is_referral'] = 1 if 'referral' in source else 0
        features['source_is_organic'] = 1 if source in ['website', 'organic', 'search'] else 0
        
        # Engagement features
        features['contact_methods_count'] = features['has_email'] + features['has_phone'] + features['has_linkedin']
        features['profile_richness'] = self._calculate_profile_richness(lead_data)
        
        # Timing features
        created_at = lead_data.get('created_at')
        if created_at:
            try:
                dt = pd.to_datetime(created_at)
                features['day_of_week'] = dt.dayofweek
                features['hour_of_day'] = dt.hour
                features['is_weekend'] = 1 if dt.dayofweek >= 5 else 0
                features['is_business_hours'] = 1 if 9 <= dt.hour <= 17 else 0
            except:
                features['day_of_week'] = 0
                features['hour_of_day'] = 12
                features['is_weekend'] = 0
                features['is_business_hours'] = 1
        else:
            features['day_of_week'] = 0
            features['hour_of_day'] = 12
            features['is_weekend'] = 0
            features['is_business_hours'] = 1
        
        return features
    
    def _is_corporate_email(self, email: str) -> bool:
        """Check if email is corporate (not free provider)"""
        if not email:
            return False
        
        free_domains = [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
            'aol.com', 'icloud.com', 'protonmail.com'
        ]
        
        domain = email.split('@')[-1].lower() if '@' in email else ''
        return domain not in free_domains
    
    def _calculate_email_quality(self, email: str) -> float:
        """Calculate email quality score"""
        if not email:
            return 0.0
        
        score = 0.5  # Base score
        
        # Corporate email bonus
        if self._is_corporate_email(email):
            score += 0.3
        
        # Professional format bonus
        if '.' in email.split('@')[0]:
            score += 0.15
        
        # Valid format check
        if '@' in email and '.' in email.split('@')[1]:
            score += 0.05
        
        return min(1.0, score)
    
    def _calculate_title_seniority(self, title: str) -> float:
        """Calculate seniority score from title"""
        if not title:
            return 0.0
        
        title_lower = title.lower()
        
        # Executive level
        if any(word in title_lower for word in ['ceo', 'president', 'founder', 'chief']):
            return 1.0
        
        # Senior management
        if any(word in title_lower for word in ['vp', 'vice president', 'director']):
            return 0.8
        
        # Management
        if any(word in title_lower for word in ['manager', 'head of', 'lead']):
            return 0.6
        
        # Senior individual contributor
        if 'senior' in title_lower:
            return 0.4
        
        # Individual contributor
        return 0.2
    
    def _calculate_decision_power(self, title: str) -> float:
        """Calculate decision-making power from title"""
        if not title:
            return 0.0
        
        title_lower = title.lower()
        
        # High decision power
        if any(word in title_lower for word in ['ceo', 'president', 'founder', 'owner']):
            return 1.0
        
        # Medium-high decision power
        if any(word in title_lower for word in ['cto', 'cfo', 'vp', 'director']):
            return 0.8
        
        # Medium decision power
        if any(word in title_lower for word in ['manager', 'head']):
            return 0.6
        
        # Some influence
        if any(word in title_lower for word in ['senior', 'lead', 'principal']):
            return 0.4
        
        return 0.2
    
    def _encode_company_size(self, size: str) -> float:
        """Encode company size as numerical score"""
        size_mapping = {
            '1-10': 0.2,
            '11-50': 0.4,
            '51-200': 0.8,
            '201-500': 1.0,
            '501-1000': 0.9,
            '1001-5000': 0.7,
            '5000+': 0.5
        }
        
        return size_mapping.get(size, 0.5)
    
    def _calculate_industry_relevance(self, industry: str) -> float:
        """Calculate industry relevance score"""
        if not industry:
            return 0.5
        
        industry_lower = industry.lower()
        
        # High relevance industries
        high_relevance = ['technology', 'software', 'ai', 'fintech', 'saas']
        if any(keyword in industry_lower for keyword in high_relevance):
            return 1.0
        
        # Medium relevance industries
        medium_relevance = ['finance', 'healthcare', 'consulting', 'manufacturing']
        if any(keyword in industry_lower for keyword in medium_relevance):
            return 0.7
        
        return 0.5
    
    def _calculate_location_score(self, location: str) -> float:
        """Calculate location relevance score"""
        if not location:
            return 0.5
        
        location_lower = location.lower()
        
        # Tier 1 locations
        tier1 = ['san francisco', 'new york', 'seattle', 'boston', 'austin', 'london']
        if any(city in location_lower for city in tier1):
            return 1.0
        
        # Tier 2 locations
        tier2 = ['chicago', 'denver', 'atlanta', 'toronto', 'berlin']
        if any(city in location_lower for city in tier2):
            return 0.7
        
        return 0.5
    
    def _calculate_completeness(self, lead_data: Dict[str, Any]) -> float:
        """Calculate data completeness score"""
        required_fields = ['first_name', 'last_name', 'email', 'title']
        optional_fields = ['phone', 'linkedin_url', 'company']
        
        required_score = sum(1 for field in required_fields if lead_data.get(field)) / len(required_fields)
        optional_score = sum(1 for field in optional_fields if lead_data.get(field)) / len(optional_fields)
        
        return (required_score * 0.7) + (optional_score * 0.3)
    
    def _calculate_freshness(self, created_at: Optional[str]) -> float:
        """Calculate data freshness score"""
        if not created_at:
            return 0.5
        
        try:
            created_date = pd.to_datetime(created_at)
            days_old = (datetime.now() - created_date.tz_localize(None)).days
            
            if days_old <= 7:
                return 1.0
            elif days_old <= 30:
                return 0.8
            elif days_old <= 90:
                return 0.6
            else:
                return 0.3
        except:
            return 0.5
    
    def _evaluate_source_quality(self, source: str) -> float:
        """Evaluate lead source quality"""
        source_scores = {
            'referral': 1.0,
            'linkedin': 0.9,
            'website': 0.8,
            'event': 0.8,
            'webinar': 0.7,
            'directory': 0.6,
            'purchased': 0.3,
            'unknown': 0.4
        }
        
        return source_scores.get(source, 0.5)
    
    def _calculate_profile_richness(self, lead_data: Dict[str, Any]) -> float:
        """Calculate overall profile richness"""
        fields_to_check = [
            'first_name', 'last_name', 'email', 'phone', 'title',
            'linkedin_url', 'company', 'notes'
        ]
        
        filled_fields = sum(1 for field in fields_to_check if lead_data.get(field))
        return filled_fields / len(fields_to_check)

# ============================
# MAIN LEAD CLASSIFIER CLASS
# ============================

class LeadClassifier:
    """
    Machine Learning classifier for lead prioritization and quality assessment
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None, model_path: str = "ai/saved_models/"):
        """
        Initialize Lead Classifier
        
        Args:
            model_config: Configuration for the ML model
            model_path: Path to save/load models
        """
        self.model_config = model_config or ModelConfig(ModelType.RANDOM_FOREST)
        self.model_path = model_path
        self.feature_engineer = LeadFeatureEngineer()
        
        # Model components
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        
        # Performance tracking
        self.training_history = []
        self.last_training_date = None
        self.prediction_count = 0
        
        # Ensure model directory exists
        os.makedirs(model_path, exist_ok=True)
        
        logger.info("🤖 Lead Classifier initialized")
    
    def prepare_training_data(self, leads_data: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from lead records
        
        Args:
            leads_data: List of lead data dictionaries with labels
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        features_list = []
        targets = []
        
        for lead in leads_data:
            # Extract features
            features = self.feature_engineer.extract_features(lead)
            features_list.append(features)
            
            # Extract target (priority/quality label)
            target = lead.get(self.model_config.target_column)
            if target:
                targets.append(target)
            else:
                # Generate synthetic target based on features for demo
                targets.append(self._generate_synthetic_target(features))
        
        features_df = pd.DataFrame(features_list)
        targets_series = pd.Series(targets)
        
        return features_df, targets_series
    
    def _generate_synthetic_target(self, features: Dict[str, Any]) -> str:
        """Generate synthetic target for demonstration"""
        score = (
            features.get('title_seniority', 0) * 0.3 +
            features.get('email_quality_score', 0) * 0.2 +
            features.get('data_completeness', 0) * 0.2 +
            features.get('industry_relevance', 0) * 0.15 +
            features.get('company_size_score', 0) * 0.15
        )
        
        if score >= 0.8:
            return 'high'
        elif score >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def train_model(self, training_data: List[Dict[str, Any]], 
                   validation_split: float = 0.2) -> TrainingResult:
        """
        Train the lead classification model
        
        Args:
            training_data: List of lead data with labels
            validation_split: Fraction of data for validation
            
        Returns:
            TrainingResult with performance metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for model training")
        
        start_time = datetime.now()
        
        # Prepare data
        X, y = self.prepare_training_data(training_data)
        
        # Create preprocessor
        self.preprocessor = self._create_preprocessor(X)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=validation_split,
            random_state=self.model_config.random_state,
            stratify=y_encoded
        )
        
        # Create and train model
        self.model = self._create_model()
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        self.model = pipeline
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # AUC score for multiclass
        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.5  # Default if AUC calculation fails
        
        # Cross-validation scores
        cv_scores = cross_val_score(pipeline, X, y_encoded, cv=self.model_config.cv_folds)
        
        # Feature importance
        feature_importance = self._get_feature_importance(X.columns)
        
        # Confusion matrix and classification report
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, 
                                           target_names=self.label_encoder.classes_)
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Create result
        result = TrainingResult(
            model=self.model,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_score=auc,
            feature_importance=feature_importance,
            training_time=training_time,
            cv_scores=cv_scores.tolist(),
            confusion_matrix=conf_matrix,
            classification_report=class_report
        )
        
        # Save training history
        self.training_history.append({
            'timestamp': start_time,
            'accuracy': accuracy,
            'f1_score': f1,
            'training_samples': len(training_data)
        })
        
        self.last_training_date = start_time
        
        # Save model
        self.save_model()
        
        logger.info(f"✅ Model trained successfully - Accuracy: {accuracy:.3f}")
        
        return result
    
    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessing pipeline"""
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Create transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
    
    def _create_model(self):
        """Create ML model based on configuration"""
        if self.model_config.model_type == ModelType.RANDOM_FOREST:
            return RandomForestClassifier(
                **self.model_config.hyperparameters,
                random_state=self.model_config.random_state
            )
        
        elif self.model_config.model_type == ModelType.GRADIENT_BOOSTING:
            return GradientBoostingClassifier(
                **self.model_config.hyperparameters,
                random_state=self.model_config.random_state
            )
        
        elif self.model_config.model_type == ModelType.LOGISTIC_REGRESSION:
            return LogisticRegression(
                **self.model_config.hyperparameters,
                random_state=self.model_config.random_state
            )
        
        elif self.model_config.model_type == ModelType.SVM:
            return SVC(
                **self.model_config.hyperparameters,
                probability=True,
                random_state=self.model_config.random_state
            )
        
        elif self.model_config.model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
            return xgb.XGBClassifier(
                **self.model_config.hyperparameters,
                random_state=self.model_config.random_state
            )
        
        elif self.model_config.model_type == ModelType.ENSEMBLE:
            # Create ensemble of multiple models
            rf = RandomForestClassifier(random_state=self.model_config.random_state)
            gb = GradientBoostingClassifier(random_state=self.model_config.random_state)
            lr = LogisticRegression(random_state=self.model_config.random_state)
            
            return VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                voting='soft'
            )
        
        else:
            # Default to Random Forest
            return RandomForestClassifier(random_state=self.model_config.random_state)
    
    def _get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from trained model"""
        try:
            if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
                importances = self.model.named_steps['classifier'].feature_importances_
                
                # Get feature names after preprocessing
                if hasattr(self.preprocessor, 'get_feature_names_out'):
                    processed_feature_names = self.preprocessor.get_feature_names_out()
                else:
                    processed_feature_names = [f"feature_{i}" for i in range(len(importances))]
                
                return dict(zip(processed_feature_names, importances))
            
            return {}
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")
            return {}
    
    def predict(self, lead_data: Dict[str, Any]) -> PredictionResult:
        """
        Predict lead priority/quality
        
        Args:
            lead_data: Lead information dictionary
            
        Returns:
            PredictionResult with prediction and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        start_time = datetime.now()
        
        # Extract features
        features = self.feature_engineer.extract_features(lead_data)
        features_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = self.model.predict(features_df)[0]
        prediction_proba = self.model.predict_proba(features_df)[0]
        
        # Decode prediction
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        # Create probability dictionary
        probability_dict = dict(zip(
            self.label_encoder.classes_,
            prediction_proba
        ))
        
        # Calculate confidence (highest probability)
        confidence = np.max(prediction_proba)
        
        # Feature contributions (simplified)
        feature_contributions = self._calculate_feature_contributions(features)
        
        # Calculate prediction time
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # Update prediction count
        self.prediction_count += 1
        
        return PredictionResult(
            predicted_class=predicted_class,
            prediction_probability=probability_dict,
            confidence=confidence,
            feature_contributions=feature_contributions,
            model_version=f"{self.model_config.model_type.value}_v1.0",
            prediction_time=prediction_time
        )
    
    def _calculate_feature_contributions(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate simplified feature contributions"""
        # This is a simplified version
        # In practice, you might use SHAP or LIME for better explanations
        contributions = {}
        
        # Key features that typically contribute to priority
        key_features = [
            'title_seniority', 'email_quality_score', 'data_completeness',
            'industry_relevance', 'company_size_score'
        ]
        
        total_score = sum(features.get(f, 0) for f in key_features)
        
        for feature in key_features:
            if total_score > 0:
                contributions[feature] = features.get(feature, 0) / total_score
            else:
                contributions[feature] = 0.0
        
        return contributions
    
    def batch_predict(self, leads_data: List[Dict[str, Any]]) -> List[PredictionResult]:
        """
        Predict multiple leads in batch
        
        Args:
            leads_data: List of lead data dictionaries
            
        Returns:
            List of PredictionResult objects
        """
        results = []
        
        for lead_data in leads_data:
            try:
                result = self.predict(lead_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Prediction failed for lead {lead_data.get('id')}: {e}")
                # Add fallback result
                results.append(PredictionResult(
                    predicted_class='medium',
                    prediction_probability={'low': 0.33, 'medium': 0.34, 'high': 0.33},
                    confidence=0.3,
                    feature_contributions={},
                    model_version='fallback',
                    prediction_time=0.0
                ))
        
        return results
    
    def save_model(self, filename: Optional[str] = None):
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        filename = filename or f"lead_classifier_{self.model_config.model_type.value}.pkl"
        filepath = os.path.join(self.model_path, filename)
        
        # Save model and associated components
        model_package = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'label_encoder': self.label_encoder,
            'feature_engineer': self.feature_engineer,
            'model_config': self.model_config,
            'training_history': self.training_history,
            'last_training_date': self.last_training_date
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)
        
        logger.info(f"💾 Model saved to {filepath}")
    
    def load_model(self, filename: Optional[str] = None):
        """Load trained model from disk"""
        filename = filename or f"lead_classifier_{self.model_config.model_type.value}.pkl"
        filepath = os.path.join(self.model_path, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)
        
        # Restore model components
        self.model = model_package['model']
        self.preprocessor = model_package['preprocessor']
        self.label_encoder = model_package['label_encoder']
        self.feature_engineer = model_package['feature_engineer']
        self.model_config = model_package.get('model_config', self.model_config)
        self.training_history = model_package.get('training_history', [])
        self.last_training_date = model_package.get('last_training_date')
        
        logger.info(f"📁 Model loaded from {filepath}")
    
    def hyperparameter_tuning(self, training_data: List[Dict[str, Any]], 
                             param_grid: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            training_data: Training data
            param_grid: Parameter grid for tuning
            
        Returns:
            Best parameters and scores
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for hyperparameter tuning")
        
        # Prepare data
        X, y = self.prepare_training_data(training_data)
        
        # Create preprocessor and encode labels
        self.preprocessor = self._create_preprocessor(X)
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Default parameter grids for different models
        if param_grid is None:
            if self.model_config.model_type == ModelType.RANDOM_FOREST:
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [None, 10, 20],
                    'classifier__min_samples_split': [2, 5, 10]
                }
            elif self.model_config.model_type == ModelType.GRADIENT_BOOSTING:
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__learning_rate': [0.05, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7]
                }
            else:
                param_grid = {}
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', self._create_model())
        ])
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=self.model_config.cv_folds,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y_encoded)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics and statistics"""
        if not self.training_history:
            return {'status': 'No training history available'}
        
        latest_training = self.training_history[-1]
        
        return {
            'model_type': self.model_config.model_type.value,
            'target_column': self.model_config.target_column,
            'latest_accuracy': latest_training.get('accuracy', 0.0),
            'latest_f1_score': latest_training.get('f1_score', 0.0),
            'training_samples': latest_training.get('training_samples', 0),
            'last_training_date': self.last_training_date,
            'total_predictions': self.prediction_count,
            'model_available': self.model is not None,
            'training_history_count': len(self.training_history)
        }
    
    def explain_prediction(self, lead_data: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
        """
        Explain model prediction with feature importance
        
        Args:
            lead_data: Lead data for explanation
            top_n: Number of top features to explain
            
        Returns:
            Explanation dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Get prediction
        prediction_result = self.predict(lead_data)
        
        # Get feature values
        features = self.feature_engineer.extract_features(lead_data)
        
        # Get feature importance from model
        feature_importance = self._get_feature_importance(list(features.keys()))
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        # Create explanation
        explanation = {
            'prediction': prediction_result.predicted_class,
            'confidence': prediction_result.confidence,
            'top_features': [],
            'feature_values': features,
            'reasoning': []
        }
        
        for feature_name, importance in sorted_features:
            # Get original feature name (remove preprocessing prefixes)
            original_name = feature_name.split('__')[-1] if '__' in feature_name else feature_name
            
            if original_name in features:
                feature_value = features[original_name]
                explanation['top_features'].append({
                    'feature': original_name,
                    'value': feature_value,
                    'importance': importance,
                    'contribution': importance * feature_value
                })
        
        # Generate reasoning
        explanation['reasoning'] = self._generate_reasoning(features, prediction_result)
        
        return explanation
    
    def _generate_reasoning(self, features: Dict[str, Any], prediction: PredictionResult) -> List[str]:
        """Generate human-readable reasoning for prediction"""
        reasoning = []
        
        # Title seniority reasoning
        if features.get('title_seniority', 0) > 0.8:
            reasoning.append("High seniority title indicates decision-making authority")
        elif features.get('title_seniority', 0) < 0.3:
            reasoning.append("Junior-level title may have limited decision authority")
        
        # Email quality reasoning
        if features.get('email_quality_score', 0) > 0.8:
            reasoning.append("High-quality professional email address")
        elif features.get('email_quality_score', 0) < 0.5:
            reasoning.append("Email quality concerns may affect deliverability")
        
        # Data completeness reasoning
        if features.get('data_completeness', 0) > 0.8:
            reasoning.append("Complete profile with all essential information")
        elif features.get('data_completeness', 0) < 0.5:
            reasoning.append("Incomplete profile missing key information")
        
        # Industry relevance reasoning
        if features.get('industry_relevance', 0) > 0.8:
            reasoning.append("Industry is highly relevant to our solution")
        elif features.get('industry_relevance', 0) < 0.5:
            reasoning.append("Industry may not be ideal fit for our solution")
        
        # Company size reasoning
        company_size = features.get('company_size_score', 0)
        if company_size > 0.8:
            reasoning.append("Company size is in our optimal target range")
        elif company_size < 0.4:
            reasoning.append("Company size may not align with our typical customers")
        
        return reasoning
    
    def retrain_with_feedback(self, feedback_data: List[Dict[str, Any]]) -> TrainingResult:
        """
        Retrain model with user feedback
        
        Args:
            feedback_data: List of lead data with actual outcomes
            
        Returns:
            TrainingResult from retraining
        """
        logger.info(f"🔄 Retraining model with {len(feedback_data)} feedback samples")
        
        # Convert feedback to training format
        training_data = []
        for feedback in feedback_data:
            lead_data = feedback.get('lead_data', {})
            actual_outcome = feedback.get('actual_outcome')
            
            # Map outcomes to priority levels
            outcome_mapping = {
                'converted': 'high',
                'qualified': 'high',
                'interested': 'medium',
                'responded': 'medium',
                'no_response': 'low',
                'rejected': 'low'
            }
            
            mapped_priority = outcome_mapping.get(actual_outcome, 'medium')
            lead_data[self.model_config.target_column] = mapped_priority
            training_data.append(lead_data)
        
        # Retrain model
        return self.train_model(training_data)
    
    def generate_training_report(self, training_result: TrainingResult) -> str:
        """Generate human-readable training report"""
        report = f"""
🤖 Lead Classifier Training Report
{'='*50}

Model Configuration:
- Type: {self.model_config.model_type.value}
- Target: {self.model_config.target_column}
- Features: {len(training_result.feature_importance)} features

Performance Metrics:
- Accuracy: {training_result.accuracy:.3f}
- Precision: {training_result.precision:.3f}
- Recall: {training_result.recall:.3f}
- F1 Score: {training_result.f1_score:.3f}
- AUC Score: {training_result.auc_score:.3f}

Cross-Validation:
- CV Scores: {[f'{score:.3f}' for score in training_result.cv_scores]}
- Mean CV Score: {np.mean(training_result.cv_scores):.3f}
- Std CV Score: {np.std(training_result.cv_scores):.3f}

Training Details:
- Training Time: {training_result.training_time:.2f} seconds
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Top Feature Importance:
"""
        
        # Add top 10 features
        sorted_features = sorted(
            training_result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for i, (feature, importance) in enumerate(sorted_features, 1):
            report += f"  {i}. {feature}: {importance:.4f}\n"
        
        report += f"\nClassification Report:\n{training_result.classification_report}"
        
        return report


# ============================
# SPECIALIZED CLASSIFIERS
# ============================

class PriorityClassifier(LeadClassifier):
    """Specialized classifier for lead priority"""
    
    def __init__(self, model_path: str = "ai/saved_models/"):
        config = ModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            target_column="priority",
            hyperparameters={
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'class_weight': 'balanced'
            }
        )
        super().__init__(config, model_path)
    
    def predict_priority(self, lead_data: Dict[str, Any]) -> str:
        """Predict lead priority level"""
        result = self.predict(lead_data)
        return result.predicted_class


class QualityClassifier(LeadClassifier):
    """Specialized classifier for lead quality"""
    
    def __init__(self, model_path: str = "ai/saved_models/"):
        config = ModelConfig(
            model_type=ModelType.GRADIENT_BOOSTING,
            target_column="quality",
            hyperparameters={
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 6
            }
        )
        super().__init__(config, model_path)
    
    def predict_quality(self, lead_data: Dict[str, Any]) -> str:
        """Predict lead quality level"""
        result = self.predict(lead_data)
        return result.predicted_class


class ConversionClassifier(LeadClassifier):
    """Specialized classifier for conversion prediction"""
    
    def __init__(self, model_path: str = "ai/saved_models/"):
        config = ModelConfig(
            model_type=ModelType.ENSEMBLE,
            target_column="conversion_likely",
            hyperparameters={}
        )
        super().__init__(config, model_path)
    
    def predict_conversion_probability(self, lead_data: Dict[str, Any]) -> float:
        """Predict probability of conversion"""
        result = self.predict(lead_data)
        # Return probability of positive class
        return result.prediction_probability.get('high', 0.0)


# ============================
# MODEL FACTORY
# ============================

class ModelFactory:
    """Factory for creating different types of lead classifiers"""
    
    @staticmethod
    def create_priority_classifier(model_path: str = "ai/saved_models/") -> PriorityClassifier:
        """Create priority classifier"""
        return PriorityClassifier(model_path)
    
    @staticmethod
    def create_quality_classifier(model_path: str = "ai/saved_models/") -> QualityClassifier:
        """Create quality classifier"""
        return QualityClassifier(model_path)
    
    @staticmethod
    def create_conversion_classifier(model_path: str = "ai/saved_models/") -> ConversionClassifier:
        """Create conversion classifier"""
        return ConversionClassifier(model_path)
    
    @staticmethod
    def create_custom_classifier(model_type: ModelType, target_column: str, 
                                hyperparameters: Dict[str, Any] = None,
                                model_path: str = "ai/saved_models/") -> LeadClassifier:
        """Create custom classifier"""
        config = ModelConfig(
            model_type=model_type,
            target_column=target_column,
            hyperparameters=hyperparameters or {}
        )
        return LeadClassifier(config, model_path)


# ============================
# ENSEMBLE CLASSIFIER
# ============================

class EnsembleLeadClassifier:
    """Ensemble of multiple classifiers for robust predictions"""
    
    def __init__(self, model_path: str = "ai/saved_models/"):
        self.model_path = model_path
        self.classifiers = {
            'priority': PriorityClassifier(model_path),
            'quality': QualityClassifier(model_path),
            'conversion': ConversionClassifier(model_path)
        }
        self.weights = {
            'priority': 0.4,
            'quality': 0.3,
            'conversion': 0.3
        }
    
    def train_all_models(self, training_data: List[Dict[str, Any]]) -> Dict[str, TrainingResult]:
        """Train all ensemble models"""
        results = {}
        
        for name, classifier in self.classifiers.items():
            logger.info(f"Training {name} classifier...")
            try:
                result = classifier.train_model(training_data)
                results[name] = result
            except Exception as e:
                logger.error(f"Failed to train {name} classifier: {e}")
                results[name] = None
        
        return results
    
    def predict_ensemble(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make ensemble prediction combining all classifiers"""
        predictions = {}
        total_score = 0.0
        
        for name, classifier in self.classifiers.items():
            try:
                if classifier.model is not None:
                    result = classifier.predict(lead_data)
                    predictions[name] = result
                    
                    # Convert prediction to score
                    score_mapping = {'low': 0.2, 'medium': 0.6, 'high': 1.0}
                    score = score_mapping.get(result.predicted_class, 0.5)
                    total_score += score * self.weights[name]
                else:
                    logger.warning(f"{name} classifier not trained")
            except Exception as e:
                logger.error(f"Prediction failed for {name} classifier: {e}")
        
        # Determine ensemble prediction
        if total_score >= 0.8:
            ensemble_prediction = 'high'
        elif total_score >= 0.5:
            ensemble_prediction = 'medium'
        else:
            ensemble_prediction = 'low'
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'ensemble_score': total_score,
            'individual_predictions': predictions,
            'confidence': min([pred.confidence for pred in predictions.values()]) if predictions else 0.0
        }
    
    def get_ensemble_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all classifiers"""
        performance = {}
        
        for name, classifier in self.classifiers.items():
            performance[name] = classifier.get_model_performance()
        
        return performance


# ============================
# USAGE EXAMPLES AND TESTING
# ============================

if __name__ == "__main__":
    # Example usage and testing
    print("🧪 Testing Lead Classifier")
    print("=" * 50)
    
    # Sample training data
    sample_leads = [
        {
            'id': 1,
            'first_name': 'John',
            'last_name': 'Smith',
            'email': 'john.smith@techcorp.com',
            'phone': '+1-555-123-4567',
            'title': 'VP of Engineering',
            'company': {
                'name': 'TechCorp',
                'industry': 'Technology',
                'size': '201-500',
                'location': 'San Francisco, CA'
            },
            'source': 'linkedin',
            'created_at': '2024-01-15T10:30:00Z',
            'priority': 'high'  # Label for training
        },
        {
            'id': 2,
            'first_name': 'Jane',
            'last_name': 'Doe',
            'email': 'jane.doe@startup.ai',
            'title': 'Software Engineer',
            'company': {
                'name': 'StartupAI',
                'industry': 'Artificial Intelligence',
                'size': '11-50',
                'location': 'Austin, TX'
            },
            'source': 'website',
            'created_at': '2024-01-20T14:15:00Z',
            'priority': 'medium'
        },
        {
            'id': 3,
            'first_name': 'Bob',
            'last_name': 'Wilson',
            'email': 'bob.wilson@gmail.com',
            'title': 'Intern',
            'company': {
                'name': 'Small Company',
                'industry': 'Retail',
                'size': '1-10',
                'location': 'Unknown'
            },
            'source': 'purchased',
            'created_at': '2023-12-01T09:00:00Z',
            'priority': 'low'
        }
    ]
    
    # Test feature engineering
    print("\n🔧 Testing Feature Engineering:")
    print("-" * 30)
    
    feature_engineer = LeadFeatureEngineer()
    features = feature_engineer.extract_features(sample_leads[0])
    
    print("Sample features extracted:")
    for key, value in list(features.items())[:10]:  # Show first 10 features
        print(f"  {key}: {value}")
    
    # Test individual classifier
    print("\n🤖 Testing Priority Classifier:")
    print("-" * 30)
    
    if SKLEARN_AVAILABLE:
        classifier = PriorityClassifier()
        
        # Generate more training data for better model
        training_data = sample_leads * 10  # Duplicate for demo
        
        try:
            # Train model
            training_result = classifier.train_model(training_data)
            print(f"✅ Model trained - Accuracy: {training_result.accuracy:.3f}")
            
            # Test prediction
            test_lead = sample_leads[0].copy()
            del test_lead['priority']  # Remove label
            
            prediction = classifier.predict(test_lead)
            print(f"🎯 Prediction: {prediction.predicted_class}")
            print(f"📊 Confidence: {prediction.confidence:.3f}")
            print(f"⏱️ Prediction time: {prediction.prediction_time:.4f}s")
            
            # Test explanation
            explanation = classifier.explain_prediction(test_lead)
            print(f"\n🔍 Explanation:")
            print(f"  Predicted: {explanation['prediction']}")
            print(f"  Top features:")
            for feature in explanation['top_features'][:3]:
                print(f"    {feature['feature']}: {feature['value']:.3f} (importance: {feature['importance']:.3f})")
            
            # Test batch prediction
            batch_results = classifier.batch_predict([test_lead] * 3)
            print(f"\n📊 Batch prediction: {len(batch_results)} predictions completed")
            
        except Exception as e:
            print(f"❌ Training/prediction failed: {e}")
    else:
        print("⚠️ Scikit-learn not available - skipping ML tests")
    
    # Test ensemble classifier
    print("\n🎭 Testing Ensemble Classifier:")
    print("-" * 30)
    
    if SKLEARN_AVAILABLE:
        ensemble = EnsembleLeadClassifier()
        
        try:
            # Train ensemble (this would take longer in practice)
            print("Training ensemble models...")
            ensemble_results = ensemble.train_all_models(sample_leads * 5)
            
            successful_models = sum(1 for result in ensemble_results.values() if result is not None)
            print(f"✅ Trained {successful_models}/{len(ensemble_results)} models successfully")
            
            # Test ensemble prediction
            test_lead = sample_leads[0].copy()
            if 'priority' in test_lead:
                del test_lead['priority']
            
            ensemble_pred = ensemble.predict_ensemble(test_lead)
            print(f"🎯 Ensemble prediction: {ensemble_pred['ensemble_prediction']}")
            print(f"📊 Ensemble score: {ensemble_pred['ensemble_score']:.3f}")
            
        except Exception as e:
            print(f"❌ Ensemble test failed: {e}")
    
    # Test model factory
    print("\n🏭 Testing Model Factory:")
    print("-" * 30)
    
    factory = ModelFactory()
    
    # Create different classifiers
    priority_clf = factory.create_priority_classifier()
    quality_clf = factory.create_quality_classifier()
    conversion_clf = factory.create_conversion_classifier()
    
    print(f"✅ Priority classifier: {type(priority_clf).__name__}")
    print(f"✅ Quality classifier: {type(quality_clf).__name__}")
    print(f"✅ Conversion classifier: {type(conversion_clf).__name__}")
    
    # Create custom classifier
    custom_clf = factory.create_custom_classifier(
        ModelType.RANDOM_FOREST,
        "custom_target",
        {'n_estimators': 50}
    )
    print(f"✅ Custom classifier: {custom_clf.model_config.model_type.value}")
    
    print(f"\n✅ Lead Classifier testing completed!")
    print(f"🎯 Ready for production use with multiple model types")
    print(f"🚀 Features: Priority/Quality/Conversion prediction, Ensemble models, Feature importance")