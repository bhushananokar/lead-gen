# ai/models/quality_scorer.py - Data Quality Assessment ML Models

# ============================
# IMPORTS
# ============================
import os
import pickle
import json
import logging
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import time

# Email validation
try:
    import email_validator
    EMAIL_VALIDATOR_AVAILABLE = True
except ImportError:
    EMAIL_VALIDATOR_AVAILABLE = False
    logging.warning("email-validator not available. Install with: pip install email-validator")

# Phone validation
try:
    import phonenumbers
    PHONENUMBERS_AVAILABLE = True
except ImportError:
    PHONENUMBERS_AVAILABLE = False
    logging.warning("phonenumbers not available. Install with: pip install phonenumbers")

# Machine Learning
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Install with: pip install scikit-learn")

# Fuzzy matching
try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    logging.warning("fuzzywuzzy not available. Install with: pip install fuzzywuzzy")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# ENUMS AND DATA CLASSES
# ============================

class QualityScore(Enum):
    """Quality score classifications"""
    EXCELLENT = "excellent"  # 85-100
    GOOD = "good"           # 70-84
    FAIR = "fair"           # 50-69
    POOR = "poor"           # 0-49

class ValidationStatus(Enum):
    """Validation status types"""
    VALID = "valid"
    INVALID = "invalid"
    UNKNOWN = "unknown"
    SUSPICIOUS = "suspicious"

class QualityDimension(Enum):
    """Quality assessment dimensions"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    FRESHNESS = "freshness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"

@dataclass
class QualityMetrics:
    """Quality assessment metrics"""
    overall_score: float = 0.0
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    consistency_score: float = 0.0
    freshness_score: float = 0.0
    validity_score: float = 0.0
    uniqueness_score: float = 0.0
    confidence: float = 0.0
    
@dataclass
class QualityIssue:
    """Represents a data quality issue"""
    dimension: QualityDimension
    severity: str  # 'critical', 'high', 'medium', 'low'
    field_name: str
    description: str
    suggestion: str
    confidence: float = 1.0
    
@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    lead_id: Optional[int] = None
    overall_score: float = 0.0
    metrics: QualityMetrics = field(default_factory=QualityMetrics)
    issues: List[QualityIssue] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    validation_results: Dict[str, ValidationStatus] = field(default_factory=dict)
    improvement_potential: float = 0.0
    assessment_timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time_ms: float = 0.0

# ============================
# ML-BASED QUALITY SCORER
# ============================

class QualityScorer:
    """
    Machine Learning-based data quality assessment engine
    """
    
    def __init__(self, model_path: str = "ai/saved_models"):
        """
        Initialize the Quality Scorer
        
        Args:
            model_path: Path to save/load ML models
        """
        self.model_path = model_path
        self.quality_models = {}
        self.feature_extractors = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Quality weights for different dimensions
        self.dimension_weights = {
            QualityDimension.COMPLETENESS: 0.25,
            QualityDimension.ACCURACY: 0.25,
            QualityDimension.CONSISTENCY: 0.20,
            QualityDimension.VALIDITY: 0.15,
            QualityDimension.FRESHNESS: 0.10,
            QualityDimension.UNIQUENESS: 0.05
        }
        
        # Common email domains for validation
        self.common_domains = {
            'free': ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com'],
            'business': ['.com', '.org', '.net', '.edu', '.gov']
        }
        
        # Initialize models
        self._initialize_models()
        
        logger.info("🎯 Quality Scorer initialized")
    
    def _initialize_models(self):
        """Initialize or load quality assessment models"""
        try:
            # Try to load existing models
            model_file = os.path.join(self.model_path, "quality_scorer_models.pkl")
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.quality_models = saved_data.get('models', {})
                    self.scalers = saved_data.get('scalers', {})
                    self.label_encoders = saved_data.get('encoders', {})
                logger.info("✅ Loaded existing quality models")
            else:
                # Create default models
                self._create_default_models()
                logger.info("🔧 Created default quality models")
                
        except Exception as e:
            logger.warning(f"⚠️ Model initialization warning: {e}")
            self._create_default_models()
    
    def _create_default_models(self):
        """Create default quality assessment models"""
        if SKLEARN_AVAILABLE:
            # Overall quality regressor
            self.quality_models['overall'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # Dimension-specific models
            for dimension in QualityDimension:
                self.quality_models[dimension.value] = RandomForestRegressor(
                    n_estimators=50,
                    max_depth=8,
                    random_state=42
                )
                
            # Scalers for feature normalization
            for model_name in self.quality_models.keys():
                self.scalers[model_name] = StandardScaler()
        
        logger.info("🤖 Default ML models created")
    
    # ============================
    # MAIN QUALITY ASSESSMENT
    # ============================
    
    def assess_quality(self, lead_data: Dict[str, Any]) -> QualityReport:
        """
        Comprehensive quality assessment of lead data
        
        Args:
            lead_data: Lead information dictionary
            
        Returns:
            QualityReport with detailed assessment
        """
        start_time = time.time()
        
        try:
            # Extract features for ML models
            features = self._extract_features(lead_data)
            
            # Initialize metrics
            metrics = QualityMetrics()
            issues = []
            suggestions = []
            validation_results = {}
            
            # Assess each quality dimension
            metrics.completeness_score = self._assess_completeness(lead_data, issues, suggestions)
            metrics.accuracy_score = self._assess_accuracy(lead_data, issues, suggestions, validation_results)
            metrics.consistency_score = self._assess_consistency(lead_data, issues, suggestions)
            metrics.validity_score = self._assess_validity(lead_data, issues, suggestions, validation_results)
            metrics.freshness_score = self._assess_freshness(lead_data, issues, suggestions)
            metrics.uniqueness_score = self._assess_uniqueness(lead_data, issues, suggestions)
            
            # Calculate overall score
            if SKLEARN_AVAILABLE and 'overall' in self.quality_models:
                # Use ML model if available
                try:
                    feature_vector = np.array(list(features.values())).reshape(1, -1)
                    if 'overall' in self.scalers and hasattr(self.scalers['overall'], 'transform'):
                        feature_vector = self.scalers['overall'].transform(feature_vector)
                    
                    predicted_score = self.quality_models['overall'].predict(feature_vector)[0]
                    metrics.overall_score = max(0, min(100, predicted_score))
                    metrics.confidence = 0.8
                except:
                    # Fallback to weighted average
                    metrics.overall_score = self._calculate_weighted_score(metrics)
                    metrics.confidence = 0.6
            else:
                # Weighted average fallback
                metrics.overall_score = self._calculate_weighted_score(metrics)
                metrics.confidence = 0.5
            
            # Calculate improvement potential
            improvement_potential = max(0, 100 - metrics.overall_score)
            
            # Processing time
            processing_time = (time.time() - start_time) * 1000
            
            report = QualityReport(
                lead_id=lead_data.get('id'),
                overall_score=round(metrics.overall_score, 2),
                metrics=metrics,
                issues=issues,
                suggestions=list(set(suggestions)),  # Remove duplicates
                validation_results=validation_results,
                improvement_potential=round(improvement_potential, 2),
                processing_time_ms=round(processing_time, 2)
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return QualityReport(
                lead_id=lead_data.get('id'),
                overall_score=0.0,
                issues=[QualityIssue(
                    dimension=QualityDimension.ACCURACY,
                    severity='critical',
                    field_name='system',
                    description=f'Assessment failed: {str(e)}',
                    suggestion='Contact system administrator'
                )]
            )
    
    def _calculate_weighted_score(self, metrics: QualityMetrics) -> float:
        """Calculate weighted overall score from dimension scores"""
        return (
            metrics.completeness_score * self.dimension_weights[QualityDimension.COMPLETENESS] +
            metrics.accuracy_score * self.dimension_weights[QualityDimension.ACCURACY] +
            metrics.consistency_score * self.dimension_weights[QualityDimension.CONSISTENCY] +
            metrics.validity_score * self.dimension_weights[QualityDimension.VALIDITY] +
            metrics.freshness_score * self.dimension_weights[QualityDimension.FRESHNESS] +
            metrics.uniqueness_score * self.dimension_weights[QualityDimension.UNIQUENESS]
        )
    
    # ============================
    # QUALITY DIMENSION ASSESSMENTS
    # ============================
    
    def _assess_completeness(self, lead_data: Dict[str, Any], issues: List[QualityIssue], suggestions: List[str]) -> float:
        """Assess data completeness"""
        total_fields = 0
        completed_fields = 0
        
        # Critical fields (high weight)
        critical_fields = ['first_name', 'last_name', 'email']
        for field in critical_fields:
            total_fields += 3  # Weight = 3
            if lead_data.get(field) and str(lead_data[field]).strip():
                completed_fields += 3
            else:
                issues.append(QualityIssue(
                    dimension=QualityDimension.COMPLETENESS,
                    severity='critical',
                    field_name=field,
                    description=f'Critical field {field} is missing',
                    suggestion=f'Add {field} information to improve lead quality'
                ))
        
        # Important fields (medium weight)
        important_fields = ['phone', 'title', 'company']
        for field in important_fields:
            total_fields += 2  # Weight = 2
            if lead_data.get(field) and str(lead_data[field]).strip():
                completed_fields += 2
            else:
                issues.append(QualityIssue(
                    dimension=QualityDimension.COMPLETENESS,
                    severity='medium',
                    field_name=field,
                    description=f'Important field {field} is missing',
                    suggestion=f'Consider adding {field} for better lead qualification'
                ))
        
        # Optional fields (low weight)
        optional_fields = ['linkedin_url', 'source', 'notes']
        for field in optional_fields:
            total_fields += 1  # Weight = 1
            if lead_data.get(field) and str(lead_data[field]).strip():
                completed_fields += 1
        
        # Company information
        company = lead_data.get('company', {})
        if isinstance(company, dict):
            company_fields = ['name', 'industry', 'size', 'location']
            for field in company_fields:
                total_fields += 2
                if company.get(field) and str(company[field]).strip():
                    completed_fields += 2
        
        completeness_score = (completed_fields / total_fields * 100) if total_fields > 0 else 0
        
        if completeness_score < 60:
            suggestions.append('Use data enrichment tools to complete missing fields')
        
        return completeness_score
    
    def _assess_accuracy(self, lead_data: Dict[str, Any], issues: List[QualityIssue], 
                        suggestions: List[str], validation_results: Dict[str, ValidationStatus]) -> float:
        """Assess data accuracy"""
        accuracy_score = 100.0
        
        # Email validation
        email = lead_data.get('email', '')
        email_status = self._validate_email(email)
        validation_results['email'] = email_status
        
        if email_status == ValidationStatus.INVALID:
            accuracy_score -= 25
            issues.append(QualityIssue(
                dimension=QualityDimension.ACCURACY,
                severity='high',
                field_name='email',
                description='Email format is invalid',
                suggestion='Verify and correct email address format'
            ))
        elif email_status == ValidationStatus.SUSPICIOUS:
            accuracy_score -= 10
            issues.append(QualityIssue(
                dimension=QualityDimension.ACCURACY,
                severity='medium',
                field_name='email',
                description='Email format appears suspicious',
                suggestion='Double-check email address accuracy'
            ))
        
        # Phone validation
        phone = lead_data.get('phone', '')
        phone_status = self._validate_phone(phone)
        validation_results['phone'] = phone_status
        
        if phone and phone_status == ValidationStatus.INVALID:
            accuracy_score -= 15
            issues.append(QualityIssue(
                dimension=QualityDimension.ACCURACY,
                severity='medium',
                field_name='phone',
                description='Phone number format is invalid',
                suggestion='Verify phone number format and country code'
            ))
        
        # Name validation
        first_name = lead_data.get('first_name', '')
        last_name = lead_data.get('last_name', '')
        name_status = self._validate_names(first_name, last_name)
        validation_results['name'] = name_status
        
        if name_status == ValidationStatus.INVALID:
            accuracy_score -= 15
            issues.append(QualityIssue(
                dimension=QualityDimension.ACCURACY,
                severity='medium',
                field_name='name',
                description='Name format appears invalid (contains numbers or special characters)',
                suggestion='Review and correct name formatting'
            ))
        
        # LinkedIn URL validation
        linkedin_url = lead_data.get('linkedin_url', '')
        if linkedin_url:
            linkedin_status = self._validate_linkedin_url(linkedin_url)
            validation_results['linkedin'] = linkedin_status
            
            if linkedin_status == ValidationStatus.INVALID:
                accuracy_score -= 10
                issues.append(QualityIssue(
                    dimension=QualityDimension.ACCURACY,
                    severity='low',
                    field_name='linkedin_url',
                    description='LinkedIn URL format is invalid',
                    suggestion='Verify LinkedIn profile URL format'
                ))
        
        return max(0, accuracy_score)
    
    def _assess_consistency(self, lead_data: Dict[str, Any], issues: List[QualityIssue], suggestions: List[str]) -> float:
        """Assess internal data consistency"""
        consistency_score = 100.0
        
        # Email-Company domain consistency
        email = lead_data.get('email', '')
        company = lead_data.get('company', {})
        
        if email and '@' in email and isinstance(company, dict):
            email_domain = email.split('@')[1].lower()
            company_domain = company.get('domain', '').lower()
            company_website = company.get('website', '').lower()
            
            # Check domain consistency
            if company_domain and email_domain != company_domain:
                consistency_score -= 15
                issues.append(QualityIssue(
                    dimension=QualityDimension.CONSISTENCY,
                    severity='medium',
                    field_name='email_domain',
                    description='Email domain does not match company domain',
                    suggestion='Verify email belongs to the correct company'
                ))
            
            # Check against common free email providers
            if email_domain in self.common_domains['free'] and company.get('name'):
                consistency_score -= 10
                suggestions.append('Consider finding corporate email address for better lead quality')
        
        # Title-Company size consistency
        title = lead_data.get('title', '').lower()
        company_size = company.get('size', '') if isinstance(company, dict) else ''
        
        executive_titles = ['ceo', 'cto', 'cfo', 'president', 'founder', 'vice president']
        is_executive = any(exec_title in title for exec_title in executive_titles)
        
        if is_executive and company_size in ['1-10', '11-50']:
            # Executives in small companies are normal
            pass
        elif is_executive and not company_size:
            suggestions.append('Verify company size for executive-level contacts')
        
        # Name consistency (capitalization)
        first_name = lead_data.get('first_name', '')
        last_name = lead_data.get('last_name', '')
        
        if first_name and not first_name[0].isupper():
            consistency_score -= 5
            suggestions.append('Standardize name capitalization')
        
        if last_name and not last_name[0].isupper():
            consistency_score -= 5
        
        return max(0, consistency_score)
    
    def _assess_validity(self, lead_data: Dict[str, Any], issues: List[QualityIssue], 
                        suggestions: List[str], validation_results: Dict[str, ValidationStatus]) -> float:
        """Assess data validity and format compliance"""
        validity_score = 100.0
        
        # Check for obviously fake or test data
        suspicious_patterns = [
            r'test@test\.',
            r'example@example\.',
            r'noreply@',
            r'donotreply@',
            r'admin@',
            r'info@.*\.test'
        ]
        
        email = lead_data.get('email', '')
        for pattern in suspicious_patterns:
            if re.search(pattern, email, re.IGNORECASE):
                validity_score -= 30
                issues.append(QualityIssue(
                    dimension=QualityDimension.VALIDITY,
                    severity='high',
                    field_name='email',
                    description='Email appears to be test or placeholder data',
                    suggestion='Replace with valid contact email'
                ))
                break
        
        # Check for suspicious names
        first_name = lead_data.get('first_name', '').lower()
        last_name = lead_data.get('last_name', '').lower()
        
        suspicious_names = ['test', 'example', 'sample', 'demo', 'placeholder']
        if first_name in suspicious_names or last_name in suspicious_names:
            validity_score -= 20
            issues.append(QualityIssue(
                dimension=QualityDimension.VALIDITY,
                severity='high',
                field_name='name',
                description='Name appears to be placeholder or test data',
                suggestion='Replace with actual contact name'
            ))
        
        # Check phone number format
        phone = lead_data.get('phone', '')
        if phone:
            # Remove formatting
            clean_phone = re.sub(r'[^\d+]', '', phone)
            if len(clean_phone) < 10:
                validity_score -= 15
                issues.append(QualityIssue(
                    dimension=QualityDimension.VALIDITY,
                    severity='medium',
                    field_name='phone',
                    description='Phone number appears too short',
                    suggestion='Verify complete phone number with area code'
                ))
        
        return max(0, validity_score)
    
    def _assess_freshness(self, lead_data: Dict[str, Any], issues: List[QualityIssue], suggestions: List[str]) -> float:
        """Assess data freshness and recency"""
        freshness_score = 100.0
        
        # Check creation date
        created_at = lead_data.get('created_at')
        if created_at:
            try:
                if isinstance(created_at, str):
                    created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    created_date = created_at
                
                days_old = (datetime.utcnow() - created_date.replace(tzinfo=None)).days
                
                if days_old > 365:  # More than a year old
                    freshness_score -= 30
                    suggestions.append('Consider updating lead information - data is over 1 year old')
                elif days_old > 180:  # More than 6 months old
                    freshness_score -= 15
                    suggestions.append('Lead data is aging - consider refreshing contact information')
                elif days_old > 90:  # More than 3 months old
                    freshness_score -= 5
                
            except (ValueError, TypeError):
                freshness_score -= 10
                issues.append(QualityIssue(
                    dimension=QualityDimension.FRESHNESS,
                    severity='low',
                    field_name='created_at',
                    description='Invalid or missing creation date',
                    suggestion='Ensure proper date tracking for lead management'
                ))
        else:
            freshness_score -= 20
            suggestions.append('Add timestamp tracking for better lead management')
        
        return max(0, freshness_score)
    
    def _assess_uniqueness(self, lead_data: Dict[str, Any], issues: List[QualityIssue], suggestions: List[str]) -> float:
        """Assess data uniqueness (placeholder for duplicate detection)"""
        # This would typically check against a database of existing leads
        # For now, return a high score as we assume the lead is unique
        uniqueness_score = 95.0
        
        # Basic duplicate indicators
        email = lead_data.get('email', '')
        phone = lead_data.get('phone', '')
        
        # Check for obvious duplicates (this would be enhanced with database queries)
        if not email and not phone:
            uniqueness_score -= 15
            suggestions.append('Add email or phone to enable duplicate detection')
        
        return uniqueness_score
    
    # ============================
    # VALIDATION METHODS
    # ============================
    
    def _validate_email(self, email: str) -> ValidationStatus:
        """Validate email format and deliverability"""
        if not email:
            return ValidationStatus.UNKNOWN
        
        # Basic format check
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            return ValidationStatus.INVALID
        
        # Use email-validator if available
        if EMAIL_VALIDATOR_AVAILABLE:
            try:
                import email_validator
                email_validator.validate_email(email)
                return ValidationStatus.VALID
            except email_validator.EmailNotValidError:
                return ValidationStatus.INVALID
            except:
                pass
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'\.{2,}',  # Multiple consecutive dots
            r'^\.|\.$',  # Starts or ends with dot
            r'@.*@',  # Multiple @ symbols
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, email):
                return ValidationStatus.SUSPICIOUS
        
        return ValidationStatus.VALID
    
    def _validate_phone(self, phone: str) -> ValidationStatus:
        """Validate phone number format"""
        if not phone:
            return ValidationStatus.UNKNOWN
        
        # Use phonenumbers library if available
        if PHONENUMBERS_AVAILABLE:
            try:
                import phonenumbers
                parsed = phonenumbers.parse(phone, None)
                if phonenumbers.is_valid_number(parsed):
                    return ValidationStatus.VALID
                else:
                    return ValidationStatus.INVALID
            except:
                pass
        
        # Basic validation
        clean_phone = re.sub(r'[^\d+]', '', phone)
        
        # Check length
        if len(clean_phone) < 10 or len(clean_phone) > 15:
            return ValidationStatus.INVALID
        
        # Check for obvious patterns
        if re.match(r'^(\d)\1+$', clean_phone):  # All same digits
            return ValidationStatus.SUSPICIOUS
        
        return ValidationStatus.VALID
    
    def _validate_names(self, first_name: str, last_name: str) -> ValidationStatus:
        """Validate name format"""
        if not first_name or not last_name:
            return ValidationStatus.UNKNOWN
        
        # Check for numbers or special characters
        name_pattern = r'^[a-zA-Z\s\-\'\.]+$'
        
        if not re.match(name_pattern, first_name) or not re.match(name_pattern, last_name):
            return ValidationStatus.INVALID
        
        # Check for suspicious patterns
        if len(first_name) < 2 or len(last_name) < 2:
            return ValidationStatus.SUSPICIOUS
        
        return ValidationStatus.VALID
    
    def _validate_linkedin_url(self, url: str) -> ValidationStatus:
        """Validate LinkedIn URL format"""
        if not url:
            return ValidationStatus.UNKNOWN
        
        linkedin_pattern = r'^https?://(www\.)?linkedin\.com/in/[a-zA-Z0-9\-]+/?$'
        
        if re.match(linkedin_pattern, url):
            return ValidationStatus.VALID
        else:
            return ValidationStatus.INVALID
    
    # ============================
    # FEATURE EXTRACTION FOR ML
    # ============================
    
    def _extract_features(self, lead_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features for ML models"""
        features = {}
        
        # Completeness features
        features['has_email'] = float(bool(lead_data.get('email')))
        features['has_phone'] = float(bool(lead_data.get('phone')))
        features['has_linkedin'] = float(bool(lead_data.get('linkedin_url')))
        features['has_title'] = float(bool(lead_data.get('title')))
        features['has_company'] = float(bool(lead_data.get('company', {}).get('name')))
        
        # Email features
        email = lead_data.get('email', '')
        features['email_length'] = len(email)
        features['email_has_dot'] = float('.' in email)
        features['email_domain_length'] = len(email.split('@')[1]) if '@' in email else 0
        features['is_free_email'] = float(any(domain in email for domain in self.common_domains['free']))
        
        # Phone features
        phone = lead_data.get('phone', '')
        clean_phone = re.sub(r'[^\d+]', '', phone)
        features['phone_length'] = len(clean_phone)
        features['phone_has_country_code'] = float(phone.startswith('+') or len(clean_phone) > 10)
        
        # Name features
        first_name = lead_data.get('first_name', '')
        last_name = lead_data.get('last_name', '')
        features['first_name_length'] = len(first_name)
        features['last_name_length'] = len(last_name)
        features['name_has_special_chars'] = float(bool(re.search(r'[^a-zA-Z\s]', first_name + last_name)))
        
        # Company features
        company = lead_data.get('company', {})
        features['company_has_domain'] = float(bool(company.get('domain')))
        features['company_has_industry'] = float(bool(company.get('industry')))
        features['company_has_size'] = float(bool(company.get('size')))
        
        # Title features
        title = lead_data.get('title', '').lower()
        features['title_length'] = len(title)
        features['is_executive'] = float(any(word in title for word in ['ceo', 'cto', 'cfo', 'president', 'founder']))
        features['is_manager'] = float(any(word in title for word in ['manager', 'director', 'head', 'lead']))
        features['is_senior'] = float('senior' in title or 'sr.' in title)
        
        # Data age features
        created_at = lead_data.get('created_at')
        if created_at:
            try:
                if isinstance(created_at, str):
                    created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    created_date = created_at
                
                days_old = (datetime.utcnow() - created_date.replace(tzinfo=None)).days
                features['days_old'] = min(days_old, 1000)  # Cap at 1000 days
                features['is_recent'] = float(days_old <= 30)
            except:
                features['days_old'] = 365  # Default to 1 year
                features['is_recent'] = 0.0
        else:
            features['days_old'] = 365
            features['is_recent'] = 0.0
        
        # Source features
        source = lead_data.get('source', '').lower()
        features['source_is_referral'] = float('referral' in source)
        features['source_is_linkedin'] = float('linkedin' in source)
        features['source_is_organic'] = float('organic' in source or 'website' in source)
        
        return features
    
    # ============================
    # MODEL TRAINING
    # ============================
    
    def train_quality_models(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train quality assessment models
        
        Args:
            training_data: List of lead data with quality labels
            
        Returns:
            Training results and performance metrics
        """
        if not SKLEARN_AVAILABLE:
            logger.error("Scikit-learn not available for training")
            return {"error": "ML libraries not available"}
        
        start_time = time.time()
        logger.info(f"🎯 Training quality models on {len(training_data)} samples...")
        
        try:
            # Prepare training data
            X_features = []
            y_overall = []
            y_dimensions = {dim.value: [] for dim in QualityDimension}
            
            for lead_data in training_data:
                # Extract features
                features = self._extract_features(lead_data)
                X_features.append(list(features.values()))
                
                # Extract labels (assuming they exist in training data)
                if 'quality_score' in lead_data:
                    y_overall.append(lead_data['quality_score'])
                else:
                    # Generate synthetic quality score for training
                    synthetic_report = self.assess_quality(lead_data)
                    y_overall.append(synthetic_report.overall_score)
                
                # Extract dimension scores
                for dim in QualityDimension:
                    if f'{dim.value}_score' in lead_data:
                        y_dimensions[dim.value].append(lead_data[f'{dim.value}_score'])
                    else:
                        # Use synthetic scores
                        if dim == QualityDimension.COMPLETENESS:
                            y_dimensions[dim.value].append(synthetic_report.metrics.completeness_score)
                        elif dim == QualityDimension.ACCURACY:
                            y_dimensions[dim.value].append(synthetic_report.metrics.accuracy_score)
                        # Add other dimensions as needed
                        else:
                            y_dimensions[dim.value].append(75.0)  # Default score
            
            X = np.array(X_features)
            feature_names = list(self._extract_features(training_data[0]).keys())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_overall, test_size=0.2, random_state=42
            )
            
            training_results = {}
            
            # Train overall quality model
            logger.info("Training overall quality model...")
            
            # Scale features
            self.scalers['overall'] = StandardScaler()
            X_train_scaled = self.scalers['overall'].fit_transform(X_train)
            X_test_scaled = self.scalers['overall'].transform(X_test)
            
            # Train model
            self.quality_models['overall'].fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.quality_models['overall'].predict(X_test_scaled)
            
            overall_results = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2_score': r2_score(y_test, y_pred),
                'feature_importance': dict(zip(feature_names, self.quality_models['overall'].feature_importances_))
            }
            
            training_results['overall'] = overall_results
            logger.info(f"Overall model - MAE: {overall_results['mae']:.2f}, R²: {overall_results['r2_score']:.3f}")
            
            # Train dimension-specific models
            for dimension in QualityDimension:
                if len(y_dimensions[dimension.value]) == len(X):
                    logger.info(f"Training {dimension.value} model...")
                    
                    y_dim = np.array(y_dimensions[dimension.value])
                    _, _, y_dim_train, y_dim_test = train_test_split(
                        X, y_dim, test_size=0.2, random_state=42
                    )
                    
                    # Scale features for this dimension
                    self.scalers[dimension.value] = StandardScaler()
                    X_train_dim_scaled = self.scalers[dimension.value].fit_transform(X_train)
                    X_test_dim_scaled = self.scalers[dimension.value].transform(X_test)
                    
                    # Train model
                    self.quality_models[dimension.value].fit(X_train_dim_scaled, y_dim_train)
                    
                    # Evaluate
                    y_dim_pred = self.quality_models[dimension.value].predict(X_test_dim_scaled)
                    
                    dim_results = {
                        'mae': mean_absolute_error(y_dim_test, y_dim_pred),
                        'rmse': np.sqrt(mean_squared_error(y_dim_test, y_dim_pred)),
                        'r2_score': r2_score(y_dim_test, y_dim_pred)
                    }
                    
                    training_results[dimension.value] = dim_results
                    logger.info(f"{dimension.value} model - MAE: {dim_results['mae']:.2f}, R²: {dim_results['r2_score']:.3f}")
            
            # Save models
            self.save_models()
            
            training_time = time.time() - start_time
            training_results['training_time_minutes'] = training_time / 60
            training_results['training_samples'] = len(training_data)
            training_results['feature_count'] = len(feature_names)
            
            logger.info(f"✅ Model training completed in {training_time:.1f} seconds")
            
            return training_results
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return {"error": str(e)}
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            model_data = {
                'models': self.quality_models,
                'scalers': self.scalers,
                'encoders': self.label_encoders,
                'dimension_weights': self.dimension_weights,
                'saved_at': datetime.utcnow().isoformat()
            }
            
            model_file = os.path.join(self.model_path, "quality_scorer_models.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"💾 Models saved to {model_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save models: {e}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            model_file = os.path.join(self.model_path, "quality_scorer_models.pkl")
            
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.quality_models = model_data.get('models', {})
                self.scalers = model_data.get('scalers', {})
                self.label_encoders = model_data.get('encoders', {})
                self.dimension_weights = model_data.get('dimension_weights', self.dimension_weights)
                
                logger.info(f"✅ Models loaded from {model_file}")
                return True
            else:
                logger.warning(f"⚠️ Model file not found: {model_file}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            return False
    
    # ============================
    # BATCH PROCESSING
    # ============================
    
    def assess_batch_quality(self, leads_data: List[Dict[str, Any]], 
                           parallel: bool = True) -> List[QualityReport]:
        """
        Assess quality for multiple leads
        
        Args:
            leads_data: List of lead data dictionaries
            parallel: Whether to use parallel processing (if available)
            
        Returns:
            List of QualityReport objects
        """
        logger.info(f"🔍 Assessing quality for {len(leads_data)} leads...")
        
        start_time = time.time()
        results = []
        
        try:
            if parallel and len(leads_data) > 10:
                # Try to use multiprocessing for large batches
                try:
                    from multiprocessing import Pool, cpu_count
                    
                    with Pool(processes=min(cpu_count(), 4)) as pool:
                        results = pool.map(self.assess_quality, leads_data)
                    
                    logger.info(f"✅ Parallel processing completed")
                    
                except ImportError:
                    # Fallback to sequential processing
                    results = [self.assess_quality(lead_data) for lead_data in leads_data]
                    logger.info(f"✅ Sequential processing completed")
            else:
                # Sequential processing
                for i, lead_data in enumerate(leads_data):
                    if i % 100 == 0 and i > 0:
                        logger.info(f"Processed {i}/{len(leads_data)} leads...")
                    
                    result = self.assess_quality(lead_data)
                    results.append(result)
            
            processing_time = time.time() - start_time
            avg_time_per_lead = processing_time / len(leads_data) * 1000  # ms
            
            logger.info(f"✅ Batch assessment completed in {processing_time:.2f}s")
            logger.info(f"📊 Average time per lead: {avg_time_per_lead:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch assessment failed: {e}")
            return [QualityReport(lead_id=lead.get('id'), overall_score=0.0) for lead in leads_data]
    
    def get_quality_statistics(self, quality_reports: List[QualityReport]) -> Dict[str, Any]:
        """
        Calculate aggregate quality statistics
        
        Args:
            quality_reports: List of quality assessment reports
            
        Returns:
            Statistical summary
        """
        if not quality_reports:
            return {}
        
        scores = [report.overall_score for report in quality_reports]
        
        stats = {
            'total_leads': len(quality_reports),
            'average_score': np.mean(scores),
            'median_score': np.median(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'score_distribution': {
                'excellent': len([s for s in scores if s >= 85]),
                'good': len([s for s in scores if 70 <= s < 85]),
                'fair': len([s for s in scores if 50 <= s < 70]),
                'poor': len([s for s in scores if s < 50])
            }
        }
        
        # Common issues analysis
        all_issues = []
        for report in quality_reports:
            all_issues.extend([issue.description for issue in report.issues])
        
        from collections import Counter
        issue_counts = Counter(all_issues)
        stats['common_issues'] = dict(issue_counts.most_common(10))
        
        # Dimension averages
        dimension_scores = {dim.value: [] for dim in QualityDimension}
        for report in quality_reports:
            dimension_scores['completeness'].append(report.metrics.completeness_score)
            dimension_scores['accuracy'].append(report.metrics.accuracy_score)
            dimension_scores['consistency'].append(report.metrics.consistency_score)
            dimension_scores['validity'].append(report.metrics.validity_score)
            dimension_scores['freshness'].append(report.metrics.freshness_score)
            dimension_scores['uniqueness'].append(report.metrics.uniqueness_score)
        
        stats['dimension_averages'] = {
            dim: np.mean(scores) for dim, scores in dimension_scores.items() if scores
        }
        
        return stats


# ============================
# QUALITY SCORER ENSEMBLE
# ============================

class EnsembleQualityScorer:
    """
    Ensemble of multiple quality assessment models
    """
    
    def __init__(self, model_path: str = "ai/saved_models"):
        """Initialize ensemble quality scorer"""
        self.model_path = model_path
        self.scorers = {}
        self.weights = {}
        
        # Initialize base scorers
        self._initialize_scorers()
        
        logger.info("🎭 Ensemble Quality Scorer initialized")
    
    def _initialize_scorers(self):
        """Initialize multiple quality scorers"""
        try:
            # Rule-based scorer (always available)
            self.scorers['rule_based'] = QualityScorer(self.model_path)
            self.weights['rule_based'] = 0.4
            
            if SKLEARN_AVAILABLE:
                # ML-based scorers with different algorithms
                self.scorers['gradient_boosting'] = QualityScorer(self.model_path)
                self.scorers['gradient_boosting'].quality_models['overall'] = GradientBoostingRegressor(
                    n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
                )
                self.weights['gradient_boosting'] = 0.35
                
                self.scorers['random_forest'] = QualityScorer(self.model_path)
                self.scorers['random_forest'].quality_models['overall'] = RandomForestRegressor(
                    n_estimators=100, max_depth=8, random_state=42
                )
                self.weights['random_forest'] = 0.25
            
            # Normalize weights
            total_weight = sum(self.weights.values())
            self.weights = {k: v/total_weight for k, v in self.weights.items()}
            
        except Exception as e:
            logger.warning(f"⚠️ Ensemble initialization warning: {e}")
    
    def assess_quality_ensemble(self, lead_data: Dict[str, Any]) -> QualityReport:
        """
        Assess quality using ensemble of models
        
        Args:
            lead_data: Lead information dictionary
            
        Returns:
            Ensemble quality report
        """
        try:
            scorer_results = {}
            
            # Get predictions from all scorers
            for scorer_name, scorer in self.scorers.items():
                try:
                    result = scorer.assess_quality(lead_data)
                    scorer_results[scorer_name] = result
                except Exception as e:
                    logger.warning(f"⚠️ Scorer {scorer_name} failed: {e}")
            
            if not scorer_results:
                logger.error("❌ All scorers failed")
                return QualityReport(lead_id=lead_data.get('id'), overall_score=0.0)
            
            # Combine results
            weighted_score = 0.0
            total_weight = 0.0
            
            for scorer_name, result in scorer_results.items():
                weight = self.weights.get(scorer_name, 0.0)
                weighted_score += result.overall_score * weight
                total_weight += weight
            
            ensemble_score = weighted_score / total_weight if total_weight > 0 else 0.0
            
            # Combine other metrics (use best scorer's results as base)
            best_scorer = max(scorer_results.items(), key=lambda x: x[1].overall_score)
            base_report = best_scorer[1]
            
            # Override with ensemble score
            base_report.overall_score = round(ensemble_score, 2)
            base_report.metrics.confidence = min(0.9, base_report.metrics.confidence + 0.1)
            
            # Add ensemble information
            ensemble_info = {
                'ensemble_method': 'weighted_average',
                'scorers_used': list(scorer_results.keys()),
                'individual_scores': {name: result.overall_score for name, result in scorer_results.items()},
                'weights_applied': {name: self.weights.get(name, 0.0) for name in scorer_results.keys()}
            }
            
            if not hasattr(base_report, 'metadata'):
                base_report.metadata = {}
            base_report.metadata = ensemble_info
            
            return base_report
            
        except Exception as e:
            logger.error(f"❌ Ensemble assessment failed: {e}")
            return QualityReport(lead_id=lead_data.get('id'), overall_score=0.0)


# ============================
# TESTING AND VALIDATION
# ============================

def test_quality_scorer():
    """Test the quality scorer functionality"""
    print("🧪 Testing Quality Scorer")
    print("=" * 50)
    
    # Sample lead data for testing
    sample_leads = [
        {
            'id': 1,
            'first_name': 'John',
            'last_name': 'Smith',
            'email': 'john.smith@example.com',
            'phone': '+1-555-123-4567',
            'title': 'Senior Software Engineer',
            'linkedin_url': 'https://www.linkedin.com/in/johnsmith',
            'company': {
                'name': 'TechCorp Inc',
                'domain': 'example.com',
                'industry': 'Technology',
                'size': '201-500',
                'location': 'San Francisco, CA'
            },
            'source': 'linkedin',
            'created_at': '2024-01-15T10:30:00Z'
        },
        {
            'id': 2,
            'first_name': 'Jane',
            'last_name': '',  # Missing last name
            'email': 'invalid-email',  # Invalid email
            'phone': '123',  # Invalid phone
            'title': 'Manager',
            'company': {
                'name': 'Small Startup'
                # Missing other company info
            },
            'source': 'purchased',
            'created_at': '2022-06-01T08:15:00Z'  # Old data
        },
        {
            'id': 3,
            'first_name': 'Test',  # Suspicious name
            'last_name': 'User',
            'email': 'test@test.com',  # Test email
            'phone': '+1-999-999-9999',
            'title': 'CEO',
            'company': {
                'name': 'Example Corp',
                'domain': 'test.com',
                'industry': 'Consulting',
                'size': '1-10'
            },
            'source': 'demo'
        }
    ]
    
    # Test basic quality scorer
    print("\n🎯 Testing Basic Quality Scorer:")
    print("-" * 30)
    
    scorer = QualityScorer()
    
    for i, lead in enumerate(sample_leads):
        print(f"\n📊 Lead {i+1} Assessment:")
        report = scorer.assess_quality(lead)
        
        print(f"Overall Score: {report.overall_score:.1f}/100")
        print(f"Completeness: {report.metrics.completeness_score:.1f}")
        print(f"Accuracy: {report.metrics.accuracy_score:.1f}")
        print(f"Consistency: {report.metrics.consistency_score:.1f}")
        print(f"Processing Time: {report.processing_time_ms:.1f}ms")
        
        if report.issues:
            print(f"Issues Found: {len(report.issues)}")
            for issue in report.issues[:3]:  # Show first 3 issues
                print(f"  - {issue.severity}: {issue.description}")
        
        if report.suggestions:
            print(f"Suggestions: {len(report.suggestions)}")
            for suggestion in report.suggestions[:2]:  # Show first 2 suggestions
                print(f"  - {suggestion}")
    
    # Test batch assessment
    print(f"\n📦 Testing Batch Assessment:")
    print("-" * 30)
    
    batch_reports = scorer.assess_batch_quality(sample_leads)
    stats = scorer.get_quality_statistics(batch_reports)
    
    print(f"Total Leads: {stats['total_leads']}")
    print(f"Average Score: {stats['average_score']:.1f}")
    print(f"Score Distribution:")
    for quality_level, count in stats['score_distribution'].items():
        print(f"  {quality_level}: {count}")
    
    # Test ensemble scorer
    print(f"\n🎭 Testing Ensemble Scorer:")
    print("-" * 30)
    
    ensemble = EnsembleQualityScorer()
    
    for i, lead in enumerate(sample_leads[:2]):  # Test first 2 leads
        print(f"\n📊 Ensemble Assessment - Lead {i+1}:")
        ensemble_report = ensemble.assess_quality_ensemble(lead)
        
        print(f"Ensemble Score: {ensemble_report.overall_score:.1f}/100")
        print(f"Confidence: {ensemble_report.metrics.confidence:.3f}")
        
        if hasattr(ensemble_report, 'metadata') and ensemble_report.metadata:
            individual_scores = ensemble_report.metadata.get('individual_scores', {})
            for scorer_name, score in individual_scores.items():
                print(f"  {scorer_name}: {score:.1f}")
    
    # Test model training (if sklearn available)
    if SKLEARN_AVAILABLE:
        print(f"\n🎓 Testing Model Training:")
        print("-" * 30)
        
        try:
            # Generate more training data
            training_data = sample_leads * 20  # Repeat for more samples
            
            # Add synthetic quality scores for training
            for lead in training_data:
                # Generate synthetic quality score based on data completeness
                base_score = 50
                if lead.get('email') and '@' in lead['email']:
                    base_score += 15
                if lead.get('phone'):
                    base_score += 10
                if lead.get('company', {}).get('name'):
                    base_score += 15
                if lead.get('linkedin_url'):
                    base_score += 10
                
                lead['quality_score'] = min(100, base_score + np.random.normal(0, 5))
            
            # Train models
            training_results = scorer.train_quality_models(training_data)
            
            if 'error' not in training_results:
                print(f"✅ Training completed!")
                print(f"Training samples: {training_results.get('training_samples', 0)}")
                print(f"Feature count: {training_results.get('feature_count', 0)}")
                print(f"Training time: {training_results.get('training_time_minutes', 0):.2f} minutes")
                
                if 'overall' in training_results:
                    overall_metrics = training_results['overall']
                    print(f"Overall model performance:")
                    print(f"  MAE: {overall_metrics.get('mae', 0):.2f}")
                    print(f"  R² Score: {overall_metrics.get('r2_score', 0):.3f}")
            else:
                print(f"❌ Training failed: {training_results['error']}")
                
        except Exception as e:
            print(f"❌ Training test failed: {e}")
    else:
        print("⚠️ Scikit-learn not available - skipping ML training tests")
    
    print(f"\n✅ Quality Scorer testing completed!")
    print(f"🎯 Ready for production use with comprehensive quality assessment")
    print(f"🚀 Features: Multi-dimensional quality scoring, ML models, Ensemble methods")


if __name__ == "__main__":
    test_quality_scorer()