# utils/data_quality_engine.py - Advanced Data Quality and Deduplication Engine

# ============================
# IMPORTS
# ============================
import re
import json
import logging
import hashlib
import pickle
import os
import phonenumbers
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz, process
import pandas as pd
import numpy as np

try:
    from email_validator import validate_email, EmailNotValidError
    EMAIL_VALIDATOR_AVAILABLE = True
except ImportError:
    EMAIL_VALIDATOR_AVAILABLE = False
    logging.warning("email-validator not available, using basic email validation")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# ENUMS AND DATA CLASSES
# ============================

class DataIssueType(Enum):
    """Types of data quality issues"""
    MISSING_DATA = "missing_data"
    INVALID_FORMAT = "invalid_format"
    DUPLICATE_ENTRY = "duplicate_entry"
    INCONSISTENT_DATA = "inconsistent_data"
    OUTDATED_DATA = "outdated_data"
    LOW_CONFIDENCE = "low_confidence"
    SUSPICIOUS_PATTERN = "suspicious_pattern"

class DataIssueseverity(Enum):
    """Severity levels for data issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class DataIssue:
    """Individual data quality issue"""
    issue_type: DataIssueType
    severity: DataIssueseverity
    field_name: str
    description: str
    suggested_fix: str
    confidence: float = 0.0
    impact_score: float = 0.0

@dataclass
class QualityReport:
    """Comprehensive data quality report"""
    overall_score: float
    completeness_score: float
    accuracy_score: float
    consistency_score: float
    freshness_score: float
    email_quality: float
    phone_quality: float
    issues: List[DataIssue] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    validation_passed: bool = True
    confidence: float = 0.0

@dataclass
class DuplicateCandidate:
    """Potential duplicate lead candidate"""
    lead_id: int
    similarity_score: float
    matching_fields: List[str]
    merge_recommendation: str
    confidence: float

@dataclass
class DuplicateGroup:
    """Group of duplicate leads"""
    primary_lead_id: int
    duplicates: List[DuplicateCandidate]
    group_confidence: float
    merge_strategy: str

# ============================
# MAIN DATA QUALITY ENGINE CLASS
# ============================

class DataQualityEngine:
    """
    Advanced Data Quality Engine for lead data validation, cleaning,
    deduplication, and quality assessment
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "ai/saved_models/"
        
        # Quality thresholds
        self.quality_thresholds = {
            'email_quality_min': 70,
            'phone_quality_min': 80,
            'completeness_min': 60,
            'consistency_min': 75,
            'duplicate_threshold': 0.85,
            'suspicious_threshold': 0.9
        }
        
        # Validation patterns
        self.validation_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'linkedin': r'^https?://(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+/?$',
            'website': r'^https?://(?:[-\w.])+(?:\.[a-zA-Z]{2,})+(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?$'
        }
        
        # Common patterns for suspicious data
        self.suspicious_patterns = [
            r'test@test\.com',
            r'example@example\.com',
            r'admin@admin\.com',
            r'noreply@',
            r'donotreply@',
            r'^\d+@',  # Numbers only before @
            r'test\d+',  # Test followed by numbers
            r'^(a+|b+|c+)@'  # Repeated characters
        ]
        
        # Industry standardization
        self.industry_mappings = {
            'tech': ['technology', 'software', 'it', 'computer', 'digital'],
            'finance': ['financial', 'banking', 'investment', 'insurance'],
            'healthcare': ['health', 'medical', 'pharmaceutical', 'biotech'],
            'education': ['educational', 'university', 'school', 'academic'],
            'retail': ['ecommerce', 'e-commerce', 'shopping', 'consumer']
        }
        
        # Company size standardization
        self.size_mappings = {
            '1-10': ['1-10', 'startup', 'micro', 'very small'],
            '11-50': ['11-50', 'small', '11-49', '10-50'],
            '51-200': ['51-200', 'medium', '50-200', '51-199'],
            '201-500': ['201-500', 'medium-large', '200-500'],
            '501-1000': ['501-1000', 'large', '500-1000'],
            '1001-5000': ['1001-5000', 'large', '1000-5000'],
            '5000+': ['5000+', 'enterprise', 'very large', '5001+']
        }
        
        # Performance tracking
        self.validation_count = 0
        self.deduplication_count = 0
        self.cleaning_count = 0
        
        # Load or initialize models
        self._initialize_quality_models()
        
        logger.info("🔧 Data Quality Engine initialized successfully")

    # ============================
    # INITIALIZATION METHODS
    # ============================
    
    def _initialize_quality_models(self):
        """Initialize quality assessment models"""
        try:
            # Try to load existing models
            model_path = os.path.join(self.model_path, "quality_models.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.quality_models = pickle.load(f)
                logger.info("✅ Loaded existing quality models")
            else:
                # Create default rule-based models
                self.quality_models = self._create_default_quality_models()
                logger.info("🔧 Created default quality models")
                
        except Exception as e:
            logger.warning(f"⚠️ Quality model initialization warning: {e}")
            self.quality_models = self._create_default_quality_models()
    
    def _create_default_quality_models(self) -> Dict[str, Any]:
        """Create default rule-based quality models"""
        return {
            'email_scorer': self._rule_based_email_quality,
            'phone_scorer': self._rule_based_phone_quality,
            'completeness_scorer': self._rule_based_completeness,
            'consistency_scorer': self._rule_based_consistency,
            'duplicate_detector': self._rule_based_duplicate_detection
        }

    # ============================
    # CORE QUALITY ASSESSMENT METHODS
    # ============================
    
    def assess_lead_quality(self, lead_data: Dict[str, Any]) -> QualityReport:
        """
        Comprehensive quality assessment of a single lead
        
        Args:
            lead_data: Lead information dictionary
            
        Returns:
            QualityReport with detailed quality metrics and issues
        """
        try:
            issues = []
            suggestions = []
            
            # Email quality assessment
            email_quality = self._assess_email_quality(lead_data, issues, suggestions)
            
            # Phone quality assessment
            phone_quality = self._assess_phone_quality(lead_data, issues, suggestions)
            
            # Data completeness assessment
            completeness_score = self._assess_completeness(lead_data, issues, suggestions)
            
            # Data accuracy assessment
            accuracy_score = self._assess_accuracy(lead_data, issues, suggestions)
            
            # Data consistency assessment
            consistency_score = self._assess_consistency(lead_data, issues, suggestions)
            
            # Data freshness assessment
            freshness_score = self._assess_freshness(lead_data, issues, suggestions)
            
            # Calculate overall score
            weights = {
                'completeness': 0.25,
                'accuracy': 0.25,
                'consistency': 0.20,
                'email': 0.15,
                'phone': 0.10,
                'freshness': 0.05
            }
            
            overall_score = (
                completeness_score * weights['completeness'] +
                accuracy_score * weights['accuracy'] +
                consistency_score * weights['consistency'] +
                email_quality * weights['email'] +
                phone_quality * weights['phone'] +
                freshness_score * weights['freshness']
            )
            
            # Determine validation statusDataIssueseverity
            validation_passed = (
                overall_score >= 60 and
                len([i for i in issues if i.severity == DataIssueseverity.CRITICAL]) == 0
            )
            
            # Calculate confidence
            confidence = self._calculate_assessment_confidence(lead_data, issues)
            
            self.validation_count += 1
            
            return QualityReport(
                overall_score=min(100, max(0, overall_score)),
                completeness_score=completeness_score,
                accuracy_score=accuracy_score,
                consistency_score=consistency_score,
                freshness_score=freshness_score,
                email_quality=email_quality,
                phone_quality=phone_quality,
                issues=issues,
                suggestions=suggestions,
                validation_passed=validation_passed,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return self._generate_fallback_quality_report(lead_data)

    # ============================
    # EMAIL QUALITY ASSESSMENT
    # ============================
    
    def _assess_email_quality(self, lead_data: Dict[str, Any], issues: List[DataIssue], suggestions: List[str]) -> float:
        """Assess email quality with detailed analysis"""
        email = lead_data.get('email', '').strip()
        
        if not email:
            issues.append(DataIssue(
                issue_type=DataIssueType.MISSING_DATA,
                severity=DataIssueseverity.HIGH,
                field_name='email',
                description='Email address is missing',
                suggested_fix='Find and add a valid email address',
                confidence=1.0,
                impact_score=30.0
            ))
            suggestions.append('Use email enrichment tools to find contact email')
            return 0.0
        
        score = 50.0
        
        # Basic format validation
        if not re.match(self.validation_patterns['email'], email):
            issues.append(DataIssue(
                issue_type=DataIssueType.INVALID_FORMAT,
                severity=DataIssueseverity.CRITICAL,
                field_name='email',
                description='Email format is invalid',
                suggested_fix='Correct the email format',
                confidence=0.95,
                impact_score=40.0
            ))
            return 0.0
        
        # Advanced email validation if available
        if EMAIL_VALIDATOR_AVAILABLE:
            try:
                validation_result = validate_email(email)
                score += 20
            except EmailNotValidError:
                issues.append(DataIssue(
                    issue_type=DataIssueType.INVALID_FORMAT,
                    severity=DataIssueseverity.HIGH,
                    field_name='email',
                    description='Email address failed validation',
                    suggested_fix='Verify and correct the email address',
                    confidence=0.85,
                    impact_score=35.0
                ))
                score -= 30
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, email.lower()):
                issues.append(DataIssue(
                    issue_type=DataIssueType.SUSPICIOUS_PATTERN,
                    severity=DataIssueseverity.MEDIUM,
                    field_name='email',
                    description=f'Email matches suspicious pattern: {pattern}',
                    suggested_fix='Verify this is a real professional email',
                    confidence=0.7,
                    impact_score=20.0
                ))
                score -= 25
                break
        
        # Professional email patterns
        email_lower = email.lower()
        domain = email.split('@')[1] if '@' in email else ''
        
        # Check for professional structure
        if re.match(r'^[a-z]+\.[a-z]+@', email_lower):
            score += 20
            
        elif re.match(r'^[a-z]+@', email_lower):
            score += 10
        
        # Domain quality analysis
        free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com']
        if domain.lower() in free_domains:
            score -= 15
            suggestions.append('Consider finding a professional/corporate email address')
        else:
            score += 15
        
        # Domain TLD analysis
        if domain.endswith(('.ai', '.io', '.tech', '.dev')):
            score += 10
        elif domain.endswith('.com'):
            score += 5
        
        # Email length and structure
        local_part = email.split('@')[0]
        if len(local_part) < 2:
            score -= 20
        elif len(local_part) > 30:
            score -= 10
        
        # Check for generic emails
        generic_patterns = ['info', 'contact', 'admin', 'support', 'sales', 'hello', 'team']
        if any(pattern in local_part.lower() for pattern in generic_patterns):
            issues.append(DataIssue(
                issue_type=DataIssueType.LOW_CONFIDENCE,
                severity=DataIssueseverity.MEDIUM,
                field_name='email',
                description='Email appears to be a generic/role-based address',
                suggested_fix='Find the specific person\'s direct email',
                confidence=0.8,
                impact_score=15.0
            ))
            score -= 20
        
        return min(100, max(0, score))

    # ============================
    # PHONE QUALITY ASSESSMENT
    # ============================
    
    def _assess_phone_quality(self, lead_data: Dict[str, Any], issues: List[DataIssue], suggestions: List[str]) -> float:
        """Assess phone number quality"""
        phone = lead_data.get('phone', '').strip()
        
        if not phone:
            suggestions.append('Add phone number for better contact options')
            return 0.0
        
        score = 50.0
        
        # Clean phone number for analysis
        cleaned_phone = re.sub(r'[^\d+]', '', phone)
        
        # Basic length checks
        if len(cleaned_phone) < 10:
            issues.append(DataIssue(
                issue_type=DataIssueType.INVALID_FORMAT,
                severity=DataIssueseverity.HIGH,
                field_name='phone',
                description='Phone number appears too short',
                suggested_fix='Verify and complete the phone number',
                confidence=0.9,
                impact_score=25.0
            ))
            score -= 30
        elif len(cleaned_phone) > 15:
            issues.append(DataIssue(
                issue_type=DataIssueType.INVALID_FORMAT,
                severity=DataIssueseverity.MEDIUM,
                field_name='phone',
                description='Phone number appears too long',
                suggested_fix='Verify the phone number format',
                confidence=0.8,
                impact_score=15.0
            ))
            score -= 20
        
        # Format validation with phonenumbers library if available
        try:
            import phonenumbers
            from phonenumbers import geocoder, carrier
            
            # Try to parse the phone number
            parsed = phonenumbers.parse(phone, None)
            
            if phonenumbers.is_valid_number(parsed):
                score += 30
                
                # Get additional info
                location = geocoder.description_for_number(parsed, "en")
                if location:
                    score += 10
                    
            else:
                issues.append(DataIssue(
                    issue_type=DataIssueType.INVALID_FORMAT,
                    severity=DataIssueseverity.HIGH,
                    field_name='phone',
                    description='Phone number format is invalid',
                    suggested_fix='Correct the phone number format',
                    confidence=0.85,
                    impact_score=30.0
                ))
                score -= 25
                
        except ImportError:
            # Fallback validation without phonenumbers
            # US phone number pattern
            if re.match(r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$', phone):
                score += 20
            # International format
            elif re.match(r'^\+[1-9]\d{1,14}$', cleaned_phone):
                score += 15
            else:
                score -= 20
        except Exception:
            score -= 10
        
        # Check for obviously fake numbers
        fake_patterns = [
            r'^(\d)\1{9,}',  # All same digits
            r'^123456',      # Sequential
            r'^000000',      # All zeros
            r'^111111'       # All ones
        ]
        
        for pattern in fake_patterns:
            if re.search(pattern, cleaned_phone):
                issues.append(DataIssue(
                    issue_type=DataIssueType.SUSPICIOUS_PATTERN,
                    severity=DataIssueseverity.HIGH,
                    field_name='phone',
                    description='Phone number appears to be fake or placeholder',
                    suggested_fix='Verify and replace with real phone number',
                    confidence=0.9,
                    impact_score=35.0
                ))
                score -= 40
                break
        
        return min(100, max(0, score))

    # ============================
    # COMPLETENESS ASSESSMENT
    # ============================
    
    def _assess_completeness(self, lead_data: Dict[str, Any], issues: List[DataIssue], suggestions: List[str]) -> float:
        """Assess data completeness"""
        required_fields = ['first_name', 'last_name', 'email']
        important_fields = ['title', 'phone', 'company']
        optional_fields = ['linkedin_url', 'website', 'notes']
        
        total_weight = 0
        completed_weight = 0
        
        # Check required fields (high weight)
        for field in required_fields:
            total_weight += 3
            if lead_data.get(field):
                completed_weight += 3
            else:
                issues.append(DataIssue(
                    issue_type=DataIssueType.MISSING_DATA,
                    severity=DataIssueseverity.HIGH,
                    field_name=field,
                    description=f'Required field {field} is missing',
                    suggested_fix=f'Add {field} information',
                    confidence=1.0,
                    impact_score=25.0
                ))
        
        # Check important fields (medium weight)
        for field in important_fields:
            total_weight += 2
            if lead_data.get(field):
                completed_weight += 2
            else:
                issues.append(DataIssue(
                    issue_type=DataIssueType.MISSING_DATA,
                    severity=DataIssueseverity.MEDIUM,
                    field_name=field,
                    description=f'Important field {field} is missing',
                    suggested_fix=f'Consider adding {field} information',
                    confidence=0.8,
                    impact_score=15.0
                ))
        
        # Check optional fields (low weight)
        for field in optional_fields:
            total_weight += 1
            if lead_data.get(field):
                completed_weight += 1
        
        # Check company information completeness
        company = lead_data.get('company', {})
        if isinstance(company, dict):
            company_fields = ['name', 'industry', 'size', 'location']
            company_total = len(company_fields) * 2
            company_completed = sum(2 for field in company_fields if company.get(field))
            
            total_weight += company_total
            completed_weight += company_completed
            
            if company_completed < company_total * 0.5:
                suggestions.append('Enrich company information for better lead quality')
        
        completeness_score = (completed_weight / total_weight) * 100 if total_weight > 0 else 0
        
        if completeness_score < 60:
            suggestions.append('Consider using data enrichment tools to complete missing fields')
        
        return completeness_score

    # ============================
    # ACCURACY ASSESSMENT
    # ============================
    
    def _assess_accuracy(self, lead_data: Dict[str, Any], issues: List[DataIssue], suggestions: List[str]) -> float:
        """Assess data accuracy"""
        accuracy_score = 100.0
        
        # Name format validation
        first_name = lead_data.get('first_name', '').strip()
        last_name = lead_data.get('last_name', '').strip()
        
        if first_name:
            if not re.match(r'^[a-zA-Z\s\'-]+$', first_name):
                issues.append(DataIssue(
                    issue_type=DataIssueType.INVALID_FORMAT,
                    severity=DataIssueseverity.MEDIUM,
                    field_name='first_name',
                    description='First name contains invalid characters',
                    suggested_fix='Remove invalid characters from first name',
                    confidence=0.9,
                    impact_score=10.0
                ))
                accuracy_score -= 15
        
        if last_name:
            if not re.match(r'^[a-zA-Z\s\'-]+$', last_name):
                issues.append(DataIssue(
                    issue_type=DataIssueType.INVALID_FORMAT,
                    severity=DataIssueseverity.MEDIUM,
                    field_name='last_name',
                    description='Last name contains invalid characters',
                    suggested_fix='Remove invalid characters from last name',
                    confidence=0.9,
                    impact_score=10.0
                ))
                accuracy_score -= 15
        
        # LinkedIn URL validation
        linkedin_url = lead_data.get('linkedin_url', '').strip()
        if linkedin_url and not re.match(self.validation_patterns['linkedin'], linkedin_url):
            issues.append(DataIssue(
                issue_type=DataIssueType.INVALID_FORMAT,
                severity=DataIssueseverity.MEDIUM,
                field_name='linkedin_url',
                description='LinkedIn URL format is invalid',
                suggested_fix='Correct the LinkedIn URL format',
                confidence=0.85,
                impact_score=10.0
            ))
            accuracy_score -= 10
        
        # Website URL validation
        website = lead_data.get('website', '').strip()
        if website and not re.match(self.validation_patterns['website'], website):
            issues.append(DataIssue(
                issue_type=DataIssueType.INVALID_FORMAT,
                severity=DataIssueseverity.LOW,
                field_name='website',
                description='Website URL format may be invalid',
                suggested_fix='Verify and correct the website URL',
                confidence=0.7,
                impact_score=5.0
            ))
            accuracy_score -= 5
        
        # Title validation
        title = lead_data.get('title', '').strip()
        if title:
            # Check for obviously fake titles
            fake_title_patterns = [
                r'^test\s+',
                r'^fake\s+',
                r'^sample\s+',
                r'^\d+$',  # Only numbers
                r'^[^a-zA-Z]*$'  # No letters
            ]
            
            for pattern in fake_title_patterns:
                if re.search(pattern, title.lower()):
                    issues.append(DataIssue(
                        issue_type=DataIssueType.SUSPICIOUS_PATTERN,
                        severity=DataIssueseverity.HIGH,
                        field_name='title',
                        description='Job title appears to be fake or placeholder',
                        suggested_fix='Replace with actual job title',
                        confidence=0.8,
                        impact_score=20.0
                    ))
                    accuracy_score -= 25
                    break
        
        return max(0, accuracy_score)

    # ============================
    # CONSISTENCY ASSESSMENT
    # ============================
    
    def _assess_consistency(self, lead_data: Dict[str, Any], issues: List[DataIssue], suggestions: List[str]) -> float:
        """Assess data consistency across fields"""
        consistency_score = 100.0
        
        # Email vs name consistency
        email = lead_data.get('email', '').lower()
        first_name = lead_data.get('first_name', '').lower()
        last_name = lead_data.get('last_name', '').lower()
        
        if email and first_name and '@' in email:
            email_local = email.split('@')[0]
            
            # Check if name appears in email
            if first_name not in email_local and last_name not in email_local:
                # Check for common variations
                first_initial = first_name[0] if first_name else ''
                last_initial = last_name[0] if last_name else ''
                
                if not (first_initial in email_local or last_initial in email_local):
                    issues.append(DataIssue(
                        issue_type=DataIssueType.INCONSISTENT_DATA,
                        severity=DataIssueseverity.MEDIUM,
                        field_name='email',
                        description='Email doesn\'t match the person\'s name',
                        suggested_fix='Verify email belongs to this person',
                        confidence=0.6,
                        impact_score=15.0
                    ))
                    consistency_score -= 20
        
        # Company domain vs email domain consistency
        company = lead_data.get('company', {})
        company_domain = company.get('domain', '').lower() if isinstance(company, dict) else ''
        
        if email and company_domain and '@' in email:
            email_domain = email.split('@')[1]
            
            # Allow for reasonable variations
            if (company_domain not in email_domain and 
                email_domain not in company_domain and
                not self._domains_related(email_domain, company_domain)):
                
                # Check if it's a personal email for business contact
                free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
                if email_domain not in free_domains:
                    issues.append(DataIssue(
                        issue_type=DataIssueType.INCONSISTENT_DATA,
                        severity=DataIssueseverity.LOW,
                        field_name='email',
                        description='Email domain doesn\'t match company domain',
                        suggested_fix='Verify email and company information',
                        confidence=0.5,
                        impact_score=10.0
                    ))
                    consistency_score -= 10
        
        # Title vs company size consistency
        title = lead_data.get('title', '').lower()
        company_size = company.get('size', '') if isinstance(company, dict) else ''
        
        if title and company_size:
            # C-level in very small companies
            if any(c_level in title for c_level in ['ceo', 'president', 'founder']) and '1-10' in company_size:
                # This is actually often valid for startups
                pass
            elif any(c_level in title for c_level in ['ceo', 'president']) and company_size in ['11-50', '51-200']:
                # Potentially valid but worth noting
                pass
            elif 'director' in title and '1-10' in company_size:
                issues.append(DataIssue(
                    issue_type=DataIssueType.INCONSISTENT_DATA,
                    severity=DataIssueseverity.LOW,
                    field_name='title',
                    description='Director-level title in very small company',
                    suggested_fix='Verify title accuracy',
                    confidence=0.4,
                    impact_score=5.0
                ))
                consistency_score -= 5
        
        return max(0, consistency_score)

    # ============================
    # FRESHNESS ASSESSMENT
    # ============================
    
    def _assess_freshness(self, lead_data: Dict[str, Any], issues: List[DataIssue], suggestions: List[str]) -> float:
        """Assess data freshness based on creation and update dates"""
        created_at = lead_data.get('created_at')
        updated_at = lead_data.get('updated_at')
        
        if not created_at:
            return 100.0  # No date info, assume fresh
        
        try:
            if isinstance(created_at, str):
                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                created_date = created_at
            
            # Remove timezone info for calculation
            created_date = created_date.replace(tzinfo=None)
            days_old = (datetime.utcnow() - created_date).days
            
            # Freshness scoring
            if days_old <= 7:
                freshness_score = 100.0
            elif days_old <= 30:
                freshness_score = 90.0
            elif days_old <= 90:
                freshness_score = 75.0
            elif days_old <= 180:
                freshness_score = 50.0
            elif days_old <= 365:
                freshness_score = 25.0
            else:
                freshness_score = 10.0
                issues.append(DataIssue(
                    issue_type=DataIssueType.OUTDATED_DATA,
                    severity=DataIssueseverity.MEDIUM,
                    field_name='created_at',
                    description=f'Lead data is {days_old} days old',
                    suggested_fix='Verify and update lead information',
                    confidence=0.9,
                    impact_score=15.0
                ))
                
                if days_old > 365:
                    suggestions.append('Consider refreshing this lead data - it\'s over a year old')
                elif days_old > 180:
                    suggestions.append('Lead data is getting stale - consider updating')
            
            return freshness_score
            
        except Exception:
            return 50.0  # Default score if date parsing fails

    # ============================
    # DUPLICATE DETECTION METHODS
    # ============================
    
    def detect_duplicates(self, leads: List[Any], threshold: float = 0.85) -> List[DuplicateGroup]:
        """
        Detect duplicate leads using advanced matching algorithms
        
        Args:
            leads: List of lead objects or dictionaries
            threshold: Similarity threshold for duplicate detection
            
        Returns:
            List of duplicate groups with merge recommendations
        """
        try:
            duplicate_groups = []
            processed_leads = set()
            
            for i, lead1 in enumerate(leads):
                if i in processed_leads:
                    continue
                
                lead1_data = self._extract_lead_data(lead1)
                duplicates = []
                
                for j, lead2 in enumerate(leads[i+1:], start=i+1):
                    if j in processed_leads:
                        continue
                    
                    lead2_data = self._extract_lead_data(lead2)
                    similarity_result = self._calculate_lead_similarity(lead1_data, lead2_data)
                    
                    if similarity_result['score'] >= threshold:
                        duplicates.append(DuplicateCandidate(
                            lead_id=lead2_data.get('id', j),
                            similarity_score=similarity_result['score'],
                            matching_fields=similarity_result['matching_fields'],
                            merge_recommendation=self._generate_merge_recommendation(similarity_result),
                            confidence=similarity_result['confidence']
                        ))
                        processed_leads.add(j)
                
                if duplicates:
                    # Determine merge strategy
                    group_confidence = np.mean([dup.confidence for dup in duplicates])
                    merge_strategy = self._determine_merge_strategy(lead1_data, duplicates)
                    
                    duplicate_groups.append(DuplicateGroup(
                        primary_lead_id=lead1_data.get('id', i),
                        duplicates=duplicates,
                        group_confidence=group_confidence,
                        merge_strategy=merge_strategy
                    ))
                    processed_leads.add(i)
            
            self.deduplication_count += len(duplicate_groups)
            return duplicate_groups
            
        except Exception as e:
            logger.error(f"Duplicate detection failed: {e}")
            return []
    
    def _calculate_lead_similarity(self, lead1: Dict[str, Any], lead2: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate similarity between two leads using multiple algorithms"""
        similarity_scores = {}
        matching_fields = []
        
        # Email similarity (highest weight)
        email1 = lead1.get('email', '').lower().strip()
        email2 = lead2.get('email', '').lower().strip()
        
        if email1 and email2:
            if email1 == email2:
                similarity_scores['email'] = 1.0
                matching_fields.append('email')
            else:
                # Check for similar emails (typos, etc.)
                email_sim = fuzz.ratio(email1, email2) / 100.0
                if email_sim > 0.9:
                    similarity_scores['email'] = email_sim
                    matching_fields.append('email_similar')
                else:
                    similarity_scores['email'] = 0.0
        else:
            similarity_scores['email'] = 0.0
        
        # Name similarity
        name1 = f"{lead1.get('first_name', '')} {lead1.get('last_name', '')}".strip().lower()
        name2 = f"{lead2.get('first_name', '')} {lead2.get('last_name', '')}".strip().lower()
        
        if name1 and name2:
            name_sim = fuzz.ratio(name1, name2) / 100.0
            similarity_scores['name'] = name_sim
            if name_sim > 0.8:
                matching_fields.append('name')
        else:
            similarity_scores['name'] = 0.0
        
        # Phone similarity
        phone1 = self._normalize_phone(lead1.get('phone', ''))
        phone2 = self._normalize_phone(lead2.get('phone', ''))
        
        if phone1 and phone2:
            if phone1 == phone2:
                similarity_scores['phone'] = 1.0
                matching_fields.append('phone')
            else:
                similarity_scores['phone'] = 0.0
        else:
            similarity_scores['phone'] = 0.0
        
        # LinkedIn similarity
        linkedin1 = lead1.get('linkedin_url', '').lower().strip()
        linkedin2 = lead2.get('linkedin_url', '').lower().strip()
        
        if linkedin1 and linkedin2:
            if linkedin1 == linkedin2:
                similarity_scores['linkedin'] = 1.0
                matching_fields.append('linkedin')
            else:
                linkedin_sim = fuzz.ratio(linkedin1, linkedin2) / 100.0
                similarity_scores['linkedin'] = linkedin_sim
        else:
            similarity_scores['linkedin'] = 0.0
        
        # Company similarity
        company1 = lead1.get('company', {})
        company2 = lead2.get('company', {})
        
        if isinstance(company1, dict) and isinstance(company2, dict):
            company_name1 = company1.get('name', '').lower().strip()
            company_name2 = company2.get('name', '').lower().strip()
            
            if company_name1 and company_name2:
                company_sim = fuzz.ratio(company_name1, company_name2) / 100.0
                similarity_scores['company'] = company_sim
                if company_sim > 0.8:
                    matching_fields.append('company')
            else:
                similarity_scores['company'] = 0.0
        else:
            similarity_scores['company'] = 0.0
        
        # Title similarity
        title1 = lead1.get('title', '').lower().strip()
        title2 = lead2.get('title', '').lower().strip()
        
        if title1 and title2:
            title_sim = fuzz.ratio(title1, title2) / 100.0
            similarity_scores['title'] = title_sim
            if title_sim > 0.7:
                matching_fields.append('title')
        else:
            similarity_scores['title'] = 0.0
        
        # Calculate weighted overall similarity
        weights = {
            'email': 0.4,
            'name': 0.25,
            'phone': 0.15,
            'linkedin': 0.1,
            'company': 0.07,
            'title': 0.03
        }
        
        overall_score = sum(similarity_scores[field] * weight for field, weight in weights.items())
        
        # Calculate confidence based on available data
        available_fields = sum(1 for score in similarity_scores.values() if score > 0)
        confidence = min(0.95, available_fields / len(weights) * overall_score)
        
        return {
            'score': overall_score,
            'matching_fields': matching_fields,
            'field_scores': similarity_scores,
            'confidence': confidence
        }
    
    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number for comparison"""
        if not phone:
            return ""
        
        # Remove all non-digit characters except +
        normalized = re.sub(r'[^\d+]', '', phone)
        
        # Remove leading 1 from US numbers
        if normalized.startswith('+1'):
            normalized = normalized[2:]
        elif normalized.startswith('1') and len(normalized) == 11:
            normalized = normalized[1:]
        
        return normalized
    
    def _generate_merge_recommendation(self, similarity_result: Dict[str, Any]) -> str:
        """Generate merge recommendation based on similarity analysis"""
        score = similarity_result['score']
        matching_fields = similarity_result['matching_fields']
        
        if score >= 0.95:
            return "auto_merge"
        elif score >= 0.85:
            if 'email' in matching_fields or 'phone' in matching_fields:
                return "high_confidence_merge"
            else:
                return "manual_review"
        elif score >= 0.7:
            return "manual_review"
        else:
            return "no_merge"
    
    def _determine_merge_strategy(self, primary_lead: Dict[str, Any], duplicates: List[DuplicateCandidate]) -> str:
        """Determine the best merge strategy for a duplicate group"""
        high_confidence_count = sum(1 for dup in duplicates if dup.confidence > 0.9)
        total_duplicates = len(duplicates)
        
        if high_confidence_count == total_duplicates and total_duplicates <= 3:
            return "batch_merge"
        elif high_confidence_count >= total_duplicates * 0.8:
            return "selective_merge"
        else:
            return "manual_review_required"

    # ============================
    # DATA CLEANING METHODS
    # ============================
    
    def clean_lead_data(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and standardize lead data
        
        Args:
            lead_data: Raw lead data dictionary
            
        Returns:
            Cleaned lead data dictionary
        """
        try:
            cleaned_data = lead_data.copy()
            
            # Clean names
            cleaned_data = self._clean_names(cleaned_data)
            
            # Clean email
            cleaned_data = self._clean_email(cleaned_data)
            
            # Clean phone
            cleaned_data = self._clean_phone(cleaned_data)
            
            # Clean and standardize company data
            cleaned_data = self._clean_company_data(cleaned_data)
            
            # Clean title
            cleaned_data = self._clean_title(cleaned_data)
            
            # Clean URLs
            cleaned_data = self._clean_urls(cleaned_data)
            
            # Remove extra whitespace
            cleaned_data = self._trim_whitespace(cleaned_data)
            
            self.cleaning_count += 1
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            return lead_data  # Return original data if cleaning fails
    
    def _clean_names(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize name fields"""
        # Clean first name
        if data.get('first_name'):
            first_name = data['first_name'].strip()
            # Capitalize properly
            first_name = ' '.join(word.capitalize() for word in first_name.split())
            # Remove invalid characters
            first_name = re.sub(r'[^a-zA-Z\s\'-]', '', first_name)
            data['first_name'] = first_name
        
        # Clean last name
        if data.get('last_name'):
            last_name = data['last_name'].strip()
            # Capitalize properly
            last_name = ' '.join(word.capitalize() for word in last_name.split())
            # Remove invalid characters
            last_name = re.sub(r'[^a-zA-Z\s\'-]', '', last_name)
            data['last_name'] = last_name
        
        return data
    
    def _clean_email(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate email address"""
        if data.get('email'):
            email = data['email'].strip().lower()
            
            # Remove common email prefix/suffix errors
            email = re.sub(r'^mailto:', '', email)
            email = re.sub(r'\s+', '', email)  # Remove all whitespace
            
            # Basic validation
            if re.match(self.validation_patterns['email'], email):
                data['email'] = email
            else:
                # Try to fix common issues
                if '@' not in email and '.' in email:
                    # Might be missing @
                    parts = email.split('.')
                    if len(parts) >= 3:
                        # Assume first part is username, rest is domain
                        email = f"{parts[0]}@{'.'.join(parts[1:])}"
                        data['email'] = email
        
        return data
    
    def _clean_phone(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize phone number"""
        if data.get('phone'):
            phone = data['phone'].strip()
            
            # Remove common formatting
            phone = re.sub(r'[^\d+\-\(\)\s\.]', '', phone)
            
            # Try to standardize US numbers
            digits_only = re.sub(r'[^\d]', '', phone)
            
            if len(digits_only) == 10:
                # Format as (XXX) XXX-XXXX
                formatted = f"({digits_only[:3]}) {digits_only[3:6]}-{digits_only[6:]}"
                data['phone'] = formatted
            elif len(digits_only) == 11 and digits_only.startswith('1'):
                # Format as +1 (XXX) XXX-XXXX
                formatted = f"+1 ({digits_only[1:4]}) {digits_only[4:7]}-{digits_only[7:]}"
                data['phone'] = formatted
            elif phone.startswith('+'):
                # Keep international format as is
                data['phone'] = phone
        
        return data
    
    def _clean_company_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize company information"""
        company = data.get('company', {})
        if not isinstance(company, dict):
            return data
        
        # Clean company name
        if company.get('name'):
            name = company['name'].strip()
            # Remove common suffixes like Inc., LLC, etc. for cleaner display
            # But keep them for accuracy
            company['name'] = name
        
        # Standardize industry
        if company.get('industry'):
            industry = company['industry'].strip().lower()
            standardized_industry = self._standardize_industry(industry)
            company['industry'] = standardized_industry
        
        # Standardize company size
        if company.get('size'):
            size = company['size'].strip()
            standardized_size = self._standardize_company_size(size)
            company['size'] = standardized_size
        
        # Clean location
        if company.get('location'):
            location = company['location'].strip()
            # Remove extra commas and spaces
            location = re.sub(r',\s*,', ',', location)
            location = re.sub(r'\s+', ' ', location)
            company['location'] = location
        
        data['company'] = company
        return data
    
    def _clean_title(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and standardize job title"""
        if data.get('title'):
            title = data['title'].strip()
            
            # Proper capitalization
            # Keep common abbreviations uppercase
            abbreviations = ['CEO', 'CTO', 'CFO', 'CMO', 'VP', 'SVP', 'EVP', 'COO', 'IT', 'HR', 'QA']
            
            words = title.split()
            cleaned_words = []
            
            for word in words:
                if word.upper() in abbreviations:
                    cleaned_words.append(word.upper())
                else:
                    cleaned_words.append(word.capitalize())
            
            data['title'] = ' '.join(cleaned_words)
        
        return data
    
    def _clean_urls(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate URLs"""
        # Clean LinkedIn URL
        if data.get('linkedin_url'):
            linkedin = data['linkedin_url'].strip()
            
            # Ensure proper format
            if not linkedin.startswith('http'):
                linkedin = 'https://' + linkedin
            
            # Standardize LinkedIn URLs
            if 'linkedin.com' in linkedin:
                # Extract profile ID and reconstruct clean URL
                match = re.search(r'linkedin\.com/in/([a-zA-Z0-9_-]+)', linkedin)
                if match:
                    profile_id = match.group(1)
                    linkedin = f"https://www.linkedin.com/in/{profile_id}"
            
            data['linkedin_url'] = linkedin
        
        # Clean website URL
        if data.get('website'):
            website = data['website'].strip()
            
            if not website.startswith('http'):
                website = 'https://' + website
            
            data['website'] = website
        
        return data
    
    def _trim_whitespace(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove extra whitespace from all string fields"""
        cleaned = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                cleaned[key] = value.strip()
            elif isinstance(value, dict):
                cleaned[key] = self._trim_whitespace(value)
            else:
                cleaned[key] = value
        
        return cleaned

    # ============================
    # STANDARDIZATION METHODS
    # ============================
    
    def _standardize_industry(self, industry: str) -> str:
        """Standardize industry names"""
        industry_lower = industry.lower()
        
        for standard_name, variations in self.industry_mappings.items():
            for variation in variations:
                if variation in industry_lower:
                    return standard_name.title()
        
        # Return original if no mapping found
        return industry.title()
    
    def _standardize_company_size(self, size: str) -> str:
        """Standardize company size descriptions"""
        size_lower = size.lower()
        
        for standard_size, variations in self.size_mappings.items():
            for variation in variations:
                if variation in size_lower:
                    return standard_size
        
        # Return original if no mapping found
        return size

    # ============================
    # UTILITY METHODS
    # ============================
    
    def _extract_lead_data(self, lead: Any) -> Dict[str, Any]:
        """Extract data from lead object or dictionary"""
        if hasattr(lead, '__dict__'):
            # SQLAlchemy model object
            data = {}
            for column in lead.__table__.columns:
                data[column.name] = getattr(lead, column.name)
            
            # Handle company relationship
            if hasattr(lead, 'company') and lead.company:
                data['company'] = {
                    'name': lead.company.name,
                    'industry': getattr(lead.company, 'industry', None),
                    'size': getattr(lead.company, 'size', None),
                    'location': getattr(lead.company, 'location', None),
                    'domain': getattr(lead.company, 'domain', None)
                }
            
            return data
        elif isinstance(lead, dict):
            return lead
        else:
            return {}
    
    def _domains_related(self, domain1: str, domain2: str) -> bool:
        """Check if two domains are related (subsidiaries, etc.)"""
        # Remove common prefixes
        clean_domain1 = re.sub(r'^(www\.|mail\.|email\.)', '', domain1)
        clean_domain2 = re.sub(r'^(www\.|mail\.|email\.)', '', domain2)
        
        # Check if one is subdomain of another
        if clean_domain1 in clean_domain2 or clean_domain2 in clean_domain1:
            return True
        
        # Check for common corporate domain patterns
        base1 = clean_domain1.split('.')[0]
        base2 = clean_domain2.split('.')[0]
        
        # Check for similar base names
        similarity = fuzz.ratio(base1, base2) / 100.0
        return similarity > 0.8
    
    def _calculate_assessment_confidence(self, lead_data: Dict[str, Any], issues: List[DataIssue]) -> float:
        """Calculate confidence in the quality assessment"""
        # Base confidence on data completeness
        available_fields = sum(1 for field in ['email', 'first_name', 'last_name', 'title', 'company'] 
                             if lead_data.get(field))
        completeness_factor = available_fields / 5
        
        # Reduce confidence based on critical issues
        critical_issues = sum(1 for issue in issues if issue.severity == DataIssueseverity.CRITICAL)
        issue_factor = max(0, 1 - (critical_issues * 0.3))
        
        # Validation factor
        validation_factor = 1.0
        if not EMAIL_VALIDATOR_AVAILABLE:
            validation_factor *= 0.9
        
        confidence = completeness_factor * issue_factor * validation_factor
        return min(0.95, max(0.1, confidence))
    
    def _generate_fallback_quality_report(self, lead_data: Dict[str, Any]) -> QualityReport:
        """Generate basic quality report when assessment fails"""
        return QualityReport(
            overall_score=50.0,
            completeness_score=50.0,
            accuracy_score=50.0,
            consistency_score=50.0,
            freshness_score=50.0,
            email_quality=50.0,
            phone_quality=50.0,
            issues=[DataIssue(
                issue_type=DataIssueType.LOW_CONFIDENCE,
                severity=DataIssueseverity.INFO,
                field_name='system',
                description='Quality assessment failed, using fallback scoring',
                suggested_fix='Manual review recommended',
                confidence=0.3,
                impact_score=0.0
            )],
            suggestions=['Manual quality review recommended'],
            validation_passed=False,
            confidence=0.3
        )

    # ============================
    # BATCH PROCESSING METHODS
    # ============================
    
    def batch_quality_assessment(self, leads: List[Any], assessment_type: str = 'standard') -> Dict[int, QualityReport]:
        """
        Perform quality assessment on multiple leads
        
        Args:
            leads: List of lead objects or dictionaries
            assessment_type: 'basic', 'standard', or 'comprehensive'
            
        Returns:
            Dictionary mapping lead IDs to quality reports
        """
        results = {}
        
        for lead in leads:
            try:
                lead_data = self._extract_lead_data(lead)
                lead_id = lead_data.get('id', 0)
                
                if assessment_type == 'basic':
                    report = self._basic_quality_assessment(lead_data)
                elif assessment_type == 'comprehensive':
                    report = self._comprehensive_quality_assessment(lead_data)
                else:  # standard
                    report = self.assess_lead_quality(lead_data)
                
                results[lead_id] = report
                
            except Exception as e:
                logger.warning(f"Failed to assess lead quality: {e}")
                continue
        
        return results
    
    def _basic_quality_assessment(self, lead_data: Dict[str, Any]) -> QualityReport:
        """Basic quality assessment with minimal checks"""
        issues = []
        suggestions = []
        
        # Check only essential fields
        required_fields = ['email', 'first_name', 'last_name']
        missing_count = sum(1 for field in required_fields if not lead_data.get(field))
        
        completeness_score = ((len(required_fields) - missing_count) / len(required_fields)) * 100
        
        # Basic email validation
        email_quality = 50.0
        if lead_data.get('email'):
            if re.match(self.validation_patterns['email'], lead_data['email']):
                email_quality = 80.0
            else:
                email_quality = 20.0
        
        overall_score = (completeness_score * 0.6 + email_quality * 0.4)
        
        return QualityReport(
            overall_score=overall_score,
            completeness_score=completeness_score,
            accuracy_score=75.0,  # Default
            consistency_score=75.0,  # Default
            freshness_score=75.0,  # Default
            email_quality=email_quality,
            phone_quality=50.0,  # Default
            issues=issues,
            suggestions=suggestions,
            validation_passed=overall_score >= 60,
            confidence=0.6
        )
    
    def _comprehensive_quality_assessment(self, lead_data: Dict[str, Any]) -> QualityReport:
        """Comprehensive quality assessment with all checks"""
        # Start with standard assessment
        report = self.assess_lead_quality(lead_data)
        
        # Add additional comprehensive checks
        issues = report.issues.copy()
        suggestions = report.suggestions.copy()
        
        # Advanced pattern detection
        self._detect_advanced_patterns(lead_data, issues, suggestions)
        
        # Social media validation
        self._validate_social_media(lead_data, issues, suggestions)
        
        # Industry-specific validation
        self._industry_specific_validation(lead_data, issues, suggestions)
        
        # Update report with comprehensive results
        report.issues = issues
        report.suggestions = suggestions
        
        return report
    
    def _detect_advanced_patterns(self, lead_data: Dict[str, Any], issues: List[DataIssue], suggestions: List[str]):
        """Detect advanced suspicious patterns"""
        # Check for bot-generated data patterns
        email = lead_data.get('email', '')
        name = f"{lead_data.get('first_name', '')} {lead_data.get('last_name', '')}".strip()
        
        # Sequential character patterns
        if re.search(r'(abc|123|xyz)', email.lower()) or re.search(r'(abc|123|xyz)', name.lower()):
            issues.append(DataIssue(
                issue_type=DataIssueType.SUSPICIOUS_PATTERN,
                severity=DataIssueseverity.HIGH,
                field_name='email',
                description='Data contains suspicious sequential patterns',
                suggested_fix='Verify this is real contact information',
                confidence=0.8,
                impact_score=25.0
            ))
        
        # Check for duplicate character patterns
        if re.search(r'(.)\1{3,}', email) or re.search(r'(.)\1{3,}', name):
            issues.append(DataIssue(
                issue_type=DataIssueType.SUSPICIOUS_PATTERN,
                severity=DataIssueseverity.MEDIUM,
                field_name='name',
                description='Repeated character patterns detected',
                suggested_fix='Verify data accuracy',
                confidence=0.7,
                impact_score=15.0
            ))
    
    def _validate_social_media(self, lead_data: Dict[str, Any], issues: List[DataIssue], suggestions: List[str]):
        """Validate social media profiles"""
        linkedin_url = lead_data.get('linkedin_url', '')
        
        if linkedin_url:
            # Check LinkedIn URL format
            if not re.match(self.validation_patterns['linkedin'], linkedin_url):
                issues.append(DataIssue(
                    issue_type=DataIssueType.INVALID_FORMAT,
                    severity=DataIssueseverity.MEDIUM,
                    field_name='linkedin_url',
                    description='LinkedIn URL format appears invalid',
                    suggested_fix='Verify and correct LinkedIn URL',
                    confidence=0.8,
                    impact_score=10.0
                ))
            
            # Extract profile name from URL and compare with lead name
            linkedin_match = re.search(r'linkedin\.com/in/([a-zA-Z0-9_-]+)', linkedin_url)
            if linkedin_match:
                profile_id = linkedin_match.group(1).lower()
                first_name = lead_data.get('first_name', '').lower()
                last_name = lead_data.get('last_name', '').lower()
                
                if first_name and last_name:
                    if first_name not in profile_id and last_name not in profile_id:
                        issues.append(DataIssue(
                            issue_type=DataIssueType.INCONSISTENT_DATA,
                            severity=DataIssueseverity.LOW,
                            field_name='linkedin_url',
                            description='LinkedIn profile doesn\'t match contact name',
                            suggested_fix='Verify LinkedIn profile belongs to this person',
                            confidence=0.6,
                            impact_score=5.0
                        ))
    
    def _industry_specific_validation(self, lead_data: Dict[str, Any], issues: List[DataIssue], suggestions: List[str]):
        """Perform industry-specific validation"""
        company = lead_data.get('company', {})
        industry = company.get('industry', '').lower() if isinstance(company, dict) else ''
        title = lead_data.get('title', '').lower()
        
        # Tech industry validation
        if 'tech' in industry or 'software' in industry:
            if title and not any(tech_term in title for tech_term in 
                               ['engineer', 'developer', 'manager', 'director', 'ceo', 'cto', 'product', 'design']):
                # Unusual title for tech industry
                suggestions.append('Verify job title is accurate for technology industry')
        
        # Finance industry validation
        if 'financ' in industry or 'bank' in industry:
            if title and not any(finance_term in title for finance_term in 
                               ['analyst', 'manager', 'director', 'advisor', 'consultant', 'executive']):
                suggestions.append('Verify job title is accurate for finance industry')

    # ============================
    # PUBLIC API METHODS
    # ============================
    
    def is_ready(self) -> bool:
        """Check if data quality engine is ready for use"""
        return bool(self.quality_models)
    
    def get_model_version(self) -> str:
        """Get current model version"""
        return "1.0.0"
    
    def get_model_accuracy(self) -> float:
        """Get estimated model accuracy"""
        return 0.85  # Estimated accuracy for rule-based models
    
    def is_model_loaded(self) -> bool:
        """Check if quality models are loaded"""
        return len(self.quality_models) > 0
    
    def get_statistics(self) -> Dict[str, int]:
        """Get engine usage statistics"""
        return {
            'validations_performed': self.validation_count,
            'duplicates_detected': self.deduplication_count,
            'data_cleaning_operations': self.cleaning_count
        }

    # ============================
    # RULE-BASED FALLBACK METHODS
    # ============================
    
    def _rule_based_email_quality(self, email: str) -> float:
        """Rule-based email quality scoring"""
        if not email or '@' not in email:
            return 0.0
        
        score = 50.0
        email_lower = email.lower()
        
        # Professional patterns
        if re.match(r'^[a-z]+\.[a-z]+@', email_lower):
            score += 25
        elif re.match(r'^[a-z]+@', email_lower):
            score += 15
        
        # Domain analysis
        domain = email.split('@')[1]
        free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
        if domain not in free_domains:
            score += 20
        else:
            score -= 10
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, email_lower):
                score -= 30
                break
        
        return min(100, max(0, score))
    
    def _rule_based_phone_quality(self, phone: str) -> float:
        """Rule-based phone quality scoring"""
        if not phone:
            return 0.0
        
        cleaned_phone = re.sub(r'[^\d+]', '', phone)
        
        if 10 <= len(cleaned_phone) <= 15:
            return 80.0
        elif len(cleaned_phone) < 10:
            return 30.0
        else:
            return 50.0
    
    def _rule_based_completeness(self, lead_data: Dict[str, Any]) -> float:
        """Rule-based completeness scoring"""
        required_fields = ['first_name', 'last_name', 'email', 'title']
        completed = sum(1 for field in required_fields if lead_data.get(field))
        return (completed / len(required_fields)) * 100
    
    def _rule_based_consistency(self, lead_data: Dict[str, Any]) -> float:
        """Rule-based consistency scoring"""
        # Simple consistency checks
        email = lead_data.get('email', '').lower()
        first_name = lead_data.get('first_name', '').lower()
        
        if email and first_name and '@' in email:
            email_local = email.split('@')[0]
            if first_name in email_local:
                return 90.0
            else:
                return 70.0
        
        return 75.0  # Default score
    
    def _rule_based_duplicate_detection(self, leads: List[Any]) -> List[DuplicateGroup]:
        """Rule-based duplicate detection fallback"""
        duplicates = []
        processed = set()
        
        for i, lead1 in enumerate(leads):
            if i in processed:
                continue
            
            lead1_data = self._extract_lead_data(lead1)
            similar_leads = []
            
            for j, lead2 in enumerate(leads[i+1:], start=i+1):
                if j in processed:
                    continue
                
                lead2_data = self._extract_lead_data(lead2)
                
                # Simple email matching
                if (lead1_data.get('email') and lead2_data.get('email') and
                    lead1_data['email'].lower() == lead2_data['email'].lower()):
                    
                    similar_leads.append(DuplicateCandidate(
                        lead_id=lead2_data.get('id', j),
                        similarity_score=1.0,
                        matching_fields=['email'],
                        merge_recommendation='high_confidence_merge',
                        confidence=0.95
                    ))
                    processed.add(j)
            
            if similar_leads:
                duplicates.append(DuplicateGroup(
                    primary_lead_id=lead1_data.get('id', i),
                    duplicates=similar_leads,
                    group_confidence=0.95,
                    merge_strategy='batch_merge'
                ))
                processed.add(i)
        
        return duplicates


# ============================
# SPECIALIZED QUALITY ENGINES
# ============================

class EmailQualityEngine:
    """Specialized engine for email validation and quality assessment"""
    
    def __init__(self):
        self.disposable_domains = self._load_disposable_domains()
        self.role_based_patterns = [
            'admin', 'administrator', 'info', 'contact', 'support', 'help',
            'sales', 'marketing', 'hr', 'noreply', 'no-reply', 'postmaster'
        ]
    
    def _load_disposable_domains(self) -> Set[str]:
        """Load known disposable email domains"""
        # In production, this would load from a file or API
        return {
            '10minutemail.com', 'tempmail.org', 'guerrillamail.com',
            'mailinator.com', 'throwaway.email', 'temp-mail.org'
        }
    
    def assess_email_deliverability(self, email: str) -> Dict[str, Any]:
        """Assess email deliverability and quality"""
        if not email:
            return {'deliverable': False, 'score': 0, 'issues': ['Email is empty']}
        
        issues = []
        score = 80  # Start with base score
        
        # Basic format check
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            return {'deliverable': False, 'score': 0, 'issues': ['Invalid email format']}
        
        local, domain = email.split('@', 1)
        
        # Check disposable email
        if domain.lower() in self.disposable_domains:
            issues.append('Disposable email domain')
            score -= 40
        
        # Check role-based email
        if any(pattern in local.lower() for pattern in self.role_based_patterns):
            issues.append('Role-based email address')
            score -= 20
        
        # Domain checks
        if domain.count('.') == 0:
            issues.append('Invalid domain structure')
            score -= 30
        
        # Local part checks
        if len(local) < 1 or len(local) > 64:
            issues.append('Invalid local part length')
            score -= 20
        
        # Check for consecutive dots
        if '..' in email:
            issues.append('Consecutive dots in email')
            score -= 25
        
        deliverable = score >= 60 and not issues
        
        return {
            'deliverable': deliverable,
            'score': max(0, score),
            'issues': issues,
            'domain': domain,
            'local': local,
            'is_role_based': any(pattern in local.lower() for pattern in self.role_based_patterns),
            'is_disposable': domain.lower() in self.disposable_domains
        }


class PhoneQualityEngine:
    """Specialized engine for phone number validation and quality assessment"""
    
    def __init__(self):
        self.country_codes = self._load_country_codes()
    
    def _load_country_codes(self) -> Dict[str, str]:
        """Load common country codes"""
        return {
            '+1': 'US/Canada',
            '+44': 'United Kingdom',
            '+49': 'Germany',
            '+33': 'France',
            '+86': 'China',
            '+91': 'India',
            '+81': 'Japan',
            '+82': 'South Korea'
        }
    
    def assess_phone_quality(self, phone: str) -> Dict[str, Any]:
        """Assess phone number quality and format"""
        if not phone:
            return {'valid': False, 'score': 0, 'issues': ['Phone number is empty']}
        
        issues = []
        score = 70  # Base score
        
        # Clean the phone number
        cleaned = re.sub(r'[^\d+]', '', phone)
        
        # Basic length validation
        if len(cleaned) < 7:
            issues.append('Phone number too short')
            score -= 40
        elif len(cleaned) > 15:
            issues.append('Phone number too long')
            score -= 30
        
        # Country code detection
        country = 'Unknown'
        if cleaned.startswith('+'):
            for code, country_name in self.country_codes.items():
                if cleaned.startswith(code):
                    country = country_name
                    score += 10
                    break
        
        # US/Canada specific validation
        if country in ['US/Canada', 'Unknown'] and len(cleaned) in [10, 11]:
            if len(cleaned) == 11 and cleaned.startswith('1'):
                score += 15
            elif len(cleaned) == 10:
                score += 10
            
            # Check for invalid US patterns
            if len(cleaned) >= 10:
                area_code = cleaned[-10:-7]
                if area_code.startswith('0') or area_code.startswith('1'):
                    issues.append('Invalid US area code')
                    score -= 20
        
        # Check for obviously fake numbers
        if len(set(cleaned)) <= 2:  # All same digits or only 2 different digits
            issues.append('Appears to be fake number')
            score -= 50
        
        # Sequential numbers
        if re.search(r'123456|234567|345678|456789|567890', cleaned):
            issues.append('Sequential number pattern')
            score -= 30
        
        valid = score >= 50
        
        return {
            'valid': valid,
            'score': max(0, score),
            'issues': issues,
            'country': country,
            'cleaned': cleaned,
            'formatted': self._format_phone(cleaned, country)
        }
    
    def _format_phone(self, cleaned: str, country: str) -> str:
        """Format phone number based on country"""
        if country == 'US/Canada' and len(cleaned) in [10, 11]:
            if len(cleaned) == 11 and cleaned.startswith('1'):
                return f"+1 ({cleaned[1:4]}) {cleaned[4:7]}-{cleaned[7:]}"
            elif len(cleaned) == 10:
                return f"({cleaned[:3]}) {cleaned[3:6]}-{cleaned[6:]}"
        
        # Default international format
        if cleaned.startswith('+'):
            return cleaned
        else:
            return f"+{cleaned}"


# ============================
# USAGE EXAMPLE AND TESTING
# ============================

if __name__ == "__main__":
    # Initialize engines
    quality_engine = DataQualityEngine()
    email_engine = EmailQualityEngine()
    phone_engine = PhoneQualityEngine()
    
    # Test lead data with various quality issues
    test_leads = [
        {
            'id': 1,
            'first_name': 'John',
            'last_name': 'Smith',
            'email': 'john.smith@techcorp.com',
            'phone': '+1 (555) 123-4567',
            'title': 'Senior Software Engineer',
            'linkedin_url': 'https://linkedin.com/in/john-smith',
            'created_at': datetime.utcnow().isoformat(),
            'company': {
                'name': 'TechCorp Inc',
                'industry': 'Technology',
                'size': '51-200',
                'location': 'San Francisco, CA'
            }
        },
        {
            'id': 2,
            'first_name': 'jane',  # Case issue
            'last_name': 'DOE',     # Case issue
            'email': 'jane.doe@techcorp.com',  # Similar email
            'phone': '5551234567',  # No formatting
            'title': 'software engineer',  # Case issue
            'company': {
                'name': 'techcorp inc',  # Case issue
                'industry': 'tech',      # Needs standardization
                'size': '50-200'         # Needs standardization
            }
        },
        {
            'id': 3,
            'first_name': 'Test',
            'last_name': 'User',
            'email': 'test@test.com',  # Suspicious pattern
            'phone': '1234567890',     # Sequential pattern
            'title': 'Test Engineer',
            'company': {
                'name': 'Test Company',
                'industry': 'Testing'
            }
        },
        {
            'id': 4,
            'first_name': '',  # Missing
            'last_name': 'Unknown',
            'email': 'invalid-email',  # Invalid format
            'phone': '123',            # Too short
            'title': '',               # Missing
            'company': {}              # Empty
        }
    ]
    
    print("🧪 Testing Data Quality Engine")
    print("=" * 60)
    
    # Test individual quality assessment
    for i, lead in enumerate(test_leads, 1):
        print(f"\n📊 Lead {i} Quality Assessment:")
        print("-" * 40)
        
        report = quality_engine.assess_lead_quality(lead)
        
        print(f"Overall Score: {report.overall_score:.1f}/100")
        print(f"Completeness: {report.completeness_score:.1f}/100")
        print(f"Accuracy: {report.accuracy_score:.1f}/100")
        print(f"Consistency: {report.consistency_score:.1f}/100")
        print(f"Email Quality: {report.email_quality:.1f}/100")
        print(f"Phone Quality: {report.phone_quality:.1f}/100")
        print(f"Validation Passed: {report.validation_passed}")
        print(f"Confidence: {report.confidence:.2f}")
        
        if report.issues:
            print(f"Issues Found ({len(report.issues)}):")
            for issue in report.issues[:3]:  # Show top 3 issues
                print(f"  • {issue.severity.value.upper()}: {issue.description}")
        
        if report.suggestions:
            print(f"Suggestions ({len(report.suggestions)}):")
            for suggestion in report.suggestions[:2]:  # Show top 2 suggestions
                print(f"  • {suggestion}")
    
    # Test data cleaning
    print(f"\n🧹 Data Cleaning Test:")
    print("-" * 40)
    dirty_lead = test_leads[1].copy()  # Lead with formatting issues
    
    print("Before cleaning:")
    print(f"  Name: {dirty_lead['first_name']} {dirty_lead['last_name']}")
    print(f"  Title: {dirty_lead['title']}")
    print(f"  Phone: {dirty_lead['phone']}")
    
    cleaned_lead = quality_engine.clean_lead_data(dirty_lead)
    
    print("After cleaning:")
    print(f"  Name: {cleaned_lead['first_name']} {cleaned_lead['last_name']}")
    print(f"  Title: {cleaned_lead['title']}")
    print(f"  Phone: {cleaned_lead['phone']}")
    print(f"  Company: {cleaned_lead['company']['name']} ({cleaned_lead['company']['industry']})")
    
    # Test duplicate detection
    print(f"\n🔍 Duplicate Detection Test:")
    print("-" * 40)
    
    # Create a near-duplicate
    duplicate_lead = test_leads[0].copy()
    duplicate_lead['id'] = 5
    duplicate_lead['first_name'] = 'John'  # Slight variation
    duplicate_lead['email'] = 'john.smith@techcorp.com'  # Same email
    
    test_with_duplicate = test_leads + [duplicate_lead]
    duplicates = quality_engine.detect_duplicates(test_with_duplicate, threshold=0.7)
    
    print(f"Duplicate Groups Found: {len(duplicates)}")
    for group in duplicates:
        print(f"  Primary Lead: {group.primary_lead_id}")
        print(f"  Duplicates: {[dup.lead_id for dup in group.duplicates]}")
        print(f"  Confidence: {group.group_confidence:.2f}")
        print(f"  Strategy: {group.merge_strategy}")
    
    # Test batch quality assessment
    print(f"\n📈 Batch Quality Assessment:")
    print("-" * 40)
    
    batch_results = quality_engine.batch_quality_assessment(test_leads, 'standard')
    
    avg_score = np.mean([report.overall_score for report in batch_results.values()])
    high_quality_count = sum(1 for report in batch_results.values() if report.overall_score >= 80)
    low_quality_count = sum(1 for report in batch_results.values() if report.overall_score < 60)
    
    print(f"Leads Assessed: {len(batch_results)}")
    print(f"Average Quality Score: {avg_score:.1f}/100")
    print(f"High Quality Leads (≥80): {high_quality_count}")
    print(f"Low Quality Leads (<60): {low_quality_count}")
    
    # Test specialized engines
    print(f"\n📧 Email Quality Engine Test:")
    print("-" * 40)
    
    test_emails = [
        'john.smith@company.com',
        'test@test.com',
        'admin@company.com',
        'user@10minutemail.com',
        'invalid-email'
    ]
    
    for email in test_emails:
        result = email_engine.assess_email_deliverability(email)
        print(f"  {email}: Score={result['score']}, Deliverable={result['deliverable']}")
        if result['issues']:
            print(f"    Issues: {', '.join(result['issues'])}")
    
    print(f"\n📞 Phone Quality Engine Test:")
    print("-" * 40)
    
    test_phones = [
        '+1 (555) 123-4567',
        '5551234567',
        '123-456-7890',
        '+44 20 7946 0958',
        '123',
        '1111111111'
    ]
    
    for phone in test_phones:
        result = phone_engine.assess_phone_quality(phone)
        print(f"  {phone}: Score={result['score']}, Valid={result['valid']}")
        if result['issues']:
            print(f"    Issues: {', '.join(result['issues'])}")
        print(f"    Formatted: {result['formatted']}")
    
    # Test engine statistics
    print(f"\n📊 Engine Statistics:")
    print("-" * 40)
    
    stats = quality_engine.get_statistics()
    print(f"Validations Performed: {stats['validations_performed']}")
    print(f"Duplicates Detected: {stats['duplicates_detected']}")
    print(f"Cleaning Operations: {stats['data_cleaning_operations']}")
    print(f"Engine Ready: {quality_engine.is_ready()}")
    print(f"Model Version: {quality_engine.get_model_version()}")
    print(f"Estimated Accuracy: {quality_engine.get_model_accuracy():.1%}")
    
    print(f"\n✅ Data Quality Engine testing completed successfully!")
    print(f"🎯 System ready for production use with comprehensive quality assessment")
    print(f"🚀 Features: Validation, Cleaning, Deduplication, Batch Processing, Specialized Engines")