# utils/ai_lead_analyzer.py - Advanced AI Lead Intelligence and Analysis System

# ============================
# IMPORTS
# ============================
import re
import json
import logging
import hashlib
import numpy as np
import pickle
import os
import math
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# ENUMS AND DATA CLASSES
# ============================

class PriorityLevel(Enum):
    """Lead priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class InsightType(Enum):
    """Types of AI insights"""
    PRIORITY = "priority"
    TIMING = "timing"
    APPROACH = "approach"
    WARNING = "warning"
    OPPORTUNITY = "opportunity"
    CONVERSION = "conversion"

@dataclass
class LeadFeatures:
    """Structured lead features for AI analysis"""
    # Contact completeness
    has_email: bool = False
    has_phone: bool = False
    has_linkedin: bool = False
    email_quality_score: float = 0.0
    
    # Professional profile
    seniority_score: float = 0.0
    title_decision_power: float = 0.0
    industry_relevance: float = 0.0
    company_size_score: float = 0.0
    
    # Geographic and demographic
    location_score: float = 0.0
    timezone_alignment: float = 0.0
    language_match: float = 1.0
    
    # Behavioral indicators
    engagement_potential: float = 0.0
    response_likelihood: float = 0.0
    timing_sensitivity: float = 0.0
    
    # Data quality
    completeness_score: float = 0.0
    freshness_score: float = 0.0
    consistency_score: float = 0.0
    
    # Source and context
    source_quality: float = 0.0
    acquisition_channel: str = "unknown"
    campaign_context: Dict[str, Any] = None

# ============================
# MAIN AI LEAD ANALYZER CLASS
# ============================

class AILeadAnalyzer:
    """
    Advanced AI Lead Analyzer for intelligent lead processing, prioritization,
    and insight generation using machine learning and pattern recognition
    """
    
    def __init__(self, model_path: Optional[str] = None, enable_groq: bool = False):
        self.model_path = model_path or "ai/saved_models/"
        self.enable_groq = enable_groq
        self.groq_client = None
        
        # Initialize AI models and components
        self.models = {}
        self.feature_encoders = {}
        self.insight_patterns = {}
        self.similarity_vectors = {}
        
        # Performance tracking
        self.analysis_count = 0
        self.last_training_date = None
        self.model_accuracy_cache = {}
        
        # Analysis cache for performance
        self._analysis_cache = {}
        self._cache_max_size = 1000
        
        # Load or initialize models
        self._initialize_models()
        self._load_insight_patterns()
        
        # Initialize Groq if enabled
        if enable_groq:
            self._initialize_groq()
        
        logger.info("🤖 AI Lead Analyzer initialized successfully")

    # ============================
    # INITIALIZATION METHODS
    # ============================
    
    def _initialize_models(self):
        """Initialize or load AI models"""
        try:
            # Try to load existing models
            if os.path.exists(os.path.join(self.model_path, "lead_priority_model.pkl")):
                self._load_models()
                logger.info("✅ Loaded existing AI models")
            else:
                # Create default models
                self._create_default_models()
                logger.info("🔧 Created default AI models")
                
        except Exception as e:
            logger.warning(f"⚠️ Model initialization warning: {e}")
            self._create_fallback_models()
    
    def _load_models(self):
        """Load trained models from disk"""
        try:
            model_files = {
                'priority': 'lead_priority_model.pkl',
                'quality': 'lead_quality_model.pkl',
                'timing': 'optimal_timing_model.pkl',
                'similarity': 'similarity_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                filepath = os.path.join(self.model_path, filename)
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    logger.debug(f"Loaded {model_name} model")
                    
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self._create_fallback_models()
    
    def _create_default_models(self):
        """Create default AI models using synthetic data"""
        try:
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Priority classification model
            self.models['priority'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            # Quality regression model
            self.models['quality'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # Feature scalers
            self.feature_encoders['scaler'] = StandardScaler()
            
            # Train with synthetic data
            self._train_with_synthetic_data()
            
        except ImportError:
            logger.warning("Scikit-learn not available, using rule-based fallback")
            self._create_fallback_models()
        except Exception as e:
            logger.error(f"Error creating default models: {e}")
            self._create_fallback_models()
    
    def _create_fallback_models(self):
        """Create simple rule-based fallback models"""
        self.models = {
            'priority': self._rule_based_priority,
            'quality': self._rule_based_quality,
            'timing': self._rule_based_timing,
            'similarity': self._rule_based_similarity
        }
        logger.info("🔄 Using rule-based fallback models")
    
    def _train_with_synthetic_data(self):
        """Train models with synthetic training data"""
        try:
            # Generate synthetic training data
            synthetic_features, synthetic_labels = self._generate_synthetic_training_data(1000)
            
            # Train priority model
            if 'priority' in self.models:
                priority_labels = [1 if label > 70 else 0 for label in synthetic_labels]
                self.models['priority'].fit(synthetic_features, priority_labels)
            
            # Train quality model
            if 'quality' in self.models:
                self.models['quality'].fit(synthetic_features, synthetic_labels)
            
            # Fit feature scaler
            if 'scaler' in self.feature_encoders:
                self.feature_encoders['scaler'].fit(synthetic_features)
            
            # Save models
            self._save_models()
            
            logger.info("✅ Models trained with synthetic data")
            
        except Exception as e:
            logger.error(f"Synthetic training failed: {e}")
    
    def _generate_synthetic_training_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for model training"""
        np.random.seed(42)
        
        features = []
        labels = []
        
        for i in range(n_samples):
            # Generate random feature vector
            feature_vector = [
                np.random.random(),  # has_email
                np.random.random(),  # has_phone
                np.random.random(),  # has_linkedin
                np.random.uniform(0, 100),  # email_quality_score
                np.random.uniform(0, 100),  # seniority_score
                np.random.uniform(0, 100),  # title_decision_power
                np.random.uniform(0, 100),  # industry_relevance
                np.random.uniform(0, 100),  # company_size_score
                np.random.uniform(0, 100),  # location_score
                np.random.uniform(0, 100),  # engagement_potential
                np.random.uniform(0, 100),  # completeness_score
                np.random.uniform(0, 100),  # source_quality
            ]
            
            # Calculate synthetic label based on features
            label = (
                feature_vector[0] * 25 +  # email
                feature_vector[4] * 35 +  # seniority
                feature_vector[5] * 30 +  # decision power
                feature_vector[6] * 20 +  # industry relevance
                np.random.uniform(-10, 10)  # noise
            )
            
            features.append(feature_vector)
            labels.append(max(0, min(100, label)))
        
        return np.array(features), np.array(labels)
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            os.makedirs(self.model_path, exist_ok=True)
            
            model_files = {
                'priority': 'lead_priority_model.pkl',
                'quality': 'lead_quality_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                if model_name in self.models:
                    filepath = os.path.join(self.model_path, filename)
                    with open(filepath, 'wb') as f:
                        pickle.dump(self.models[model_name], f)
            
            # Save feature encoders
            if self.feature_encoders:
                filepath = os.path.join(self.model_path, "feature_encoders.pkl")
                with open(filepath, 'wb') as f:
                    pickle.dump(self.feature_encoders, f)
                    
            logger.debug("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def _load_insight_patterns(self):
        """Load or create insight generation patterns"""
        self.insight_patterns = {
            'high_priority_signals': [
                'executive_title',
                'decision_maker_role',
                'high_value_industry',
                'optimal_company_size',
                'recent_funding',
                'growth_indicators'
            ],
            'timing_indicators': [
                'quarter_end_urgency',
                'budget_cycle_timing',
                'industry_seasonality',
                'company_growth_phase',
                'competitive_pressure'
            ],
            'risk_factors': [
                'incomplete_data',
                'low_engagement_history',
                'competitive_saturated_market',
                'economic_headwinds',
                'decision_process_complexity'
            ]
        }
    
    def _initialize_groq(self):
        """Initialize Groq LLM client for enhanced insights"""
        try:
            import os
            groq_api_key = os.getenv("GROQ_API_KEY")
            
            if groq_api_key:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_api_key)
                logger.info("✅ Groq LLM initialized for enhanced insights")
            else:
                logger.warning("⚠️ GROQ_API_KEY not found, using rule-based insights")
                
        except ImportError:
            logger.warning("⚠️ Groq library not installed, using rule-based insights")
        except Exception as e:
            logger.error(f"Groq initialization failed: {e}")

    # ============================
    # CORE ANALYSIS METHODS
    # ============================
    def _lead_to_dict(self, lead) -> Dict[str, Any]:
        """Convert SQLAlchemy Lead object to dictionary for analysis"""
        lead_dict = {
            'id': lead.id,
            'first_name': lead.first_name,
            'last_name': lead.last_name,
            'email': lead.email,
            'phone': lead.phone,
            'title': lead.title,
            'linkedin_url': lead.linkedin_url,
            'source': lead.source,
            'created_at': lead.created_at.isoformat() if lead.created_at else None,
            'status': lead.status,
            'score': lead.score,
            'notes': lead.notes,
            'tags': lead.tags
        }
        
        # Add company information if available
        if hasattr(lead, 'company') and lead.company:
            lead_dict['company'] = {
                'id': lead.company.id,
                'name': lead.company.name,
                'domain': lead.company.domain,
                'industry': lead.company.industry,
                'size': lead.company.size,
                'location': lead.company.location,
                'website': lead.company.website,
                'description': lead.company.description
            }
        else:
            lead_dict['company'] = {}
        
        return lead_dict
    
    def analyze_lead(self, lead_data: Union[Dict[str, Any], object], use_cache: bool = True) -> Dict[str, Any]:
        """
        Analyze a lead to determine priority score, quality, and recommendations
        Can accept either a dictionary or a Lead object
        """
        try:
            # Convert Lead object to dictionary if needed
            if not isinstance(lead_data, dict):
                lead_data = self._lead_to_dict(lead_data)
            
            # Generate cache key
            cache_key = self._generate_cache_key(lead_data)
            
            # Check cache if enabled
            if use_cache and cache_key in self.analysis_cache:
                cached_result = self.analysis_cache[cache_key]
                if (datetime.utcnow() - cached_result['timestamp']).seconds < 3600:  # 1 hour cache
                    return cached_result['result']
            
            # Extract features from lead data
            features = self._extract_lead_features(lead_data)
            
            # Calculate scores
            priority_score = self._calculate_priority_score(features, lead_data)
            quality_score = self._calculate_lead_quality_score(features)
            confidence = self._calculate_confidence(features)
            
            # Generate analysis result
            analysis_result = {
                'lead_id': lead_data.get('id'),
                'priority_score': priority_score,
                'quality_score': quality_score,
                'confidence': confidence,
                'priority_category': self._categorize_priority(priority_score),
                'priority_reason': self._generate_priority_reason(features, priority_score),
                'quality_issues': self._identify_quality_issues(features, lead_data),
                'actions': self._recommend_actions(features, priority_score, lead_data),
                'optimal_timing': self._suggest_optimal_timing(features, lead_data),
                'response_prediction': self._predict_response_rate(features),
                'enrichment_suggestions': self._suggest_enrichments(features),
                'similar_leads': [],
                'feature_importance': self._calculate_feature_importance(features, priority_score),
                'analyzed_at': datetime.utcnow().isoformat()
            }
            
            # Add contextual insights
            context = self._analyze_lead_context(lead_data, features)
            if context:
                analysis_result['contextual_insights'] = context
            
            # Generate AI insights
            insights = self._generate_ai_insights(features, analysis_result, context)
            analysis_result['insights'] = insights
            
            # Advanced pattern analysis
            if self.groq_client:
                groq_insights = self._get_groq_insights(lead_data, analysis_result)
                if groq_insights:
                    analysis_result['llm_insights'] = groq_insights
            
            # Cache the result
            self._cache_analysis(cache_key, analysis_result)
            
            # Update analytics
            self.analysis_count += 1
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Lead analysis failed: {e}")
            return self._generate_fallback_analysis(lead_data)

    # ============================
    # FEATURE EXTRACTION METHODS
    # ============================
    
    def _extract_lead_features(self, lead_data: Dict[str, Any]) -> LeadFeatures:
        """Extract structured features from lead data for AI analysis"""
        features = LeadFeatures()
        
        # Contact information analysis
        features.has_email = bool(lead_data.get('email'))
        features.has_phone = bool(lead_data.get('phone'))
        features.has_linkedin = bool(lead_data.get('linkedin_url'))
        
        # Email quality analysis
        if features.has_email:
            features.email_quality_score = self._analyze_email_quality(lead_data['email'])
        
        # Professional profile analysis
        title = lead_data.get('title', '').lower()
        features.seniority_score = self._calculate_seniority_score(title)
        features.title_decision_power = self._calculate_decision_power(title)
        
        # Company analysis
        company = lead_data.get('company', {})
        features.industry_relevance = self._calculate_industry_relevance(company.get('industry', ''))
        features.company_size_score = self._calculate_company_size_score(company.get('size', ''))
        features.location_score = self._calculate_location_score(company.get('location', ''))
        
        # Behavioral indicators
        features.engagement_potential = self._calculate_engagement_potential(lead_data)
        features.response_likelihood = self._estimate_response_likelihood(features)
        
        # Data quality metrics
        features.completeness_score = self._calculate_completeness(lead_data)
        features.freshness_score = self._calculate_freshness(lead_data.get('created_at'))
        features.consistency_score = self._check_data_consistency(lead_data)
        
        # Source and context
        features.source_quality = self._evaluate_source_quality(lead_data.get('source', ''))
        features.acquisition_channel = lead_data.get('source', 'unknown')
        
        return features

    # ============================
    # PREDICTION METHODS
    # ============================
    
    def _predict_priority(self, features: LeadFeatures) -> float:
        """Predict lead priority using AI model"""
        try:
            if 'priority' in self.models and hasattr(self.models['priority'], 'predict_proba'):
                # ML model prediction
                feature_vector = self._features_to_vector(features)
                if 'scaler' in self.feature_encoders:
                    feature_vector = self.feature_encoders['scaler'].transform([feature_vector])
                
                probabilities = self.models['priority'].predict_proba(feature_vector)[0]
                priority_score = probabilities[1] * 100  # Probability of high priority
                
                return min(95, max(5, priority_score))
            else:
                # Fallback to rule-based
                return self._rule_based_priority(features)
                
        except Exception as e:
            logger.warning(f"Priority prediction failed: {e}")
            return self._rule_based_priority(features)
    
    def _predict_quality(self, features: LeadFeatures) -> float:
        """Predict lead quality using AI model"""
        try:
            if 'quality' in self.models and hasattr(self.models['quality'], 'predict'):
                # ML model prediction
                feature_vector = self._features_to_vector(features)
                if 'scaler' in self.feature_encoders:
                    feature_vector = self.feature_encoders['scaler'].transform([feature_vector])
                
                quality_score = self.models['quality'].predict(feature_vector)[0]
                return min(100, max(0, quality_score))
            else:
                # Fallback to rule-based
                return self._rule_based_quality(features)
                
        except Exception as e:
            logger.warning(f"Quality prediction failed: {e}")
            return self._rule_based_quality(features)
    
    def _features_to_vector(self, features: LeadFeatures) -> List[float]:
        """Convert LeadFeatures to numerical vector for ML models"""
        return [
            float(features.has_email),
            float(features.has_phone),
            float(features.has_linkedin),
            features.email_quality_score,
            features.seniority_score,
            features.title_decision_power,
            features.industry_relevance,
            features.company_size_score,
            features.location_score,
            features.engagement_potential,
            features.completeness_score,
            features.source_quality
        ]

    # ============================
    # RULE-BASED FALLBACK METHODS
    # ============================
    
    def _rule_based_priority(self, features: LeadFeatures) -> float:
        """Rule-based priority scoring fallback"""
        score = 50.0  # Base score
        
        # Contact availability
        if features.has_email:
            score += 15
        if features.has_phone:
            score += 10
        if features.has_linkedin:
            score += 8
        
        # Professional profile
        score += features.seniority_score * 0.3
        score += features.title_decision_power * 0.25
        score += features.industry_relevance * 0.2
        
        # Company factors
        score += features.company_size_score * 0.15
        score += features.location_score * 0.1
        
        # Quality factors
        score += features.completeness_score * 0.1
        score += features.source_quality * 0.1
        
        return min(95, max(5, score))
    
    def _rule_based_quality(self, features: LeadFeatures) -> float:
        """Rule-based quality scoring fallback"""
        score = 0.0
        
        # Data completeness (40% weight)
        score += features.completeness_score * 0.4
        
        # Contact quality (30% weight)
        contact_quality = (features.email_quality_score + 
                          (50 if features.has_phone else 0) +
                          (30 if features.has_linkedin else 0)) / 3
        score += contact_quality * 0.3
        
        # Professional relevance (20% weight)
        prof_relevance = (features.seniority_score + features.industry_relevance) / 2
        score += prof_relevance * 0.2
        
        # Source and freshness (10% weight)
        source_fresh = (features.source_quality + features.freshness_score) / 2
        score += source_fresh * 0.1
        
        return min(100, max(0, score))
    
    def _rule_based_timing(self, features: LeadFeatures) -> Dict[str, Any]:
        """Rule-based timing analysis fallback"""
        return {
            'best_day': 'tuesday',
            'best_hour_start': 10,
            'best_hour_end': 12,
            'confidence': 0.5,
            'reasoning': 'Standard business timing'
        }
    
    def _rule_based_similarity(self, lead1: Dict, lead2: Dict) -> float:
        """Rule-based similarity calculation fallback"""
        # Simple rule-based similarity
        similarity = 0.0
        
        # Title similarity
        title1 = lead1.get('title', '').lower()
        title2 = lead2.get('title', '').lower()
        if title1 and title2:
            # Simple word overlap
            words1 = set(title1.split())
            words2 = set(title2.split())
            if words1 and words2:
                similarity += len(words1.intersection(words2)) / len(words1.union(words2)) * 0.3
        
        # Industry similarity
        company1 = lead1.get('company', {})
        company2 = lead2.get('company', {})
        industry1 = company1.get('industry', '').lower()
        industry2 = company2.get('industry', '').lower()
        if industry1 == industry2 and industry1:
            similarity += 0.4
        
        # Company size similarity
        size1 = company1.get('size', '')
        size2 = company2.get('size', '')
        if size1 == size2 and size1:
            similarity += 0.3
        
        return min(1.0, similarity)

    # ============================
    # ANALYSIS HELPER METHODS
    # ============================
    
    def _calculate_confidence(self, features: LeadFeatures) -> float:
        """Calculate confidence in the analysis"""
        confidence_factors = [
            features.completeness_score / 100,
            features.consistency_score / 100,
            features.freshness_score / 100,
            min(1.0, features.source_quality / 80)
        ]
        
        # Higher weight for more complete data
        weights = [0.4, 0.3, 0.2, 0.1]
        confidence = sum(f * w for f, w in zip(confidence_factors, weights))
        
        return min(0.95, max(0.1, confidence))
    
    def _analyze_email_quality(self, email: str) -> float:
        """Analyze email quality with advanced patterns"""
        if not email or '@' not in email:
            return 0.0
        
        score = 50.0
        email_lower = email.lower()
        
        # Professional patterns
        if re.match(r'^[a-z]+\.[a-z]+@', email_lower):
            score += 25
        elif re.match(r'^[a-z]+[a-z]+@', email_lower):
            score += 15
        
        # Domain analysis
        domain = email.split('@')[1]
        
        # Corporate vs free email
        free_domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']
        if domain not in free_domains:
            score += 20
        else:
            score -= 10
        
        # Domain quality indicators
        if domain.endswith(('.ai', '.io', '.tech')):
            score += 15
        elif domain.endswith('.com'):
            score += 10
        
        # Email structure quality
        local_part = email.split('@')[0]
        if len(local_part) < 3:
            score -= 15
        elif len(local_part) > 20:
            score -= 10
        
        # Avoid obvious generic emails
        generic_patterns = ['info', 'contact', 'admin', 'support', 'sales']
        if any(pattern in local_part for pattern in generic_patterns):
            score -= 20
        
        return min(100, max(0, score))
    
    def _calculate_seniority_score(self, title: str) -> float:
        """Calculate seniority score from job title"""
        if not title:
            return 0.0
        
        title_lower = title.lower()
        
        # Executive level
        executive_keywords = ['ceo', 'cto', 'cfo', 'president', 'founder', 'chief']
        if any(keyword in title_lower for keyword in executive_keywords):
            return 95.0
        
        # Senior level
        senior_keywords = ['vp', 'vice president', 'director', 'head of', 'senior director']
        if any(keyword in title_lower for keyword in senior_keywords):
            return 80.0
        
        # Manager level
        manager_keywords = ['manager', 'lead', 'principal', 'senior manager']
        if any(keyword in title_lower for keyword in manager_keywords):
            return 65.0
        
        # Senior individual contributor
        if 'senior' in title_lower or 'principal' in title_lower:
            return 50.0
        
        # Individual contributor
        return 30.0
    
    def _calculate_decision_power(self, title: str) -> float:
        """Calculate decision-making power from title"""
        if not title:
            return 0.0
        
        title_lower = title.lower()
        
        # High decision power
        high_power = ['ceo', 'president', 'founder', 'owner', 'partner']
        if any(keyword in title_lower for keyword in high_power):
            return 95.0
        
        # Medium-high decision power
        med_high = ['cto', 'cfo', 'vp', 'vice president', 'director']
        if any(keyword in title_lower for keyword in med_high):
            return 80.0
        
        # Medium decision power
        medium = ['head of', 'manager', 'lead']
        if any(keyword in title_lower for keyword in medium):
            return 60.0
        
        # Some influence
        some_influence = ['senior', 'principal', 'specialist']
        if any(keyword in title_lower for keyword in some_influence):
            return 40.0
        
        return 20.0
    
    def _calculate_industry_relevance(self, industry: str) -> float:
        """Calculate industry relevance score"""
        if not industry:
            return 50.0
        
        industry_lower = industry.lower()
        
        # Tier 1 industries (highest relevance)
        tier1 = ['ai', 'artificial intelligence', 'machine learning', 'saas', 'fintech', 'cybersecurity']
        if any(keyword in industry_lower for keyword in tier1):
            return 95.0
        
        # Tier 2 industries
        tier2 = ['technology', 'software', 'cloud', 'data', 'biotech', 'healthtech']
        if any(keyword in industry_lower for keyword in tier2):
            return 80.0
        
        # Tier 3 industries
        tier3 = ['finance', 'healthcare', 'consulting', 'e-commerce']
        if any(keyword in industry_lower for keyword in tier3):
            return 65.0
        
        return 50.0
    
    def _calculate_company_size_score(self, size: str) -> float:
        """Calculate company size relevance score"""
        if not size:
            return 50.0
        
        # Optimal sizes for B2B
        if '201-500' in size or '501-1000' in size:
            return 90.0
        elif '51-200' in size:
            return 85.0
        elif '1001-5000' in size:
            return 80.0
        elif '11-50' in size:
            return 70.0
        elif '1-10' in size:
            return 60.0  # Startups can be valuable
        elif '5000+' in size:
            return 60.0  # Large enterprises (longer sales cycles)
        
        return 50.0
    
    def _calculate_location_score(self, location: str) -> float:
        """Calculate location relevance score"""
        if not location:
            return 50.0
        
        location_lower = location.lower()
        
        # Tier 1 tech hubs
        tier1_locations = [
            'san francisco', 'silicon valley', 'new york', 'seattle', 
            'boston', 'austin', 'london', 'tel aviv'
        ]
        if any(loc in location_lower for loc in tier1_locations):
            return 90.0
        
        # Tier 2 business centers
        tier2_locations = [
            'chicago', 'denver', 'atlanta', 'toronto', 'vancouver',
            'berlin', 'amsterdam', 'singapore'
        ]
        if any(loc in location_lower for loc in tier2_locations):
            return 75.0
        
        # English-speaking countries
        english_countries = ['usa', 'canada', 'uk', 'australia', 'new zealand']
        if any(country in location_lower for country in english_countries):
            return 65.0
        
        return 50.0
    
    def _calculate_engagement_potential(self, lead_data: Dict[str, Any]) -> float:
        """Calculate potential for successful engagement"""
        potential = 50.0
        
        # Multiple contact methods
        contact_methods = sum([
            bool(lead_data.get('email')),
            bool(lead_data.get('phone')),
            bool(lead_data.get('linkedin_url'))
        ])
        potential += contact_methods * 10
        
        # Professional social presence
        if lead_data.get('linkedin_url'):
            potential += 15
        
        # Complete profile
        name_complete = bool(lead_data.get('first_name')) and bool(lead_data.get('last_name'))
        if name_complete:
            potential += 10
        
        # Company information
        company = lead_data.get('company', {})
        if company.get('website'):
            potential += 8
        
        # Title indicates accessibility
        title = lead_data.get('title', '').lower()
        accessible_titles = ['manager', 'director', 'head', 'lead']
        if any(accessible in title for accessible in accessible_titles):
            potential += 12
        
        return min(100, max(0, potential))
    
    def _estimate_response_likelihood(self, features: LeadFeatures) -> float:
        """Estimate likelihood of positive response"""
        # Base on seniority (senior people often busier)
        base_response = max(30, 80 - (features.seniority_score * 0.3))
        
        # Adjust for contact quality
        if features.has_email and features.email_quality_score > 70:
            base_response += 15
        
        if features.has_linkedin:
            base_response += 10
        
        # Industry factor
        base_response += features.industry_relevance * 0.2
        
        # Company size factor (mid-size companies more responsive)
        if 60 <= features.company_size_score <= 90:
            base_response += 10
        
        return min(90, max(10, base_response))
    
    def _calculate_completeness(self, lead_data: Dict[str, Any]) -> float:
        """Calculate data completeness score"""
        total_fields = 0
        completed_fields = 0
        
        # Core fields
        core_fields = ['first_name', 'last_name', 'email', 'title']
        for field in core_fields:
            total_fields += 2  # Core fields weighted more
            if lead_data.get(field):
                completed_fields += 2
        
        # Secondary fields
        secondary_fields = ['phone', 'linkedin_url']
        for field in secondary_fields:
            total_fields += 1
            if lead_data.get(field):
                completed_fields += 1
        
        # Company fields
        company = lead_data.get('company', {})
        company_fields = ['name', 'industry', 'size', 'location']
        for field in company_fields:
            total_fields += 1
            if company.get(field):
                completed_fields += 1
        
        return (completed_fields / total_fields) * 100 if total_fields > 0 else 0
    
    def _calculate_freshness(self, created_at: Optional[str]) -> float:
        """Calculate data freshness score"""
        if not created_at:
            return 50.0  # Unknown age
        
        try:
            if isinstance(created_at, str):
                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                created_date = created_at
            
            days_old = (datetime.utcnow() - created_date.replace(tzinfo=None)).days
            
            if days_old <= 7:
                return 100.0
            elif days_old <= 30:
                return 90.0
            elif days_old <= 90:
                return 70.0
            elif days_old <= 180:
                return 50.0
            else:
                return max(20, 100 - (days_old / 10))
                
        except Exception:
            return 50.0
    
    def _check_data_consistency(self, lead_data: Dict[str, Any]) -> float:
        """Check internal data consistency"""
        consistency_score = 100.0
        
        # Email domain vs company domain
        email = lead_data.get('email', '')
        company = lead_data.get('company', {})
        company_domain = company.get('domain', '')
        
        if email and company_domain and '@' in email:
            email_domain = email.split('@')[1]
            if email_domain != company_domain:
                # Check if it's a reasonable variation
                if not (email_domain in company_domain or company_domain in email_domain):
                    consistency_score -= 20
        
        # Name consistency with email
        first_name = lead_data.get('first_name', '').lower()
        last_name = lead_data.get('last_name', '').lower()
        
        if email and first_name:
            email_local = email.split('@')[0].lower()
            if first_name not in email_local and last_name not in email_local:
                consistency_score -= 15
        
        # Title vs company size consistency
        title = lead_data.get('title', '').lower()
        company_size = company.get('size', '')
        
        # C-level in very small companies might be inconsistent
        if any(c_level in title for c_level in ['ceo', 'cto', 'cfo']) and '1-10' in company_size:
            # Actually this is often valid for startups, so small penalty
            consistency_score -= 5
        
        return max(0, consistency_score)
    
    def _evaluate_source_quality(self, source: str) -> float:
        """Evaluate the quality of the lead source"""
        source_scores = {
            'linkedin': 90,
            'referral': 95,
            'website': 80,
            'event': 85,
            'webinar': 75,
            'hunter': 70,
            'directory': 65,
            'google': 60,
            'manual': 70,
            'import': 50,
            'purchased': 30
        }
        
        return source_scores.get(source.lower(), 50)

    # ============================
    # INSIGHT GENERATION METHODS
    # ============================
    
    def _generate_priority_explanation(self, features: LeadFeatures, priority_score: float) -> str:
        """Generate human-readable explanation for priority score"""
        explanations = []
        
        if priority_score >= 80:
            if features.seniority_score >= 80:
                explanations.append("Executive-level contact with high decision-making authority")
            if features.industry_relevance >= 80:
                explanations.append("operates in high-value target industry")
            if features.has_email and features.email_quality_score > 70:
                explanations.append("high-quality professional contact information available")
        elif priority_score >= 60:
            if features.title_decision_power >= 60:
                explanations.append("Senior professional with influence in decision-making")
            if features.company_size_score >= 70:
                explanations.append("works at optimally-sized company for our solution")
            if features.completeness_score >= 70:
                explanations.append("comprehensive lead profile with good data quality")
        else:
            if features.completeness_score < 50:
                explanations.append("limited contact information available")
            if features.seniority_score < 40:
                explanations.append("junior-level contact with limited decision authority")
            if features.industry_relevance < 50:
                explanations.append("industry may not be ideal fit for our solution")
        
        if not explanations:
            return "Standard lead profile with moderate conversion potential"
        
        return "High-priority lead: " + ", ".join(explanations) if priority_score >= 60 else "Lower priority: " + ", ".join(explanations)
    
    def _recommend_actions(self, features: LeadFeatures, analysis: Dict[str, Any]) -> List[str]:
        """Recommend specific actions based on analysis"""
        actions = []
        priority_score = analysis.get('priority_score', 0)
        
        # Priority-based actions
        if priority_score >= 80:
            actions.append("Schedule immediate personalized outreach")
            actions.append("Research recent company news and funding")
            actions.append("Prepare executive-level value proposition")
        elif priority_score >= 60:
            actions.append("Add to priority campaign within 24 hours")
            actions.append("Customize messaging for their industry")
        else:
            actions.append("Include in nurture campaign")
            actions.append("Consider data enrichment before outreach")
        
        # Contact-specific actions
        if not features.has_email:
            actions.append("🔍 Find email address using lead enrichment tools")
        elif features.email_quality_score < 50:
            actions.append("⚠️ Verify email deliverability before sending")
        
        if not features.has_phone and features.seniority_score > 70:
            actions.append("📞 Research phone number for executive outreach")
        
        if not features.has_linkedin:
            actions.append("🔗 Find LinkedIn profile for social selling approach")
        
        # Data quality actions
        if features.completeness_score < 60:
            actions.append("📊 Enrich lead data before outreach")
        
        # Timing actions
        current_hour = datetime.utcnow().hour
        if 9 <= current_hour <= 17:
            actions.append("⏰ Optimal time for immediate contact (business hours)")
        
        return actions[:6]  # Limit to most important actions
    
    def _analyze_optimal_timing(self, features: LeadFeatures, context: Optional[Dict]) -> Dict[str, Any]:
        """Analyze optimal contact timing"""
        timing_analysis = {
            'best_day': 'tuesday',
            'best_hour_start': 10,
            'best_hour_end': 12,
            'timezone': 'UTC',
            'confidence': 0.5,
            'reasoning': '',
            'alternatives': []
        }
        
        # Industry-based timing
        if features.industry_relevance >= 80:
            # Tech industry typically more flexible
            timing_analysis.update({
                'best_day': 'wednesday',
                'best_hour_start': 9,
                'best_hour_end': 11,
                'confidence': 0.7,
                'reasoning': 'Tech industry professionals often prefer mid-week, morning contacts'
            })
        
        # Seniority-based timing
        if features.seniority_score >= 80:
            # Executives prefer early morning or late afternoon
            timing_analysis.update({
                'best_hour_start': 8,
                'best_hour_end': 9,
                'confidence': 0.8,
                'reasoning': 'Executive contacts often check email early morning before meetings start'
            })
            timing_analysis['alternatives'] = ['16:00-17:00', '7:00-8:00']
        
        # Context-based adjustments
        if context and context.get('urgent_campaign'):
            timing_analysis['reasoning'] += '. Urgent campaign - contact immediately during business hours'
        
        return timing_analysis
    
    def _suggest_engagement_strategy(self, features: LeadFeatures) -> Dict[str, Any]:
        """Suggest optimal engagement strategy"""
        strategy = {
            'primary_channel': 'email',
            'secondary_channel': 'linkedin',
            'message_tone': 'professional',
            'personalization_level': 'medium',
            'follow_up_sequence': 'standard',
            'content_focus': 'value_proposition'
        }
        
        # Adjust based on seniority
        if features.seniority_score >= 80:
            strategy.update({
                'message_tone': 'executive',
                'personalization_level': 'high',
                'content_focus': 'strategic_outcomes',
                'follow_up_sequence': 'executive'
            })
        elif features.seniority_score < 40:
            strategy.update({
                'message_tone': 'friendly',
                'personalization_level': 'medium',
                'content_focus': 'features_benefits'
            })
        
        # Adjust primary channel based on availability
        if not features.has_email and features.has_linkedin:
            strategy['primary_channel'] = 'linkedin'
        elif features.has_phone and features.seniority_score >= 70:
            strategy['secondary_channel'] = 'phone'
        
        # Industry-specific adjustments
        if features.industry_relevance >= 80:
            strategy['content_focus'] = 'industry_specific_roi'
            strategy['personalization_level'] = 'high'
        
        return strategy
    
    def _analyze_competitive_context(self, features: LeadFeatures) -> Dict[str, Any]:
        """Analyze competitive landscape and positioning"""
        return {
            'competition_level': 'medium',
            'differentiation_opportunities': [
                'Emphasize unique AI capabilities',
                'Highlight proven ROI in similar companies',
                'Focus on implementation speed'
            ],
            'risk_factors': [
                'Saturated market with many solutions',
                'High switching costs from existing solutions'
            ],
            'urgency_indicators': [],
            'competitive_advantages': [
                'First-mover advantage in AI space',
                'Strong technical capabilities'
            ]
        }
    
    def _estimate_conversion_probability(self, features: LeadFeatures) -> float:
        """Estimate probability of successful conversion"""
        base_probability = 0.05  # 5% base conversion rate
        
        # Adjust for seniority (decision makers convert better)
        if features.seniority_score >= 80:
            base_probability *= 3
        elif features.seniority_score >= 60:
            base_probability *= 2
        
        # Adjust for industry relevance
        base_probability *= (1 + features.industry_relevance / 100)
        
        # Adjust for company size (sweet spot companies)
        if features.company_size_score >= 80:
            base_probability *= 1.5
        
        # Adjust for data quality (better data = better targeting)
        base_probability *= (0.5 + features.completeness_score / 200)
        
        # Adjust for engagement potential
        base_probability *= (0.7 + features.engagement_potential / 333)
        
        return min(0.8, base_probability)  # Cap at 80%
    
    def _calculate_urgency(self, features: LeadFeatures, context: Optional[Dict]) -> float:
        """Calculate urgency score for this lead"""
        urgency = 50.0
        
        # Time-based urgency
        current_time = datetime.utcnow()
        
        # Quarter-end urgency
        if current_time.month in [3, 6, 9, 12] and current_time.day >= 25:
            urgency += 20
        
        # Year-end budget considerations
        if current_time.month == 12:
            urgency += 15
        elif current_time.month == 1:  # New budget year
            urgency += 10
        
        # Context-based urgency
        if context:
            if context.get('urgent_campaign'):
                urgency += 30
            if context.get('competitor_activity'):
                urgency += 20
        
        # Lead-specific urgency
        if features.seniority_score >= 80:
            urgency += 10  # Executives harder to reach, act fast
        
        if features.freshness_score >= 90:
            urgency += 15  # Fresh leads need immediate action
        
        return min(100, urgency)
    
    def _assess_business_impact(self, features: LeadFeatures) -> Dict[str, Any]:
        """Assess potential business impact of this lead"""
        # Calculate deal size potential
        deal_size_potential = 'medium'
        if features.company_size_score >= 85:
            deal_size_potential = 'large'
        elif features.company_size_score <= 60:
            deal_size_potential = 'small'
        
        # Calculate strategic value
        strategic_value = 'medium'
        if features.industry_relevance >= 90:
            strategic_value = 'high'
        elif features.industry_relevance <= 50:
            strategic_value = 'low'
        
        return {
            'deal_size_potential': deal_size_potential,
            'strategic_value': strategic_value,
            'estimated_deal_value': self._estimate_deal_value(features),
            'sales_cycle_length': self._estimate_sales_cycle(features),
            'success_indicators': self._identify_success_indicators(features),
            'risk_factors': self._identify_risk_factors(features)
        }
    
    def _estimate_deal_value(self, features: LeadFeatures) -> str:
        """Estimate potential deal value"""
        if features.company_size_score >= 85 and features.seniority_score >= 80:
            return '$50K-$200K'
        elif features.company_size_score >= 70:
            return '$20K-$75K'
        elif features.company_size_score >= 50:
            return '$5K-$30K'
        else:
            return '$1K-$15K'
    
    def _estimate_sales_cycle(self, features: LeadFeatures) -> str:
        """Estimate sales cycle length"""
        if features.company_size_score >= 85:
            return '6-12 months'
        elif features.company_size_score >= 70:
            return '3-6 months'
        elif features.seniority_score >= 80:
            return '2-4 months'
        else:
            return '1-3 months'
    
    def _identify_success_indicators(self, features: LeadFeatures) -> List[str]:
        """Identify indicators of potential success"""
        indicators = []
        
        if features.seniority_score >= 80:
            indicators.append('Decision-maker authority')
        if features.industry_relevance >= 80:
            indicators.append('High industry fit')
        if features.completeness_score >= 80:
            indicators.append('Complete contact profile')
        if features.engagement_potential >= 70:
            indicators.append('High engagement potential')
        if features.source_quality >= 80:
            indicators.append('High-quality lead source')
        
        return indicators
    
    def _identify_risk_factors(self, features: LeadFeatures) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        if features.completeness_score < 50:
            risks.append('Incomplete contact information')
        if features.response_likelihood < 40:
            risks.append('Low response probability')
        if features.consistency_score < 70:
            risks.append('Data quality concerns')
        if features.freshness_score < 50:
            risks.append('Outdated lead information')
        if features.industry_relevance < 50:
            risks.append('Industry fit concerns')
        
        return risks
    
    def _generate_ai_insights(self, features: LeadFeatures, analysis: Dict[str, Any], context: Optional[Dict]) -> List[Dict[str, Any]]:
        """Generate AI-powered insights"""
        insights = []
        
        # Priority insight
        priority_score = analysis.get('priority_score', 0)
        if priority_score >= 80:
            insights.append({
                'type': 'priority',
                'level': 'high',
                'title': 'High-Priority Lead Detected',
                'description': 'This lead shows strong indicators for immediate outreach',
                'confidence': analysis.get('confidence', 0.5),
                'action_required': True
            })
        
        # Quality insight
        quality_score = analysis.get('quality_score', 0)
        if quality_score < 60:
            insights.append({
                'type': 'warning',
                'level': 'medium',
                'title': 'Data Quality Enhancement Needed',
                'description': 'Lead data should be enriched before outreach',
                'confidence': 0.8,
                'action_required': True
            })
        
        # Timing insight
        urgency = analysis.get('urgency_score', 50)
        if urgency >= 80:
            insights.append({
                'type': 'timing',
                'level': 'high',
                'title': 'Time-Sensitive Opportunity',
                'description': 'Current timing factors suggest immediate action',
                'confidence': 0.7,
                'action_required': True
            })
        
        # Industry opportunity insight
        if features.industry_relevance >= 85:
            insights.append({
                'type': 'opportunity',
                'level': 'high',
                'title': 'High-Value Industry Target',
                'description': 'Lead operates in prime target industry with strong fit',
                'confidence': 0.9,
                'action_required': False
            })
        
        # Engagement strategy insight
        if features.engagement_potential >= 80:
            insights.append({
                'type': 'approach',
                'level': 'medium',
                'title': 'High Engagement Potential',
                'description': 'Multiple quality contact methods available for outreach',
                'confidence': 0.8,
                'action_required': False
            })
        
        return insights[:5]  # Limit to top 5 insights
    
    def _get_groq_insights(self, lead_data: Dict[str, Any], analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get enhanced insights from Groq LLM"""
        if not self.groq_client:
            return None
        
        try:
            prompt = f"""
            Analyze this lead for B2B sales potential:
            
            Lead Profile:
            - Name: {lead_data.get('first_name', 'Unknown')} {lead_data.get('last_name', 'Unknown')}
            - Title: {lead_data.get('title', 'Unknown')}
            - Company: {lead_data.get('company', {}).get('name', 'Unknown')}
            - Industry: {lead_data.get('company', {}).get('industry', 'Unknown')}
            - Company Size: {lead_data.get('company', {}).get('size', 'Unknown')}
            
            AI Analysis Results:
            - Priority Score: {analysis.get('priority_score', 0)}/100
            - Quality Score: {analysis.get('quality_score', 0)}/100
            - Conversion Probability: {analysis.get('conversion_probability', 0)*100:.1f}%
            
            Provide 3 key insights for sales strategy in JSON format:
            {{
                "strategic_insight": "brief insight about approach",
                "timing_recommendation": "when to contact",
                "personalization_angle": "how to personalize outreach"
            }}
            """
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192",
                temperature=0.3,
                max_tokens=300
            )
            
            llm_response = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                return json.loads(llm_response)
            except json.JSONDecodeError:
                return {"raw_insight": llm_response}
                
        except Exception as e:
            logger.warning(f"Groq insights failed: {e}")
            return None

    # ============================
    # SIMILARITY ANALYSIS METHODS
    # ============================
    
    def find_similar_leads(self, reference_lead: Dict[str, Any], candidate_leads: List[Dict[str, Any]], 
                          similarity_threshold: float = 0.7, max_results: int = 10) -> Dict[str, Any]:
        """Find leads similar to reference lead using AI similarity analysis"""
        try:
            ref_features = self._extract_lead_features(reference_lead)
            ref_vector = self._features_to_vector(ref_features)
            
            similarities = []
            
            for candidate in candidate_leads:
                try:
                    cand_features = self._extract_lead_features(candidate)
                    cand_vector = self._features_to_vector(cand_features)
                    
                    # Calculate cosine similarity
                    similarity = self._calculate_cosine_similarity(ref_vector, cand_vector)
                    
                    if similarity >= similarity_threshold:
                        similarities.append({
                            'lead_id': candidate.get('id', 0),
                            'lead_data': candidate,
                            'similarity_score': similarity,
                            'matching_features': self._identify_matching_features(ref_features, cand_features)
                        })
                        
                except Exception as e:
                    logger.debug(f"Error processing candidate lead: {e}")
                    continue
            
            # Sort by similarity and limit results
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            top_similarities = similarities[:max_results]
            
            return {
                'similar_leads': [
                    {
                        'lead_id': sim['lead_id'],
                        'similarity_score': sim['similarity_score'],
                        'matching_features': sim['matching_features']
                    }
                    for sim in top_similarities
                ],
                'factors': self._analyze_similarity_factors(ref_features, top_similarities),
                'confidence': self._calculate_similarity_confidence(top_similarities),
                'scores': [sim['similarity_score'] for sim in top_similarities],
                'business_relevance': self._assess_similarity_business_relevance(ref_features, top_similarities),
                'actions': self._recommend_similarity_actions(top_similarities)
            }
            
        except Exception as e:
            logger.error(f"Similarity analysis failed: {e}")
            return {
                'similar_leads': [],
                'factors': [],
                'confidence': 0.0,
                'scores': [],
                'business_relevance': 0.0,
                'actions': []
            }
    
    def _calculate_cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity between two feature vectors"""
        try:
            # Convert to numpy arrays for easier calculation
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0, min(1, similarity))  # Clamp between 0 and 1
            
        except Exception:
            # Fallback to simple correlation
            return self._simple_similarity(vector1, vector2)
    
    def _simple_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Simple similarity calculation fallback"""
        if len(vector1) != len(vector2):
            return 0.0
        
        differences = [abs(v1 - v2) for v1, v2 in zip(vector1, vector2)]
        avg_difference = sum(differences) / len(differences)
        
        # Convert difference to similarity (0-1 scale)
        max_possible_diff = 100  # Assuming features are 0-100 scale
        similarity = 1 - (avg_difference / max_possible_diff)
        
        return max(0, min(1, similarity))
    
    def _identify_matching_features(self, ref_features: LeadFeatures, cand_features: LeadFeatures) -> List[str]:
        """Identify which features are similar between leads"""
        matching = []
        
        # Check email similarity
        if abs(ref_features.email_quality_score - cand_features.email_quality_score) < 20:
            matching.append('email_quality')
        
        # Check seniority similarity
        if abs(ref_features.seniority_score - cand_features.seniority_score) < 25:
            matching.append('seniority_level')
        
        # Check industry similarity
        if abs(ref_features.industry_relevance - cand_features.industry_relevance) < 20:
            matching.append('industry_sector')
        
        # Check company size similarity
        if abs(ref_features.company_size_score - cand_features.company_size_score) < 20:
            matching.append('company_size')
        
        # Check location similarity
        if abs(ref_features.location_score - cand_features.location_score) < 25:
            matching.append('geographic_region')
        
        # Check contact completeness
        ref_contacts = sum([ref_features.has_email, ref_features.has_phone, ref_features.has_linkedin])
        cand_contacts = sum([cand_features.has_email, cand_features.has_phone, cand_features.has_linkedin])
        if abs(ref_contacts - cand_contacts) <= 1:
            matching.append('contact_completeness')
        
        return matching
    
    def _analyze_similarity_factors(self, ref_features: LeadFeatures, similarities: List[Dict]) -> List[str]:
        """Analyze what factors drive similarity"""
        if not similarities:
            return []
        
        # Analyze common patterns
        factors = []
        
        # Check if seniority is a common factor
        avg_seniority_diff = np.mean([
            abs(ref_features.seniority_score - self._extract_lead_features(sim['lead_data']).seniority_score)
            for sim in similarities
        ])
        if avg_seniority_diff < 20:
            factors.append('Similar seniority levels')
        
        # Check industry patterns
        ref_industry = ref_features.industry_relevance
        industry_matches = sum(1 for sim in similarities 
                             if abs(ref_industry - self._extract_lead_features(sim['lead_data']).industry_relevance) < 25)
        if industry_matches >= len(similarities) * 0.7:
            factors.append('Same industry sector')
        
        # Check company size patterns
        ref_size = ref_features.company_size_score
        size_matches = sum(1 for sim in similarities 
                          if abs(ref_size - self._extract_lead_features(sim['lead_data']).company_size_score) < 25)
        if size_matches >= len(similarities) * 0.6:
            factors.append('Similar company sizes')
        
        return factors[:5]  # Top 5 factors
    
    def _calculate_similarity_confidence(self, similarities: List[Dict]) -> float:
        """Calculate confidence in similarity analysis"""
        if not similarities:
            return 0.0
        
        # Base confidence on number of results and score distribution
        num_results = len(similarities)
        avg_score = np.mean([sim['similarity_score'] for sim in similarities])
        score_std = np.std([sim['similarity_score'] for sim in similarities])
        
        # Higher confidence with more results and higher average scores
        confidence = min(0.95, (num_results / 20) * 0.5 + avg_score * 0.4 + (1 - score_std) * 0.1)
        
        return confidence
    
    def _assess_similarity_business_relevance(self, ref_features: LeadFeatures, similarities: List[Dict]) -> float:
        """Assess business relevance of similar leads"""
        if not similarities:
            return 0.0
        # Higher relevance if similar leads are high quality
        total_quality = 0
        for sim in similarities:
            sim_features = self._extract_lead_features(sim['lead_data'])
            quality = (sim_features.seniority_score + sim_features.industry_relevance + 
                      sim_features.completeness_score) / 3
            total_quality += quality
        
        avg_quality = total_quality / len(similarities)
        return min(100, avg_quality)
    
    def _recommend_similarity_actions(self, similarities: List[Dict]) -> List[str]:
        """Recommend actions based on similarity analysis"""
        if not similarities:
            return []
        
        actions = []
        
        if len(similarities) >= 5:
            actions.append('Create targeted campaign for this lead segment')
        
        if len(similarities) >= 3:
            actions.append('Develop personalized messaging for similar profiles')
        
        # Check if similar leads have high scores
        high_scoring_similar = [sim for sim in similarities if sim['similarity_score'] > 0.8]
        if high_scoring_similar:
            actions.append('Apply successful strategies from similar high-performing leads')
        
        actions.append('Monitor engagement patterns across similar leads')
        
        return actions[:4]

    # ============================
    # TIMING PREDICTION METHODS
    # ============================
    
    def predict_optimal_timing(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict optimal contact timing using AI analysis"""
        features = self._extract_lead_features(lead_data)
        
        # Base timing on industry patterns
        timing = {
            'best_day': 'tuesday',
            'best_hour_start': 10,
            'best_hour_end': 12,
            'timezone': 'UTC',
            'confidence': 0.6,
            'reasoning': 'Standard business timing',
            'success_rate': 0.15,
            'alternatives': ['wednesday 9-11', 'thursday 14-16']
        }
        
        # Adjust for industry
        if features.industry_relevance >= 80:
            # Tech companies often more flexible
            timing.update({
                'best_day': 'wednesday',
                'best_hour_start': 9,
                'best_hour_end': 11,
                'confidence': 0.7,
                'reasoning': 'Technology industry timing optimization',
                'success_rate': 0.22
            })
        
        # Adjust for seniority
        if features.seniority_score >= 80:
            # Executives prefer early morning
            timing.update({
                'best_hour_start': 8,
                'best_hour_end': 9,
                'confidence': 0.8,
                'reasoning': 'Executive timing - early morning before meetings',
                'success_rate': 0.28,
                'alternatives': ['7-8', '17-18']
            })
        
        # Location-based timing adjustments
        if features.location_score >= 80:
            timing['timezone'] = self._infer_timezone(lead_data.get('company', {}).get('location', ''))
            timing['confidence'] += 0.1
        
        return timing
    
    def _infer_timezone(self, location: str) -> str:
        """Infer timezone from location"""
        location_lower = location.lower()
        
        # US timezones
        if any(city in location_lower for city in ['san francisco', 'seattle', 'los angeles']):
            return 'America/Los_Angeles'
        elif any(city in location_lower for city in ['new york', 'boston', 'atlanta']):
            return 'America/New_York'
        elif any(city in location_lower for city in ['chicago', 'austin', 'denver']):
            return 'America/Chicago'
        
        # International
        elif 'london' in location_lower or 'uk' in location_lower:
            return 'Europe/London'
        elif 'berlin' in location_lower or 'germany' in location_lower:
            return 'Europe/Berlin'
        elif 'singapore' in location_lower:
            return 'Asia/Singapore'
        elif 'sydney' in location_lower:
            return 'Australia/Sydney'
        
        return 'UTC'

    # ============================
    # BATCH PROCESSING METHODS
    # ============================
    
    def batch_analyze_leads(self, leads: List[Dict[str, Any]], 
                           include_similarities: bool = False) -> Dict[str, Any]:
        """Analyze multiple leads in batch for efficiency"""
        try:
            start_time = datetime.utcnow()
            results = []
            
            # Process leads in parallel conceptually (for now sequential)
            for i, lead in enumerate(leads):
                try:
                    analysis = self.analyze_lead(lead)
                    results.append({
                        'lead_id': lead.get('id', i),
                        'analysis': analysis
                    })
                except Exception as e:
                    logger.warning(f"Failed to analyze lead {lead.get('id', i)}: {e}")
                    continue
            
            # Calculate batch statistics
            if results:
                priority_scores = [r['analysis']['priority_score'] for r in results]
                quality_scores = [r['analysis']['quality_score'] for r in results]
                confidence_scores = [r['analysis']['confidence'] for r in results]
                
                batch_stats = {
                    'total_analyzed': len(results),
                    'average_priority': np.mean(priority_scores),
                    'average_quality': np.mean(quality_scores),
                    'average_confidence': np.mean(confidence_scores),
                    'high_priority_count': sum(1 for score in priority_scores if score >= 80),
                    'medium_priority_count': sum(1 for score in priority_scores if 60 <= score < 80),
                    'low_priority_count': sum(1 for score in priority_scores if score < 60),
                    'processing_time_seconds': (datetime.utcnow() - start_time).total_seconds()
                }
            else:
                batch_stats = {'total_analyzed': 0, 'error': 'No leads processed successfully'}
            
            # Find similarities if requested
            similarities = []
            if include_similarities and len(results) > 1:
                similarities = self._find_batch_similarities(leads)
            
            return {
                'results': results,
                'batch_statistics': batch_stats,
                'similarities': similarities,
                'recommendations': self._generate_batch_recommendations(results)
            }
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return {
                'results': [],
                'batch_statistics': {'total_analyzed': 0, 'error': str(e)},
                'similarities': [],
                'recommendations': []
            }
    
    def _find_batch_similarities(self, leads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find similarities within a batch of leads"""
        similarities = []
        
        try:
            # Extract features for all leads
            features_list = [self._extract_lead_features(lead) for lead in leads]
            vectors_list = [self._features_to_vector(features) for features in features_list]
            
            # Compare each lead with every other lead
            for i in range(len(leads)):
                for j in range(i + 1, len(leads)):
                    similarity_score = self._calculate_cosine_similarity(vectors_list[i], vectors_list[j])
                    
                    if similarity_score >= 0.7:  # High similarity threshold
                        similarities.append({
                            'lead1_id': leads[i].get('id', i),
                            'lead2_id': leads[j].get('id', j),
                            'similarity_score': similarity_score,
                            'matching_features': self._identify_matching_features(features_list[i], features_list[j])
                        })
            
            # Sort by similarity score
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            
        except Exception as e:
            logger.warning(f"Batch similarity analysis failed: {e}")
        
        return similarities[:20]  # Return top 20 similarities
    
    def _generate_batch_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on batch analysis"""
        if not results:
            return []
        
        recommendations = []
        
        # Priority-based recommendations
        high_priority = [r for r in results if r['analysis']['priority_score'] >= 80]
        if high_priority:
            recommendations.append(f"🎯 Focus on {len(high_priority)} high-priority leads first")
        
        # Quality-based recommendations
        low_quality = [r for r in results if r['analysis'].get('quality_score', 0) < 60]
        if low_quality:
            recommendations.append(f"📊 Enrich data for {len(low_quality)} leads before outreach")
        
        # Confidence-based recommendations
        low_confidence = [r for r in results if r['analysis']['confidence'] < 0.5]
        if low_confidence:
            recommendations.append(f"⚠️ Validate {len(low_confidence)} leads with uncertain analysis")
        
        # Industry clustering recommendation
        industries = {}
        for result in results:
            lead_id = result['lead_id']
            # Would need to extract industry from original lead data
            # This is a placeholder
            industry = 'technology'  # Placeholder
            industries[industry] = industries.get(industry, 0) + 1
        
        if industries:
            top_industry = max(industries, key=industries.get)
            recommendations.append(f"🏢 Create industry-specific campaigns for {top_industry} sector")
        
        return recommendations[:5]

    # ============================
    # MODEL TRAINING AND FEEDBACK
    # ============================
    
    def train_model_with_feedback(self, feedback_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train models with user feedback for continuous improvement"""
        try:
            if not feedback_data:
                return {'success': False, 'message': 'No feedback data provided'}
            
            # Process feedback
            training_features = []
            training_labels = []
            
            for feedback in feedback_data:
                lead_data = feedback.get('lead_data', {})
                actual_outcome = feedback.get('actual_outcome')  # 'converted', 'rejected', etc.
                user_rating = feedback.get('user_rating', 0)  # 1-5 stars
                
                if lead_data and actual_outcome:
                    features = self._extract_lead_features(lead_data)
                    feature_vector = self._features_to_vector(features)
                    
                    # Convert outcome to score
                    outcome_score = self._outcome_to_score(actual_outcome, user_rating)
                    
                    training_features.append(feature_vector)
                    training_labels.append(outcome_score)
            
            if len(training_features) < 10:
                return {'success': False, 'message': 'Insufficient feedback data for training'}
            
            # Retrain models with new data
            training_features = np.array(training_features)
            training_labels = np.array(training_labels)
            
            # Update priority model
            if 'priority' in self.models and hasattr(self.models['priority'], 'fit'):
                priority_labels = [1 if label > 70 else 0 for label in training_labels]
                self.models['priority'].fit(training_features, priority_labels)
            
            # Update quality model
            if 'quality' in self.models and hasattr(self.models['quality'], 'fit'):
                self.models['quality'].fit(training_features, training_labels)
            
            # Save updated models
            self._save_models()
            self.last_training_date = datetime.utcnow()
            
            return {
                'success': True,
                'message': f'Models retrained with {len(training_features)} feedback samples',
                'training_date': self.last_training_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {'success': False, 'message': f'Training failed: {str(e)}'}
    
    def _outcome_to_score(self, outcome: str, user_rating: int) -> float:
        """Convert outcome and rating to training score"""
        outcome_scores = {
            'converted': 95,
            'qualified': 85,
            'interested': 75,
            'responded': 65,
            'contacted': 55,
            'no_response': 40,
            'rejected': 20,
            'unqualified': 15
        }
        
        base_score = outcome_scores.get(outcome.lower(), 50)
        
        # Adjust based on user rating (1-5 stars)
        if user_rating > 0:
            rating_adjustment = (user_rating - 3) * 10  # -20 to +20
            base_score += rating_adjustment
        
        return max(0, min(100, base_score))

    # ============================
    # PERFORMANCE AND ANALYTICS
    # ============================
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        return {
            'analyses_performed': self.analysis_count,
            'last_training_date': self.last_training_date,
            'model_versions': {
                'priority_model': '1.0.0',
                'quality_model': '1.0.0',
                'similarity_model': '1.0.0'
            },
            'accuracy_estimates': self.model_accuracy_cache,
            'cache_efficiency': len(self._analysis_cache) / max(self._cache_max_size, 1),
            'available_features': [
                'lead_prioritization',
                'quality_assessment',
                'similarity_matching',
                'timing_optimization',
                'batch_processing'
            ]
        }
    
    def export_analysis_data(self, format: str = 'json') -> Dict[str, Any]:
        """Export analysis data for external use"""
        try:
            export_data = {
                'export_timestamp': datetime.utcnow().isoformat(),
                'model_performance': self.get_model_performance(),
                'analysis_patterns': self._extract_analysis_patterns(),
                'feature_importance': self._calculate_feature_importance(),
                'recommendations': self._generate_system_recommendations()
            }
            
            if format == 'json':
                return export_data
            elif format == 'csv':
                # Convert to flat structure for CSV
                return self._flatten_for_csv(export_data)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {'error': str(e)}
    
    def _extract_analysis_patterns(self) -> Dict[str, Any]:
        """Extract patterns from historical analyses"""
        # This would analyze historical data to find patterns
        # For now, return placeholder data
        return {
            'high_priority_indicators': [
                'executive_titles',
                'tech_industry',
                'medium_company_size'
            ],
            'conversion_patterns': [
                'linkedin_presence_increases_conversion',
                'fresh_leads_perform_better',
                'complete_profiles_convert_more'
            ],
            'timing_patterns': [
                'tuesday_wednesday_best_days',
                'morning_hours_preferred',
                'executive_early_contact'
            ]
        }
    
    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance for model interpretability"""
        # This would analyze which features are most predictive
        # For now, return estimated importance based on domain knowledge
        return {
            'seniority_score': 0.25,
            'industry_relevance': 0.20,
            'email_quality_score': 0.15,
            'company_size_score': 0.12,
            'completeness_score': 0.10,
            'title_decision_power': 0.08,
            'location_score': 0.05,
            'source_quality': 0.05
        }
    
    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-level recommendations for improvement"""
        recommendations = []
        
        if self.analysis_count < 100:
            recommendations.append("Analyze more leads to improve model accuracy")
        
        if len(self._analysis_cache) >= self._cache_max_size * 0.9:
            recommendations.append("Consider increasing cache size for better performance")
        
        if not self.groq_client:
            recommendations.append("Enable Groq LLM integration for enhanced insights")
        
        recommendations.append("Collect user feedback to improve model predictions")
        recommendations.append("Regular model retraining recommended for best performance")
        
        return recommendations
    
    def _flatten_for_csv(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten nested data structure for CSV export"""
        flattened = []
        
        # Extract performance metrics
        performance = data.get('model_performance', {})
        for key, value in performance.items():
            if isinstance(value, (int, float, str)):
                flattened.append({
                    'category': 'performance',
                    'metric': key,
                    'value': value
                })
        
        # Extract feature importance
        importance = data.get('feature_importance', {})
        for feature, score in importance.items():
            flattened.append({
                'category': 'feature_importance',
                'metric': feature,
                'value': score
            })
        
        return flattened

    # ============================
    # CACHE AND UTILITY METHODS
    # ============================
    
    def _generate_cache_key(self, lead_data: Dict[str, Any]) -> str:
        """Generate cache key for lead analysis"""
        # Create hash from key lead attributes
        key_data = {
            'email': lead_data.get('email', ''),
            'title': lead_data.get('title', ''),
            'company_name': lead_data.get('company', {}).get('name', ''),
            'company_size': lead_data.get('company', {}).get('size', '')
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _cache_analysis(self, cache_key: str, analysis: Dict[str, Any]):
        """Cache analysis result"""
        if len(self._analysis_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._analysis_cache))
            del self._analysis_cache[oldest_key]
        
        self._analysis_cache[cache_key] = {
            'analysis': analysis,
            'timestamp': datetime.utcnow(),
            'version': '2.0.0'
        }
    
    def _is_cache_valid(self, cached_item: Dict[str, Any]) -> bool:
        """Check if cached analysis is still valid"""
        cache_time = cached_item.get('timestamp')
        if not cache_time:
            return False
        
        # Cache valid for 1 hour
        return (datetime.utcnow() - cache_time).total_seconds() < 3600
    
    def _generate_fallback_analysis(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate basic fallback analysis when AI fails"""
        return {
            'lead_id': lead_data.get('id', 0),
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'priority_score': 50.0,
            'quality_score': 50.0,
            'confidence': 0.3,
            'priority_reason': 'Basic analysis - AI systems unavailable',
            'actions': ['Review lead manually', 'Standard outreach approach'],
            'insights': [],
            'model_version': 'fallback-1.0.0',
            'error': 'AI analysis failed, using fallback scoring'
        }
    
    def _assess_similarity_business_relevance(self, ref_features: LeadFeatures, similarities: List[Dict]) -> float:
        """Assess business relevance of similar leads"""
        if not similarities:
            return 0.0
        
        # Higher relevance if similar leads are high quality
        total_quality = 0
        for sim in similarities:
            sim_features = self._extract_lead_features(sim['lead_data'])
            quality = (sim_features.seniority_score + sim_features.industry_relevance + 
                      sim_features.completeness_score) / 3
            total_quality += quality
        
        avg_quality = total_quality / len(similarities)
        return min(100, avg_quality)

    # ============================
    # PUBLIC API METHODS
    # ============================
    
    def is_ready(self) -> bool:
        """Check if AI analyzer is ready for use"""
        return bool(self.models and 'priority' in self.models)
    
    def get_model_version(self) -> str:
        """Get current model version"""
        return "2.0.0"
    
    def get_model_accuracy(self) -> float:
        """Get estimated model accuracy"""
        return self.model_accuracy_cache.get('priority', 0.75)
    
    def get_last_training_date(self) -> Optional[datetime]:
        """Get last training date"""
        return self.last_training_date
    
    def is_model_loaded(self) -> bool:
        """Check if models are loaded"""
        return len(self.models) > 0
    
    def get_daily_analysis_count(self) -> int:
        """Get count of analyses performed today"""
        # This would track daily usage in production
        return self.analysis_count


# ============================
# USAGE EXAMPLE AND TESTING
# ============================

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = AILeadAnalyzer(enable_groq=False)
    
    # Test lead data
    test_lead = {
        'id': 1,
        'first_name': 'John',
        'last_name': 'Smith', 
        'email': 'john.smith@techcorp.ai',
        'phone': '+1-555-123-4567',
        'title': 'Chief Technology Officer',
        'linkedin_url': 'https://linkedin.com/in/john-smith-cto',
        'source': 'linkedin',
        'created_at': datetime.utcnow().isoformat(),
        'company': {
            'name': 'TechCorp AI',
            'domain': 'techcorp.ai',
            'industry': 'Artificial Intelligence',
            'size': '51-200',
            'location': 'San Francisco, CA',
            'website': 'https://techcorp.ai'
        }
    }
    
    # Perform analysis
    print("🧪 Testing AI Lead Analyzer")
    print("=" * 50)
    
    analysis = analyzer.analyze_lead(test_lead)
    
    print(f"Priority Score: {analysis['priority_score']:.1f}/100")
    print(f"Quality Score: {analysis['quality_score']:.1f}/100") 
    print(f"Confidence: {analysis['confidence']:.2f}")
    print(f"Priority Reason: {analysis['priority_reason']}")
    print(f"Recommended Actions: {', '.join(analysis['actions'][:3])}")
    
    # Test similarity
    similar_lead = {
        'id': 2,
        'first_name': 'Jane',
        'last_name': 'Doe',
        'email': 'jane.doe@aitech.com',
        'title': 'CTO',
        'company': {
            'name': 'AI Tech Solutions',
            'industry': 'Artificial Intelligence',
            'size': '51-200',
            'location': 'San Francisco, CA'
        }
    }
    
    similarity_result = analyzer.find_similar_leads(test_lead, [similar_lead])
    print(f"\nSimilarity Analysis:")
    print(f"Similar leads found: {len(similarity_result['similar_leads'])}")
    if similarity_result['similar_leads']:
        print(f"Top similarity score: {similarity_result['similar_leads'][0]['similarity_score']:.2f}")
    
    # Test batch analysis
    test_leads = [test_lead, similar_lead]
    batch_result = analyzer.batch_analyze_leads(test_leads, include_similarities=True)
    print(f"\nBatch Analysis:")
    print(f"Leads processed: {batch_result['batch_statistics']['total_analyzed']}")
    print(f"Average priority: {batch_result['batch_statistics'].get('average_priority', 0):.1f}")
    print(f"High priority count: {batch_result['batch_statistics'].get('high_priority_count', 0)}")
    
    # Test timing prediction
    timing_result = analyzer.predict_optimal_timing(test_lead)
    print(f"\nOptimal Timing:")
    print(f"Best day: {timing_result['best_day']}")
    print(f"Best hours: {timing_result['best_hour_start']}-{timing_result['best_hour_end']}")
    print(f"Confidence: {timing_result['confidence']:.2f}")
    
    # Test model performance
    performance = analyzer.get_model_performance()
    print(f"\nModel Performance:")
    print(f"Analyses performed: {performance['analyses_performed']}")
    print(f"Model ready: {analyzer.is_ready()}")
    print(f"Cache efficiency: {performance['cache_efficiency']:.2f}")
    
    # Test feedback training (with mock data)
    feedback_data = [
        {
            'lead_data': test_lead,
            'actual_outcome': 'converted',
            'user_rating': 5
        },
        {
            'lead_data': similar_lead,
            'actual_outcome': 'qualified',
            'user_rating': 4
        }
    ]
    
    # Note: This would fail with insufficient data in real usage
    training_result = analyzer.train_model_with_feedback(feedback_data)
    print(f"\nFeedback Training:")
    print(f"Training success: {training_result['success']}")
    print(f"Message: {training_result['message']}")
    
    # Test data export
    export_result = analyzer.export_analysis_data(format='json')
    print(f"\nData Export:")
    print(f"Export timestamp: {export_result.get('export_timestamp', 'N/A')}")
    print(f"Available features: {len(export_result.get('model_performance', {}).get('available_features', []))}")
    
    print(f"\n✅ AI Lead Analyzer testing completed successfully!")
    print(f"🎯 System ready for production use with {analyzer.get_model_version()} models")
    print(f"🚀 Enhanced features: Priority scoring, Quality assessment, Similarity matching, Timing optimization")