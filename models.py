# models.py - Enhanced with AI Tables
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

# ============================
# EXISTING MODELS (UNCHANGED)
# ============================

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_premium = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    leads = relationship("Lead", back_populates="owner")
    campaigns = relationship("Campaign", back_populates="owner")

class Company(Base):
    __tablename__ = "companies"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    domain = Column(String, unique=True, index=True)
    industry = Column(String)
    size = Column(String)
    location = Column(String)
    description = Column(Text)
    website = Column(String)
    linkedin_url = Column(String)
    phone = Column(String)
    revenue = Column(String)
    technologies = Column(Text)  # Store as JSON string for SQLite
    social_media = Column(Text)  # Store as JSON string for SQLite
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    leads = relationship("Lead", back_populates="company")

class Lead(Base):
    __tablename__ = "leads"
    
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String)
    last_name = Column(String)
    email = Column(String, index=True)
    phone = Column(String)
    title = Column(String)
    linkedin_url = Column(String)
    twitter_url = Column(String)
    source = Column(String)  # linkedin, website, manual, etc.
    status = Column(String, default="new")  # new, contacted, qualified, converted, rejected
    score = Column(Float, default=0.0)
    notes = Column(Text)
    tags = Column(Text)  # Store as JSON string for SQLite
    
    # Relationships
    company_id = Column(Integer, ForeignKey("companies.id"))
    owner_id = Column(Integer, ForeignKey("users.id"))
    campaign_id = Column(Integer, ForeignKey("campaigns.id"))
    
    company = relationship("Company", back_populates="leads")
    owner = relationship("User", back_populates="leads")
    campaign = relationship("Campaign", back_populates="leads")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Campaign(Base):
    __tablename__ = "campaigns"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(Text)
    target_criteria = Column(Text)  # Store as JSON string for SQLite
    status = Column(String, default="draft")  # draft, active, paused, completed
    
    # Relationships
    owner_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="campaigns")
    leads = relationship("Lead", back_populates="campaign")
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

# ============================
# NEW AI-ENHANCED MODELS
# ============================

class LeadPrediction(Base):
    """Store AI model predictions for leads"""
    __tablename__ = "lead_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"), index=True)
    model_type = Column(String(50), index=True)  # 'priority', 'quality', 'conversion', 'timing'
    model_version = Column(String(20))  # Track model version used
    prediction_score = Column(Float)  # 0-100 prediction score
    confidence = Column(Float)  # 0-1 confidence level
    explanation = Column(Text)  # Human-readable explanation
    feature_importance = Column(Text)  # JSON string of feature weights
    raw_features = Column(Text)  # JSON string of input features for debugging
    processing_time_ms = Column(Float)  # Time taken for prediction
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    lead = relationship("Lead", backref="ai_predictions")

class DataQualityScore(Base):
    """Track data quality metrics over time"""
    __tablename__ = "data_quality_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"), index=True)
    overall_score = Column(Float, index=True)  # 0-100 overall quality score
    email_quality = Column(Float)  # Email validation score
    completeness_score = Column(Float)  # Data completeness percentage
    accuracy_score = Column(Float)  # Data accuracy estimate
    consistency_score = Column(Float)  # Internal consistency check
    freshness_score = Column(Float)  # Data recency score
    issues_found = Column(Text)  # JSON string of specific issues
    suggestions = Column(Text)  # JSON string of improvement suggestions
    quality_trend = Column(String(20))  # 'improving', 'declining', 'stable'
    previous_score = Column(Float)  # Previous quality score for comparison
    improvement_potential = Column(Float)  # Estimated improvement if suggestions followed
    assessment_method = Column(String(50))  # 'automated', 'manual', 'hybrid'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    lead = relationship("Lead", backref="quality_assessments")

class LeadInsight(Base):
    """Store AI-generated insights about leads"""
    __tablename__ = "lead_insights"
    
    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"), index=True)
    insight_type = Column(String(50), index=True)  # 'priority', 'timing', 'approach', 'warning'
    insight_category = Column(String(50))  # 'conversion', 'engagement', 'data_quality', 'competitive'
    insight_text = Column(Text)  # Human-readable insight
    short_summary = Column(String(200))  # Brief summary for UI
    action_items = Column(Text)  # JSON string of suggested actions
    priority_level = Column(String(20), index=True)  # 'critical', 'high', 'medium', 'low'
    confidence_score = Column(Float)  # 0-1 confidence in the insight
    impact_score = Column(Float)  # 0-100 estimated business impact
    urgency_score = Column(Float)  # 0-100 how urgent this insight is
    supporting_evidence = Column(Text)  # JSON string of evidence/data points
    related_insights = Column(Text)  # JSON array of related insight IDs
    status = Column(String(20), default="active")  # 'active', 'resolved', 'dismissed'
    user_feedback = Column(String(20))  # 'helpful', 'not_helpful', 'irrelevant'
    generated_by = Column(String(50))  # AI model or system that generated this
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    lead = relationship("Lead", backref="ai_insights")

class ModelMetadata(Base):
    """Track ML model versions and performance"""
    __tablename__ = "model_metadata"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), index=True)  # Model identifier
    model_type = Column(String(50))  # 'classifier', 'regressor', 'clustering'
    version = Column(String(20))  # Model version (e.g., 'v1.2.3')
    algorithm = Column(String(50))  # 'random_forest', 'gradient_boosting', etc.
    accuracy_score = Column(Float)  # Model accuracy on test set
    precision_score = Column(Float)  # Precision metric
    recall_score = Column(Float)  # Recall metric
    f1_score = Column(Float)  # F1 score
    training_samples = Column(Integer)  # Number of training samples
    validation_samples = Column(Integer)  # Number of validation samples
    feature_count = Column(Integer)  # Number of features used
    training_date = Column(DateTime(timezone=True))
    training_duration_minutes = Column(Float)  # Training time
    model_path = Column(String(500))  # Path to saved model file
    hyperparameters = Column(Text)  # JSON string of hyperparameters
    feature_names = Column(Text)  # JSON array of feature names
    is_active = Column(Boolean, default=True, index=True)  # Currently deployed model
    deployment_date = Column(DateTime(timezone=True))
    performance_metrics = Column(Text)  # JSON string of additional metrics
    cross_validation_scores = Column(Text)  # JSON array of CV scores
    model_size_mb = Column(Float)  # Model file size
    predictions_made = Column(Integer, default=0)  # Count of predictions
    avg_prediction_time_ms = Column(Float)  # Average prediction time
    last_used = Column(DateTime(timezone=True))  # Last prediction timestamp
    notes = Column(Text)  # Additional notes about the model
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class LeadSimilarity(Base):
    """Store lead similarity relationships for recommendations"""
    __tablename__ = "lead_similarities"
    
    id = Column(Integer, primary_key=True, index=True)
    lead_id_1 = Column(Integer, ForeignKey("leads.id"), index=True)
    lead_id_2 = Column(Integer, ForeignKey("leads.id"), index=True)
    similarity_score = Column(Float, index=True)  # 0-1 similarity score
    similarity_type = Column(String(50))  # 'company', 'title', 'industry', 'behavioral'
    matching_features = Column(Text)  # JSON array of features that match
    difference_features = Column(Text)  # JSON array of differing features
    calculation_method = Column(String(50))  # 'cosine', 'jaccard', 'euclidean'
    confidence = Column(Float)  # Confidence in similarity calculation
    business_relevance = Column(Float)  # How relevant this similarity is for business
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    lead_1 = relationship("Lead", foreign_keys=[lead_id_1])
    lead_2 = relationship("Lead", foreign_keys=[lead_id_2])

class DuplicateDetection(Base):
    """Track detected duplicate leads"""
    __tablename__ = "duplicate_detections"
    
    id = Column(Integer, primary_key=True, index=True)
    primary_lead_id = Column(Integer, ForeignKey("leads.id"), index=True)
    duplicate_lead_id = Column(Integer, ForeignKey("leads.id"), index=True)
    match_score = Column(Float, index=True)  # 0-100 duplicate confidence score
    match_type = Column(String(50))  # 'exact', 'fuzzy', 'probable'
    matching_fields = Column(Text)  # JSON array of fields that match
    conflicting_fields = Column(Text)  # JSON array of conflicting data
    detection_method = Column(String(50))  # Algorithm used for detection
    auto_merge_recommended = Column(Boolean, default=False)  # Safe to auto-merge
    merge_strategy = Column(Text)  # JSON object with merge recommendations
    status = Column(String(20), default="detected")  # 'detected', 'merged', 'dismissed', 'reviewed'
    reviewed_by = Column(Integer, ForeignKey("users.id"))  # User who reviewed
    review_decision = Column(String(20))  # 'merge', 'keep_separate', 'need_review'
    review_notes = Column(Text)  # Notes from reviewer
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    reviewed_at = Column(DateTime(timezone=True))
    
    # Relationships
    primary_lead = relationship("Lead", foreign_keys=[primary_lead_id])
    duplicate_lead = relationship("Lead", foreign_keys=[duplicate_lead_id])
    reviewer = relationship("User", foreign_keys=[reviewed_by])

class AIProcessingLog(Base):
    """Log AI processing activities for monitoring and debugging"""
    __tablename__ = "ai_processing_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    operation_type = Column(String(50), index=True)  # 'prediction', 'analysis', 'training'
    entity_type = Column(String(50))  # 'lead', 'company', 'campaign'
    entity_id = Column(Integer, index=True)  # ID of the processed entity
    model_name = Column(String(100))  # Model used for processing
    model_version = Column(String(20))  # Version of model used
    input_data_hash = Column(String(64))  # Hash of input data for debugging
    processing_status = Column(String(20), index=True)  # 'started', 'completed', 'failed'
    start_time = Column(DateTime(timezone=True), server_default=func.now())
    end_time = Column(DateTime(timezone=True))
    processing_duration_ms = Column(Float)  # Time taken for processing
    cpu_usage_percent = Column(Float)  # CPU usage during processing
    memory_usage_mb = Column(Float)  # Memory usage during processing
    error_message = Column(Text)  # Error details if failed
    warning_messages = Column(Text)  # JSON array of warnings
    result_summary = Column(Text)  # JSON summary of results
    user_id = Column(Integer, ForeignKey("users.id"))  # User who triggered processing
    batch_id = Column(String(50))  # Group related operations
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User")

class AIFeedback(Base):
    """Store user feedback on AI predictions and insights"""
    __tablename__ = "ai_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("lead_predictions.id"))
    insight_id = Column(Integer, ForeignKey("lead_insights.id"))
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    feedback_type = Column(String(20), index=True)  # 'rating', 'correction', 'comment'
    rating = Column(Integer)  # 1-5 star rating
    is_accurate = Column(Boolean)  # Whether prediction was accurate
    is_helpful = Column(Boolean)  # Whether insight was helpful
    actual_outcome = Column(String(50))  # Actual result for validation
    improvement_suggestion = Column(Text)  # How to improve the AI
    feedback_text = Column(Text)  # Additional comments
    sentiment = Column(String(20))  # 'positive', 'negative', 'neutral'
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    prediction = relationship("LeadPrediction")
    insight = relationship("LeadInsight")
    user = relationship("User")

class OptimalContactTiming(Base):
    """Store predictions for optimal contact timing"""
    __tablename__ = "optimal_contact_timing"
    
    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"), index=True)
    best_day_of_week = Column(String(10))  # 'monday', 'tuesday', etc.
    best_hour_start = Column(Integer)  # 24-hour format
    best_hour_end = Column(Integer)  # 24-hour format
    timezone = Column(String(50))  # Lead's timezone
    confidence_score = Column(Float)  # 0-1 confidence in timing
    reasoning = Column(Text)  # Explanation for timing recommendation
    industry_pattern = Column(Boolean)  # Based on industry patterns
    title_pattern = Column(Boolean)  # Based on job title patterns
    geographic_pattern = Column(Boolean)  # Based on location patterns
    historical_success = Column(Float)  # Success rate for this timing
    last_updated = Column(DateTime(timezone=True), server_default=func.now())
    model_version = Column(String(20))  # Model version used
    
    # Relationships
    lead = relationship("Lead", backref="optimal_timing")

class EnrichmentSuggestion(Base):
    """Store AI-generated suggestions for data enrichment"""
    __tablename__ = "enrichment_suggestions"
    
    id = Column(Integer, primary_key=True, index=True)
    lead_id = Column(Integer, ForeignKey("leads.id"), index=True)
    missing_field = Column(String(50), index=True)  # Field that needs enrichment
    suggested_value = Column(String(500))  # Suggested value for the field
    confidence_score = Column(Float)  # 0-1 confidence in suggestion
    data_source = Column(String(100))  # Where suggestion came from
    enrichment_method = Column(String(50))  # Method used to find suggestion
    priority = Column(String(20), index=True)  # 'critical', 'high', 'medium', 'low'
    estimated_impact = Column(Float)  # Expected improvement in lead quality
    cost_estimate = Column(Float)  # Estimated cost to obtain this data
    effort_estimate = Column(String(20))  # 'low', 'medium', 'high' effort needed
    status = Column(String(20), default="pending")  # 'pending', 'applied', 'rejected', 'expired'
    expires_at = Column(DateTime(timezone=True))  # When suggestion expires
    applied_at = Column(DateTime(timezone=True))  # When suggestion was applied
    applied_by = Column(Integer, ForeignKey("users.id"))  # User who applied suggestion
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    lead = relationship("Lead", backref="enrichment_suggestions")
    applied_by_user = relationship("User")