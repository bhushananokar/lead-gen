# schemas.py - PYDANTIC MODELS ONLY (NOT SQLAlchemy) - Enhanced with AI Schemas
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

# ============================
# EXISTING SCHEMAS (UNCHANGED)
# ============================

class UserBase(BaseModel):
    email: EmailStr
    username: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    is_premium: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class CompanyBase(BaseModel):
    name: str
    domain: Optional[str] = None
    industry: Optional[str] = None
    size: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None
    website: Optional[str] = None

class CompanyCreate(CompanyBase):
    pass

class Company(CompanyBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True

class LeadBase(BaseModel):
    first_name: str
    last_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    title: Optional[str] = None
    linkedin_url: Optional[str] = None
    source: Optional[str] = "new"
    status: Optional[str] = "new"

class LeadCreate(LeadBase):
    company_id: Optional[int] = None
    campaign_id: Optional[int] = None

class Lead(LeadBase):
    id: int
    score: float
    company: Optional[Company] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class CampaignBase(BaseModel):
    name: str
    description: Optional[str] = None
    target_criteria: Optional[Dict[str, Any]] = None

class CampaignCreate(CampaignBase):
    pass

class Campaign(CampaignBase):
    id: int
    status: str
    created_at: datetime
    leads_count: Optional[int] = 0
    
    class Config:
        from_attributes = True

class LeadScrapeRequest(BaseModel):
    query: str
    source: str = "linkedin"  # linkedin, google, website
    max_results: int = 50
    filters: Optional[Dict[str, Any]] = None

class BulkLeadImport(BaseModel):
    leads: List[LeadCreate]
    campaign_id: Optional[int] = None

# ============================
# NEW AI-ENHANCED SCHEMAS
# ============================

# Enums for consistent values
class PriorityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class InsightType(str, Enum):
    PRIORITY = "priority"
    TIMING = "timing"
    APPROACH = "approach"
    WARNING = "warning"
    OPPORTUNITY = "opportunity"

class ModelType(str, Enum):
    PRIORITY = "priority"
    QUALITY = "quality"
    CONVERSION = "conversion"
    TIMING = "timing"
    SIMILARITY = "similarity"

class ProcessingStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING = "pending"

# Core AI Response Schemas
class AILeadInsight(BaseModel):
    """AI-generated lead insight response"""
    lead_id: int
    priority_score: float = Field(..., ge=0, le=100, description="Priority score from 0-100")
    priority_reason: str = Field(..., description="Human-readable explanation for priority")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level from 0-1")
    suggested_actions: List[str] = Field(default_factory=list, description="List of recommended actions")
    optimal_contact_time: Optional[str] = None
    expected_response_rate: Optional[float] = Field(None, ge=0, le=100)
    insight_type: InsightType = InsightType.PRIORITY
    urgency_score: Optional[float] = Field(None, ge=0, le=100)
    business_impact: Optional[float] = Field(None, ge=0, le=100)
    
    @validator('priority_score')
    def validate_priority_score(cls, v):
        return round(v, 2)
    
    class Config:
        from_attributes = True

class DataQualityReport(BaseModel):
    """Data quality assessment response"""
    lead_id: int
    overall_score: float = Field(..., ge=0, le=100, description="Overall quality score")
    quality_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of quality scores by category"
    )
    issues_found: List[str] = Field(default_factory=list, description="List of data quality issues")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
    duplicate_candidates: List[int] = Field(default_factory=list, description="Potential duplicate lead IDs")
    completeness_score: Optional[float] = Field(None, ge=0, le=100)
    accuracy_score: Optional[float] = Field(None, ge=0, le=100)
    freshness_score: Optional[float] = Field(None, ge=0, le=100)
    consistency_score: Optional[float] = Field(None, ge=0, le=100)
    improvement_potential: Optional[float] = Field(None, ge=0, le=100)
    
    class Config:
        from_attributes = True

class BatchAnalysisResult(BaseModel):
    """Batch lead analysis response"""
    total_analyzed: int = Field(..., ge=0)
    high_priority_leads: List[int] = Field(default_factory=list, description="High priority lead IDs")
    medium_priority_leads: List[int] = Field(default_factory=list)
    low_priority_leads: List[int] = Field(default_factory=list)
    quality_issues_found: int = Field(default=0, ge=0)
    duplicates_detected: int = Field(default=0, ge=0)
    insights_generated: int = Field(default=0, ge=0)
    processing_time_seconds: float = Field(..., ge=0)
    average_quality_score: Optional[float] = Field(None, ge=0, le=100)
    recommendations_count: int = Field(default=0, ge=0)
    
    class Config:
        from_attributes = True

class SmartLeadRecommendation(BaseModel):
    """Smart lead recommendation response"""
    recommended_leads: List[AILeadInsight]
    filtering_criteria: Dict[str, Any] = Field(default_factory=dict)
    total_matches: int = Field(..., ge=0)
    quality_threshold: float = Field(..., ge=0, le=100)
    confidence_threshold: float = Field(..., ge=0, le=1)
    recommendation_reason: str
    similar_patterns: List[str] = Field(default_factory=list)
    
    class Config:
        from_attributes = True

class AIInsightsSummary(BaseModel):
    """AI insights dashboard summary"""
    total_leads_analyzed: int = Field(..., ge=0)
    high_priority_count: int = Field(..., ge=0)
    medium_priority_count: int = Field(..., ge=0)
    low_priority_count: int = Field(..., ge=0)
    critical_priority_count: int = Field(default=0, ge=0)
    average_quality_score: float = Field(..., ge=0, le=100)
    average_confidence: float = Field(..., ge=0, le=1)
    top_insights: List[str] = Field(default_factory=list)
    data_quality_trend: List[Dict[str, Any]] = Field(default_factory=list)
    recent_improvements: int = Field(default=0, ge=0)
    pending_actions: int = Field(default=0, ge=0)
    model_performance: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        from_attributes = True

class LeadSimilarityResult(BaseModel):
    """Similar leads analysis result"""
    reference_lead_id: int
    similar_leads: List[Dict[str, Any]] = Field(default_factory=list)
    similarity_factors: List[str] = Field(default_factory=list, description="Factors that make leads similar")
    confidence: float = Field(..., ge=0, le=1)
    similarity_scores: List[float] = Field(default_factory=list)
    business_relevance: float = Field(..., ge=0, le=100)
    recommended_actions: List[str] = Field(default_factory=list)
    
    class Config:
        from_attributes = True

class OptimalContactTiming(BaseModel):
    """Optimal contact timing prediction"""
    lead_id: int
    best_day_of_week: str = Field(..., pattern="^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$")
    best_hour_range: str = Field(..., description="Optimal hour range (e.g., '9-11 AM')")
    timezone: str = Field(..., description="Lead's timezone")
    confidence: float = Field(..., ge=0, le=1)
    reasoning: str = Field(..., description="Explanation for timing recommendation")
    success_probability: Optional[float] = Field(None, ge=0, le=100)
    alternative_times: List[str] = Field(default_factory=list)
    
    class Config:
        from_attributes = True

class LeadEnrichmentSuggestion(BaseModel):
    """Lead enrichment suggestions"""
    lead_id: int
    missing_fields: List[str] = Field(default_factory=list)
    enrichment_opportunities: List[Dict[str, Any]] = Field(default_factory=list)
    estimated_improvement: float = Field(..., ge=0, le=100, description="Expected quality score improvement")
    priority: PriorityLevel = PriorityLevel.MEDIUM
    suggested_sources: List[str] = Field(default_factory=list)
    cost_estimate: Optional[str] = None
    effort_estimate: Optional[str] = None
    
    class Config:
        from_attributes = True

# Advanced AI Schemas
class DuplicateDetectionResult(BaseModel):
    """Duplicate detection analysis result"""
    primary_lead_id: int
    duplicate_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    total_duplicates_found: int = Field(..., ge=0)
    auto_merge_recommended: List[int] = Field(default_factory=list)
    manual_review_required: List[int] = Field(default_factory=list)
    confidence_scores: Dict[int, float] = Field(default_factory=dict)
    merge_strategies: Dict[int, str] = Field(default_factory=dict)
    
    class Config:
        from_attributes = True

class ModelPerformanceMetrics(BaseModel):
    """AI model performance metrics"""
    model_name: str
    model_version: str
    accuracy: float = Field(..., ge=0, le=1)
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    predictions_made: int = Field(..., ge=0)
    average_prediction_time_ms: float = Field(..., ge=0)
    last_training_date: datetime
    is_active: bool
    confidence_distribution: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        from_attributes = True

class AIProcessingStatus(BaseModel):
    """AI processing operation status"""
    operation_id: str
    operation_type: str
    status: ProcessingStatus
    progress_percentage: float = Field(..., ge=0, le=100)
    estimated_completion_time: Optional[datetime] = None
    results_preview: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Request Schemas for AI Operations
class AIAnalysisRequest(BaseModel):
    """Request for AI analysis operations"""
    lead_ids: Optional[List[int]] = None
    analysis_types: List[str] = Field(default_factory=lambda: ["priority", "quality"])
    include_insights: bool = True
    include_timing: bool = False
    confidence_threshold: float = Field(0.5, ge=0, le=1)
    priority_threshold: float = Field(50.0, ge=0, le=100)
    
    @validator('analysis_types')
    def validate_analysis_types(cls, v):
        valid_types = ["priority", "quality", "timing", "similarity", "duplicates"]
        for analysis_type in v:
            if analysis_type not in valid_types:
                raise ValueError(f"Invalid analysis type: {analysis_type}")
        return v

class SimilarLeadsRequest(BaseModel):
    """Request for finding similar leads"""
    reference_lead_id: int
    similarity_threshold: float = Field(0.7, ge=0, le=1)
    max_results: int = Field(10, ge=1, le=100)
    include_company_similarity: bool = True
    include_title_similarity: bool = True
    include_industry_similarity: bool = True
    
class DuplicateDetectionRequest(BaseModel):
    """Request for duplicate detection"""
    lead_ids: Optional[List[int]] = None
    detection_threshold: float = Field(0.8, ge=0, le=1)
    include_fuzzy_matching: bool = True
    auto_merge_threshold: float = Field(0.95, ge=0, le=1)
    exclude_reviewed: bool = True

class DataQualityAssessmentRequest(BaseModel):
    """Request for data quality assessment"""
    lead_ids: Optional[List[int]] = None
    assessment_depth: str = Field("standard", pattern="^(basic|standard|comprehensive)$")
    include_enrichment_suggestions: bool = True
    include_duplicate_check: bool = True
    quality_threshold: float = Field(70.0, ge=0, le=100)

class ModelTrainingRequest(BaseModel):
    """Request for model training/retraining"""
    model_type: ModelType
    training_data_size: Optional[int] = None
    use_recent_feedback: bool = True
    hyperparameter_tuning: bool = False
    cross_validation_folds: int = Field(5, ge=2, le=10)
    test_size_percentage: float = Field(20.0, ge=10, le=40)

# Response aggregation schemas
class ComprehensiveLeadAnalysis(BaseModel):
    """Complete AI analysis for a lead"""
    lead_id: int
    ai_insights: AILeadInsight
    quality_report: DataQualityReport
    timing_prediction: Optional[OptimalContactTiming] = None
    enrichment_suggestions: Optional[LeadEnrichmentSuggestion] = None
    similar_leads: Optional[LeadSimilarityResult] = None
    duplicate_check: Optional[DuplicateDetectionResult] = None
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float
    
    class Config:
        from_attributes = True

class AISystemHealth(BaseModel):
    """Overall AI system health status"""
    system_status: str = Field(..., pattern="^(healthy|degraded|critical)$")
    active_models: List[str]
    model_performance: Dict[str, ModelPerformanceMetrics]
    recent_predictions: int
    average_response_time_ms: float
    error_rate_percentage: float
    last_training_date: Optional[datetime]
    pending_operations: int
    system_load_percentage: float
    available_features: List[str]
    
    class Config:
        from_attributes = True

class AIInsightHistory(BaseModel):
    """Historical AI insights for tracking"""
    lead_id: int
    insights: List[AILeadInsight]
    quality_trend: List[float]
    score_history: List[float]
    improvement_timeline: List[Dict[str, Any]]
    user_feedback_summary: Dict[str, int]
    
    class Config:
        from_attributes = True

# Utility schemas
class AIExplanation(BaseModel):
    """Detailed explanation of AI decisions"""
    decision_type: str
    explanation_text: str
    contributing_factors: List[Dict[str, Any]]
    confidence_breakdown: Dict[str, float]
    alternative_scenarios: List[str]
    business_context: str
    
class AIRecommendationAction(BaseModel):
    """Specific actionable recommendation"""
    action_type: str
    description: str
    priority: PriorityLevel
    estimated_impact: float = Field(..., ge=0, le=100)
    effort_required: str = Field(..., pattern="^(low|medium|high)$")
    deadline: Optional[datetime] = None
    dependencies: List[str] = Field(default_factory=list)
    success_metrics: List[str] = Field(default_factory=list)

# Feedback and learning schemas
class AIFeedbackSubmission(BaseModel):
    """User feedback on AI predictions"""
    prediction_id: Optional[int] = None
    insight_id: Optional[int] = None
    rating: int = Field(..., ge=1, le=5)
    is_accurate: Optional[bool] = None
    is_helpful: Optional[bool] = None
    actual_outcome: Optional[str] = None
    improvement_suggestion: Optional[str] = None
    feedback_text: Optional[str] = None
    
class AILearningUpdate(BaseModel):
    """Update from AI learning process"""
    model_improvements: List[str]
    accuracy_changes: Dict[str, float]
    new_insights_available: bool
    recommended_actions: List[str]
    next_training_date: Optional[datetime]