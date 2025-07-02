# routers/leads.py - Enhanced with AI Integration
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Any, Dict, List, Optional
import json
import time
from datetime import datetime

from database import get_db, get_db_for_ai
from models import User, Lead, Company, LeadPrediction, DataQualityScore, LeadInsight
from schemas import (
    Lead as LeadSchema, 
    LeadCreate, 
    LeadScrapeRequest, 
    BulkLeadImport,
    AILeadInsight,
    DataQualityReport,
    BatchAnalysisResult,
    LeadSimilarityResult,
    DuplicateDetectionResult,
    ComprehensiveLeadAnalysis
)
from routers.auth import get_current_user

router = APIRouter()

# ============================
# EXISTING LEAD ENDPOINTS (ENHANCED)
# ============================

@router.get("/", response_model=List[LeadSchema])
def get_leads(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    min_score: Optional[float] = None,
    include_ai_insights: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get leads with optional AI insights and filtering"""
    query = db.query(Lead).filter(Lead.owner_id == current_user.id)
    
    if status:
        query = query.filter(Lead.status == status)
    
    if min_score is not None:
        query = query.filter(Lead.score >= min_score)
    
    leads = query.offset(skip).limit(limit).all()
    
    # Optionally include AI insights
    if include_ai_insights:
        for lead in leads:
            # Get latest AI prediction
            latest_prediction = db.query(LeadPrediction).filter(
                LeadPrediction.lead_id == lead.id
            ).order_by(LeadPrediction.created_at.desc()).first()
            
            if latest_prediction:
                lead.ai_prediction = {
                    "score": latest_prediction.prediction_score,
                    "confidence": latest_prediction.confidence,
                    "explanation": latest_prediction.explanation
                }
    
    return leads

@router.post("/", response_model=LeadSchema)
def create_lead(
    lead: LeadCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create lead with automatic AI analysis"""
    try:
        # Enhanced lead scoring with AI
        from utils.lead_scorer import LeadScorer
        from utils.ai_analyzer_2 import GroqEnhancedAnalyzer
        
        scorer = LeadScorer()
        ai_analyzer = GroqEnhancedAnalyzer()
        
        # Calculate traditional score
        traditional_score = scorer.calculate_score(lead.dict())
        
        # Create lead
        db_lead = Lead(**lead.dict(), owner_id=current_user.id, score=traditional_score)
        db.add(db_lead)
        db.commit()
        db.refresh(db_lead)
        
        # Schedule AI analysis in background
        background_tasks.add_task(
            _perform_ai_analysis_background,
            db_lead.id,
            current_user.id
        )
        
        return db_lead
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating lead: {str(e)}")

@router.get("/{lead_id}", response_model=LeadSchema)
def get_lead(
    lead_id: int,
    include_ai_data: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get single lead with optional AI data"""
    lead = db.query(Lead).filter(
        Lead.id == lead_id,
        Lead.owner_id == current_user.id
    ).first()
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    if include_ai_data:
        # Attach AI predictions and insights
        predictions = db.query(LeadPrediction).filter(
            LeadPrediction.lead_id == lead_id
        ).order_by(LeadPrediction.created_at.desc()).limit(5).all()
        
        insights = db.query(LeadInsight).filter(
            LeadInsight.lead_id == lead_id,
            LeadInsight.status == "active"
        ).order_by(LeadInsight.created_at.desc()).limit(3).all()
        
        lead.ai_predictions = predictions
        lead.ai_insights = insights
    
    return lead

@router.put("/{lead_id}", response_model=LeadSchema)
def update_lead(
    lead_id: int,
    lead_update: LeadCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update lead with AI re-analysis"""
    lead = db.query(Lead).filter(
        Lead.id == lead_id,
        Lead.owner_id == current_user.id
    ).first()
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    # Update lead fields
    for field, value in lead_update.dict(exclude_unset=True).items():
        setattr(lead, field, value)
    
    # Recalculate score with updated data
    from utils.lead_scorer import LeadScorer
    scorer = LeadScorer()
    lead.score = scorer.calculate_score(lead_update.dict())
    
    db.commit()
    db.refresh(lead)
    
    # Schedule AI re-analysis
    background_tasks.add_task(
        _perform_ai_analysis_background,
        lead.id,
        current_user.id
    )
    
    return lead

@router.delete("/{lead_id}")
def delete_lead(
    lead_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete lead and associated AI data"""
    lead = db.query(Lead).filter(
        Lead.id == lead_id,
        Lead.owner_id == current_user.id
    ).first()
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    # Delete associated AI data
    db.query(LeadPrediction).filter(LeadPrediction.lead_id == lead_id).delete()
    db.query(DataQualityScore).filter(DataQualityScore.lead_id == lead_id).delete()
    db.query(LeadInsight).filter(LeadInsight.lead_id == lead_id).delete()
    
    # Delete lead
    db.delete(lead)
    db.commit()
    
    return {"message": "Lead and associated AI data deleted successfully"}

# ============================
# NEW AI-ENHANCED ENDPOINTS
# ============================

@router.get("/{lead_id}/ai-insights", response_model=AILeadInsight)
def get_lead_ai_insights(
    lead_id: int,
    force_refresh: bool = False,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Get AI-generated insights for a specific lead - STANDALONE VERSION"""
    lead = db.query(Lead).filter(
        Lead.id == lead_id,
        Lead.owner_id == current_user.id
    ).first()
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    try:
        # STEP 1: Check for existing stored data (unless refresh requested)
        if not force_refresh:
            latest_prediction = db.query(LeadPrediction).filter(
                LeadPrediction.lead_id == lead_id,
                LeadPrediction.model_type == "priority"
            ).order_by(LeadPrediction.created_at.desc()).first()
            
            latest_insight = db.query(LeadInsight).filter(
                LeadInsight.lead_id == lead_id,
                LeadInsight.insight_type == "priority",
                LeadInsight.status == "active"
            ).order_by(LeadInsight.created_at.desc()).first()
            
            # If we have recent data (within last 24 hours), use it
            if latest_prediction and latest_insight:
                from datetime import datetime, timedelta
                if (datetime.utcnow() - latest_prediction.created_at).total_seconds() < 86400:  # 24 hours
                    
                    suggested_actions = []
                    if latest_insight.action_items:
                        try:
                            suggested_actions = json.loads(latest_insight.action_items)
                        except:
                            suggested_actions = [latest_insight.action_items] if latest_insight.action_items else []
                    
                    return AILeadInsight(
                        lead_id=lead.id,
                        priority_score=float(latest_prediction.prediction_score),
                        priority_reason=latest_prediction.explanation or latest_insight.insight_text or "AI analysis completed",
                        confidence=latest_prediction.confidence or 0.5,
                        suggested_actions=suggested_actions,
                        insight_type=latest_insight.insight_type,
                        urgency_score=latest_insight.urgency_score,
                        business_impact=latest_insight.impact_score
                    )
        
        # STEP 2: Perform fresh analysis with inline safe conversion
        from utils.ai_analyzer_2 import GroqEnhancedAnalyzer
        analyzer = GroqEnhancedAnalyzer()
        
        # INLINE SAFE CONVERSION - No helper function needed
        lead_dict = {
            'id': getattr(lead, 'id', 0),
            'first_name': getattr(lead, 'first_name', '') or '',
            'last_name': getattr(lead, 'last_name', '') or '',
            'email': getattr(lead, 'email', '') or '',
            'phone': getattr(lead, 'phone', '') or '',
            'title': getattr(lead, 'title', '') or '',
            'linkedin_url': getattr(lead, 'linkedin_url', '') or '',
            'source': getattr(lead, 'source', '') or '',
            'status': getattr(lead, 'status', '') or '',
            'score': getattr(lead, 'score', 0) or 0,
            'notes': getattr(lead, 'notes', '') or '',
            'tags': getattr(lead, 'tags', '') or '',
            'created_at': getattr(lead, 'created_at', None),
            'updated_at': getattr(lead, 'updated_at', None)
        }
        
        # Handle company information safely
        company_info = {}
        if hasattr(lead, 'company') and lead.company:
            # Company is a relationship object
            company_info = {
                'name': getattr(lead.company, 'name', '') or '',
                'industry': getattr(lead.company, 'industry', '') or '',
                'website': getattr(lead.company, 'website', '') or '',
                'size': getattr(lead.company, 'size', '') or '',
                'location': getattr(lead.company, 'location', '') or ''
            }
        else:
            # Company info might be stored directly on lead
            company_info = {
                'name': getattr(lead, 'company', '') or getattr(lead, 'company_name', '') or '',
                'industry': getattr(lead, 'industry', '') or '',
                'website': getattr(lead, 'website', '') or '',
                'size': getattr(lead, 'company_size', '') or '',
                'location': getattr(lead, 'location', '') or ''
            }
        
        # Add company info to lead dict
        lead_dict['company'] = company_info
        lead_dict['industry'] = company_info.get('industry', '')
        lead_dict['website'] = company_info.get('website', '')
        
        # Perform fresh analysis
        fresh_analysis = analyzer.analyze_lead(lead_dict)
        
        # STEP 3: Store the fresh analysis results (inline - no helper function)
        try:
            # Update or create prediction record
            existing_prediction = db.query(LeadPrediction).filter(
                LeadPrediction.lead_id == lead_id,
                LeadPrediction.model_type == "priority"
            ).order_by(LeadPrediction.created_at.desc()).first()
            
            if existing_prediction:
                existing_prediction.prediction_score = fresh_analysis.get('priority_score', 50)
                existing_prediction.confidence = fresh_analysis.get('confidence', 0.5)
                existing_prediction.explanation = fresh_analysis.get('reason', 'Fresh AI analysis')
                existing_prediction.model_version = "2.0.0"
            else:
                new_prediction = LeadPrediction(
                    lead_id=lead_id,
                    model_type="priority",
                    model_version="2.0.0",
                    prediction_score=fresh_analysis.get('priority_score', 50),
                    confidence=fresh_analysis.get('confidence', 0.5),
                    explanation=fresh_analysis.get('reason', 'Fresh AI analysis'),
                    processing_time_ms=100.0
                )
                db.add(new_prediction)
            
            # Update or create insight record
            existing_insight = db.query(LeadInsight).filter(
                LeadInsight.lead_id == lead_id,
                LeadInsight.insight_type == "priority",
                LeadInsight.status == "active"
            ).first()
            
            # Helper function to get priority level (inline)
            def get_priority_level(score: float) -> str:
                if score >= 85:
                    return "critical"
                elif score >= 70:
                    return "high"
                elif score >= 50:
                    return "medium"
                else:
                    return "low"
            
            if existing_insight:
                existing_insight.insight_text = fresh_analysis.get('reason', 'Updated AI analysis')
                existing_insight.action_items = json.dumps(fresh_analysis.get('actions', []))
                existing_insight.confidence_score = fresh_analysis.get('confidence', 0.5)
                existing_insight.impact_score = fresh_analysis.get('priority_score', 50)
                existing_insight.urgency_score = fresh_analysis.get('urgency_score', 50)
                existing_insight.priority_level = get_priority_level(fresh_analysis.get('priority_score', 50))
            else:
                new_insight = LeadInsight(
                    lead_id=lead_id,
                    insight_type="priority",
                    insight_text=fresh_analysis.get('reason', 'Fresh AI analysis'),
                    action_items=json.dumps(fresh_analysis.get('actions', [])),
                    priority_level=get_priority_level(fresh_analysis.get('priority_score', 50)),
                    confidence_score=fresh_analysis.get('confidence', 0.5),
                    impact_score=fresh_analysis.get('priority_score', 50),
                    urgency_score=fresh_analysis.get('urgency_score', 50),
                    generated_by="ai_analyzer_2",
                    status="active"
                )
                db.add(new_insight)
            
            db.commit()
            
        except Exception as storage_error:
            print(f"⚠️ Failed to store analysis results: {storage_error}")
            db.rollback()
        
        # Return the fresh analysis
        return AILeadInsight(
            lead_id=lead.id,
            priority_score=float(fresh_analysis.get('priority_score', 50)),
            priority_reason=fresh_analysis.get('reason', 'Fresh AI analysis'),
            confidence=fresh_analysis.get('confidence', 0.5),
            suggested_actions=fresh_analysis.get('actions', []),
            insight_type="priority",
            urgency_score=fresh_analysis.get('urgency_score', 50),
            business_impact=fresh_analysis.get('priority_score', 50)
        )
        
    except Exception as e:
        print(f"❌ AI insights generation failed for lead {lead_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI insights generation failed: {str(e)}")


def _generate_suggested_actions(insights: Dict[str, Any]) -> List[str]:
    """Generate specific suggested actions based on the analysis"""
    actions = []
    
    priority_score = insights.get('priority_score', 0)
    priority_level = insights.get('priority_level', 'medium')
    
    # Priority-based actions
    if priority_level == 'critical':
        actions.extend([
            "Schedule immediate outreach within 2 hours",
            "Prepare executive-level value proposition",
            "Research company background thoroughly"
        ])
    elif priority_level == 'high':
        actions.extend([
            "Prioritize for outreach today",
            "Customize messaging based on role and industry",
            "Prepare relevant case studies"
        ])
    elif priority_level == 'medium':
        actions.extend([
            "Add to weekly outreach queue",
            "Research company and role context"
        ])
    else:
        actions.extend([
            "Qualify and enrich lead data first",
            "Research company needs before outreach"
        ])
    
    # Role-specific actions
    role_analysis = insights.get('role_analysis', {})
    if role_analysis.get('score', 0) >= 85:
        actions.append("Focus on strategic business impact")
    elif role_analysis.get('score', 0) >= 60:
        actions.append("Highlight operational benefits")
    
    # Company-specific actions
    company_analysis = insights.get('company_analysis', {})
    company_factors = company_analysis.get('factors', [])
    if any('high-value industry' in factor.lower() for factor in company_factors):
        actions.append("Emphasize industry-specific solutions")
    
    # Contact quality actions
    contact_analysis = insights.get('contact_analysis', {})
    contact_issues = contact_analysis.get('issues', [])
    if contact_issues:
        actions.append("Enhance contact data quality before outreach")
    
    return actions[:5]  # Limit to top 5 actions

def _estimate_response_rate(insights: Dict[str, Any]) -> float:
    """Estimate response rate based on lead quality factors"""
    priority_score = insights.get('priority_score', 50)
    
    # Base response rate estimation
    if priority_score >= 85:
        return 25.0  # 25% for exceptional leads
    elif priority_score >= 70:
        return 18.0  # 18% for high priority
    elif priority_score >= 50:
        return 12.0  # 12% for medium priority
    else:
        return 6.0   # 6% for low priority
def _is_high_value_lead(lead_dict: Dict) -> bool:
    """Detect if this is a high-value lead based on real indicators"""
    
    # Check for executive titles
    title = (lead_dict.get('title', '') or '').lower()
    executive_keywords = ['ceo', 'cto', 'cmo', 'vp', 'director', 'president', 'founder', 'head of', 'chief']
    has_executive_title = any(keyword in title for keyword in executive_keywords)
    
    # Check for high-value companies/industries
    company = (lead_dict.get('company', '') or '').lower()
    industry = (lead_dict.get('industry', '') or '').lower()
    high_value_industries = ['technology', 'software', 'ai', 'fintech', 'healthcare', 'enterprise']
    in_high_value_industry = any(industry_term in (company + ' ' + industry) for industry_term in high_value_industries)
    
    # Check for complete contact info
    has_email = bool(lead_dict.get('email'))
    has_phone = bool(lead_dict.get('phone'))
    complete_contact = has_email and has_phone
    
    return has_executive_title or in_high_value_industry or complete_contact


def _has_high_potential_indicators(lead_dict: Dict) -> bool:
    """Check for indicators of high conversion potential"""
    
    # Check for decision-maker titles
    title = (lead_dict.get('title', '') or '').lower()
    decision_maker_keywords = ['manager', 'senior', 'lead', 'specialist', 'analyst', 'coordinator']
    is_decision_maker = any(keyword in title for keyword in decision_maker_keywords)
    
    # Check for professional email domains
    email = lead_dict.get('email', '') or ''
    professional_domains = ['.com', '.org', '.net', '.io', '.co']
    has_professional_email = any(domain in email for domain in professional_domains) and '@' in email
    
    # Check for LinkedIn or other professional sources
    source = (lead_dict.get('source', '') or '').lower()
    professional_source = 'linkedin' in source or 'professional' in source
    
    return is_decision_maker or has_professional_email or professional_source


def _has_strong_conversion_potential(lead_dict: Dict) -> bool:
    """Check for strong conversion indicators that warrant high priority"""
    
    # Check lead description/text for quality indicators
    title = (lead_dict.get('title', '') or '').lower()
    company = (lead_dict.get('company', '') or '').lower()
    
    # Quality indicators in the actual lead data
    quality_indicators = [
        'strong conversion potential' in (title + company).lower(),
        'high-quality' in (title + company).lower(),
        len(lead_dict.get('email', '')) > 0 and '@' in lead_dict.get('email', ''),
        len(lead_dict.get('phone', '')) > 0,
        len(lead_dict.get('title', '')) > 0,
        len(lead_dict.get('company', '')) > 0
    ]
    
    # If most quality indicators are met, this should be high priority
    return sum(quality_indicators) >= 4


def _generate_realistic_insight_text(lead_dict: Dict, priority_level: str, score: float) -> str:
    """Generate realistic insight text that matches the priority level"""
    
    name = f"{lead_dict.get('first_name', '')} {lead_dict.get('last_name', '')}".strip()
    title = lead_dict.get('title', 'Professional')
    company = lead_dict.get('company', 'their company')
    
    if priority_level == "critical":
        return f"Critical priority lead: {name} is a {title} at {company} with exceptional conversion potential and immediate outreach opportunity"
    elif priority_level == "high":
        return f"High-quality lead with strong conversion potential: {name} ({title}) shows excellent fit and engagement indicators"
    elif priority_level == "medium":
        return f"Qualified prospect: {name} at {company} demonstrates good potential with moderate priority for outreach"
    else:
        return f"Standard lead requiring nurturing: {name} shows basic qualification but needs further development"


def _generate_realistic_actions(priority_level: str) -> List[str]:
    """Generate actions that match the priority level"""
    
    if priority_level == "critical":
        return [
            "Contact immediately",
            "Prepare executive proposal",
            "Schedule priority follow-up",
            "Research company background"
        ]
    elif priority_level == "high":
        return [
            "Contact immediately",
            "Prepare proposal",
            "Schedule follow-up call"
        ]
    elif priority_level == "medium":
        return [
            "Add to weekly outreach",
            "Research background",
            "Prepare standard proposal"
        ]
    else:
        return [
            "Add to nurture campaign",
            "Collect more information",
            "Monitor for engagement"
        ]
        
# Update the background analysis function in leads.py
def _perform_ai_analysis_background(lead_id: int, user_id: int):
    """Background task for AI analysis - FIXED PRIORITY CALCULATION"""
    try:
        from database import SessionLocal
        from utils.ai_analyzer_2 import GroqEnhancedAnalyzer
        from utils.data_quality_engine import DataQualityEngine
        
        db = SessionLocal()
        analyzer = GroqEnhancedAnalyzer()
        quality_engine = DataQualityEngine()
        
        lead = db.query(Lead).filter(Lead.id == lead_id).first()
        if not lead:
            db.close()
            return
        
        print(f"🔍 Processing lead {lead_id}: {getattr(lead, 'first_name', '')} {getattr(lead, 'last_name', '')}")
        
        # Convert Lead object to dictionary for analysis
        lead_dict = {
            'id': lead.id,
            'first_name': lead.first_name or '',
            'last_name': lead.last_name or '',
            'email': lead.email or '',
            'phone': lead.phone or '',
            'title': lead.title or '',
            'company': lead.company or '',
            'industry': getattr(lead, 'industry', '') or '',
            'website': getattr(lead, 'website', '') or '',
            'source': lead.source or '',
            'created_at': lead.created_at,
            'updated_at': lead.updated_at
        }
        
        # AI analysis with the safely converted dictionary
        print(f"🤖 Starting AI analysis for lead {lead_id}")
        insights = analyzer.analyze_lead(lead_dict)
        priority_score = insights.get('priority_score', 50)
        print(f"✅ AI analysis completed for lead {lead_id}, score: {priority_score}")
        
        # FIXED: Correct priority level calculation that matches actual lead quality
        # This logic should reflect the real lead potential
        if priority_score >= 85 or _is_high_value_lead(lead_dict):
            priority_level = "critical"
        elif priority_score >= 70 or _has_high_potential_indicators(lead_dict):
            priority_level = "high"  
        elif priority_score >= 50:
            priority_level = "medium"
        else:
            priority_level = "low"
        
        # Override priority level if lead shows strong conversion indicators
        if _has_strong_conversion_potential(lead_dict):
            priority_level = "high"
            priority_score = max(priority_score, 75)  # Boost score to match priority
            print(f"📈 Lead {lead_id} upgraded to HIGH priority due to strong conversion potential")
        
        # Store AI prediction
        prediction = LeadPrediction(
            lead_id=lead_id,
            model_type="priority",
            model_version="2.0.0",
            prediction_score=priority_score,
            confidence=insights.get('confidence', 0.8),
            explanation=insights.get('reason', 'Enhanced AI analysis'),
            processing_time_ms=100.0
        )
        db.add(prediction)
        
        # Quality assessment
        try:
            quality_result = quality_engine.assess_lead_quality(lead_dict)
            if isinstance(quality_result, dict):
                quality_report = quality_result
            else:
                quality_report = {
                    'overall_score': getattr(quality_result, 'overall_score', 75),
                    'email_quality': getattr(quality_result, 'email_quality', 80),
                    'completeness_score': getattr(quality_result, 'completeness_score', 70),
                    'accuracy_score': getattr(quality_result, 'accuracy_score', 85),
                    'issues': [],
                    'suggestions': []
                }
            
            quality_score = DataQualityScore(
                lead_id=lead_id,
                overall_score=quality_report.get('overall_score', 75),
                email_quality=quality_report.get('email_quality', 80),
                completeness_score=quality_report.get('completeness_score', 70),
                accuracy_score=quality_report.get('accuracy_score', 85),
                issues_found=json.dumps(quality_report.get('issues', [])),
                suggestions=json.dumps(quality_report.get('suggestions', [])),
                assessment_method="enhanced_analysis"
            )
            db.add(quality_score)
            
        except Exception as e:
            print(f"⚠️ Quality assessment failed for lead {lead_id}: {e}")
        
        # FIXED: Create insight with correct priority and meaningful description
        insight_text = _generate_realistic_insight_text(lead_dict, priority_level, priority_score)
        
        insight = LeadInsight(
            lead_id=lead_id,
            insight_type="priority",
            insight_text=insight_text,
            action_items=json.dumps(_generate_realistic_actions(priority_level)),
            priority_level=priority_level,  # This is what shows in the UI
            confidence_score=insights.get('confidence', 0.8),
            impact_score=priority_score,
            urgency_score=insights.get('urgency_score', priority_score * 0.8),
            generated_by="groq_enhanced_analyzer",
            status="active"
        )
        db.add(insight)
        
        db.commit()
        print(f"✅ Enhanced AI analysis completed for lead {lead_id} with priority: {priority_level}")
        
    except Exception as e:
        print(f"❌ Enhanced AI analysis failed for lead {lead_id}: {e}")
    finally:
        try:
            db.close()
        except:
            pass


@router.get("/{lead_id}/data-quality", response_model=DataQualityReport)
def assess_lead_data_quality(
    lead_id: int,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Assess data quality for a specific lead - FIXED VERSION"""
    lead = db.query(Lead).filter(
        Lead.id == lead_id,
        Lead.owner_id == current_user.id
    ).first()
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    try:
        from utils.data_quality_engine import DataQualityEngine
        quality_engine = DataQualityEngine()
        
        # Convert Lead object to dictionary
        lead_dict = {
            'id': lead.id,
            'first_name': lead.first_name or '',
            'last_name': lead.last_name or '',
            'email': lead.email or '',
            'phone': lead.phone or '',
            'title': lead.title or '',
            'company': lead.company or '',
            'industry': lead.industry or '',
            'website': lead.website or '',
            'source': lead.source or ''
        }
        
        # Assess quality with dictionary
        quality_result = quality_engine.assess_lead_quality(lead_dict)
        
        # Handle the response properly
        if hasattr(quality_result, '__dict__'):
            # Convert object to dict
            quality_data = {
                'overall_score': getattr(quality_result, 'overall_score', 50),
                'quality_breakdown': {
                    'email_quality': getattr(quality_result, 'email_quality', 0),
                    'completeness_score': getattr(quality_result, 'completeness_score', 0),
                    'accuracy_score': getattr(quality_result, 'accuracy_score', 0)
                },
                'issues_found': getattr(quality_result, 'issues_found', []),
                'improvement_suggestions': getattr(quality_result, 'improvement_suggestions', [])
            }
        elif isinstance(quality_result, dict):
            quality_data = {
                'overall_score': quality_result.get('overall_score', 50),
                'quality_breakdown': {
                    'email_quality': quality_result.get('email_quality', 0),
                    'completeness_score': quality_result.get('completeness_score', 0),
                    'accuracy_score': quality_result.get('accuracy_score', 0)
                },
                'issues_found': quality_result.get('issues', []),
                'improvement_suggestions': quality_result.get('suggestions', [])
            }
        else:
            # Fallback
            quality_data = {
                'overall_score': 50,
                'quality_breakdown': {'email_quality': 0, 'completeness_score': 0, 'accuracy_score': 0},
                'issues_found': [],
                'improvement_suggestions': []
            }
        
        return DataQualityReport(
            lead_id=lead_id,
            **quality_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {str(e)}")

@router.post("/analyze-batch", response_model=BatchAnalysisResult)
def analyze_leads_batch(
    background_tasks: BackgroundTasks,
    limit: int = 100,
    quality_threshold: float = 70.0,
    priority_threshold: float = 50.0,   
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Batch analyze leads with AI for prioritization and quality"""
    start_time = time.time()
    
    try:
        # Get user's leads
        leads = db.query(Lead).filter(
            Lead.owner_id == current_user.id
        ).limit(limit).all()
        
        if not leads:
            raise HTTPException(status_code=404, detail="No leads found for analysis")
        
        from utils.ai_analyzer_2 import GroqEnhancedAnalyzer
        from utils.data_quality_engine import DataQualityEngine
        
        analyzer = GroqEnhancedAnalyzer()
        quality_engine = DataQualityEngine()
        
        high_priority_leads = []
        medium_priority_leads = []
        low_priority_leads = []
        quality_issues = 0
        duplicates_detected = 0
        insights_generated = 0
        total_quality_score = 0
        
        # Process leads in batches for better performance
        batch_size = 10
        for i in range(0, len(leads), batch_size):
            batch_leads = leads[i:i + batch_size]
            
            for lead in batch_leads:
                try:
                    # AI analysis
                    insights = analyzer.analyze_lead(lead)
                    priority_score = insights.get('priority_score', 50)
                    
                    # Categorize by priority
                    if priority_score >= 80:
                        high_priority_leads.append(lead.id)
                    elif priority_score >= 60:
                        medium_priority_leads.append(lead.id)
                    else:
                        low_priority_leads.append(lead.id)
                    
                    # Quality assessment
                    quality_report = quality_engine.assess_lead_quality(lead)
                    quality_score = quality_report.get('overall_score', 50)
                    total_quality_score += quality_score
                    
                    if quality_score < quality_threshold:
                        quality_issues += 1
                    
                    # Store predictions and quality scores
                    prediction = LeadPrediction(
                        lead_id=lead.id,
                        model_type="priority",
                        model_version="1.0.0",
                        prediction_score=priority_score,
                        confidence=insights.get('confidence', 0.5),
                        explanation=insights.get('reason', 'Automated AI analysis'),
                        processing_time_ms=50.0  # Approximate
                    )
                    db.add(prediction)
                    
                    quality_score_record = DataQualityScore(
                        lead_id=lead.id,
                        overall_score=quality_score,
                        email_quality=quality_report.get('email_quality', 0),
                        completeness_score=quality_report.get('completeness_score', 0),
                        accuracy_score=quality_report.get('accuracy_score', 0),
                        issues_found=json.dumps(quality_report.get('issues', [])),
                        suggestions=json.dumps(quality_report.get('suggestions', [])),
                        assessment_method="batch_automated"
                    )
                    db.add(quality_score_record)
                    
                    insights_generated += 1
                    
                except Exception as e:
                    print(f"Error processing lead {lead.id}: {e}")
                    continue
        
        # Commit all changes
        db.commit()
        
        # Schedule duplicate detection in background
        background_tasks.add_task(
            _detect_duplicates_background,
            current_user.id,
            [lead.id for lead in leads]
        )
        
        processing_time = time.time() - start_time
        average_quality = total_quality_score / len(leads) if leads else 0
        
        return BatchAnalysisResult(
            total_analyzed=len(leads),
            high_priority_leads=high_priority_leads,
            medium_priority_leads=medium_priority_leads,
            low_priority_leads=low_priority_leads,
            quality_issues_found=quality_issues,
            duplicates_detected=duplicates_detected,  # Will be updated by background task
            insights_generated=insights_generated,
            processing_time_seconds=processing_time,
            average_quality_score=average_quality,
            recommendations_count=insights_generated
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.post("/find-similar", response_model=LeadSimilarityResult)
def find_similar_leads(
    reference_lead_id: int,
    similarity_threshold: float = 0.7,
    max_results: int = 10,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Find leads similar to a reference lead using AI"""
    reference_lead = db.query(Lead).filter(
        Lead.id == reference_lead_id,
        Lead.owner_id == current_user.id
    ).first()
    
    if not reference_lead:
        raise HTTPException(status_code=404, detail="Reference lead not found")
    
    try:
        from utils.ai_analyzer_2 import GroqEnhancedAnalyzer
        analyzer = GroqEnhancedAnalyzer()
        
        # Get all user's leads for comparison
        all_leads = db.query(Lead).filter(
            Lead.owner_id == current_user.id,
            Lead.id != reference_lead_id
        ).all()
        
        similar_leads = analyzer.find_similar_leads(
            reference_lead, 
            all_leads, 
            similarity_threshold, 
            max_results
        )
        
        return LeadSimilarityResult(
            reference_lead_id=reference_lead_id,
            similar_leads=similar_leads.get('similar_leads', []),
            similarity_factors=similar_leads.get('factors', []),
            confidence=similar_leads.get('confidence', 0.0),
            similarity_scores=similar_leads.get('scores', []),
            business_relevance=similar_leads.get('business_relevance', 50.0),
            recommended_actions=similar_leads.get('actions', [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity analysis failed: {str(e)}")

@router.post("/detect-duplicates", response_model=DuplicateDetectionResult)
def detect_duplicate_leads(
    detection_threshold: float = 0.8,
    auto_merge_threshold: float = 0.95,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Detect duplicate leads using AI"""
    try:
        from utils.data_quality_engine import DataQualityEngine
        quality_engine = DataQualityEngine()
        
        # Get all user's leads
        leads = db.query(Lead).filter(Lead.owner_id == current_user.id).all()
        
        if len(leads) < 2:
            return DuplicateDetectionResult(
                primary_lead_id=0,
                duplicate_candidates=[],
                total_duplicates_found=0,
                auto_merge_recommended=[],
                manual_review_required=[],
                confidence_scores={},
                merge_strategies={}
            )
        
        duplicates = quality_engine.detect_duplicates(leads, detection_threshold)
        
        auto_merge = []
        manual_review = []
        confidence_scores = {}
        merge_strategies = {}
        
        for duplicate_group in duplicates:
            primary_id = duplicate_group['primary_lead_id']
            candidates = duplicate_group['duplicates']
            
            for candidate in candidates:
                candidate_id = candidate['lead_id']
                confidence = candidate['confidence']
                confidence_scores[candidate_id] = confidence
                
                if confidence >= auto_merge_threshold:
                    auto_merge.append(candidate_id)
                    merge_strategies[candidate_id] = "auto_merge"
                else:
                    manual_review.append(candidate_id)
                    merge_strategies[candidate_id] = "manual_review"
        
        # Store duplicate detection results
        from models import DuplicateDetection
        
        for duplicate_group in duplicates:
            primary_id = duplicate_group['primary_lead_id']
            for candidate in duplicate_group['duplicates']:
                duplicate_record = DuplicateDetection(
                    primary_lead_id=primary_id,
                    duplicate_lead_id=candidate['lead_id'],
                    match_score=candidate['confidence'] * 100,
                    match_type="fuzzy" if candidate['confidence'] < 0.9 else "exact",
                    matching_fields=json.dumps(candidate.get('matching_fields', [])),
                    detection_method="ai_fuzzy_matching",
                    auto_merge_recommended=candidate['confidence'] >= auto_merge_threshold,
                    merge_strategy=json.dumps(candidate.get('merge_strategy', {}))
                )
                db.add(duplicate_record)
        
        db.commit()
        
        return DuplicateDetectionResult(
            primary_lead_id=duplicates[0]['primary_lead_id'] if duplicates else 0,
            duplicate_candidates=[{
                'lead_id': dup['lead_id'],
                'confidence': dup['confidence'],
                'matching_fields': dup.get('matching_fields', [])
            } for group in duplicates for dup in group['duplicates']],
            total_duplicates_found=sum(len(group['duplicates']) for group in duplicates),
            auto_merge_recommended=auto_merge,
            manual_review_required=manual_review,
            confidence_scores=confidence_scores,
            merge_strategies=merge_strategies
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Duplicate detection failed: {str(e)}")

@router.get("/{lead_id}/comprehensive-analysis", response_model=ComprehensiveLeadAnalysis)
def get_comprehensive_lead_analysis(
    lead_id: int,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Get complete AI analysis for a lead"""
    lead = db.query(Lead).filter(
        Lead.id == lead_id,
        Lead.owner_id == current_user.id
    ).first()
    
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    start_time = time.time()
    
    try:
        # Get AI insights
        ai_insights = get_lead_ai_insights(lead_id, False, db, current_user)
        
        # Get quality report
        quality_report = assess_lead_data_quality(lead_id, db, current_user)
        
        # Get similar leads
        try:
            similar_leads = find_similar_leads(lead_id, 0.7, 5, db, current_user)
        except:
            similar_leads = None
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return ComprehensiveLeadAnalysis(
            lead_id=lead_id,
            ai_insights=ai_insights,
            quality_report=quality_report,
            timing_prediction=None,  # Could be implemented later
            enrichment_suggestions=None,  # Could be implemented later
            similar_leads=similar_leads,
            duplicate_check=None,  # Could be implemented later
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")

# ============================
# EXISTING SCRAPING ENDPOINTS (ENHANCED)
# ============================

@router.post("/scrape")
def scrape_real_leads(
    request: LeadScrapeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Scrape real leads from various sources with AI enhancement"""
    try:
        print(f"🔍 Starting AI-enhanced scraping for user {current_user.id}")
        print(f"📋 Request: {request.dict()}")
        
        from utils.lead_scraper import LeadScraper
        from utils.email_finder import EmailFinder
        from utils.lead_scorer import LeadScorer
        
        scraper = LeadScraper()
        email_finder = EmailFinder()
        scorer = LeadScorer()
        
        # Get real leads from scraper
        scraped_leads = scraper.scrape_leads(
            query=request.query,
            source=request.source,
            max_results=request.max_results
        )
        
        print(f"📊 Scraper returned {len(scraped_leads)} leads")
        
        if not scraped_leads:
            return {
                "message": "No leads found for your search query. Try different keywords or sources.",
                "status": "completed",
                "count": 0
            }
        
        created_leads = []
        
        for lead_data in scraped_leads:
            try:
                # Extract company data
                company_data = lead_data.pop('company', {})
                company_id = None
                
                # Create or find company
                if company_data and company_data.get('name'):
                    existing_company = db.query(Company).filter(
                        Company.domain == company_data.get('domain')
                    ).first()
                    
                    if existing_company:
                        company_id = existing_company.id
                    else:
                        new_company = Company(
                            name=company_data.get('name', '')[:255],
                            domain=company_data.get('domain', '')[:255],
                            industry=company_data.get('industry', '')[:255],
                            size=company_data.get('size', '')[:50],
                            location=company_data.get('location', '')[:255],
                            website=company_data.get('website', '')[:500],
                            description=company_data.get('description', '')[:1000]
                        )
                        db.add(new_company)
                        db.flush()
                        company_id = new_company.id
                
                # Enrich email if missing
                if not lead_data.get('email') and lead_data.get('first_name') and lead_data.get('last_name'):
                    if company_data.get('domain'):
                        email_result = email_finder.find_email(
                            lead_data['first_name'],
                            lead_data['last_name'],
                            company_data['domain']
                        )
                        if email_result and email_result.get('email'):
                            lead_data['email'] = email_result['email']
                
                # Calculate lead score
                score = scorer.calculate_score(lead_data)
                
                # Create lead
                lead = Lead(
                    first_name=lead_data.get('first_name', '')[:100],
                    last_name=lead_data.get('last_name', '')[:100],
                    email=lead_data.get('email', '')[:255],
                    phone=lead_data.get('phone', '')[:50],
                    title=lead_data.get('title', '')[:255],
                    linkedin_url=lead_data.get('linkedin_url', '')[:500],
                    source=lead_data.get('source', request.source)[:50],
                    status="new",
                    score=score,
                    owner_id=current_user.id,
                    company_id=company_id
                )
                db.add(lead)
                created_leads.append(lead)
                
            except Exception as e:
                print(f"❌ Error creating lead: {e}")
                continue
        
        db.commit()
        
        print(f"✅ Successfully created {len(created_leads)} leads in database")
        
        # Schedule AI analysis for all new leads in background
        if created_leads:
            background_tasks.add_task(
                _batch_ai_analysis_background,
                [lead.id for lead in created_leads],
                current_user.id
            )
        
        return {
            "message": f"Successfully scraped {len(created_leads)} leads! AI analysis running in background.", 
            "status": "completed",
            "count": len(created_leads),
            "ai_analysis_scheduled": True
        }
        
    except Exception as e:
        print(f"❌ Error in scraping: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Scraping failed: {str(e)}")

# ============================
# BACKGROUND TASK FUNCTIONS
# ============================

def _perform_ai_analysis_background(lead_id: int, user_id: int):
    """Background task for AI analysis of a single lead"""
    try:
        from database import SessionLocal
        from utils.ai_analyzer_2 import GroqEnhancedAnalyzer
        from utils.data_quality_engine import DataQualityEngine
        
        db = SessionLocal()
        analyzer = GroqEnhancedAnalyzer()
        quality_engine = DataQualityEngine()
        
        lead = db.query(Lead).filter(Lead.id == lead_id).first()
        if not lead:
            return
        
        # AI analysis
        insights = analyzer.analyze_lead(lead)
        
        # Store AI prediction
        prediction = LeadPrediction(
            lead_id=lead_id,
            model_type="priority",
            model_version="1.0.0",
            prediction_score=insights.get('priority_score', 50),
            confidence=insights.get('confidence', 0.5),
            explanation=insights.get('reason', 'Background AI analysis'),
            processing_time_ms=100.0
        )
        db.add(prediction)
        
        # Quality assessment
        quality_report = quality_engine.assess_lead_quality(lead)
        
        quality_score = DataQualityScore(
            lead_id=lead_id,
            overall_score=quality_report.get('overall_score', 50),
            email_quality=quality_report.get('email_quality', 0),
            completeness_score=quality_report.get('completeness_score', 0),
            accuracy_score=quality_report.get('accuracy_score', 0),
            issues_found=json.dumps(quality_report.get('issues', [])),
            suggestions=json.dumps(quality_report.get('suggestions', [])),
            assessment_method="background_analysis"
        )
        db.add(quality_score)
        
        # Create insight record
        insight = LeadInsight(
            lead_id=lead_id,
            insight_type="priority",
            insight_text=insights.get('reason', 'AI analysis completed'),
            action_items=json.dumps(insights.get('actions', [])),
            priority_level=_get_priority_level(insights.get('priority_score', 50)),
            confidence_score=insights.get('confidence', 0.5),
            impact_score=insights.get('priority_score', 50),
            generated_by="background_ai_analyzer"
        )
        db.add(insight)
        
        db.commit()
        db.close()
        
        print(f"✅ Background AI analysis completed for lead {lead_id}")
        
    except Exception as e:
        print(f"❌ Background AI analysis failed for lead {lead_id}: {e}")

def _batch_ai_analysis_background(lead_ids: List[int], user_id: int):
    """Background task for batch AI analysis"""
    try:
        from database import SessionLocal
        from utils.ai_analyzer_2 import GroqEnhancedAnalyzer
        from utils.data_quality_engine import DataQualityEngine
        
        db = SessionLocal()
        analyzer = GroqEnhancedAnalyzer()
        quality_engine = DataQualityEngine()
        
        print(f"🔄 Starting background batch AI analysis for {len(lead_ids)} leads")
        
        for lead_id in lead_ids:
            try:
                lead = db.query(Lead).filter(Lead.id == lead_id).first()
                if not lead:
                    continue
                
                # AI analysis
                insights = analyzer.analyze_lead(lead)
                
                # Store prediction
                prediction = LeadPrediction(
                    lead_id=lead_id,
                    model_type="priority",
                    model_version="1.0.0",
                    prediction_score=insights.get('priority_score', 50),
                    confidence=insights.get('confidence', 0.5),
                    explanation=insights.get('reason', 'Batch AI analysis'),
                    processing_time_ms=75.0
                )
                db.add(prediction)
                
                # Quality assessment
                quality_report = quality_engine.assess_lead_quality(lead)
                
                quality_score = DataQualityScore(
                    lead_id=lead_id,
                    overall_score=quality_report.get('overall_score', 50),
                    email_quality=quality_report.get('email_quality', 0),
                    completeness_score=quality_report.get('completeness_score', 0),
                    accuracy_score=quality_report.get('accuracy_score', 0),
                    issues_found=json.dumps(quality_report.get('issues', [])),
                    suggestions=json.dumps(quality_report.get('suggestions', [])),
                    assessment_method="batch_background"
                )
                db.add(quality_score)
                
            except Exception as e:
                print(f"❌ Error processing lead {lead_id} in batch: {e}")
                continue
        
        db.commit()
        db.close()
        
        print(f"✅ Background batch AI analysis completed for {len(lead_ids)} leads")
        
    except Exception as e:
        print(f"❌ Background batch AI analysis failed: {e}")

def _detect_duplicates_background(user_id: int, lead_ids: List[int]):
    """Background task for duplicate detection"""
    try:
        from database import SessionLocal
        from utils.data_quality_engine import DataQualityEngine
        from models import DuplicateDetection
        
        db = SessionLocal()
        quality_engine = DataQualityEngine()
        
        print(f"🔄 Starting background duplicate detection for user {user_id}")
        
        # Get leads to analyze
        leads = db.query(Lead).filter(
            Lead.owner_id == user_id,
            Lead.id.in_(lead_ids)
        ).all()
        
        if len(leads) < 2:
            return
        
        duplicates = quality_engine.detect_duplicates(leads, 0.8)
        
        for duplicate_group in duplicates:
            primary_id = duplicate_group['primary_lead_id']
            for candidate in duplicate_group['duplicates']:
                # Check if already exists
                existing = db.query(DuplicateDetection).filter(
                    DuplicateDetection.primary_lead_id == primary_id,
                    DuplicateDetection.duplicate_lead_id == candidate['lead_id']
                ).first()
                
                if not existing:
                    duplicate_record = DuplicateDetection(
                        primary_lead_id=primary_id,
                        duplicate_lead_id=candidate['lead_id'],
                        match_score=candidate['confidence'] * 100,
                        match_type="fuzzy" if candidate['confidence'] < 0.9 else "exact",
                        matching_fields=json.dumps(candidate.get('matching_fields', [])),
                        detection_method="background_ai_detection",
                        auto_merge_recommended=candidate['confidence'] >= 0.95
                    )
                    db.add(duplicate_record)
        
        db.commit()
        db.close()
        
        print(f"✅ Background duplicate detection completed")
        
    except Exception as e:
        print(f"❌ Background duplicate detection failed: {e}")

# ============================
# ADDITIONAL AI ENDPOINTS
# ============================

@router.post("/bulk-import", response_model=dict)
def bulk_import_leads(
    import_data: BulkLeadImport,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Import multiple leads at once with AI analysis"""
    try:
        from utils.lead_scorer import LeadScorer
        
        scorer = LeadScorer()
        created_leads = []
        
        for lead_data in import_data.leads:
            score = scorer.calculate_score(lead_data.dict())
                
            db_lead = Lead(
                **lead_data.dict(),
                owner_id=current_user.id,
                campaign_id=import_data.campaign_id,
                score=score
            )
            db.add(db_lead)
            created_leads.append(db_lead)
        
        db.commit()
        
        # Get lead IDs for background processing
        lead_ids = [lead.id for lead in created_leads]
        
        # Schedule AI analysis
        background_tasks.add_task(
            _batch_ai_analysis_background,
            lead_ids,
            current_user.id
        )
        
        return {
            "message": f"Successfully imported {len(created_leads)} leads",
            "count": len(created_leads),
            "ai_analysis_scheduled": True
        }
        
    except Exception as e:
        print(f"❌ Bulk import failed: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Bulk import failed: {str(e)}")

@router.get("/export")
def export_leads(
    format: str = "csv",
    status: Optional[str] = None,
    include_ai_data: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Export leads in various formats with optional AI data"""
    try:
        from utils.export_service import ExportService
        
        # Get leads to export
        query = db.query(Lead).filter(Lead.owner_id == current_user.id)
        if status:
            query = query.filter(Lead.status == status)
        leads = query.all()
        
        # Convert to dictionary format
        leads_data = []
        for lead in leads:
            lead_dict = {
                "id": lead.id,
                "first_name": lead.first_name,
                "last_name": lead.last_name,
                "email": lead.email,
                "phone": lead.phone,
                "title": lead.title,
                "linkedin_url": lead.linkedin_url,
                "source": lead.source,
                "status": lead.status,
                "score": lead.score,
                "created_at": lead.created_at,
                "company": {
                    "name": lead.company.name if lead.company else "",
                    "domain": lead.company.domain if lead.company else "",
                    "industry": lead.company.industry if lead.company else "",
                    "location": lead.company.location if lead.company else ""
                }
            }
            
            # Include AI data if requested
            if include_ai_data:
                # Get latest AI prediction
                latest_prediction = db.query(LeadPrediction).filter(
                    LeadPrediction.lead_id == lead.id
                ).order_by(LeadPrediction.created_at.desc()).first()
                
                if latest_prediction:
                    lead_dict["ai_priority_score"] = latest_prediction.prediction_score
                    lead_dict["ai_confidence"] = latest_prediction.confidence
                    lead_dict["ai_explanation"] = latest_prediction.explanation
                
                # Get latest quality score
                latest_quality = db.query(DataQualityScore).filter(
                    DataQualityScore.lead_id == lead.id
                ).order_by(DataQualityScore.created_at.desc()).first()
                
                if latest_quality:
                    lead_dict["data_quality_score"] = latest_quality.overall_score
                    lead_dict["completeness_score"] = latest_quality.completeness_score
                    lead_dict["accuracy_score"] = latest_quality.accuracy_score
            
            leads_data.append(lead_dict)
        
        export_service = ExportService()
        export_data = export_service.export_leads(leads_data, format)
        
        return {
            "message": f"Exported {len(leads_data)} leads in {format} format",
            "data": export_data.getvalue().decode() if format == "csv" else export_data.getvalue(),
            "count": len(leads_data),
            "includes_ai_data": include_ai_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.post("/enrich/{lead_id}")
def enrich_specific_lead(
    lead_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Enrich a specific lead with additional data"""
    try:
        from utils.email_finder import EmailFinder
        from utils.data_enrichment import DataEnrichment
        
        lead = db.query(Lead).filter(
            Lead.id == lead_id,
            Lead.owner_id == current_user.id
        ).first()
        
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        
        email_finder = EmailFinder()
        enrichment_service = DataEnrichment()
        enriched_fields = []
        
        # Enrich email if missing
        if not lead.email and lead.first_name and lead.last_name and lead.company:
            email_result = email_finder.find_email(
                lead.first_name,
                lead.last_name,
                lead.company.domain
            )
            if email_result and email_result.get('email'):
                lead.email = email_result['email']
                enriched_fields.append("email")
        
        # Enrich company data if available
        if lead.company and lead.company.domain:
            try:
                company_data = enrichment_service.enrich_company_data(lead.company.domain)
                if company_data and company_data.get('company_info'):
                    info = company_data['company_info']
                    if not lead.company.size and info.get('employees'):
                        lead.company.size = info['employees']
                        enriched_fields.append("company_size")
                    
                    if not lead.company.revenue and info.get('annual_revenue'):
                        lead.company.revenue = info['annual_revenue']
                        enriched_fields.append("company_revenue")
                    
                    if info.get('technologies'):
                        lead.company.technologies = json.dumps(info['technologies'])
                        enriched_fields.append("technologies")
            except:
                pass  # Enrichment is optional
        
        # Enrich person data if email available
        if lead.email:
            try:
                person_data = enrichment_service.enrich_person_data(lead.email)
                if person_data and person_data.get('social_profiles'):
                    if not lead.linkedin_url and person_data['social_profiles'].get('linkedin'):
                        lead.linkedin_url = person_data['social_profiles']['linkedin']
                        enriched_fields.append("linkedin_url")
            except:
                pass  # Enrichment is optional
        
        db.commit()
        
        # Schedule AI re-analysis if data was enriched
        if enriched_fields:
            background_tasks.add_task(
                _perform_ai_analysis_background,
                lead.id,
                current_user.id
            )
        
        return {
            "message": "Lead enrichment completed" if enriched_fields else "No additional data found",
            "lead": lead,
            "enriched_fields": enriched_fields,
            "ai_reanalysis_scheduled": len(enriched_fields) > 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lead enrichment failed: {str(e)}")

@router.get("/stats")
def get_lead_statistics(
    include_ai_metrics: bool = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive lead statistics with AI metrics"""
    try:
        from sqlalchemy import func
        
        # Total leads
        total_leads = db.query(func.count(Lead.id)).filter(Lead.owner_id == current_user.id).scalar()
        
        # Leads by status
        status_stats = db.query(
            Lead.status,
            func.count(Lead.id).label('count')
        ).filter(Lead.owner_id == current_user.id).group_by(Lead.status).all()
        
        # Leads by source
        source_stats = db.query(
            Lead.source,
            func.count(Lead.id).label('count')
        ).filter(Lead.owner_id == current_user.id).group_by(Lead.source).all()
        
        # Email coverage
        leads_with_email = db.query(func.count(Lead.id)).filter(
            Lead.owner_id == current_user.id,
            Lead.email.isnot(None),
            Lead.email != ''
        ).scalar()
        
        # Average score
        avg_score = db.query(func.avg(Lead.score)).filter(Lead.owner_id == current_user.id).scalar()
        
        stats = {
            "total_leads": total_leads,
            "email_coverage": round((leads_with_email / total_leads * 100), 2) if total_leads > 0 else 0,
            "average_score": round(avg_score, 2) if avg_score else 0,
            "status_breakdown": [{"status": status, "count": count} for status, count in status_stats],
            "source_breakdown": [{"source": source or "unknown", "count": count} for source, count in source_stats]
        }
        
        # Add AI metrics if requested
        if include_ai_metrics:
            # AI predictions stats
            ai_predictions = db.query(func.count(LeadPrediction.id)).join(Lead).filter(
                Lead.owner_id == current_user.id
            ).scalar()
            
            # Average AI confidence
            avg_ai_confidence = db.query(func.avg(LeadPrediction.confidence)).join(Lead).filter(
                Lead.owner_id == current_user.id
            ).scalar()
            
            # High priority leads (AI score > 80)
            high_priority_leads = db.query(func.count(LeadPrediction.id)).join(Lead).filter(
                Lead.owner_id == current_user.id,
                LeadPrediction.prediction_score > 80
            ).scalar()
            
            # Average data quality
            avg_quality = db.query(func.avg(DataQualityScore.overall_score)).join(Lead).filter(
                Lead.owner_id == current_user.id
            ).scalar()
            
            # Quality issues count
            quality_issues = db.query(func.count(DataQualityScore.id)).join(Lead).filter(
                Lead.owner_id == current_user.id,
                DataQualityScore.overall_score < 70
            ).scalar()
            
            stats["ai_metrics"] = {
                "leads_analyzed": ai_predictions,
                "average_ai_confidence": round(avg_ai_confidence, 3) if avg_ai_confidence else 0,
                "high_priority_leads": high_priority_leads,
                "average_data_quality": round(avg_quality, 2) if avg_quality else 0,
                "quality_issues_detected": quality_issues,
                "ai_coverage": round((ai_predictions / total_leads * 100), 2) if total_leads > 0 else 0
            }
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statistics calculation failed: {str(e)}")

@router.post("/ai-feedback")
def submit_ai_feedback(
    lead_id: int,
    feedback_type: str,  # 'helpful', 'not_helpful', 'incorrect'
    feedback_text: Optional[str] = None,
    actual_outcome: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Submit feedback on AI predictions for continuous improvement"""
    try:
        from models import AIFeedback
        
        # Get latest prediction for this lead
        latest_prediction = db.query(LeadPrediction).filter(
            LeadPrediction.lead_id == lead_id
        ).order_by(LeadPrediction.created_at.desc()).first()
        
        if not latest_prediction:
            raise HTTPException(status_code=404, detail="No AI prediction found for this lead")
        
        # Create feedback record
        feedback = AIFeedback(
            prediction_id=latest_prediction.id,
            user_id=current_user.id,
            feedback_type="rating",
            is_helpful=feedback_type == "helpful",
            is_accurate=feedback_type != "incorrect",
            actual_outcome=actual_outcome,
            feedback_text=feedback_text,
            sentiment="positive" if feedback_type == "helpful" else "negative"
        )
        
        db.add(feedback)
        db.commit()
        
        return {
            "message": "Feedback submitted successfully",
            "feedback_id": feedback.id,
            "will_improve_ai": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

# ============================
# UTILITY FUNCTIONS
# ============================

def _get_priority_level(score: float) -> str:
    """Convert numeric score to priority level"""
    if score >= 85:
        return "critical"
    elif score >= 70:
        return "high"
    elif score >= 50:
        return "medium"
    else:
        return "low"

def _calculate_business_impact(lead_data: dict) -> float:
    """Calculate business impact score for a lead"""
    impact = 50.0  # Base score
    
    # Company size impact
    size = lead_data.get('company', {}).get('size', '')
    if '500+' in size:
        impact += 20
    elif '200-500' in size:
        impact += 15
    elif '50-200' in size:
        impact += 10
    
    # Title seniority impact
    title = lead_data.get('title', '').lower()
    if any(word in title for word in ['ceo', 'president', 'founder']):
        impact += 25
    elif any(word in title for word in ['vp', 'director', 'head']):
        impact += 15
    elif any(word in title for word in ['manager', 'lead']):
        impact += 10
    
    # Industry impact
    industry = lead_data.get('company', {}).get('industry', '').lower()
    if any(word in industry for word in ['technology', 'software', 'ai', 'tech']):
        impact += 15
    elif any(word in industry for word in ['finance', 'healthcare', 'consulting']):
        impact += 10
    
    return min(impact, 100.0)

def _validate_ai_operation_permissions(user: User) -> bool:
    """Validate if user has permissions for AI operations"""
    # Could implement premium features, rate limiting, etc.
    return True

def _log_ai_operation(operation_type: str, entity_id: int, user_id: int, status: str, details: dict = None):
    """Log AI operations for monitoring and debugging"""
    try:
        from database import SessionLocal
        from models import AIProcessingLog
        
        db = SessionLocal()
        
        log_entry = AIProcessingLog(
            operation_type=operation_type,
            entity_type="lead",
            entity_id=entity_id,
            processing_status=status,
            user_id=user_id,
            result_summary=json.dumps(details) if details else None,
            end_time=datetime.utcnow()
        )
        
        db.add(log_entry)
        db.commit()
        db.close()
        
    except Exception as e:
        print(f"Failed to log AI operation: {e}")

# Export router for main application
__all__ = ['router']