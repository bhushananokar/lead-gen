# routers/ai_insights.py - Dedicated AI Insights and Management Router
from asyncio.log import logger
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_, text
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import asyncio

from database import get_db, get_db_for_ai
from models import (
    User, Lead, Company, Campaign,
    LeadPrediction, DataQualityScore, LeadInsight,
    ModelMetadata, AIProcessingLog, AIFeedback,
    LeadSimilarity, DuplicateDetection, OptimalContactTiming,
    EnrichmentSuggestion
)
from schemas import (
    AILeadInsight, DataQualityReport, BatchAnalysisResult,
    LeadSimilarityResult, DuplicateDetectionResult, 
    ModelPerformanceMetrics, AISystemHealth,
    AIAnalysisRequest, SimilarLeadsRequest, DuplicateDetectionRequest,
    DataQualityAssessmentRequest, ModelTrainingRequest,
    AIProcessingStatus, OptimalContactTiming as ContactTimingSchema,
    LeadEnrichmentSuggestion, AIFeedbackSubmission
)
from routers.auth import get_current_user

router = APIRouter()

# ============================
# AI SYSTEM STATUS & HEALTH
# ============================

@router.get("/status", response_model=AISystemHealth)
def get_ai_system_status(
    db: Session = Depends(get_db_for_ai)
):
    """Get comprehensive AI system health and status"""
    try:
        # Check active models
        active_models = db.query(ModelMetadata).filter(
            ModelMetadata.is_active == True
        ).all()
        
        model_names = [model.model_name for model in active_models]
        
        # Get model performance metrics
        model_performance = {}
        for model in active_models:
            model_performance[model.model_name] = ModelPerformanceMetrics(
                model_name=model.model_name,
                model_version=model.version,
                accuracy=model.accuracy_score or 0,
                precision=model.precision_score or 0,
                recall=model.recall_score or 0,
                f1_score=model.f1_score or 0,
                predictions_made=model.predictions_made or 0,
                average_prediction_time_ms=model.avg_prediction_time_ms or 0,
                last_training_date=model.training_date or datetime.utcnow(),
                is_active=model.is_active,
                confidence_distribution=json.loads(model.performance_metrics) if model.performance_metrics else {}
            )
        
        # Recent predictions count
        recent_predictions = db.query(func.count(LeadPrediction.id)).filter(
            LeadPrediction.created_at >= datetime.utcnow() - timedelta(hours=24)
        ).scalar()
        
        # Average response time
        avg_response_time = db.query(
            func.avg(AIProcessingLog.processing_duration_ms)
        ).filter(
            AIProcessingLog.start_time >= datetime.utcnow() - timedelta(hours=24),
            AIProcessingLog.processing_status == 'completed'
        ).scalar()
        
        # Error rate
        total_ops = db.query(func.count(AIProcessingLog.id)).filter(
            AIProcessingLog.start_time >= datetime.utcnow() - timedelta(hours=24)
        ).scalar()
        
        failed_ops = db.query(func.count(AIProcessingLog.id)).filter(
            AIProcessingLog.start_time >= datetime.utcnow() - timedelta(hours=24),
            AIProcessingLog.processing_status == 'failed'
        ).scalar()
        
        error_rate = (failed_ops / max(total_ops, 1)) * 100
        
        # System load (estimate based on recent activity)
        system_load = min(100, (recent_predictions / 1000) * 100)
        
        # Pending operations
        pending_ops = db.query(func.count(AIProcessingLog.id)).filter(
            AIProcessingLog.processing_status == 'started'
        ).scalar()
        
        # Determine system status
        if error_rate > 20 or not active_models:
            status = "critical"
        elif error_rate > 10 or system_load > 80:
            status = "degraded"
        else:
            status = "healthy"
        
        # Available features based on active models
        available_features = []
        for model in active_models:
            if "priority" in model.model_name or "classifier" in model.model_name:
                available_features.extend(["lead_prioritization", "smart_scoring"])
            if "quality" in model.model_name:
                available_features.extend(["data_quality_assessment", "automated_cleanup"])
            if "similarity" in model.model_name:
                available_features.extend(["lead_similarity", "duplicate_detection"])
        
        # Remove duplicates
        available_features = list(set(available_features))
        
        return AISystemHealth(
            system_status=status,
            active_models=model_names,
            model_performance=model_performance,
            recent_predictions=recent_predictions or 0,
            average_response_time_ms=round(avg_response_time or 0, 2),
            error_rate_percentage=round(error_rate, 2),
            last_training_date=max([model.training_date for model in active_models], default=None),
            pending_operations=pending_ops or 0,
            system_load_percentage=round(system_load, 2),
            available_features=available_features
        )
        
    except Exception as e:
        # Return minimal status if there's an error
        return AISystemHealth(
            system_status="critical",
            active_models=[],
            model_performance={},
            recent_predictions=0,
            average_response_time_ms=0,
            error_rate_percentage=100,
            last_training_date=None,
            pending_operations=0,
            system_load_percentage=0,
            available_features=[]
        )

# FIXED VERSION: AI Dashboard Endpoint
# Replace the existing get_ai_insights_dashboard function in ai_insights.py

@router.get("/dashboard")
def get_ai_insights_dashboard(
    time_range_days: int = 7,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Get AI insights dashboard with user-specific metrics - FIXED VERSION"""
    try:
        start_date = datetime.utcnow() - timedelta(days=time_range_days)
        
        # Initialize default response structure
        response_data = {
            "summary": {
                "leads_analyzed": 0,
                "total_predictions": 0,
                "average_priority_score": 0,
                "average_confidence": 0,
                "time_range_days": time_range_days
            },
            "priority_distribution": [],
            "recent_insights": [],
            "quality_trend": [],
            "model_usage": []
        }
        
        # Safety check: Verify user exists and tables exist
        if not current_user or not current_user.id:
            logger.warning("No current user found for AI dashboard request")
            return response_data
        
        # Check if required tables exist
        required_tables = ['leads', 'lead_predictions', 'lead_insights', 'data_quality_scores']
        for table in required_tables:
            try:
                db.execute(text(f"SELECT COUNT(*) FROM {table} LIMIT 1")).fetchone()
            except Exception as e:
                logger.warning(f"Table {table} not accessible: {e}")
                return response_data
        
        # User's AI activity summary - with safety checks
        try:
            ai_activity = db.query(
                func.count(func.distinct(LeadPrediction.lead_id)).label('leads_analyzed'),
                func.count(LeadPrediction.id).label('total_predictions'),
                func.avg(LeadPrediction.prediction_score).label('avg_priority'),
                func.avg(LeadPrediction.confidence).label('avg_confidence')
            ).join(Lead, LeadPrediction.lead_id == Lead.id).filter(
                Lead.owner_id == current_user.id,
                LeadPrediction.created_at >= start_date
            ).first()
            
            if ai_activity:
                response_data["summary"].update({
                    "leads_analyzed": ai_activity.leads_analyzed or 0,
                    "total_predictions": ai_activity.total_predictions or 0,
                    "average_priority_score": round(ai_activity.avg_priority or 0, 2),
                    "average_confidence": round(ai_activity.avg_confidence or 0, 3)
                })
                
        except Exception as e:
            logger.warning(f"Failed to get AI activity summary: {e}")
        
        # Priority distribution - with safety checks
        try:
            priority_distribution = db.query(
                func.case([
                    (LeadPrediction.prediction_score >= 85, 'critical'),
                    (LeadPrediction.prediction_score >= 70, 'high'),
                    (LeadPrediction.prediction_score >= 50, 'medium')
                ], else_='low').label('priority'),
                func.count(LeadPrediction.id).label('count')
            ).join(Lead, LeadPrediction.lead_id == Lead.id).filter(
                Lead.owner_id == current_user.id,
                LeadPrediction.created_at >= start_date
            ).group_by('priority').all()
            
            response_data["priority_distribution"] = [
                {"priority": priority, "count": count}
                for priority, count in priority_distribution
            ]
            
        except Exception as e:
            logger.warning(f"Failed to get priority distribution: {e}")
        
        # Recent insights - with safety checks
        try:
            recent_insights = db.query(LeadInsight).join(
                Lead, LeadInsight.lead_id == Lead.id
            ).filter(
                Lead.owner_id == current_user.id,
                LeadInsight.status == 'active',
                LeadInsight.created_at >= start_date
            ).order_by(
                LeadInsight.priority_level.desc(), 
                LeadInsight.created_at.desc()
            ).limit(10).all()
            
            response_data["recent_insights"] = [
                {
                    "id": insight.id,
                    "lead_id": insight.lead_id,
                    "type": insight.insight_type,
                    "text": insight.insight_text,
                    "priority": insight.priority_level,
                    "confidence": insight.confidence_score,
                    "created_at": insight.created_at.isoformat() if insight.created_at else None
                }
                for insight in recent_insights
            ]
            
        except Exception as e:
            logger.warning(f"Failed to get recent insights: {e}")
        
        # Data quality trends - with safety checks
        try:
            quality_trend = db.query(
                func.date(DataQualityScore.created_at).label('date'),
                func.avg(DataQualityScore.overall_score).label('avg_quality')
            ).join(Lead, DataQualityScore.lead_id == Lead.id).filter(
                Lead.owner_id == current_user.id,
                DataQualityScore.created_at >= start_date
            ).group_by(func.date(DataQualityScore.created_at)).order_by('date').all()
            
            response_data["quality_trend"] = [
                {
                    "date": str(date),
                    "average_quality": round(float(quality), 2) if quality else 0
                }
                for date, quality in quality_trend
            ]
            
        except Exception as e:
            logger.warning(f"Failed to get quality trend: {e}")
        
        # Model usage statistics - with safety checks
        try:
            model_usage = db.query(
                LeadPrediction.model_type,
                func.count(LeadPrediction.id).label('usage_count')
            ).join(Lead, LeadPrediction.lead_id == Lead.id).filter(
                Lead.owner_id == current_user.id,
                LeadPrediction.created_at >= start_date
            ).group_by(LeadPrediction.model_type).all()
            
            response_data["model_usage"] = [
                {"model_type": model_type or "unknown", "usage_count": count}
                for model_type, count in model_usage
            ]
            
        except Exception as e:
            logger.warning(f"Failed to get model usage: {e}")
        
        logger.info(f"AI dashboard loaded successfully for user {current_user.id}")
        return response_data
        
    except Exception as e:
        logger.error(f"AI dashboard critical error: {str(e)}")
        # Return minimal safe response instead of 500 error
        return {
            "summary": {
                "leads_analyzed": 0,
                "total_predictions": 0,
                "average_priority_score": 0,
                "average_confidence": 0,
                "time_range_days": time_range_days
            },
            "priority_distribution": [],
            "recent_insights": [],
            "quality_trend": [],
            "model_usage": [],
            "error_message": "Dashboard data temporarily unavailable"
        }

# ============================
# ADVANCED AI ANALYSIS
# ============================

@router.post("/analyze", response_model=BatchAnalysisResult)
def perform_ai_analysis(
    request: AIAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Perform comprehensive AI analysis on leads"""
    try:
        from utils.ai_lead_analyzer import AILeadAnalyzer
        from utils.data_quality_engine import DataQualityEngine
        
        # Get leads to analyze
        if request.lead_ids:
            leads = db.query(Lead).filter(
                Lead.id.in_(request.lead_ids),
                Lead.owner_id == current_user.id
            ).all()
        else:
            # Analyze all user's leads
            leads = db.query(Lead).filter(Lead.owner_id == current_user.id).limit(100).all()
        
        if not leads:
            raise HTTPException(status_code=404, detail="No leads found for analysis")
        
        analyzer = AILeadAnalyzer()
        quality_engine = DataQualityEngine()
        
        results = {
            "total_analyzed": len(leads),
            "high_priority_leads": [],
            "medium_priority_leads": [],
            "low_priority_leads": [],
            "quality_issues_found": 0,
            "insights_generated": 0,
            "processing_time_seconds": 0
        }
        
        start_time = datetime.utcnow()
        
        for lead in leads:
            try:
                # Perform requested analysis types
                if "priority" in request.analysis_types:
                    insights = analyzer.analyze_lead(lead)
                    priority_score = insights.get('priority_score', 50)
                    
                    if priority_score >= 80:
                        results["high_priority_leads"].append(lead.id)
                    elif priority_score >= 60:
                        results["medium_priority_leads"].append(lead.id)
                    else:
                        results["low_priority_leads"].append(lead.id)
                    
                    # Store prediction
                    if priority_score >= request.priority_threshold:
                        prediction = LeadPrediction(
                            lead_id=lead.id,
                            model_type="priority",
                            model_version="1.0.0",
                            prediction_score=priority_score,
                            confidence=insights.get('confidence', 0.5),
                            explanation=insights.get('reason', 'AI analysis'),
                            processing_time_ms=50.0
                        )
                        db.add(prediction)
                        results["insights_generated"] += 1
                
                if "quality" in request.analysis_types:
                    quality_report = quality_engine.assess_lead_quality(lead)
                    
                    if quality_report.get('overall_score', 100) < 70:
                        results["quality_issues_found"] += 1
                    
                    # Store quality assessment
                    quality_score = DataQualityScore(
                        lead_id=lead.id,
                        overall_score=quality_report.get('overall_score', 50),
                        email_quality=quality_report.get('email_quality', 0),
                        completeness_score=quality_report.get('completeness_score', 0),
                        accuracy_score=quality_report.get('accuracy_score', 0),
                        issues_found=json.dumps(quality_report.get('issues', [])),
                        suggestions=json.dumps(quality_report.get('suggestions', [])),
                        assessment_method="ai_batch_analysis"
                    )
                    db.add(quality_score)
                
                if "timing" in request.analysis_types and request.include_timing:
                    # Add optimal timing analysis
                    timing_analysis = analyzer.predict_optimal_timing(lead)
                    if timing_analysis:
                        timing_record = OptimalContactTiming(
                            lead_id=lead.id,
                            best_day_of_week=timing_analysis.get('best_day', 'tuesday'),
                            best_hour_start=timing_analysis.get('best_hour_start', 10),
                            best_hour_end=timing_analysis.get('best_hour_end', 12),
                            timezone=timing_analysis.get('timezone', 'UTC'),
                            confidence_score=timing_analysis.get('confidence', 0.5),
                            reasoning=timing_analysis.get('reasoning', 'AI timing analysis')
                        )
                        db.add(timing_record)
                
            except Exception as e:
                print(f"Error analyzing lead {lead.id}: {e}")
                continue
        
        # Commit all changes
        db.commit()
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        results["processing_time_seconds"] = round(processing_time, 2)
        
        # Schedule similarity analysis if requested
        if "similarity" in request.analysis_types:
            background_tasks.add_task(
                _batch_similarity_analysis,
                [lead.id for lead in leads],
                current_user.id
            )
        
        return BatchAnalysisResult(**results)
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")


@router.post("/find-similar", response_model=LeadSimilarityResult)
def find_similar_leads(
    request: SimilarLeadsRequest,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Find leads similar to a reference lead using advanced AI - FIXED VERSION"""
    try:
        from utils.ai_lead_analyzer import AILeadAnalyzer
        
        # Get reference lead
        reference_lead = db.query(Lead).filter(
            Lead.id == request.reference_lead_id,
            Lead.owner_id == current_user.id
        ).first()
        
        if not reference_lead:
            raise HTTPException(status_code=404, detail="Reference lead not found")
        
        # Get comparison leads
        comparison_leads = db.query(Lead).filter(
            Lead.owner_id == current_user.id,
            Lead.id != request.reference_lead_id
        ).all()
        
        if not comparison_leads:
            return LeadSimilarityResult(
                reference_lead_id=request.reference_lead_id,
                similar_leads=[],
                similarity_factors=[],
                confidence=0.0,
                similarity_scores=[],
                business_relevance=0.0,
                recommended_actions=[]
            )
        
        # FIXED: Convert Lead objects to dictionaries before analysis
        def lead_to_dict(lead):
            """Convert SQLAlchemy Lead object to dictionary"""
            return {
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
                'updated_at': getattr(lead, 'updated_at', None),
                'company': {
                    'name': getattr(lead, 'company', '') or getattr(lead, 'company_name', '') or '',
                    'industry': getattr(lead, 'industry', '') or '',
                    'website': getattr(lead, 'website', '') or '',
                    'size': getattr(lead, 'company_size', '') or '',
                    'location': getattr(lead, 'location', '') or ''
                }
            }
        
        # Convert leads to dictionaries
        reference_lead_dict = lead_to_dict(reference_lead)
        comparison_leads_dicts = [lead_to_dict(lead) for lead in comparison_leads]
        
        # Perform similarity analysis with dictionaries
        analyzer = AILeadAnalyzer()
        similarity_results = analyzer.find_similar_leads(
            reference_lead_dict,  # Now a dictionary
            comparison_leads_dicts,  # Now a list of dictionaries
            request.similarity_threshold,
            request.max_results
        )
        
        # Store similarity relationships
        try:
            for similar_lead in similarity_results.get('similar_leads', []):
                similarity_record = LeadSimilarity(
                    lead_id_1=request.reference_lead_id,
                    lead_id_2=similar_lead['lead_id'],
                    similarity_score=similar_lead['similarity_score'],
                    similarity_type='ai_analysis',
                    matching_features=json.dumps(similar_lead.get('matching_features', [])),
                    calculation_method='ai_similarity_engine',
                    confidence=similarity_results.get('confidence', 0.5),
                    business_relevance=similarity_results.get('business_relevance', 50.0)
                )
                db.add(similarity_record)
            
            db.commit()
        except Exception as storage_error:
            print(f"⚠️ Failed to store similarity results: {storage_error}")
            db.rollback()
        
        return LeadSimilarityResult(
            reference_lead_id=request.reference_lead_id,
            similar_leads=similarity_results.get('similar_leads', []),
            similarity_factors=similarity_results.get('factors', []),
            confidence=similarity_results.get('confidence', 0.0),
            similarity_scores=similarity_results.get('scores', []),
            business_relevance=similarity_results.get('business_relevance', 0.0),
            recommended_actions=similarity_results.get('actions', [])
        )
        
    except Exception as e:
        print(f"❌ Similarity analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Similarity analysis failed: {str(e)}")

@router.post("/detect-duplicates", response_model=DuplicateDetectionResult)
def detect_duplicates(
    request: DuplicateDetectionRequest,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Detect duplicate leads using advanced AI matching"""
    try:
        from utils.data_quality_engine import DataQualityEngine
        
        # Get leads to analyze
        if request.lead_ids:
            leads = db.query(Lead).filter(
                Lead.id.in_(request.lead_ids),
                Lead.owner_id == current_user.id
            ).all()
        else:
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
        
        quality_engine = DataQualityEngine()
        duplicates = quality_engine.detect_duplicates(
            leads, 
            request.detection_threshold,
            include_fuzzy=request.include_fuzzy_matching
        )
        
        auto_merge = []
        manual_review = []
        confidence_scores = {}
        merge_strategies = {}
        total_duplicates = 0
        
        for duplicate_group in duplicates:
            primary_id = duplicate_group['primary_lead_id']
            candidates = duplicate_group['duplicates']
            total_duplicates += len(candidates)
            
            for candidate in candidates:
                candidate_id = candidate['lead_id']
                confidence = candidate['confidence']
                confidence_scores[candidate_id] = confidence
                
                if confidence >= request.auto_merge_threshold:
                    auto_merge.append(candidate_id)
                    merge_strategies[candidate_id] = "auto_merge_safe"
                else:
                    manual_review.append(candidate_id)
                    merge_strategies[candidate_id] = "manual_review_required"
                
                # Store duplicate detection record
                if not request.exclude_reviewed:
                    duplicate_record = DuplicateDetection(
                        primary_lead_id=primary_id,
                        duplicate_lead_id=candidate_id,
                        match_score=confidence * 100,
                        match_type="fuzzy" if confidence < 0.9 else "exact",
                        matching_fields=json.dumps(candidate.get('matching_fields', [])),
                        detection_method="ai_advanced_matching",
                        auto_merge_recommended=confidence >= request.auto_merge_threshold,
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
            total_duplicates_found=total_duplicates,
            auto_merge_recommended=auto_merge,
            manual_review_required=manual_review,
            confidence_scores=confidence_scores,
            merge_strategies=merge_strategies
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Duplicate detection failed: {str(e)}")

# ============================
# LEAD OPTIMIZATION & ENRICHMENT
# ============================

@router.get("/optimal-timing/{lead_id}", response_model=ContactTimingSchema)
def get_optimal_contact_timing(
    lead_id: int,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Get AI-predicted optimal contact timing for a lead"""
    try:
        # Check if lead exists and belongs to user
        lead = db.query(Lead).filter(
            Lead.id == lead_id,
            Lead.owner_id == current_user.id
        ).first()
        
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        
        # Check for existing timing prediction
        existing_timing = db.query(OptimalContactTiming).filter(
            OptimalContactTiming.lead_id == lead_id
        ).order_by(OptimalContactTiming.last_updated.desc()).first()
        
        if existing_timing:
            return ContactTimingSchema(
                lead_id=lead_id,
                best_day_of_week=existing_timing.best_day_of_week,
                best_hour_range=f"{existing_timing.best_hour_start}-{existing_timing.best_hour_end}",
                timezone=existing_timing.timezone,
                confidence=existing_timing.confidence_score,
                reasoning=existing_timing.reasoning,
                success_probability=existing_timing.historical_success * 100 if existing_timing.historical_success else None,
                alternative_times=[]
            )
        
        # Generate new timing prediction
        from utils.ai_lead_analyzer import AILeadAnalyzer
        analyzer = AILeadAnalyzer()
        
        timing_prediction = analyzer.predict_optimal_timing(lead)
        
        # Store the prediction
        timing_record = OptimalContactTiming(
            lead_id=lead_id,
            best_day_of_week=timing_prediction.get('best_day', 'tuesday'),
            best_hour_start=timing_prediction.get('best_hour_start', 10),
            best_hour_end=timing_prediction.get('best_hour_end', 12),
            timezone=timing_prediction.get('timezone', 'UTC'),
            confidence_score=timing_prediction.get('confidence', 0.5),
            reasoning=timing_prediction.get('reasoning', 'AI timing analysis'),
            industry_pattern=timing_prediction.get('industry_based', False),
            title_pattern=timing_prediction.get('title_based', False),
            geographic_pattern=timing_prediction.get('location_based', False),
            historical_success=timing_prediction.get('success_rate', 0.5),
            model_version="1.0.0"
        )
        db.add(timing_record)
        db.commit()
        
        return ContactTimingSchema(
            lead_id=lead_id,
            best_day_of_week=timing_prediction.get('best_day', 'tuesday'),
            best_hour_range=f"{timing_prediction.get('best_hour_start', 10)}-{timing_prediction.get('best_hour_end', 12)}",
            timezone=timing_prediction.get('timezone', 'UTC'),
            confidence=timing_prediction.get('confidence', 0.5),
            reasoning=timing_prediction.get('reasoning', 'AI timing analysis'),
            success_probability=timing_prediction.get('success_rate', 50.0),
            alternative_times=timing_prediction.get('alternatives', [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Timing prediction failed: {str(e)}")

@router.get("/enrichment-suggestions/{lead_id}", response_model=LeadEnrichmentSuggestion)
def get_enrichment_suggestions(
    lead_id: int,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Get AI-generated suggestions for lead data enrichment"""
    try:
        # Check if lead exists and belongs to user
        lead = db.query(Lead).filter(
            Lead.id == lead_id,
            Lead.owner_id == current_user.id
        ).first()
        
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        
        from utils.data_quality_engine import DataQualityEngine
        quality_engine = DataQualityEngine()
        
        # Analyze current lead data for enrichment opportunities
        enrichment_analysis = quality_engine.suggest_enrichments(lead)
        
        # Get existing enrichment suggestions
        existing_suggestions = db.query(EnrichmentSuggestion).filter(
            EnrichmentSuggestion.lead_id == lead_id,
            EnrichmentSuggestion.status == 'pending'
        ).all()
        
        # Create new suggestions if none exist
        if not existing_suggestions and enrichment_analysis.get('opportunities'):
            for opportunity in enrichment_analysis['opportunities']:
                suggestion = EnrichmentSuggestion(
                    lead_id=lead_id,
                    missing_field=opportunity['field'],
                    suggested_value=opportunity.get('suggested_value', ''),
                    confidence_score=opportunity.get('confidence', 0.5),
                    data_source=opportunity.get('source', 'ai_analysis'),
                    enrichment_method=opportunity.get('method', 'automated'),
                    priority=opportunity.get('priority', 'medium'),
                    estimated_impact=opportunity.get('impact', 10.0),
                    cost_estimate=opportunity.get('cost', 0.0),
                    effort_estimate=opportunity.get('effort', 'low'),
                    expires_at=datetime.utcnow() + timedelta(days=30)
                )
                db.add(suggestion)
            
            db.commit()
        
        return LeadEnrichmentSuggestion(
            lead_id=lead_id,
            missing_fields=enrichment_analysis.get('missing_fields', []),
            enrichment_opportunities=enrichment_analysis.get('opportunities', []),
            estimated_improvement=enrichment_analysis.get('estimated_improvement', 0.0),
            priority=enrichment_analysis.get('priority', 'medium'),
            suggested_sources=enrichment_analysis.get('sources', []),
            cost_estimate=enrichment_analysis.get('cost_estimate'),
            effort_estimate=enrichment_analysis.get('effort_estimate')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrichment suggestions failed: {str(e)}")

# ============================
# MODEL MANAGEMENT & TRAINING
# ============================

@router.get("/models", response_model=List[ModelPerformanceMetrics])
def list_ai_models(
    active_only: bool = True,
    db: Session = Depends(get_db_for_ai)
):
    """List all AI models with performance metrics"""
    try:
        query = db.query(ModelMetadata)
        
        if active_only:
            query = query.filter(ModelMetadata.is_active == True)
        
        models = query.order_by(ModelMetadata.accuracy_score.desc()).all()
        
        model_metrics = []
        for model in models:
            metrics = ModelPerformanceMetrics(
                model_name=model.model_name,
                model_version=model.version,
                accuracy=model.accuracy_score or 0,
                precision=model.precision_score or 0,
                recall=model.recall_score or 0,
                f1_score=model.f1_score or 0,
                predictions_made=model.predictions_made or 0,
                average_prediction_time_ms=model.avg_prediction_time_ms or 0,
                last_training_date=model.training_date or datetime.utcnow(),
                is_active=model.is_active,
                confidence_distribution=json.loads(model.performance_metrics) if model.performance_metrics else {}
            )
            model_metrics.append(metrics)
        
        return model_metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model listing failed: {str(e)}")

@router.post("/train-model")
def initiate_model_training(
    request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Initiate AI model training with user data"""
    try:
        # Validate user has sufficient data for training
        lead_count = db.query(func.count(Lead.id)).filter(
            Lead.owner_id == current_user.id
        ).scalar()
        
        if lead_count < 50:
            raise HTTPException(
                status_code=400, 
                detail="Insufficient data for training. Need at least 50 leads."
            )
        
        # Check if there's already a training in progress
        existing_training = db.query(AIProcessingLog).filter(
            AIProcessingLog.user_id == current_user.id,
            AIProcessingLog.operation_type == 'training',
            AIProcessingLog.processing_status == 'started'
        ).first()
        
        if existing_training:
            raise HTTPException(
                status_code=409,
                detail="Model training already in progress"
            )
        
        # Create training log entry
        training_log = AIProcessingLog(
            operation_type='training',
            entity_type='model',
            entity_id=0,  # Will be updated when model is created
            model_name=f"{request.model_type.value}_custom",
            model_version="1.0.0",
            processing_status='started',
            user_id=current_user.id,
            batch_id=f"training_{current_user.id}_{int(datetime.utcnow().timestamp())}"
        )
        db.add(training_log)
        db.commit()
        
        # Schedule background training
        background_tasks.add_task(
            _train_model_background,
            request.dict(),
            current_user.id,
            training_log.id
        )
        
        return {
            "message": "Model training initiated",
            "training_id": training_log.id,
            "model_type": request.model_type.value,
            "estimated_duration_minutes": _estimate_training_duration(lead_count, request),
            "status": "started"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training initiation failed: {str(e)}")

@router.get("/training-status/{training_id}", response_model=AIProcessingStatus)
def get_training_status(
    training_id: int,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Get status of model training process"""
    try:
        training_log = db.query(AIProcessingLog).filter(
            AIProcessingLog.id == training_id,
            AIProcessingLog.user_id == current_user.id,
            AIProcessingLog.operation_type == 'training'
        ).first()
        
        if not training_log:
            raise HTTPException(status_code=404, detail="Training process not found")
        
        # Calculate progress
        if training_log.processing_status == 'completed':
            progress = 100.0
        elif training_log.processing_status == 'failed':
            progress = 0.0
        else:
            # Estimate progress based on time elapsed
            if training_log.start_time:
                elapsed = (datetime.utcnow() - training_log.start_time).total_seconds()
                estimated_total = 300  # 5 minutes default
                progress = min(95.0, (elapsed / estimated_total) * 100)
            else:
                progress = 10.0
        
        # Parse results if available
        results_preview = None
        if training_log.result_summary:
            try:
                results_preview = json.loads(training_log.result_summary)
            except:
                pass
        
        return AIProcessingStatus(
            operation_id=str(training_log.id),
            operation_type=training_log.operation_type,
            status=training_log.processing_status,
            progress_percentage=progress,
            estimated_completion_time=training_log.end_time,
            results_preview=results_preview,
            error_message=training_log.error_message,
            started_at=training_log.start_time,
            updated_at=training_log.end_time or training_log.start_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training status check failed: {str(e)}")

# ============================
# FEEDBACK & LEARNING
# ============================

@router.post("/feedback")
def submit_ai_feedback(
    feedback: AIFeedbackSubmission,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Submit feedback on AI predictions for continuous improvement"""
    try:
        # Get the prediction or insight being rated
        prediction = None
        insight = None
        
        if feedback.prediction_id:
            prediction = db.query(LeadPrediction).filter(
                LeadPrediction.id == feedback.prediction_id
            ).first()
            if not prediction:
                raise HTTPException(status_code=404, detail="Prediction not found")
        
        if feedback.insight_id:
            insight = db.query(LeadInsight).filter(
                LeadInsight.id == feedback.insight_id
            ).first()
            if not insight:
                raise HTTPException(status_code=404, detail="Insight not found")
        
        if not prediction and not insight:
            raise HTTPException(status_code=400, detail="Either prediction_id or insight_id must be provided")
        
        # Create feedback record
        ai_feedback = AIFeedback(
            prediction_id=feedback.prediction_id,
            insight_id=feedback.insight_id,
            user_id=current_user.id,
            feedback_type="rating",
            rating=feedback.rating,
            is_accurate=feedback.is_accurate,
            is_helpful=feedback.is_helpful,
            actual_outcome=feedback.actual_outcome,
            improvement_suggestion=feedback.improvement_suggestion,
            feedback_text=feedback.feedback_text,
            sentiment="positive" if feedback.rating >= 4 else "negative" if feedback.rating <= 2 else "neutral"
        )
        
        db.add(ai_feedback)
        db.commit()
        
        # Update model feedback counters
        if prediction:
            model = db.query(ModelMetadata).filter(
                ModelMetadata.model_name.contains(prediction.model_type),
                ModelMetadata.is_active == True
            ).first()
            
            if model:
                # Simple feedback integration - could be more sophisticated
                if feedback.is_helpful and feedback.rating >= 4:
                    model.predictions_made = (model.predictions_made or 0) + 1
                
        return {
            "message": "Feedback submitted successfully",
            "feedback_id": ai_feedback.id,
            "will_improve_ai": True,
            "thank_you": "Your feedback helps improve AI accuracy!"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@router.get("/feedback-summary")
def get_feedback_summary(
    days: int = 30,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Get summary of user's AI feedback and its impact"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Feedback statistics
        feedback_stats = db.query(
            func.count(AIFeedback.id).label('total_feedback'),
            func.avg(AIFeedback.rating).label('avg_rating'),
            func.sum(func.case([(AIFeedback.is_helpful == True, 1)], else_=0)).label('helpful_count'),
            func.sum(func.case([(AIFeedback.is_accurate == True, 1)], else_=0)).label('accurate_count')
        ).filter(
            AIFeedback.user_id == current_user.id,
            AIFeedback.created_at >= start_date
        ).first()
        
        # Feedback by sentiment
        sentiment_breakdown = db.query(
            AIFeedback.sentiment,
            func.count(AIFeedback.id).label('count')
        ).filter(
            AIFeedback.user_id == current_user.id,
            AIFeedback.created_at >= start_date
        ).group_by(AIFeedback.sentiment).all()
        
        # Recent feedback trends
        daily_feedback = db.query(
            func.date(AIFeedback.created_at).label('date'),
            func.count(AIFeedback.id).label('feedback_count'),
            func.avg(AIFeedback.rating).label('avg_rating')
        ).filter(
            AIFeedback.user_id == current_user.id,
            AIFeedback.created_at >= start_date
        ).group_by(func.date(AIFeedback.created_at)).order_by('date').all()
        
        total_feedback = feedback_stats.total_feedback or 0
        helpful_rate = (feedback_stats.helpful_count / max(total_feedback, 1)) * 100
        accuracy_rate = (feedback_stats.accurate_count / max(total_feedback, 1)) * 100
        
        return {
            "summary": {
                "total_feedback_submitted": total_feedback,
                "average_rating": round(feedback_stats.avg_rating or 0, 2),
                "helpfulness_rate": round(helpful_rate, 2),
                "accuracy_rate": round(accuracy_rate, 2),
                "time_period_days": days
            },
            "sentiment_breakdown": [
                {"sentiment": sentiment, "count": count}
                for sentiment, count in sentiment_breakdown
            ],
            "daily_trends": [
                {
                    "date": str(date),
                    "feedback_count": count,
                    "avg_rating": round(float(rating), 2) if rating else 0
                }
                for date, count, rating in daily_feedback
            ],
            "impact_message": _generate_feedback_impact_message(helpful_rate, accuracy_rate, total_feedback)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback summary failed: {str(e)}")

# ============================
# QUALITY ASSESSMENT
# ============================

@router.post("/assess-quality", response_model=DataQualityReport)
def assess_data_quality(
    request: DataQualityAssessmentRequest,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Perform comprehensive data quality assessment"""
    try:
        from utils.data_quality_engine import DataQualityEngine
        
        # Get leads to assess
        if request.lead_ids:
            leads = db.query(Lead).filter(
                Lead.id.in_(request.lead_ids),
                Lead.owner_id == current_user.id
            ).all()
        else:
            leads = db.query(Lead).filter(Lead.owner_id == current_user.id).limit(100).all()
        
        if not leads:
            raise HTTPException(status_code=404, detail="No leads found for assessment")
        
        quality_engine = DataQualityEngine()
        
        # Perform assessment based on depth
        if request.assessment_depth == "comprehensive":
            assessment_results = quality_engine.comprehensive_quality_assessment(leads)
        elif request.assessment_depth == "basic":
            assessment_results = quality_engine.basic_quality_assessment(leads)
        else:  # standard
            assessment_results = quality_engine.standard_quality_assessment(leads)
        
        # Store assessment results
        for lead in leads:
            lead_quality = assessment_results.get(lead.id, {})
            
            quality_score = DataQualityScore(
                lead_id=lead.id,
                overall_score=lead_quality.get('overall_score', 50),
                email_quality=lead_quality.get('email_quality', 0),
                completeness_score=lead_quality.get('completeness_score', 0),
                accuracy_score=lead_quality.get('accuracy_score', 0),
                consistency_score=lead_quality.get('consistency_score', 0),
                freshness_score=lead_quality.get('freshness_score', 100),
                issues_found=json.dumps(lead_quality.get('issues', [])),
                suggestions=json.dumps(lead_quality.get('suggestions', [])),
                assessment_method=f"batch_{request.assessment_depth}"
            )
            db.add(quality_score)
        
        db.commit()
        
        # Aggregate results
        overall_score = sum(result.get('overall_score', 0) for result in assessment_results.values()) / len(assessment_results)
        all_issues = []
        all_suggestions = []
        duplicates = []
        
        for result in assessment_results.values():
            all_issues.extend(result.get('issues', []))
            all_suggestions.extend(result.get('suggestions', []))
            duplicates.extend(result.get('duplicates', []))
        
        # Remove duplicate issues and suggestions
        unique_issues = list(set(all_issues))
        unique_suggestions = list(set(all_suggestions))
        unique_duplicates = list(set(duplicates))
        
        return DataQualityReport(
            lead_id=0,  # Indicates batch assessment
            overall_score=round(overall_score, 2),
            quality_breakdown={
                "email_quality": round(sum(r.get('email_quality', 0) for r in assessment_results.values()) / len(assessment_results), 2),
                "completeness": round(sum(r.get('completeness_score', 0) for r in assessment_results.values()) / len(assessment_results), 2),
                "accuracy": round(sum(r.get('accuracy_score', 0) for r in assessment_results.values()) / len(assessment_results), 2),
                "consistency": round(sum(r.get('consistency_score', 0) for r in assessment_results.values()) / len(assessment_results), 2)
            },
            issues_found=unique_issues[:20],  # Limit to top 20 issues
            improvement_suggestions=unique_suggestions[:10],  # Limit to top 10 suggestions
            duplicate_candidates=unique_duplicates,
            completeness_score=round(sum(r.get('completeness_score', 0) for r in assessment_results.values()) / len(assessment_results), 2),
            accuracy_score=round(sum(r.get('accuracy_score', 0) for r in assessment_results.values()) / len(assessment_results), 2),
            freshness_score=100.0,  # Assume fresh data for now
            consistency_score=round(sum(r.get('consistency_score', 0) for r in assessment_results.values()) / len(assessment_results), 2),
            improvement_potential=round(100 - overall_score, 2) if overall_score < 100 else 0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {str(e)}")

@router.get("/quality-report")
def get_quality_report(
    include_trends: bool = True,
    days: int = 30,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Get comprehensive data quality report for user's leads"""
    try:
        # Overall quality statistics
        quality_stats = db.query(
            func.count(DataQualityScore.id).label('total_assessments'),
            func.avg(DataQualityScore.overall_score).label('avg_overall'),
            func.avg(DataQualityScore.email_quality).label('avg_email'),
            func.avg(DataQualityScore.completeness_score).label('avg_completeness'),
            func.avg(DataQualityScore.accuracy_score).label('avg_accuracy')
        ).join(Lead).filter(Lead.owner_id == current_user.id).first()
        
        # Recent improvements
        start_date = datetime.utcnow() - timedelta(days=days)
        recent_improvements = db.query(func.count(DataQualityScore.id)).join(Lead).filter(
            Lead.owner_id == current_user.id,
            DataQualityScore.created_at >= start_date,
            DataQualityScore.overall_score >= 80
        ).scalar()
        
        return {
            "overall_score": round(quality_stats.avg_overall or 0, 2),
            "total_assessments": quality_stats.total_assessments or 0,
            "recent_improvements": recent_improvements or 0,
            "quality_breakdown": {
                "email_quality": round(quality_stats.avg_email or 0, 2),
                "completeness": round(quality_stats.avg_completeness or 0, 2),
                "accuracy": round(quality_stats.avg_accuracy or 0, 2)
            },
            "recommendations": _generate_quality_recommendations(quality_stats)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality report failed: {str(e)}")

# ============================
# INSIGHTS MANAGEMENT
# ============================

@router.get("/insights")
def get_user_insights(
    priority_filter: Optional[str] = None,
    limit: int = 20,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Get user's AI insights with filtering"""
    try:
        query = db.query(LeadInsight).join(Lead).filter(
            Lead.owner_id == current_user.id,
            LeadInsight.status == 'active'
        )
        
        if priority_filter:
            query = query.filter(LeadInsight.priority_level == priority_filter)
        
        insights = query.order_by(
            LeadInsight.priority_level.desc(),
            LeadInsight.created_at.desc()
        ).limit(limit).all()
        
        insight_data = []
        for insight in insights:
            insight_data.append({
                "id": insight.id,
                "lead_id": insight.lead_id,
                "type": insight.insight_type,
                "category": insight.insight_category,
                "text": insight.insight_text,
                "summary": insight.short_summary,
                "priority": insight.priority_level,
                "confidence": insight.confidence_score,
                "impact_score": insight.impact_score,
                "urgency_score": insight.urgency_score,
                "action_items": json.loads(insight.action_items) if insight.action_items else [],
                "created_at": insight.created_at,
                "generated_by": insight.generated_by
            })
        
        return {
            "insights": insight_data,
            "total_count": len(insight_data),
            "filters_applied": {"priority": priority_filter} if priority_filter else {},
            "summary": _generate_insights_summary(insight_data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insights retrieval failed: {str(e)}")

@router.put("/insights/{insight_id}/status")
def update_insight_status(
    insight_id: int,
    status: str,
    feedback: Optional[str] = None,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Update the status of an AI insight"""
    try:
        insight = db.query(LeadInsight).join(Lead).filter(
            LeadInsight.id == insight_id,
            Lead.owner_id == current_user.id
        ).first()
        
        if not insight:
            raise HTTPException(status_code=404, detail="Insight not found")
        
        valid_statuses = ['active', 'resolved', 'dismissed']
        if status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
        
        insight.status = status
        if feedback:
            insight.user_feedback = feedback
        
        db.commit()
        
        return {
            "message": "Insight status updated successfully",
            "insight_id": insight_id,
            "new_status": status
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insight status update failed: {str(e)}")

# ============================
# BACKGROUND TASK FUNCTIONS
# ============================

def _batch_similarity_analysis(lead_ids: List[int], user_id: int):
    """Background task for batch similarity analysis"""
    try:
        from database import SessionLocal
        from utils.ai_lead_analyzer import AILeadAnalyzer
        
        db = SessionLocal()
        analyzer = AILeadAnalyzer()
        
        print(f"🔄 Starting background similarity analysis for {len(lead_ids)} leads")
        
        leads = db.query(Lead).filter(Lead.id.in_(lead_ids)).all()
        
        # Perform pairwise similarity analysis
        for i, lead1 in enumerate(leads):
            for j, lead2 in enumerate(leads[i+1:], i+1):
                try:
                    similarity_score = analyzer.calculate_similarity(lead1, lead2)
                    
                    if similarity_score > 0.7:  # Only store high similarity
                        similarity_record = LeadSimilarity(
                            lead_id_1=lead1.id,
                            lead_id_2=lead2.id,
                            similarity_score=similarity_score,
                            similarity_type='background_analysis',
                            calculation_method='ai_batch_analysis',
                            confidence=0.8,
                            business_relevance=similarity_score * 100
                        )
                        db.add(similarity_record)
                
                except Exception as e:
                    print(f"Error calculating similarity between {lead1.id} and {lead2.id}: {e}")
                    continue
        
        db.commit()
        db.close()
        
        print(f"✅ Background similarity analysis completed")
        
    except Exception as e:
        print(f"❌ Background similarity analysis failed: {e}")

def _train_model_background(training_request: dict, user_id: int, training_log_id: int):
    """Background task for model training"""
    try:
        from database import SessionLocal
        from ai.training.model_trainer import ModelTrainer
        
        db = SessionLocal()
        
        # Update training log
        training_log = db.query(AIProcessingLog).filter(
            AIProcessingLog.id == training_log_id
        ).first()
        
        if not training_log:
            return
        
        trainer = ModelTrainer()
        
        # Get user's data for training
        user_leads = db.query(Lead).filter(Lead.owner_id == user_id).all()
        
        if len(user_leads) < 50:
            training_log.processing_status = 'failed'
            training_log.error_message = 'Insufficient training data'
            training_log.end_time = datetime.utcnow()
            db.commit()
            return
        
        # Perform training
        model_type = training_request['model_type']
        training_results = trainer.train_user_model(
            model_type=model_type,
            user_data=user_leads,
            user_id=user_id,
            hyperparameters=training_request.get('hyperparameter_tuning', False)
        )
        
        if training_results['success']:
            # Update training log with success
            training_log.processing_status = 'completed'
            training_log.end_time = datetime.utcnow()
            training_log.processing_duration_ms = (training_log.end_time - training_log.start_time).total_seconds() * 1000
            training_log.result_summary = json.dumps({
                'accuracy': training_results.get('accuracy', 0),
                'model_path': training_results.get('model_path', ''),
                'training_samples': len(user_leads)
            })
            
            # Create model metadata entry
            model_metadata = ModelMetadata(
                model_name=f"{model_type}_user_{user_id}",
                model_type='classifier',
                version="1.0.0",
                algorithm=training_results.get('algorithm', 'random_forest'),
                accuracy_score=training_results.get('accuracy', 0),
                precision_score=training_results.get('precision', 0),
                recall_score=training_results.get('recall', 0),
                f1_score=training_results.get('f1_score', 0),
                training_samples=len(user_leads),
                validation_samples=training_results.get('validation_samples', 0),
                feature_count=training_results.get('feature_count', 0),
                training_date=datetime.utcnow(),
                training_duration_minutes=training_log.processing_duration_ms / 60000,
                model_path=training_results.get('model_path', ''),
                is_active=True,
                deployment_date=datetime.utcnow(),
                notes=f"Custom model trained for user {user_id}"
            )
            db.add(model_metadata)
            
        else:
            training_log.processing_status = 'failed'
            training_log.error_message = training_results.get('error', 'Training failed')
            training_log.end_time = datetime.utcnow()
        
        db.commit()
        db.close()
        
        print(f"✅ Model training completed for user {user_id}")
        
    except Exception as e:
        print(f"❌ Model training failed for user {user_id}: {e}")
        
        # Update training log with error
        try:
            db = SessionLocal()
            training_log = db.query(AIProcessingLog).filter(
                AIProcessingLog.id == training_log_id
            ).first()
            
            if training_log:
                training_log.processing_status = 'failed'
                training_log.error_message = str(e)
                training_log.end_time = datetime.utcnow()
                db.commit()
            
            db.close()
        except:
            pass

# ============================
# UTILITY FUNCTIONS
# ============================

def _estimate_training_duration(lead_count: int, request: ModelTrainingRequest) -> int:
    """Estimate training duration in minutes"""
    base_time = 2  # 2 minutes base
    data_factor = lead_count / 100  # Additional minute per 100 leads
    complexity_factor = 2 if request.hyperparameter_tuning else 1
    
    return int(base_time + data_factor * complexity_factor)

def _generate_feedback_impact_message(helpful_rate: float, accuracy_rate: float, total_feedback: int) -> str:
    """Generate a message about feedback impact"""
    if total_feedback == 0:
        return "No feedback submitted yet. Your feedback helps improve AI accuracy!"
    elif helpful_rate > 80 and accuracy_rate > 80:
        return "Excellent! Your feedback shows our AI is performing very well for you."
    elif helpful_rate > 60:
        return "Good feedback! We're continuously improving based on your input."
    else:
        return "Thank you for your feedback. We're working to improve AI performance based on your suggestions."

def _generate_quality_recommendations(quality_stats) -> List[str]:
    """Generate quality improvement recommendations"""
    recommendations = []
    
    avg_overall = quality_stats.avg_overall or 0
    avg_email = quality_stats.avg_email or 0
    avg_completeness = quality_stats.avg_completeness or 0
    
    if avg_overall < 70:
        recommendations.append("Overall data quality needs improvement - consider data cleanup")
    
    if avg_email < 60:
        recommendations.append("Email quality is low - validate and enrich email addresses")
    
    if avg_completeness < 70:
        recommendations.append("Data completeness is low - fill in missing lead information")
    
    if not recommendations:
        recommendations.append("Great job! Your data quality is excellent.")
    
    return recommendations

def _generate_insights_summary(insights: List[Dict]) -> Dict[str, Any]:
    """Generate summary of insights"""
    if not insights:
        return {"message": "No insights available"}
    
    priority_counts = {}
    for insight in insights:
        priority = insight['priority']
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    
    avg_impact = sum(insight.get('impact_score', 0) for insight in insights) / len(insights)
    
    return {
        "total_insights": len(insights),
        "priority_breakdown": priority_counts,
        "average_impact_score": round(avg_impact, 2),
        "top_priority": max(priority_counts.items(), key=lambda x: x[1])[0] if priority_counts else "none"
    }

# Export router for main application
__all__ = ['router']