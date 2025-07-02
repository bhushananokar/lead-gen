# routers/analytics.py - Enhanced with AI Analytics Dashboard
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_, text
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

from database import get_db, get_db_for_ai
from models import (
    User, Lead, Company, Campaign, 
    LeadPrediction, DataQualityScore, LeadInsight, 
    AIProcessingLog, ModelMetadata, LeadSimilarity,
    DuplicateDetection, AIFeedback
)
from schemas import AIInsightsSummary, ModelPerformanceMetrics
from routers.auth import get_current_user

router = APIRouter()

# ============================
# EXISTING ANALYTICS ENDPOINTS (ENHANCED)
# ============================

@router.get("/dashboard")
def get_dashboard_analytics(
    include_ai_metrics: bool = True,
    time_range_days: int = 30,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Enhanced dashboard with AI metrics and insights"""
    try:
        # Basic counts
        total_leads = db.query(func.count(Lead.id)).filter(Lead.owner_id == current_user.id).scalar()
        total_companies = db.query(func.count(Company.id)).scalar()
        total_campaigns = db.query(func.count(Campaign.id)).filter(Campaign.owner_id == current_user.id).scalar()
        
        # Lead status breakdown
        lead_statuses = db.query(
            Lead.status,
            func.count(Lead.id).label('count')
        ).filter(Lead.owner_id == current_user.id).group_by(Lead.status).all()
        
        # Recent activity (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_leads = db.query(func.count(Lead.id)).filter(
            Lead.owner_id == current_user.id,
            Lead.created_at >= week_ago
        ).scalar()
        
        # Top performing campaigns
        top_campaigns = db.query(
            Campaign.name,
            func.count(Lead.id).label('leads_count')
        ).join(Lead, isouter=True).filter(
            Campaign.owner_id == current_user.id
        ).group_by(Campaign.id, Campaign.name).order_by(desc('leads_count')).limit(5).all()
        
        # Lead sources
        lead_sources = db.query(
            Lead.source,
            func.count(Lead.id).label('count')
        ).filter(Lead.owner_id == current_user.id).group_by(Lead.source).all()
        
        # Base analytics
        analytics = {
            "overview": {
                "total_leads": total_leads,
                "total_companies": total_companies,
                "total_campaigns": total_campaigns,
                "recent_leads": recent_leads
            },
            "lead_statuses": [{"status": status, "count": count} for status, count in lead_statuses],
            "top_campaigns": [{"name": name, "leads_count": count} for name, count in top_campaigns],
            "lead_sources": [{"source": source or "unknown", "count": count} for source, count in lead_sources]
        }
        
        # Add AI metrics if requested
        if include_ai_metrics:
            ai_metrics = _get_ai_dashboard_metrics(db, current_user.id, time_range_days)
            analytics.update(ai_metrics)
        
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard analytics failed: {str(e)}")

@router.get("/lead-trends")
def get_lead_trends(
    days: int = 30,
    include_ai_trends: bool = True,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get lead trends with AI enhancement metrics"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Basic lead trends
        trends = db.query(
            func.date(Lead.created_at).label('date'),
            func.count(Lead.id).label('count')
        ).filter(
            Lead.owner_id == current_user.id,
            Lead.created_at >= start_date
        ).group_by(func.date(Lead.created_at)).order_by('date').all()
        
        trend_data = [{"date": str(date), "count": count} for date, count in trends]
        
        # Add AI trends if requested
        if include_ai_trends:
            # AI analysis trends
            ai_trends = db.query(
                func.date(LeadPrediction.created_at).label('date'),
                func.count(LeadPrediction.id).label('ai_analyses'),
                func.avg(LeadPrediction.prediction_score).label('avg_priority'),
                func.avg(LeadPrediction.confidence).label('avg_confidence')
            ).join(Lead).filter(
                Lead.owner_id == current_user.id,
                LeadPrediction.created_at >= start_date
            ).group_by(func.date(LeadPrediction.created_at)).order_by('date').all()
            
            # Quality trends
            quality_trends = db.query(
                func.date(DataQualityScore.created_at).label('date'),
                func.avg(DataQualityScore.overall_score).label('avg_quality')
            ).join(Lead).filter(
                Lead.owner_id == current_user.id,
                DataQualityScore.created_at >= start_date
            ).group_by(func.date(DataQualityScore.created_at)).order_by('date').all()
            
            # Merge AI trends with basic trends
            ai_trend_dict = {str(date): {
                'ai_analyses': analyses,
                'avg_priority': round(float(priority), 2) if priority else 0,
                'avg_confidence': round(float(confidence), 3) if confidence else 0
            } for date, analyses, priority, confidence in ai_trends}
            
            quality_dict = {str(date): round(float(quality), 2) if quality else 0 
                          for date, quality in quality_trends}
            
            # Enhanced trend data
            for trend in trend_data:
                date_str = trend['date']
                trend.update(ai_trend_dict.get(date_str, {
                    'ai_analyses': 0,
                    'avg_priority': 0,
                    'avg_confidence': 0
                }))
                trend['avg_quality'] = quality_dict.get(date_str, 0)
        
        return trend_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lead trends analysis failed: {str(e)}")

# ============================
# NEW AI ANALYTICS ENDPOINTS
# ============================

@router.get("/ai-dashboard", response_model=AIInsightsSummary)
def get_ai_dashboard(
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Comprehensive AI analytics dashboard - FIXED VERSION"""
    try:
        # Get all AI predictions for user's leads
        ai_stats = db.query(
            func.count(LeadPrediction.id).label('total_analyzed'),
            func.avg(LeadPrediction.prediction_score).label('avg_score'),
            func.avg(LeadPrediction.confidence).label('avg_confidence')
        ).join(Lead).filter(Lead.owner_id == current_user.id).first()
        
        # FIXED: Priority breakdown using BOTH sources for accuracy
        # Method 1: From LeadPrediction (prediction_score)
        prediction_breakdown = db.query(
            func.case([
                (LeadPrediction.prediction_score >= 85, 'critical'),
                (LeadPrediction.prediction_score >= 70, 'high'),
                (LeadPrediction.prediction_score >= 50, 'medium')
            ], else_='low').label('priority'),
            func.count(LeadPrediction.id).label('count')
        ).join(Lead).filter(
            Lead.owner_id == current_user.id
        ).group_by('priority').all()
        
        # Method 2: From LeadInsight (priority_level) - this is what AI Insights page uses
        insight_breakdown = db.query(
            LeadInsight.priority_level.label('priority'),
            func.count(LeadInsight.id).label('count')
        ).join(Lead).filter(
            Lead.owner_id == current_user.id,
            LeadInsight.status == 'active',
            LeadInsight.insight_type == 'priority'
        ).group_by(LeadInsight.priority_level).all()
        
        # Convert to counts with defaults
        priority_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }
        
        # Use prediction_breakdown as primary source
        for priority, count in prediction_breakdown:
            if priority in priority_counts:
                priority_counts[priority] = count
        
        # Add counts from insight_breakdown (may be higher/more accurate)
        insight_counts = {}
        for priority, count in insight_breakdown:
            if priority in ['critical', 'high', 'medium', 'low']:
                insight_counts[priority] = count
        
        # Use the MAXIMUM count from both sources (most accurate)
        for priority in priority_counts:
            insight_count = insight_counts.get(priority, 0)
            priority_counts[priority] = max(priority_counts[priority], insight_count)
        
        # ALTERNATIVE: Get high priority count directly from LeadInsight table
        high_priority_direct = db.query(func.count(LeadInsight.id)).join(Lead).filter(
            Lead.owner_id == current_user.id,
            LeadInsight.status == 'active',
            LeadInsight.priority_level == 'high'
        ).scalar() or 0
        
        critical_priority_direct = db.query(func.count(LeadInsight.id)).join(Lead).filter(
            Lead.owner_id == current_user.id,
            LeadInsight.status == 'active',
            LeadInsight.priority_level == 'critical'
        ).scalar() or 0
        
        # Use direct counts if they're higher (more accurate)
        priority_counts['high'] = max(priority_counts['high'], high_priority_direct)
        priority_counts['critical'] = max(priority_counts['critical'], critical_priority_direct)
        
        # Data quality trends (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        quality_trends = db.query(
            func.date(DataQualityScore.created_at).label('date'),
            func.avg(DataQualityScore.overall_score).label('avg_quality')
        ).join(Lead).filter(
            Lead.owner_id == current_user.id,
            DataQualityScore.created_at >= week_ago
        ).group_by(func.date(DataQualityScore.created_at)).order_by('date').all()
        
        # Top insights
        top_insights = db.query(LeadInsight).join(Lead).filter(
            Lead.owner_id == current_user.id,
            LeadInsight.priority_level.in_(['high', 'critical']),
            LeadInsight.status == 'active'
        ).order_by(LeadInsight.created_at.desc()).limit(5).all()
        
        # Handle case where no data exists
        total_analyzed = 0
        avg_score = 0.0
        avg_confidence = 0.0
        
        if ai_stats and ai_stats.total_analyzed:
            total_analyzed = ai_stats.total_analyzed
            avg_score = ai_stats.avg_score or 0.0
            avg_confidence = ai_stats.avg_confidence or 0.0
        
        print(f"🔍 DEBUG - Priority counts: {priority_counts}")
        print(f"🔍 DEBUG - High priority direct: {high_priority_direct}")
        print(f"🔍 DEBUG - Critical priority direct: {critical_priority_direct}")
        
        return AIInsightsSummary(
            total_leads_analyzed=total_analyzed,
            high_priority_count=priority_counts.get('high', 0),
            medium_priority_count=priority_counts.get('medium', 0),
            low_priority_count=priority_counts.get('low', 0),
            critical_priority_count=priority_counts.get('critical', 0),
            average_quality_score=round(float(avg_score), 2),
            average_confidence=round(float(avg_confidence), 3),
            top_insights=[insight.insight_text for insight in top_insights if insight.insight_text][:10]
        )
        
    except Exception as e:
        print(f"❌ AI dashboard error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI dashboard failed: {str(e)}")

@router.get("/ai-performance")
def get_ai_performance_metrics(
    model_name: Optional[str] = None,
    days: int = 30,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Get detailed AI model performance metrics"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Base query for user's AI operations
        base_query = db.query(AIProcessingLog).filter(
            AIProcessingLog.user_id == current_user.id,
            AIProcessingLog.start_time >= start_date
        )
        
        if model_name:
            base_query = base_query.filter(AIProcessingLog.model_name == model_name)
        
        # Performance statistics
        performance_stats = base_query.filter(
            AIProcessingLog.processing_status == 'completed'
        ).with_entities(
            func.count(AIProcessingLog.id).label('total_operations'),
            func.avg(AIProcessingLog.processing_duration_ms).label('avg_duration'),
            func.avg(AIProcessingLog.cpu_usage_percent).label('avg_cpu'),
            func.avg(AIProcessingLog.memory_usage_mb).label('avg_memory')
        ).first()
        
        # Error rate
        total_ops = base_query.count()
        failed_ops = base_query.filter(AIProcessingLog.processing_status == 'failed').count()
        error_rate = (failed_ops / total_ops * 100) if total_ops > 0 else 0
        
        # Daily operation trends
        daily_trends = base_query.filter(
            AIProcessingLog.processing_status == 'completed'
        ).with_entities(
            func.date(AIProcessingLog.start_time).label('date'),
            func.count(AIProcessingLog.id).label('operations'),
            func.avg(AIProcessingLog.processing_duration_ms).label('avg_duration')
        ).group_by(func.date(AIProcessingLog.start_time)).order_by('date').all()
        
        # Model accuracy trends (from feedback)
        accuracy_trends = db.query(
            func.date(AIFeedback.created_at).label('date'),
            func.avg(func.case([(AIFeedback.is_accurate == True, 1.0)], else_=0.0)).label('accuracy')
        ).filter(
            AIFeedback.user_id == current_user.id,
            AIFeedback.created_at >= start_date
        ).group_by(func.date(AIFeedback.created_at)).order_by('date').all()
        
        return {
            "performance_summary": {
                "total_operations": performance_stats.total_operations or 0,
                "average_duration_ms": round(performance_stats.avg_duration or 0, 2),
                "average_cpu_usage": round(performance_stats.avg_cpu or 0, 2),
                "average_memory_mb": round(performance_stats.avg_memory or 0, 2),
                "error_rate_percent": round(error_rate, 2),
                "success_rate_percent": round(100 - error_rate, 2)
            },
            "daily_trends": [{
                "date": str(date),
                "operations": operations,
                "avg_duration_ms": round(float(duration), 2) if duration else 0
            } for date, operations, duration in daily_trends],
            "accuracy_trends": [{
                "date": str(date),
                "accuracy_percent": round(float(accuracy) * 100, 2) if accuracy else 0
            } for date, accuracy in accuracy_trends],
            "time_range_days": days,
            "model_filter": model_name
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI performance metrics failed: {str(e)}")

@router.get("/data-quality-report")
def get_data_quality_report(
    include_trends: bool = True,
    days: int = 30,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Comprehensive data quality analysis"""
    try:
        # Overall quality statistics
        quality_stats = db.query(
            func.count(DataQualityScore.id).label('total_assessments'),
            func.avg(DataQualityScore.overall_score).label('avg_overall'),
            func.avg(DataQualityScore.email_quality).label('avg_email'),
            func.avg(DataQualityScore.completeness_score).label('avg_completeness'),
            func.avg(DataQualityScore.accuracy_score).label('avg_accuracy'),
            func.avg(DataQualityScore.consistency_score).label('avg_consistency')
        ).join(Lead).filter(Lead.owner_id == current_user.id).first()
        
        # Quality distribution
        quality_distribution = db.query(
            func.case([
                (DataQualityScore.overall_score >= 85, 'excellent'),
                (DataQualityScore.overall_score >= 70, 'good'),
                (DataQualityScore.overall_score >= 50, 'fair')
            ], else_='poor').label('quality_tier'),
            func.count(DataQualityScore.id).label('count')
        ).join(Lead).filter(
            Lead.owner_id == current_user.id
        ).group_by('quality_tier').all()
        
        # Common issues analysis
        all_issues = db.query(DataQualityScore.issues_found).join(Lead).filter(
            Lead.owner_id == current_user.id,
            DataQualityScore.issues_found.isnot(None)
        ).all()
        
        issue_counts = {}
        for issues_json in all_issues:
            if issues_json.issues_found:
                try:
                    issues = json.loads(issues_json.issues_found)
                    for issue in issues:
                        issue_counts[issue] = issue_counts.get(issue, 0) + 1
                except:
                    continue
        
        # Top issues (most common)
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        quality_report = {
            "summary": {
                "total_assessments": quality_stats.total_assessments or 0,
                "average_overall_score": round(quality_stats.avg_overall or 0, 2),
                "average_email_quality": round(quality_stats.avg_email or 0, 2),
                "average_completeness": round(quality_stats.avg_completeness or 0, 2),
                "average_accuracy": round(quality_stats.avg_accuracy or 0, 2),
                "average_consistency": round(quality_stats.avg_consistency or 0, 2)
            },
            "distribution": [
                {"tier": tier, "count": count} 
                for tier, count in quality_distribution
            ],
            "top_issues": [
                {"issue": issue, "count": count, "percentage": round(count / quality_stats.total_assessments * 100, 1)}
                for issue, count in top_issues
            ] if quality_stats.total_assessments > 0 else []
        }
        
        # Add trends if requested
        if include_trends:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            quality_trends = db.query(
                func.date(DataQualityScore.created_at).label('date'),
                func.avg(DataQualityScore.overall_score).label('avg_quality'),
                func.count(DataQualityScore.id).label('assessments')
            ).join(Lead).filter(
                Lead.owner_id == current_user.id,
                DataQualityScore.created_at >= start_date
            ).group_by(func.date(DataQualityScore.created_at)).order_by('date').all()
            
            quality_report["trends"] = [{
                "date": str(date),
                "avg_quality": round(float(quality), 2) if quality else 0,
                "assessments": assessments
            } for date, quality, assessments in quality_trends]
        
        return quality_report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data quality report failed: {str(e)}")

@router.get("/duplicate-analysis")
def get_duplicate_analysis(
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Analysis of duplicate detection results"""
    try:
        # Base query for duplicates involving user's leads
        base_query = db.query(DuplicateDetection).join(
            Lead, DuplicateDetection.primary_lead_id == Lead.id
        ).filter(Lead.owner_id == current_user.id)
        
        if status_filter:
            base_query = base_query.filter(DuplicateDetection.status == status_filter)
        
        # Duplicate statistics
        duplicate_stats = base_query.with_entities(
            func.count(DuplicateDetection.id).label('total_duplicates'),
            func.sum(func.case([(DuplicateDetection.auto_merge_recommended == True, 1)], else_=0)).label('auto_merge_candidates'),
            func.avg(DuplicateDetection.match_score).label('avg_match_score')
        ).first()
        
        # Status breakdown
        status_breakdown = base_query.with_entities(
            DuplicateDetection.status,
            func.count(DuplicateDetection.id).label('count')
        ).group_by(DuplicateDetection.status).all()
        
        # Match score distribution
        score_distribution = base_query.with_entities(
            func.case([
                (DuplicateDetection.match_score >= 95, 'very_high'),
                (DuplicateDetection.match_score >= 85, 'high'),
                (DuplicateDetection.match_score >= 75, 'medium')
            ], else_='low').label('score_tier'),
            func.count(DuplicateDetection.id).label('count')
        ).group_by('score_tier').all()
        
        # Recent duplicate detection activity
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_activity = base_query.filter(
            DuplicateDetection.detected_at >= week_ago
        ).count()
        
        return {
            "summary": {
                "total_duplicates_detected": duplicate_stats.total_duplicates or 0,
                "auto_merge_candidates": int(duplicate_stats.auto_merge_candidates or 0),
                "average_match_score": round(duplicate_stats.avg_match_score or 0, 2),
                "recent_detections": recent_activity
            },
            "status_breakdown": [
                {"status": status, "count": count}
                for status, count in status_breakdown
            ],
            "score_distribution": [
                {"score_tier": tier, "count": count}
                for tier, count in score_distribution
            ],
            "recommendations": _generate_duplicate_recommendations(duplicate_stats)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Duplicate analysis failed: {str(e)}")

@router.get("/conversion-analysis")
def get_conversion_analysis(
    days: int = 90,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Analyze lead conversion patterns with AI insights"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Overall conversion metrics
        total_leads = db.query(func.count(Lead.id)).filter(
            Lead.owner_id == current_user.id,
            Lead.created_at >= start_date
        ).scalar()
        
        converted_leads = db.query(func.count(Lead.id)).filter(
            Lead.owner_id == current_user.id,
            Lead.status == 'converted',
            Lead.created_at >= start_date
        ).scalar()
        
        conversion_rate = (converted_leads / total_leads * 100) if total_leads > 0 else 0
        
        # AI score vs conversion correlation
        ai_conversion_analysis = db.query(
            func.case([
                (LeadPrediction.prediction_score >= 80, 'high_ai_score'),
                (LeadPrediction.prediction_score >= 60, 'medium_ai_score')
            ], else_='low_ai_score').label('ai_score_tier'),
            func.count(Lead.id).label('total_leads'),
            func.sum(func.case([(Lead.status == 'converted', 1)], else_=0)).label('converted')
        ).join(LeadPrediction).filter(
            Lead.owner_id == current_user.id,
            Lead.created_at >= start_date
        ).group_by('ai_score_tier').all()
        
        # Source performance
        source_performance = db.query(
            Lead.source,
            func.count(Lead.id).label('total'),
            func.sum(func.case([(Lead.status == 'converted', 1)], else_=0)).label('converted'),
            func.avg(Lead.score).label('avg_score')
        ).filter(
            Lead.owner_id == current_user.id,
            Lead.created_at >= start_date
        ).group_by(Lead.source).all()
        
        # Time to conversion analysis
        conversion_times = db.query(
            func.extract('epoch', Lead.updated_at - Lead.created_at).label('time_to_convert')
        ).filter(
            Lead.owner_id == current_user.id,
            Lead.status == 'converted',
            Lead.created_at >= start_date
        ).all()
        
        avg_time_to_convert = sum(time.time_to_convert for time in conversion_times) / len(conversion_times) if conversion_times else 0
        avg_days_to_convert = avg_time_to_convert / 86400  # Convert seconds to days
        
        return {
            "overall_metrics": {
                "total_leads": total_leads,
                "converted_leads": converted_leads,
                "conversion_rate_percent": round(conversion_rate, 2),
                "average_days_to_convert": round(avg_days_to_convert, 1),
                "time_period_days": days
            },
            "ai_score_correlation": [{
                "ai_score_tier": tier,
                "total_leads": total,
                "converted": int(converted),
                "conversion_rate": round((int(converted) / total * 100), 2) if total > 0 else 0
            } for tier, total, converted in ai_conversion_analysis],
            "source_performance": [{
                "source": source,
                "total_leads": total,
                "converted": int(converted),
                "conversion_rate": round((int(converted) / total * 100), 2) if total > 0 else 0,
                "average_score": round(float(avg_score), 2) if avg_score else 0
            } for source, total, converted, avg_score in source_performance],
            "insights": _generate_conversion_insights(ai_conversion_analysis, source_performance, conversion_rate)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion analysis failed: {str(e)}")

@router.get("/model-comparison")
def get_model_comparison(
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Compare performance of different AI models"""
    try:
        # Get all active models
        models = db.query(ModelMetadata).filter(
            ModelMetadata.is_active == True
        ).all()
        
        model_comparison = []
        
        for model in models:
            # Get user-specific model performance
            user_predictions = db.query(
                func.count(LeadPrediction.id).label('predictions_made'),
                func.avg(LeadPrediction.confidence).label('avg_confidence'),
                func.avg(LeadPrediction.processing_time_ms).label('avg_processing_time')
            ).join(Lead).filter(
                Lead.owner_id == current_user.id,
                LeadPrediction.model_type == model.model_name.split('_')[0]  # Extract model type
            ).first()
            
            # Get user feedback for this model type
            user_feedback = db.query(
                func.avg(func.case([(AIFeedback.is_helpful == True, 1.0)], else_=0.0)).label('helpfulness'),
                func.avg(func.case([(AIFeedback.is_accurate == True, 1.0)], else_=0.0)).label('accuracy')
            ).join(LeadPrediction).join(Lead).filter(
                Lead.owner_id == current_user.id,
                LeadPrediction.model_type == model.model_name.split('_')[0]
            ).first()
            
            model_comparison.append({
                "model_name": model.model_name,
                "model_type": model.model_type,
                "version": model.version,
                "algorithm": model.algorithm,
                "global_metrics": {
                    "accuracy": model.accuracy_score,
                    "precision": model.precision_score,
                    "recall": model.recall_score,
                    "f1_score": model.f1_score,
                    "training_samples": model.training_samples
                },
                "user_specific_metrics": {
                    "predictions_made": user_predictions.predictions_made or 0,
                    "avg_confidence": round(user_predictions.avg_confidence or 0, 3),
                    "avg_processing_time_ms": round(user_predictions.avg_processing_time or 0, 2),
                    "user_helpfulness": round(user_feedback.helpfulness or 0, 3),
                    "user_accuracy": round(user_feedback.accuracy or 0, 3)
                },
                "last_training": model.training_date,
                "model_size_mb": model.model_size_mb
            })
        
        # Rank models by user-specific performance
        model_comparison.sort(key=lambda x: (
            x['user_specific_metrics']['user_accuracy'] * 0.4 +
            x['user_specific_metrics']['user_helpfulness'] * 0.3 +
            x['user_specific_metrics']['avg_confidence'] * 0.3
        ), reverse=True)
        
        return {
            "model_comparison": model_comparison,
            "total_models": len(model_comparison),
            "recommendation": _get_model_recommendation(model_comparison)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")

@router.get("/roi-analysis")
def get_roi_analysis(
    days: int = 90,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Calculate ROI of AI features and lead generation efforts"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Time savings from AI automation
        ai_operations = db.query(
            func.count(AIProcessingLog.id).label('total_operations'),
            func.sum(AIProcessingLog.processing_duration_ms).label('total_processing_time')
        ).filter(
            AIProcessingLog.user_id == current_user.id,
            AIProcessingLog.start_time >= start_date,
            AIProcessingLog.processing_status == 'completed'
        ).first()
        
        # Manual time that would have been spent (estimated)
        manual_time_per_lead = 300000  # 5 minutes in milliseconds
        leads_analyzed = db.query(func.count(distinct(LeadPrediction.lead_id))).join(Lead).filter(
            Lead.owner_id == current_user.id,
            LeadPrediction.created_at >= start_date
        ).scalar()
        
        estimated_manual_time = leads_analyzed * manual_time_per_lead
        actual_ai_time = ai_operations.total_processing_time or 0
        time_saved_ms = estimated_manual_time - actual_ai_time
        time_saved_hours = time_saved_ms / (1000 * 60 * 60)
        
        # Lead quality improvements
        quality_improvements = db.query(
            func.count(DataQualityScore.id).label('assessments'),
            func.avg(DataQualityScore.overall_score).label('avg_quality')
        ).join(Lead).filter(
            Lead.owner_id == current_user.id,
            DataQualityScore.created_at >= start_date
        ).first()
        
        # Conversion improvements (AI vs non-AI leads)
        ai_leads_conversion = db.query(
            func.count(Lead.id).label('total'),
            func.sum(func.case([(Lead.status == 'converted', 1)], else_=0)).label('converted')
        ).join(LeadPrediction).filter(
            Lead.owner_id == current_user.id,
            Lead.created_at >= start_date
        ).first()
        
        non_ai_leads_conversion = db.query(
            func.count(Lead.id).label('total'),
            func.sum(func.case([(Lead.status == 'converted', 1)], else_=0)).label('converted')
        ).outerjoin(LeadPrediction).filter(
            Lead.owner_id == current_user.id,
            Lead.created_at >= start_date,
            LeadPrediction.id == None
        ).first()
        
        # Calculate conversion rates
        ai_conversion_rate = (ai_leads_conversion.converted / ai_leads_conversion.total * 100) if ai_leads_conversion.total > 0 else 0
        non_ai_conversion_rate = (non_ai_leads_conversion.converted / non_ai_leads_conversion.total * 100) if non_ai_leads_conversion.total > 0 else 0
        conversion_improvement = ai_conversion_rate - non_ai_conversion_rate
        
        # Duplicate detection savings
        duplicates_detected = db.query(func.count(DuplicateDetection.id)).join(
            Lead, DuplicateDetection.primary_lead_id == Lead.id
        ).filter(
            Lead.owner_id == current_user.id,
            DuplicateDetection.detected_at >= start_date
        ).scalar()
        
        # Estimated cost savings (assuming $50/hour for manual work)
        hourly_rate = 50
        cost_savings = time_saved_hours * hourly_rate
        
        # Quality-based lead value improvement
        high_quality_leads = db.query(func.count(DataQualityScore.id)).join(Lead).filter(
            Lead.owner_id == current_user.id,
            DataQualityScore.overall_score >= 80,
            DataQualityScore.created_at >= start_date
        ).scalar()
        
        return {
            "roi_summary": {
                "time_period_days": days,
                "time_saved_hours": round(time_saved_hours, 2),
                "estimated_cost_savings": round(cost_savings, 2),
                "leads_analyzed_by_ai": leads_analyzed,
                "duplicates_detected": duplicates_detected,
                "high_quality_leads_identified": high_quality_leads
            },
            "efficiency_metrics": {
                "manual_vs_ai_time_ratio": round(estimated_manual_time / max(actual_ai_time, 1), 2),
                "average_processing_time_ms": round((actual_ai_time / max(ai_operations.total_operations, 1)), 2),
                "automation_coverage_percent": round((leads_analyzed / max(db.query(func.count(Lead.id)).filter(
                    Lead.owner_id == current_user.id,
                    Lead.created_at >= start_date
                ).scalar(), 1) * 100), 2)
            },
            "quality_impact": {
                "average_data_quality": round(quality_improvements.avg_quality or 0, 2),
                "quality_assessments_performed": quality_improvements.assessments or 0,
                "ai_conversion_rate": round(ai_conversion_rate, 2),
                "non_ai_conversion_rate": round(non_ai_conversion_rate, 2),
                "conversion_improvement": round(conversion_improvement, 2)
            },
            "value_drivers": _calculate_value_drivers(time_saved_hours, conversion_improvement, duplicates_detected)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ROI analysis failed: {str(e)}")

@router.get("/predictive-insights")
def get_predictive_insights(
    forecast_days: int = 30,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Generate predictive insights based on AI analysis"""
    try:
        # Historical data for prediction (last 90 days)
        lookback_date = datetime.utcnow() - timedelta(days=90)
        
        # Lead generation trends
        historical_leads = db.query(
            func.date(Lead.created_at).label('date'),
            func.count(Lead.id).label('count')
        ).filter(
            Lead.owner_id == current_user.id,
            Lead.created_at >= lookback_date
        ).group_by(func.date(Lead.created_at)).order_by('date').all()
        
        # Calculate trend
        if len(historical_leads) >= 7:
            recent_avg = sum(day.count for day in historical_leads[-7:]) / 7
            older_avg = sum(day.count for day in historical_leads[-14:-7]) / 7 if len(historical_leads) >= 14 else recent_avg
            growth_rate = (recent_avg - older_avg) / max(older_avg, 1)
        else:
            recent_avg = sum(day.count for day in historical_leads) / max(len(historical_leads), 1)
            growth_rate = 0
        
        # Predicted lead generation
        predicted_leads = max(0, recent_avg * (1 + growth_rate) * forecast_days)
        
        # Quality predictions based on recent trends
        recent_quality = db.query(
            func.avg(DataQualityScore.overall_score).label('avg_quality')
        ).join(Lead).filter(
            Lead.owner_id == current_user.id,
            DataQualityScore.created_at >= datetime.utcnow() - timedelta(days=14)
        ).scalar()
        
        # Conversion predictions
        recent_conversions = db.query(
            func.count(Lead.id).label('total'),
            func.sum(func.case([(Lead.status == 'converted', 1)], else_=0)).label('converted')
        ).filter(
            Lead.owner_id == current_user.id,
            Lead.created_at >= datetime.utcnow() - timedelta(days=30)
        ).first()
        
        conversion_rate = (recent_conversions.converted / max(recent_conversions.total, 1)) if recent_conversions.total > 0 else 0
        predicted_conversions = predicted_leads * conversion_rate
        
        # AI performance predictions
        ai_usage_trend = db.query(
            func.count(LeadPrediction.id).label('predictions')
        ).join(Lead).filter(
            Lead.owner_id == current_user.id,
            LeadPrediction.created_at >= datetime.utcnow() - timedelta(days=7)
        ).scalar()
        
        return {
            "forecast_period_days": forecast_days,
            "predictions": {
                "expected_new_leads": round(predicted_leads, 0),
                "expected_conversions": round(predicted_conversions, 1),
                "predicted_conversion_rate": round(conversion_rate * 100, 2),
                "expected_data_quality": round(recent_quality or 0, 2)
            },
            "trends": {
                "lead_generation_growth_rate": round(growth_rate * 100, 2),
                "recent_daily_average": round(recent_avg, 1),
                "ai_usage_weekly": ai_usage_trend
            },
            "recommendations": _generate_predictive_recommendations(growth_rate, recent_quality, conversion_rate),
            "confidence_factors": {
                "data_points_available": len(historical_leads),
                "trend_stability": "high" if abs(growth_rate) < 0.2 else "moderate" if abs(growth_rate) < 0.5 else "low",
                "prediction_accuracy": "moderate"  # Could be improved with more sophisticated modeling
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predictive insights failed: {str(e)}")

@router.get("/export-analytics")
def export_analytics_data(
    format: str = "json",
    include_ai_data: bool = True,
    days: int = 30,
    db: Session = Depends(get_db_for_ai),
    current_user: User = Depends(get_current_user)
):
    """Export comprehensive analytics data"""
    try:
        from utils.export_service import ExportService
        
        # Gather all analytics data
        dashboard_data = get_dashboard_analytics(include_ai_data, days, db, current_user)
        trends_data = get_lead_trends(days, include_ai_data, db, current_user)
        
        analytics_export = {
            "export_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "user_id": current_user.id,
                "time_period_days": days,
                "includes_ai_data": include_ai_data
            },
            "dashboard_summary": dashboard_data,
            "trends": trends_data
        }
        
        if include_ai_data:
            ai_dashboard = get_ai_dashboard(db, current_user)
            quality_report = get_data_quality_report(True, days, db, current_user)
            
            analytics_export.update({
                "ai_insights": ai_dashboard.dict(),
                "data_quality": quality_report
            })
        
        # Export using service
        export_service = ExportService()
        
        if format == "json":
            import json
            export_data = json.dumps(analytics_export, indent=2, default=str)
        else:
            # Convert to tabular format for CSV/Excel
            flattened_data = _flatten_analytics_for_export(analytics_export)
            from io import BytesIO
            export_data = export_service.export_leads(flattened_data, format).getvalue()
        
        return {
            "message": f"Analytics data exported in {format} format",
            "data": export_data,
            "export_size_bytes": len(str(export_data)),
            "includes_ai_data": include_ai_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics export failed: {str(e)}")

# ============================
# UTILITY FUNCTIONS
# ============================

def _get_ai_dashboard_metrics(db: Session, user_id: int, days: int) -> Dict[str, Any]:
    """Get AI-specific dashboard metrics"""
    try:
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # AI predictions summary
        ai_summary = db.query(
            func.count(LeadPrediction.id).label('total_predictions'),
            func.avg(LeadPrediction.prediction_score).label('avg_priority'),
            func.avg(LeadPrediction.confidence).label('avg_confidence')
        ).join(Lead).filter(
            Lead.owner_id == user_id,
            LeadPrediction.created_at >= start_date
        ).first()
        
        # Quality metrics
        quality_summary = db.query(
            func.avg(DataQualityScore.overall_score).label('avg_quality'),
            func.count(DataQualityScore.id).label('quality_checks')
        ).join(Lead).filter(
            Lead.owner_id == user_id,
            DataQualityScore.created_at >= start_date
        ).first()
        
        return {
            "ai_metrics": {
                "leads_analyzed": ai_summary.total_predictions or 0,
                "average_priority_score": round(ai_summary.avg_priority or 0, 2),
                "average_confidence": round(ai_summary.avg_confidence or 0, 3),
                "average_data_quality": round(quality_summary.avg_quality or 0, 2),
                "quality_assessments": quality_summary.quality_checks or 0
            }
        }
    except Exception as e:
        print(f"Error getting AI dashboard metrics: {e}")
        return {"ai_metrics": {}}

def _generate_duplicate_recommendations(duplicate_stats) -> List[str]:
    """Generate recommendations based on duplicate analysis"""
    recommendations = []
    
    if duplicate_stats.total_duplicates and duplicate_stats.total_duplicates > 0:
        if duplicate_stats.auto_merge_candidates and duplicate_stats.auto_merge_candidates > 0:
            recommendations.append(f"Consider auto-merging {duplicate_stats.auto_merge_candidates} high-confidence duplicates")
        
        manual_review = (duplicate_stats.total_duplicates or 0) - (duplicate_stats.auto_merge_candidates or 0)
        if manual_review > 0:
            recommendations.append(f"Review {manual_review} potential duplicates manually")
        
        if duplicate_stats.avg_match_score and duplicate_stats.avg_match_score < 85:
            recommendations.append("Consider adjusting duplicate detection threshold for better accuracy")
    else:
        recommendations.append("No duplicates detected - your lead data quality is excellent!")
    
    return recommendations

def _generate_conversion_insights(ai_analysis, source_performance, overall_rate) -> List[str]:
    """Generate insights from conversion analysis"""
    insights = []
    
    # AI score correlation insights
    for analysis in ai_analysis:
        if analysis[0] == 'high_ai_score' and analysis[2] > 0:
            high_ai_rate = (analysis[2] / analysis[1] * 100) if analysis[1] > 0 else 0
            if high_ai_rate > overall_rate * 1.5:
                insights.append(f"High AI-scored leads convert {high_ai_rate:.1f}% vs {overall_rate:.1f}% overall - focus on AI-prioritized leads")
    
    # Source performance insights
    best_source = max(source_performance, key=lambda x: (x[2] / x[1]) if x[1] > 0 else 0, default=None)
    if best_source and best_source[1] > 5:  # At least 5 leads for statistical significance
        best_rate = (best_source[2] / best_source[1] * 100) if best_source[1] > 0 else 0
        insights.append(f"'{best_source[0]}' source has the highest conversion rate at {best_rate:.1f}%")
    
    # Overall performance insight
    if overall_rate > 15:
        insights.append("Excellent conversion rate - consider scaling successful strategies")
    elif overall_rate < 5:
        insights.append("Low conversion rate detected - review lead qualification criteria")
    
    return insights

def _get_model_recommendation(model_comparison) -> str:
    """Get recommendation for best performing model"""
    if not model_comparison:
        return "No models available for comparison"
    
    best_model = model_comparison[0]
    user_accuracy = best_model['user_specific_metrics']['user_accuracy']
    
    if user_accuracy > 0.8:
        return f"'{best_model['model_name']}' is performing excellently for your data - continue using"
    elif user_accuracy > 0.6:
        return f"'{best_model['model_name']}' is performing well - consider providing more feedback to improve"
    else:
        return "Model performance could be improved - consider retraining with more data"

def _calculate_value_drivers(time_saved, conversion_improvement, duplicates) -> List[Dict[str, Any]]:
    """Calculate key value drivers from AI implementation"""
    drivers = []
    
    if time_saved > 0:
        drivers.append({
            "driver": "Time Automation",
            "value": f"{time_saved:.1f} hours saved",
            "impact": "high" if time_saved > 20 else "medium" if time_saved > 5 else "low"
        })
    
    if conversion_improvement > 0:
        drivers.append({
            "driver": "Conversion Improvement", 
            "value": f"+{conversion_improvement:.1f}% conversion rate",
            "impact": "high" if conversion_improvement > 5 else "medium" if conversion_improvement > 2 else "low"
        })
    
    if duplicates > 0:
        drivers.append({
            "driver": "Data Quality",
            "value": f"{duplicates} duplicates detected",
            "impact": "medium"
        })
    
    return drivers

def _generate_predictive_recommendations(growth_rate, quality, conversion_rate) -> List[str]:
    """Generate recommendations based on predictive insights"""
    recommendations = []
    
    if growth_rate > 0.1:
        recommendations.append("Lead generation is growing - consider scaling successful acquisition channels")
    elif growth_rate < -0.1:
        recommendations.append("Lead generation is declining - review and optimize acquisition strategies")
    
    if quality and quality < 70:
        recommendations.append("Data quality needs improvement - focus on better lead sources and validation")
    elif quality and quality > 85:
        recommendations.append("Excellent data quality - maintain current standards")
    
    if conversion_rate < 0.05:
        recommendations.append("Low conversion rate - improve lead qualification and nurturing processes")
    elif conversion_rate > 0.15:
        recommendations.append("High conversion rate - document and replicate successful approaches")
    
    return recommendations

def _flatten_analytics_for_export(analytics_data) -> List[Dict[str, Any]]:
    """Flatten nested analytics data for tabular export"""
    flattened = []
    
    # Extract key metrics into flat structure
    overview = analytics_data.get("dashboard_summary", {}).get("overview", {})
    for key, value in overview.items():
        flattened.append({
            "metric_type": "overview",
            "metric_name": key,
            "value": value,
            "category": "basic"
        })
    
    # Add AI metrics if available
    ai_metrics = analytics_data.get("ai_insights", {})
    for key, value in ai_metrics.items():
        if isinstance(value, (int, float)):
            flattened.append({
                "metric_type": "ai_performance",
                "metric_name": key,
                "value": value,
                "category": "artificial_intelligence"
            })
    
    return flattened

# Additional helper function for calculating distinct values
def distinct(column):
    """SQLAlchemy distinct helper"""
    return func.distinct(column)

# Export router for main application
__all__ = ['router']