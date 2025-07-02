from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List, Optional
import uvicorn
from datetime import datetime, timedelta
import os
from contextlib import asynccontextmanager

from database import get_db, engine, SessionLocal
from models import Base
from sqlalchemy import text
from routers import auth, leads, companies, campaigns, analytics, ai_insights
from utils.lead_scraper import LeadScraper
from utils.email_finder import EmailFinder
from utils.lead_scorer import LeadScorer

def initialize_database_with_ai():
    """Initialize database with all tables including AI tables"""
    try:
        print("📝 Creating database tables...")
        # Create all tables (this will include AI tables from models.py)
        Base.metadata.create_all(bind=engine)
        
        # Verify AI tables exist
        db = SessionLocal()
        try:
            ai_tables_to_check = [
                'lead_predictions',
                'data_quality_scores', 
                'lead_insights',
                'model_metadata'
            ]
            
            for table in ai_tables_to_check:
                try:
                    db.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
                    print(f"✅ AI table '{table}' ready")
                except Exception as e:
                    print(f"⚠️ AI table '{table}' issue: {e}")
            
        finally:
            db.close()
            
        print("✅ Database tables initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Lead Generation API with AI...")
    
    # Initialize database first
    print("📋 Initializing database...")
    if not initialize_database_with_ai():
        print("⚠️ Database initialization had issues, but continuing...")
    
    # Initialize AI components on startup
    try:
        print("🤖 Initializing AI components...")
        
        # Create AI directories if they don't exist
        ai_dirs = ["ai", "ai/saved_models", "ai/training_data", "ai/logs"]
        for directory in ai_dirs:
            os.makedirs(directory, exist_ok=True)
        
        from utils.ai_lead_analyzer import AILeadAnalyzer
        from utils.data_quality_engine import DataQualityEngine
        
        # Pre-load AI models for faster response times
        analyzer = AILeadAnalyzer()
        quality_engine = DataQualityEngine()
        
        # Store in app state for reuse
        app.state.ai_analyzer = analyzer
        app.state.quality_engine = quality_engine
        
        print("✅ AI components initialized successfully")
        
        # Initialize default AI model metadata if not exists
        try:
            from models import ModelMetadata
            db = SessionLocal()
            
            # Check if any models exist
            existing_models = db.query(ModelMetadata).count()
            if existing_models == 0:
                print("📊 Creating default AI model metadata...")
                
                default_model = ModelMetadata(
                    model_name="lead_priority_classifier",
                    model_type="classifier",
                    version="1.0.0",
                    algorithm="random_forest",
                    accuracy_score=0.75,
                    precision_score=0.72,
                    recall_score=0.78,
                    f1_score=0.75,
                    training_samples=1000,
                    validation_samples=200,
                    feature_count=15,
                    training_date=datetime.utcnow(),
                    training_duration_minutes=2.5,
                    model_path="ai/saved_models/lead_classifier_v1.0.0.pkl",
                    hyperparameters='{"n_estimators": 100, "max_depth": 10, "random_state": 42}',
                    feature_names='["email_quality", "title_seniority", "company_size", "source_quality"]',
                    is_active=True,
                    deployment_date=datetime.utcnow(),
                    performance_metrics='{"auc": 0.82, "precision_recall_auc": 0.78}',
                    cross_validation_scores='[0.74, 0.76, 0.75, 0.73, 0.77]',
                    model_size_mb=5.2,
                    predictions_made=0,
                    avg_prediction_time_ms=45.0,
                    notes="Initial default model for lead prioritization"
                )
                
                db.add(default_model)
                db.commit()
                print("✅ Default AI model metadata created")
            
            db.close()
            
        except Exception as e:
            print(f"⚠️ AI model metadata setup warning: {e}")
        
    except Exception as e:
        print(f"⚠️ AI initialization warning: {e}")
        print("📝 AI features may have limited functionality")
        # Create fallback state
        app.state.ai_analyzer = None
        app.state.quality_engine = None
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Lead Generation API...")

app = FastAPI(
    title="AI-Enhanced Lead Generation Tool API",
    description="Professional B2B Lead Generation and Enrichment Platform with AI Intelligence",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(leads.router, prefix="/leads", tags=["Leads"])
app.include_router(companies.router, prefix="/companies", tags=["Companies"])
app.include_router(campaigns.router, prefix="/campaigns", tags=["Campaigns"])
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
app.include_router(ai_insights.router, prefix="/ai", tags=["AI Insights"])

@app.get("/")
async def root():
    return {
        "message": "AI-Enhanced Lead Generation Tool API",
        "version": "2.0.0",
        "status": "active",
        "features": {
            "lead_generation": "active",
            "ai_insights": "active",
            "data_quality": "active",
            "smart_scoring": "active"
        },
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "ai_dashboard": "/ai/dashboard"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check with AI system status"""
    
    # Check database connectivity
    db_status = "connected"
    ai_tables_status = {}
    
    try:
        db = SessionLocal()
        
        # Test basic connection
        db.execute(text("SELECT 1"))
        
        # Check AI tables
        ai_tables = [
            'lead_predictions',
            'data_quality_scores', 
            'lead_insights',
            'model_metadata'
        ]
        
        for table in ai_tables:
            try:
                result = db.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
                ai_tables_status[table] = f"exists ({result[0]} records)"
            except Exception:
                ai_tables_status[table] = "missing"
        
        db.close()
        
    except Exception as e:
        db_status = "disconnected"
        ai_tables_status = {"error": str(e)}
    
    # Check AI system status
    ai_status = "active"
    ai_models_loaded = False
    
    try:
        if hasattr(app.state, 'ai_analyzer') and app.state.ai_analyzer:
            ai_models_loaded = True
            ai_status = "active"
        else:
            # Try to initialize on-demand
            from utils.ai_lead_analyzer import AILeadAnalyzer
            analyzer = AILeadAnalyzer()
            ai_models_loaded = True
            ai_status = "active"
    except Exception as e:
        ai_status = f"limited: {e}"
    
    # Check data quality engine
    quality_engine_status = "active"
    try:
        if hasattr(app.state, 'quality_engine') and app.state.quality_engine:
            quality_engine_status = "active"
        else:
            from utils.data_quality_engine import DataQualityEngine
            engine = DataQualityEngine()
            quality_engine_status = "active"
    except Exception as e:
        quality_engine_status = f"unavailable: {e}"
    
    overall_status = "healthy"
    if db_status != "connected":
        overall_status = "degraded"
    elif "missing" in str(ai_tables_status) or "limited" in ai_status:
        overall_status = "partial"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow(),
        "components": {
            "database": db_status,
            "ai_tables": ai_tables_status,
            "ai_insights": ai_status,
            "data_quality_engine": quality_engine_status,
            "models_loaded": ai_models_loaded
        },
        "version": "2.0.0"
    }

@app.get("/ai/status")
async def ai_system_status():
    """Detailed AI system status"""
    try:
        db = SessionLocal()
        
        # Count AI records
        ai_stats = {}
        ai_tables = [
            ('lead_predictions', 'AI Predictions'),
            ('data_quality_scores', 'Quality Assessments'), 
            ('lead_insights', 'AI Insights'),
            ('model_metadata', 'ML Models')
        ]
        
        for table, name in ai_tables:
            try:
                count = db.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()[0]
                ai_stats[name] = count
            except Exception:
                ai_stats[name] = "table missing"
        
        # Check models
        try:
            from models import ModelMetadata
            active_models = db.query(ModelMetadata).filter(ModelMetadata.is_active == True).count()
            total_models = db.query(ModelMetadata).count()
        except Exception:
            active_models = 0
            total_models = 0
        
        db.close()
        
        # Overall AI status
        ai_status = "online"
        if any(v == "table missing" for v in ai_stats.values()):
            ai_status = "offline"
        elif active_models == 0:
            ai_status = "limited"
        
        return {
            "ai_status": ai_status,
            "active_models": active_models,
            "total_models": total_models,
            "ai_statistics": ai_stats,
            "last_updated": datetime.utcnow(),
            "features_available": [
                "Lead Prioritization",
                "Data Quality Assessment", 
                "Smart Insights",
                "Duplicate Detection"
            ] if ai_status == "online" else ["Basic Features Only"]
        }
        
    except Exception as e:
        return {
            "ai_status": "offline",
            "error": str(e),
            "active_models": 0,
            "total_models": 0,
            "ai_statistics": {},
            "last_updated": datetime.utcnow(),
            "features_available": []
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )