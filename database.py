# Updated database.py with better error logging

import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQLite database URL
DATABASE_URL = "sqlite:///./leadgen.db"

# Create engine with optimized settings for SQLite
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,  # Required for SQLite
        "timeout": 30  # Increase timeout to 30 seconds
    },
    pool_pre_ping=True,  # Enable connection health checks
    pool_size=5,
    max_overflow=10,
    echo=False  # Set to True for SQL query debugging
)

# Configure for better performance with AI workloads
with engine.connect() as conn:
    conn.execute(text("PRAGMA journal_mode=WAL"))  # Write-Ahead Logging for better concurrency
    conn.execute(text("PRAGMA synchronous=NORMAL"))  # Balance between safety and speed
    conn.execute(text("PRAGMA temp_store=MEMORY"))  # Use memory for temporary tables
    conn.execute(text("PRAGMA mmap_size=536870912"))  # Use memory-mapped I/O (512MB)
    conn.execute(text("PRAGMA cache_size=-64000"))  # Use 64MB for cache
    conn.commit()

logger.info("SQLite database configured with optimizations for AI workloads")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency to get database session with enhanced error handling"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {type(e).__name__}: {str(e)}")
        try:
            db.rollback()
        except Exception as rollback_error:
            logger.error(f"Rollback failed: {rollback_error}")
        raise
    finally:
        try:
            db.close()
        except Exception as close_error:
            logger.error(f"Database close error: {close_error}")

def get_db_for_ai():
    """Simplified database session for AI operations - same as regular session"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"AI database session error: {type(e).__name__}: {str(e)}")
        try:
            db.rollback()
        except Exception as rollback_error:
            logger.error(f"AI rollback failed: {rollback_error}")
        raise
    finally:
        try:
            db.close()
        except Exception as close_error:
            logger.error(f"AI database close error: {close_error}")

def _table_exists(db, table_name):
    """Check if a table exists in SQLite"""
    try:
        db.execute(text(f"SELECT COUNT(*) FROM {table_name} LIMIT 1")).fetchone()
        return True
    except Exception:
        return False

def check_database_health():
    """Check database connectivity and performance"""
    try:
        db = SessionLocal()
        
        # Test basic connectivity
        result = db.execute(text("SELECT 1")).fetchone()
        
        # Check if core tables exist
        core_tables = ['users', 'leads', 'companies', 'campaigns']
        existing_core_tables = []
        
        for table in core_tables:
            if _table_exists(db, table):
                existing_core_tables.append(table)
        
        # Check if AI tables exist
        ai_tables = [
            'lead_predictions',
            'data_quality_scores', 
            'lead_insights',
            'model_metadata'
        ]
        
        existing_ai_tables = []
        missing_ai_tables = []
        
        for table in ai_tables:
            if _table_exists(db, table):
                existing_ai_tables.append(table)
            else:
                missing_ai_tables.append(table)
        
        # Get table counts
        table_counts = {}
        for table in existing_core_tables + existing_ai_tables:
            try:
                count = db.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()[0]
                table_counts[table] = count
            except Exception as e:
                logger.warning(f"Could not count rows in {table}: {e}")
                table_counts[table] = 0
        
        db.close()
        
        return {
            "status": "healthy" if result else "unhealthy",
            "core_tables": existing_core_tables,
            "ai_tables": existing_ai_tables,
            "missing_ai_tables": missing_ai_tables,
            "table_counts": table_counts,
            "database_file": os.path.abspath(DATABASE_URL.replace("sqlite:///", "")),
            "ai_features_available": len(existing_ai_tables) >= 4,
            "recommendations": _get_health_recommendations(missing_ai_tables, existing_ai_tables)
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {type(e).__name__}: {str(e)}")
        return {
            "status": "critical",
            "error": str(e),
            "recommendations": ["Check database file permissions", "Ensure database file exists", "Run database initialization"]
        }

def _get_health_recommendations(missing_ai_tables, existing_ai_tables):
    """Generate health recommendations based on database state"""
    recommendations = []
    
    if missing_ai_tables:
        recommendations.append(f"Run database initialization to create missing AI tables: {', '.join(missing_ai_tables)}")
        recommendations.append("Execute: python init_sqlite_db.py")
    
    if not existing_ai_tables:
        recommendations.append("No AI tables found - AI features will be offline")
        recommendations.append("Initialize database with AI support")
    elif len(existing_ai_tables) < 4:
        recommendations.append("Some AI tables missing - AI features may be limited")
    
    if not recommendations:
        recommendations.append("Database is healthy - all tables present")
    
    return recommendations

def create_ai_tables():
    """Create AI tables if they don't exist"""
    try:
        logger.info("Creating AI tables...")
        
        # Import models to ensure they're registered
        from models import Base
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("AI tables created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create AI tables: {type(e).__name__}: {str(e)}")
        return False

def ensure_ai_tables():
    """Ensure AI tables exist, create them if they don't"""
    try:
        health = check_database_health()
        
        if health["missing_ai_tables"]:
            logger.info(f"Missing AI tables detected: {health['missing_ai_tables']}")
            return create_ai_tables()
        
        return True
        
    except Exception as e:
        logger.error(f"Error ensuring AI tables: {type(e).__name__}: {str(e)}")
        return False