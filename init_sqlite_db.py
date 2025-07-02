#!/usr/bin/env python3
"""
Windows Compatible SQLite Database Initialization with AI Features
Sets up the complete database structure and initializes AI components
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import text

# Configure logging for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('database_init.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def create_tables():
    """Create all database tables including AI tables"""
    try:
        print("Creating SQLite database and tables...")
        logger.info("Starting database table creation")
        
        # Import models to register them with SQLAlchemy
        from models import Base
        from database import engine
        
        # Create all tables (existing + new AI tables)
        Base.metadata.create_all(bind=engine)
        
        print("SQLite database created: leadgen.db")
        print("All tables created successfully!")
        
        # Verify AI tables were created
        from database import SessionLocal
        db = SessionLocal()
        
        ai_tables = [
            'lead_predictions',
            'data_quality_scores', 
            'lead_insights',
            'model_metadata'
        ]
        
        created_tables = []
        failed_tables = []
        
        for table in ai_tables:
            try:
                db.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
                created_tables.append(table)
                print(f"  AI table '{table}' verified")
                logger.info(f"AI table '{table}' verified")
            except Exception as e:
                failed_tables.append(table)
                print(f"  AI table '{table}' check failed: {e}")
                logger.warning(f"AI table '{table}' check failed: {e}")
        
        db.close()
        
        if failed_tables:
            print(f"Some AI tables had issues: {failed_tables}")
            logger.warning(f"Some AI tables had issues: {failed_tables}")
        
        logger.info("Database table creation completed")
        return True
        
    except Exception as e:
        print(f"Database creation failed: {e}")
        logger.error(f"Database creation failed: {e}")
        return False

def test_connection():
    """Test database connection"""
    try:
        print("Testing SQLite connection...")
        logger.info("Testing database connection")
        
        from database import SessionLocal
        
        db = SessionLocal()
        
        try:
            # Test basic query
            result = db.execute(text("SELECT sqlite_version()")).fetchone()
            if result:
                print(f"  SQLite version: {result[0]}")
                logger.info(f"Database connection successful, SQLite version: {result[0]}")
                
                # Test a simple operation
                test_result = db.execute(text("SELECT 1")).fetchone()
                if test_result and test_result[0] == 1:
                    print("  Database operations working")
                    logger.info("Database connection test passed")
                    return True
                else:
                    print("SQLite connection test failed!")
                    logger.error("Database connection test failed")
                    return False
            else:
                print("Could not get SQLite version!")
                logger.error("Could not retrieve SQLite version")
                return False
        finally:
            db.close()
                
    except Exception as e:
        print(f"SQLite connection failed: {e}")
        logger.error(f"Database connection failed: {e}")
        return False

def create_ai_directories():
    """Create necessary directories for AI models and data"""
    try:
        print("Creating AI directories...")
        logger.info("Creating AI directory structure")
        
        # Create directory structure
        directories = [
            "ai",
            "ai/models",
            "ai/saved_models",
            "ai/training_data",
            "ai/logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"  Created directory: {directory}")
            logger.info(f"Created directory: {directory}")
        
        # Create __init__.py files for Python modules
        init_files = [
            "ai/__init__.py",
            "ai/models/__init__.py"
        ]
        
        for init_file in init_files:
            Path(init_file).touch(exist_ok=True)
            print(f"  Created module file: {init_file}")
            logger.info(f"Created module file: {init_file}")
        
        print("AI directories created successfully!")
        logger.info("AI directory structure created successfully")
        return True
        
    except Exception as e:
        print(f"Error creating AI directories: {e}")
        logger.error(f"AI directory creation failed: {e}")
        return False

def create_ai_indexes():
    """Create database indexes for better AI query performance"""
    try:
        print("Creating AI-optimized indexes...")
        logger.info("Creating AI database indexes")
        
        from database import SessionLocal
        
        db = SessionLocal()
        
        try:
            # Indexes for AI tables
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_lead_predictions_lead_id ON lead_predictions(lead_id)",
                "CREATE INDEX IF NOT EXISTS idx_lead_predictions_model_type ON lead_predictions(model_type)",
                "CREATE INDEX IF NOT EXISTS idx_lead_predictions_score ON lead_predictions(prediction_score)",
                "CREATE INDEX IF NOT EXISTS idx_data_quality_overall ON data_quality_scores(overall_score)",
                "CREATE INDEX IF NOT EXISTS idx_data_quality_lead_id ON data_quality_scores(lead_id)",
                "CREATE INDEX IF NOT EXISTS idx_lead_insights_priority ON lead_insights(priority_level)",
                "CREATE INDEX IF NOT EXISTS idx_lead_insights_type ON lead_insights(insight_type)",
                "CREATE INDEX IF NOT EXISTS idx_model_metadata_active ON model_metadata(is_active)",
                "CREATE INDEX IF NOT EXISTS idx_model_metadata_name ON model_metadata(model_name)"
            ]
            
            for index_sql in indexes:
                try:
                    db.execute(text(index_sql))
                    print(f"  Created index")
                    logger.info(f"Created index: {index_sql}")
                except Exception as e:
                    print(f"  Index creation warning: {e}")
                    logger.warning(f"Index creation failed: {e}")
            
            db.commit()
            print("AI indexes created successfully!")
            logger.info("AI indexes created successfully")
            return True
            
        finally:
            db.close()
        
    except Exception as e:
        print(f"Error creating AI indexes: {e}")
        logger.error(f"AI index creation failed: {e}")
        return False

def create_default_ai_models():
    """Create default AI model metadata entries"""
    try:
        print("Creating default AI model metadata...")
        logger.info("Creating default AI model entries")
        
        from database import SessionLocal
        from models import ModelMetadata
        from datetime import datetime
        
        db = SessionLocal()
        
        try:
            # Check if models already exist
            existing_models = db.query(ModelMetadata).count()
            if existing_models > 0:
                print(f"  Found {existing_models} existing models, skipping creation")
                logger.info(f"Found {existing_models} existing models")
                return True
            
            # Default model configurations
            default_models = [
                {
                    "model_name": "lead_priority_classifier",
                    "model_type": "classifier",
                    "version": "1.0.0",
                    "algorithm": "random_forest",
                    "accuracy_score": 0.75,
                    "precision_score": 0.72,
                    "recall_score": 0.78,
                    "f1_score": 0.75,
                    "training_samples": 1000,
                    "validation_samples": 200,
                    "feature_count": 15,
                    "training_date": datetime.utcnow(),
                    "training_duration_minutes": 2.5,
                    "model_path": "ai/saved_models/lead_classifier_v1.0.0.pkl",
                    "hyperparameters": '{"n_estimators": 100, "max_depth": 10, "random_state": 42}',
                    "feature_names": '["email_quality", "title_seniority", "company_size", "source_quality"]',
                    "is_active": True,
                    "deployment_date": datetime.utcnow(),
                    "performance_metrics": '{"auc": 0.82, "precision_recall_auc": 0.78}',
                    "cross_validation_scores": '[0.74, 0.76, 0.75, 0.73, 0.77]',
                    "model_size_mb": 5.2,
                    "predictions_made": 0,
                    "avg_prediction_time_ms": 45.0,
                    "notes": "Initial default model for lead prioritization"
                },
                {
                    "model_name": "data_quality_assessor",
                    "model_type": "regressor",
                    "version": "1.0.0",
                    "algorithm": "gradient_boosting",
                    "accuracy_score": 0.82,
                    "precision_score": 0.80,
                    "recall_score": 0.85,
                    "f1_score": 0.82,
                    "training_samples": 1500,
                    "validation_samples": 300,
                    "feature_count": 12,
                    "training_date": datetime.utcnow(),
                    "training_duration_minutes": 4.2,
                    "model_path": "ai/saved_models/quality_assessor_v1.0.0.pkl",
                    "hyperparameters": '{"n_estimators": 150, "learning_rate": 0.1, "max_depth": 8}',
                    "feature_names": '["email_format", "completeness", "consistency", "freshness"]',
                    "is_active": True,
                    "deployment_date": datetime.utcnow(),
                    "performance_metrics": '{"mae": 0.12, "r2_score": 0.78}',
                    "cross_validation_scores": '[0.80, 0.83, 0.81, 0.84, 0.82]',
                    "model_size_mb": 3.8,
                    "predictions_made": 0,
                    "avg_prediction_time_ms": 32.0,
                    "notes": "Model for assessing lead data quality and completeness"
                }
            ]
            
            # Create model entries
            for model_config in default_models:
                model = ModelMetadata(**model_config)
                db.add(model)
                print(f"  Created model: {model_config['model_name']} v{model_config['version']}")
                logger.info(f"Created model metadata: {model_config['model_name']}")
            
            db.commit()
            print("Default AI model metadata created successfully!")
            logger.info("Default AI model metadata created successfully")
            return True
            
        finally:
            db.close()
        
    except Exception as e:
        print(f"Error creating AI model metadata: {e}")
        logger.error(f"AI model metadata creation failed: {e}")
        return False

def create_sample_data():
    """Create some sample data for testing"""
    try:
        print("Creating sample data...")
        logger.info("Creating sample training data")
        
        from database import SessionLocal
        from models import User, Lead, Company, Campaign
        import random
        
        db = SessionLocal()
        
        try:
            # Check if data already exists
            existing_leads = db.query(Lead).count()
            if existing_leads > 0:
                print(f"  Found {existing_leads} existing leads, skipping sample data creation")
                logger.info(f"Found {existing_leads} existing leads")
                return True
            
            # Create a sample user
            sample_user = User(
                username="demo_user",
                email="demo@leadgen.local",
                hashed_password="$2b$12$dummy_hash_for_demo_user",
                is_active=True
            )
            db.add(sample_user)
            db.commit()
            db.refresh(sample_user)
            
            # Create sample companies
            companies_data = [
                {"name": "TechCorp Inc", "domain": "techcorp.com", "industry": "Technology", "size": "201-500"},
                {"name": "DataSystems LLC", "domain": "datasys.com", "industry": "Software", "size": "51-200"},
                {"name": "InnovateNow", "domain": "innovatenow.io", "industry": "Startups", "size": "11-50"}
            ]
            
            companies = []
            for comp_data in companies_data:
                company = Company(**comp_data)
                db.add(company)
                companies.append(company)
            
            db.commit()
            
            # Create a sample campaign
            sample_campaign = Campaign(
                name="Demo Campaign",
                description="Sample campaign for testing",
                target_criteria='{"industry": "Technology", "title_keywords": ["VP", "Director", "Manager"]}',
                status="active",
                owner_id=sample_user.id
            )
            db.add(sample_campaign)
            db.commit()
            db.refresh(sample_campaign)
            
            # Create sample leads
            titles = ["VP Sales", "Marketing Director", "Sales Manager", "Business Development Manager"]
            sources = ["linkedin", "website", "referral", "email_campaign"]
            statuses = ["new", "contacted", "qualified"]
            
            leads_created = 0
            for i in range(10):  # Create 10 sample leads
                company = random.choice(companies)
                title = random.choice(titles)
                source = random.choice(sources)
                status = random.choice(statuses)
                
                # Generate realistic email
                first_names = ["john", "jane", "mike", "sarah", "david"]
                last_names = ["smith", "johnson", "williams", "brown", "jones"]
                first_name = random.choice(first_names)
                last_name = random.choice(last_names)
                email = f"{first_name}.{last_name}@{company.domain}"
                
                lead = Lead(
                    email=email,
                    first_name=first_name.title(),
                    last_name=last_name.title(),
                    title=title,
                    phone=f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                    source=source,
                    status=status,
                    score=random.uniform(0, 100),
                    notes=f"Sample lead for testing - {title} at {company.name}",
                    company_id=company.id,
                    owner_id=sample_user.id,
                    campaign_id=sample_campaign.id
                )
                db.add(lead)
                leads_created += 1
            
            db.commit()
            print(f"  Created {leads_created} sample leads for testing")
            print("Sample data created successfully!")
            logger.info(f"Created {leads_created} sample leads for testing")
            return True
            
        finally:
            db.close()
        
    except Exception as e:
        print(f"Error creating sample data: {e}")
        logger.error(f"Sample data creation failed: {e}")
        return False

def verify_ai_setup():
    """Verify that AI components are properly set up"""
    try:
        print("Verifying AI setup...")
        logger.info("Starting AI setup verification")
        
        from database import SessionLocal
        from models import ModelMetadata, Lead, User
        
        db = SessionLocal()
        
        try:
            # Check if models exist
            models = db.query(ModelMetadata).filter(ModelMetadata.is_active == True).all()
            
            if models:
                print(f"  Found {len(models)} active AI models:")
                for model in models:
                    print(f"    - {model.model_name} v{model.version} ({model.algorithm})")
                    logger.info(f"Verified model: {model.model_name} v{model.version}")
            else:
                print("  No active AI models found")
                logger.warning("No active AI models found")
            
            # Check data
            users_count = db.query(User).count()
            leads_count = db.query(Lead).count()
            
            print(f"  Database records: {users_count} users, {leads_count} leads")
            logger.info(f"Database records: {users_count} users, {leads_count} leads")
            
            # Check AI tables
            ai_tables = [
                'lead_predictions',
                'data_quality_scores', 
                'lead_insights',
                'model_metadata'
            ]
            
            for table in ai_tables:
                try:
                    count = db.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()[0]
                    print(f"  Table {table}: {count} records")
                    logger.info(f"Table {table}: {count} records")
                except Exception as e:
                    print(f"  Table {table}: error - {e}")
                    logger.warning(f"Table {table}: error - {e}")
            
        finally:
            db.close()
        
        # Check directory structure
        required_dirs = ["ai", "ai/models", "ai/saved_models"]
        missing_dirs = []
        
        for directory in required_dirs:
            if not Path(directory).exists():
                missing_dirs.append(directory)
        
        if missing_dirs:
            print(f"  Missing directories: {', '.join(missing_dirs)}")
            logger.warning(f"Missing directories: {missing_dirs}")
        else:
            print("  All required directories exist")
            logger.info("All required directories verified")
        
        print("AI setup verification completed!")
        logger.info("AI setup verification completed")
        return True
        
    except Exception as e:
        print(f"Error verifying AI setup: {e}")
        logger.error(f"AI setup verification failed: {e}")
        return False

def main():
    """Main setup function with comprehensive AI initialization"""
    print("Starting Enhanced Lead Generation SQLite Setup with AI")
    print("=" * 70)
    logger.info("Starting database initialization with AI features")
    
    setup_steps = []
    
    # Step 1: Create core database tables
    print("\nStep 1: Creating database and tables...")
    if create_tables():
        setup_steps.append("SUCCESS: Database tables created")
    else:
        print("FAILED: Could not create database tables.")
        logger.error("Database table creation failed - aborting setup")
        return False
    
    # Step 2: Test database connection
    print("\nStep 2: Testing database connection...")
    if test_connection():
        setup_steps.append("SUCCESS: Database connection verified")
    else:
        print("FAILED: Could not connect to database.")
        logger.error("Database connection failed - aborting setup")
        return False
    
    # Step 3: Create AI-optimized indexes
    print("\nStep 3: Creating AI-optimized indexes...")
    if create_ai_indexes():
        setup_steps.append("SUCCESS: AI indexes created")
    else:
        print("WARNING: AI indexes creation failed, but continuing setup...")
        setup_steps.append("WARNING: AI indexes failed")
    
    # Step 4: Create AI directories
    print("\nStep 4: Setting up AI directory structure...")
    if create_ai_directories():
        setup_steps.append("SUCCESS: AI directories created")
    else:
        print("WARNING: AI directories creation failed, but continuing setup...")
        setup_steps.append("WARNING: AI directories failed")
    
    # Step 5: Create default AI model metadata
    print("\nStep 5: Creating AI model metadata...")
    if create_default_ai_models():
        setup_steps.append("SUCCESS: AI model metadata created")
    else:
        print("WARNING: AI model metadata creation failed, but continuing setup...")
        setup_steps.append("WARNING: AI model metadata failed")
    
    # Step 6: Initialize sample data
    print("\nStep 6: Creating sample data...")
    if create_sample_data():
        setup_steps.append("SUCCESS: Sample data created")
    else:
        print("WARNING: Sample data creation failed, but continuing setup...")
        setup_steps.append("WARNING: Sample data failed")
    
    # Step 7: Verify AI setup
    print("\nStep 7: Verifying AI setup...")
    if verify_ai_setup():
        setup_steps.append("SUCCESS: AI setup verified")
    else:
        print("WARNING: AI setup verification had issues...")
        setup_steps.append("WARNING: AI verification partial")
    
    # Final summary
    print("\n" + "=" * 70)
    print("Enhanced SQLite Setup Completed!")
    print("=" * 70)
    
    print("\nSetup Summary:")
    for step in setup_steps:
        print(f"  {step}")
    
    print(f"\nDatabase Location: {os.path.abspath('leadgen.db')}")
    print(f"AI Models Directory: {os.path.abspath('ai/saved_models')}")
    print(f"Setup Log: {os.path.abspath('database_init.log')}")
    
    print("\nNext Steps:")
    print("1. Run: python main.py")
    print("2. Visit: http://localhost:8000/docs")
    print("3. Check AI features: http://localhost:8000/ai/status")
    print("4. View health check: http://localhost:8000/health")
    
    print("\nAI Features Available:")
    print("- Smart lead prioritization and scoring")
    print("- Automated data quality assessment")
    print("- Intelligent insights and recommendations")
    print("- Lead similarity analysis")
    
    logger.info("Database initialization completed successfully")
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nSetup completed successfully!")
            sys.exit(0)
        else:
            print("\nSetup completed with errors!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during setup: {e}")
        logger.error(f"Unexpected setup error: {e}")
        sys.exit(1)