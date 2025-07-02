# ai/training/model_trainer.py - Model Training Scripts and Management
import sys
import os
from pathlib import Path

def fix_python_path():
    """Fix Python path to include project root"""
    # Get the current script's directory
    current_file = Path(__file__).resolve()
    
    # Find the project root (directory containing 'ai' folder)
    project_root = current_file
    while project_root.parent != project_root:  # Don't go beyond filesystem root
        if (project_root / 'ai').is_dir():
            break
        project_root = project_root.parent
    
    # Add project root to Python path if not already there
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
        print(f"Added to Python path: {project_root_str}")

# Call the fix function
fix_python_path()
# ============================
# IMPORTS
# ============================
import os
import pickle
import json
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Machine Learning
try:
    from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
    from sklearn.svm import SVC, SVR
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
    )
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Install with: pip install scikit-learn")

# XGBoost (optional)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# ENUMS AND DATA CLASSES
# ============================

class ModelType(Enum):
    """Available model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    SVM = "svm"
    XGBOOST = "xgboost"
    ENSEMBLE = "ensemble"

class TrainingStatus(Enum):
    """Training job status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ModelPurpose(Enum):
    """Model purpose/application"""
    LEAD_PRIORITY = "lead_priority"
    DATA_QUALITY = "data_quality"
    LEAD_SIMILARITY = "lead_similarity"
    CONVERSION_PREDICTION = "conversion_prediction"
    TIMING_OPTIMIZATION = "timing_optimization"

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    model_type: ModelType
    model_purpose: ModelPurpose
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    test_size: float = 0.2
    validation_size: float = 0.1
    cross_validation_folds: int = 5
    random_state: int = 42
    enable_hyperparameter_tuning: bool = False
    hyperparameter_search: str = "grid"  # 'grid', 'random'
    max_training_time_minutes: int = 30
    early_stopping: bool = True
    feature_selection: bool = True
    ensemble_size: int = 3

@dataclass
class TrainingResult:
    """Results from model training"""
    model_id: str
    model_type: ModelType
    model_purpose: ModelPurpose
    training_status: TrainingStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    auc_score: float = 0.0
    r2_score: float = 0.0  # For regression
    mae: float = 0.0  # Mean absolute error
    rmse: float = 0.0  # Root mean square error
    
    # Training details
    training_samples: int = 0
    validation_samples: int = 0
    feature_count: int = 0
    training_duration_seconds: float = 0.0
    cross_validation_scores: List[float] = field(default_factory=list)
    best_hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Model artifacts
    model_path: str = ""
    model_size_mb: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: str = ""
    
    # Error handling
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)

@dataclass
class TrainingJob:
    """Represents a training job"""
    job_id: str
    config: TrainingConfig
    training_data: List[Dict[str, Any]]
    user_id: Optional[int] = None
    priority: int = 1  # 1=low, 5=high
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    estimated_duration_minutes: int = 10
    progress_percentage: float = 0.0
    status: TrainingStatus = TrainingStatus.QUEUED
    result: Optional[TrainingResult] = None

# ============================
# MAIN MODEL TRAINER CLASS
# ============================

class ModelTrainer:
    """
    Comprehensive model training and management system
    """
    
    def __init__(self, models_path: str = "ai/saved_models", max_concurrent_jobs: int = 2):
        """
        Initialize the Model Trainer
        
        Args:
            models_path: Directory to save trained models
            max_concurrent_jobs: Maximum number of concurrent training jobs
        """
        self.models_path = models_path
        self.max_concurrent_jobs = max_concurrent_jobs
        
        # Training job management
        self.training_queue: List[TrainingJob] = []
        self.active_jobs: Dict[str, TrainingJob] = {}
        self.completed_jobs: Dict[str, TrainingJob] = {}
        self.job_executor = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        
        # Model registry
        self.model_registry: Dict[str, Dict[str, Any]] = {}
        
        # Default model configurations
        self.default_configs = self._initialize_default_configs()
        
        # Ensure directories exist
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(f"{models_path}/logs", exist_ok=True)
        
        logger.info("🎓 Model Trainer initialized")
    
    def _initialize_default_configs(self) -> Dict[ModelPurpose, TrainingConfig]:
        """Initialize default training configurations for different model purposes"""
        configs = {}
        
        # Lead Priority Classification
        configs[ModelPurpose.LEAD_PRIORITY] = TrainingConfig(
            model_type=ModelType.RANDOM_FOREST,
            model_purpose=ModelPurpose.LEAD_PRIORITY,
            hyperparameters={
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            enable_hyperparameter_tuning=True
        )
        
        # Data Quality Assessment (Regression)
        configs[ModelPurpose.DATA_QUALITY] = TrainingConfig(
            model_type=ModelType.GRADIENT_BOOSTING,
            model_purpose=ModelPurpose.DATA_QUALITY,
            hyperparameters={
                'n_estimators': 150,
                'learning_rate': 0.1,
                'max_depth': 8,
                'min_samples_split': 4,
                'random_state': 42
            },
            enable_hyperparameter_tuning=True
        )
        
        # Lead Similarity (Clustering/Classification)
        configs[ModelPurpose.LEAD_SIMILARITY] = TrainingConfig(
            model_type=ModelType.RANDOM_FOREST,
            model_purpose=ModelPurpose.LEAD_SIMILARITY,
            hyperparameters={
                'n_estimators': 80,
                'max_depth': 12,
                'min_samples_split': 3,
                'random_state': 42
            }
        )
        
        # Conversion Prediction
        configs[ModelPurpose.CONVERSION_PREDICTION] = TrainingConfig(
            model_type=ModelType.GRADIENT_BOOSTING,
            model_purpose=ModelPurpose.CONVERSION_PREDICTION,
            hyperparameters={
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 6,
                'random_state': 42
            },
            enable_hyperparameter_tuning=True
        )
        
        # Timing Optimization
        configs[ModelPurpose.TIMING_OPTIMIZATION] = TrainingConfig(
            model_type=ModelType.LOGISTIC_REGRESSION,
            model_purpose=ModelPurpose.TIMING_OPTIMIZATION,
            hyperparameters={
                'C': 1.0,
                'penalty': 'l2',
                'random_state': 42,
                'max_iter': 1000
            }
        )
        
        return configs
    
    # ============================
    # TRAINING JOB MANAGEMENT
    # ============================
    
    def submit_training_job(self, 
                           model_purpose: ModelPurpose,
                           training_data: List[Dict[str, Any]],
                           config: Optional[TrainingConfig] = None,
                           user_id: Optional[int] = None,
                           priority: int = 1) -> str:
        """
        Submit a new training job
        
        Args:
            model_purpose: Purpose/type of model to train
            training_data: Training data
            config: Training configuration (uses default if None)
            user_id: User requesting the training
            priority: Job priority (1=low, 5=high)
            
        Returns:
            Job ID
        """
        # Use default config if none provided
        if config is None:
            config = self.default_configs.get(model_purpose)
            if config is None:
                raise ValueError(f"No default config for model purpose: {model_purpose}")
        
        # Generate job ID
        job_id = f"{model_purpose.value}_{int(time.time())}_{np.random.randint(1000, 9999)}"
        
        # Estimate training duration
        estimated_duration = self._estimate_training_duration(len(training_data), config)
        
        # Create training job
        job = TrainingJob(
            job_id=job_id,
            config=config,
            training_data=training_data,
            user_id=user_id,
            priority=priority,
            estimated_duration_minutes=estimated_duration
        )
        
        # Add to queue
        self.training_queue.append(job)
        self.training_queue.sort(key=lambda x: (-x.priority, x.created_at))  # Sort by priority, then time
        
        logger.info(f"🎯 Training job submitted: {job_id}")
        
        # Start processing if there's capacity
        self._process_queue()
        
        return job_id
    
    def _estimate_training_duration(self, data_size: int, config: TrainingConfig) -> int:
        """Estimate training duration in minutes"""
        base_time = 2  # Base 2 minutes
        
        # Data size factor
        size_factor = max(1, data_size / 1000)  # 1 minute per 1000 samples
        
        # Model complexity factor
        complexity_factors = {
            ModelType.LOGISTIC_REGRESSION: 0.5,
            ModelType.LINEAR_REGRESSION: 0.5,
            ModelType.RANDOM_FOREST: 1.0,
            ModelType.GRADIENT_BOOSTING: 1.5,
            ModelType.SVM: 2.0,
            ModelType.XGBOOST: 1.2,
            ModelType.ENSEMBLE: 3.0
        }
        
        complexity_factor = complexity_factors.get(config.model_type, 1.0)
        
        # Hyperparameter tuning factor
        tuning_factor = 5 if config.enable_hyperparameter_tuning else 1
        
        estimated_minutes = int(base_time * size_factor * complexity_factor * tuning_factor)
        return min(estimated_minutes, config.max_training_time_minutes)
    
    def _process_queue(self):
        """Process training queue"""
        while (len(self.active_jobs) < self.max_concurrent_jobs and 
               self.training_queue):
            
            # Get next job
            job = self.training_queue.pop(0)
            
            # Move to active jobs
            self.active_jobs[job.job_id] = job
            job.status = TrainingStatus.RUNNING
            job.started_at = datetime.utcnow()
            
            # Submit for execution
            future = self.job_executor.submit(self._execute_training_job, job)
            future.add_done_callback(lambda f, job_id=job.job_id: self._job_completed(job_id, f))
            
            logger.info(f"🔄 Started training job: {job.job_id}")
    
    def _execute_training_job(self, job: TrainingJob) -> TrainingResult:
        """Execute a training job"""
        try:
            logger.info(f"🎓 Training model: {job.config.model_purpose.value}")
            
            # Update progress
            job.progress_percentage = 10.0
            
            # Prepare data
            X, y = self._prepare_training_data(job.training_data, job.config)
            job.progress_percentage = 30.0
            
            # Train model
            result = self._train_model(X, y, job.config, job.job_id)
            job.progress_percentage = 80.0
            
            # Save model
            self._save_trained_model(result, job.config)
            job.progress_percentage = 90.0
            
            # Update result
            result.training_status = TrainingStatus.COMPLETED
            result.end_time = datetime.utcnow()
            result.training_duration_seconds = (result.end_time - result.start_time).total_seconds()
            
            job.progress_percentage = 100.0
            
            logger.info(f"✅ Training completed: {job.job_id}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Training failed: {job.job_id} - {e}")
            
            result = TrainingResult(
                model_id=job.job_id,
                model_type=job.config.model_type,
                model_purpose=job.config.model_purpose,
                training_status=TrainingStatus.FAILED,
                start_time=job.started_at or datetime.utcnow(),
                end_time=datetime.utcnow(),
                error_message=str(e)
            )
            
            return result
    
    def _job_completed(self, job_id: str, future):
        """Handle job completion"""
        try:
            # Get job and result
            job = self.active_jobs.pop(job_id, None)
            if job:
                job.result = future.result()
                job.status = job.result.training_status
                self.completed_jobs[job_id] = job
                
                # Update model registry
                if job.result.training_status == TrainingStatus.COMPLETED:
                    self._register_model(job.result)
            
            # Process next job in queue
            self._process_queue()
            
        except Exception as e:
            logger.error(f"❌ Job completion error: {job_id} - {e}")
    
    # ============================
    # MODEL TRAINING CORE
    # ============================
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]], 
                             config: TrainingConfig) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for model training"""
        
        # Import feature engineering
        from ai.models.lead_classifier import LeadFeatureEngineer
        from ai.models.quality_scorer import QualityScorer
        
        feature_engineer = LeadFeatureEngineer()
        features_list = []
        targets = []
        
        for lead_data in training_data:
            try:
                # Extract features based on model purpose
                if config.model_purpose == ModelPurpose.LEAD_PRIORITY:
                    features = feature_engineer.extract_features(lead_data)
                    target = lead_data.get('priority', 'medium')
                    
                elif config.model_purpose == ModelPurpose.DATA_QUALITY:
                    quality_scorer = QualityScorer()
                    features = quality_scorer._extract_features(lead_data)
                    target = lead_data.get('quality_score', 75.0)  # Default quality score
                    
                elif config.model_purpose == ModelPurpose.LEAD_SIMILARITY:
                    features = feature_engineer.extract_features(lead_data)
                    target = lead_data.get('similarity_group', 0)
                    
                elif config.model_purpose == ModelPurpose.CONVERSION_PREDICTION:
                    features = feature_engineer.extract_features(lead_data)
                    target = lead_data.get('converted', False)
                    
                elif config.model_purpose == ModelPurpose.TIMING_OPTIMIZATION:
                    features = feature_engineer.extract_features(lead_data)
                    target = lead_data.get('optimal_contact_time', 'business_hours')
                    
                else:
                    features = feature_engineer.extract_features(lead_data)
                    target = lead_data.get('target', 'unknown')
                
                features_list.append(features)
                targets.append(target)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to process lead data: {e}")
                continue
        
        # Convert to DataFrames
        X = pd.DataFrame(features_list)
        y = pd.Series(targets)
        
        # Handle missing values
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(X.mode().iloc[0] if not X.mode().empty else 0)
        
        logger.info(f"📊 Prepared training data: {len(X)} samples, {len(X.columns)} features")
        
        return X, y
    
    def _train_model(self, X: pd.DataFrame, y: pd.Series, 
                    config: TrainingConfig, model_id: str) -> TrainingResult:
        """Train the machine learning model"""
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("Scikit-learn is required for model training")
        
        start_time = datetime.utcnow()
        
        # Initialize result
        result = TrainingResult(
            model_id=model_id,
            model_type=config.model_type,
            model_purpose=config.model_purpose,
            training_status=TrainingStatus.RUNNING,
            start_time=start_time,
            training_samples=len(X),
            feature_count=len(X.columns)
        )
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=y if self._is_classification_task(config.model_purpose) else None
            )
            
            result.validation_samples = len(X_test)
            
            # Create preprocessor
            preprocessor = self._create_preprocessor(X)
            
            # Create and configure model
            model = self._create_model(config)
            
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Hyperparameter tuning
            if config.enable_hyperparameter_tuning:
                pipeline, best_params = self._hyperparameter_tuning(
                    pipeline, X_train, y_train, config
                )
                result.best_hyperparameters = best_params
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics based on task type
            if self._is_classification_task(config.model_purpose):
                result.accuracy = accuracy_score(y_test, y_pred)
                result.precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                result.recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                result.f1_score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # AUC score for binary classification
                if len(np.unique(y)) == 2:
                    try:
                        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                        result.auc_score = roc_auc_score(y_test, y_pred_proba)
                    except:
                        result.auc_score = 0.5
                
                result.confusion_matrix = confusion_matrix(y_test, y_pred)
                result.classification_report = classification_report(y_test, y_pred)
                
            else:  # Regression task
                result.mae = mean_absolute_error(y_test, y_pred)
                result.rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                result.r2_score = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(
                pipeline, X_train, y_train, 
                cv=config.cross_validation_folds,
                scoring='accuracy' if self._is_classification_task(config.model_purpose) else 'r2'
            )
            result.cross_validation_scores = cv_scores.tolist()
            
            # Feature importance
            if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                importance_values = pipeline.named_steps['model'].feature_importances_
                feature_names = X.columns
                result.feature_importance = dict(zip(feature_names, importance_values))
            
            # Save trained pipeline
            model_filename = f"{model_id}_{config.model_purpose.value}.pkl"
            model_path = os.path.join(self.models_path, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump(pipeline, f)
            
            result.model_path = model_path
            result.model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            logger.info(f"🎯 Model training completed: {model_id}")
            logger.info(f"📊 Performance - Accuracy: {result.accuracy:.3f}, F1: {result.f1_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            result.training_status = TrainingStatus.FAILED
            result.error_message = str(e)
            return result
    
    def _is_classification_task(self, model_purpose: ModelPurpose) -> bool:
        """Determine if the task is classification or regression"""
        classification_tasks = [
            ModelPurpose.LEAD_PRIORITY,
            ModelPurpose.LEAD_SIMILARITY,
            ModelPurpose.CONVERSION_PREDICTION,
            ModelPurpose.TIMING_OPTIMIZATION
        ]
        return model_purpose in classification_tasks
    
    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Create preprocessing pipeline"""
        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
        
        # Create transformers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
    
    def _create_model(self, config: TrainingConfig):
        """Create ML model based on configuration"""
        hyperparams = config.hyperparameters
        
        if config.model_type == ModelType.RANDOM_FOREST:
            if self._is_classification_task(config.model_purpose):
                return RandomForestClassifier(**hyperparams)
            else:
                return RandomForestRegressor(**hyperparams)
                
        elif config.model_type == ModelType.GRADIENT_BOOSTING:
            if self._is_classification_task(config.model_purpose):
                return GradientBoostingClassifier(**hyperparams)
            else:
                return GradientBoostingRegressor(**hyperparams)
                
        elif config.model_type == ModelType.LOGISTIC_REGRESSION:
            return LogisticRegression(**hyperparams)
            
        elif config.model_type == ModelType.LINEAR_REGRESSION:
            return LinearRegression(**hyperparams)
            
        elif config.model_type == ModelType.SVM:
            if self._is_classification_task(config.model_purpose):
                return SVC(**hyperparams)
            else:
                return SVR(**hyperparams)
                
        elif config.model_type == ModelType.XGBOOST and XGBOOST_AVAILABLE:
            if self._is_classification_task(config.model_purpose):
                return xgb.XGBClassifier(**hyperparams)
            else:
                return xgb.XGBRegressor(**hyperparams)
        
        else:
            # Default to RandomForest
            if self._is_classification_task(config.model_purpose):
                return RandomForestClassifier(random_state=42)
            else:
                return RandomForestRegressor(random_state=42)
    
    def _hyperparameter_tuning(self, pipeline, X_train, y_train, 
                              config: TrainingConfig) -> Tuple[Pipeline, Dict[str, Any]]:
        """Perform hyperparameter tuning"""
        logger.info("🔧 Starting hyperparameter tuning...")
        
        # Define parameter grids based on model type
        param_grids = self._get_hyperparameter_grids(config)
        
        if config.hyperparameter_search == "grid":
            search = GridSearchCV(
                pipeline,
                param_grids,
                cv=config.cross_validation_folds,
                scoring='accuracy' if self._is_classification_task(config.model_purpose) else 'r2',
                n_jobs=1,
                verbose=1
            )
        else:  # Random search
            search = RandomizedSearchCV(
                pipeline,
                param_grids,
                cv=config.cross_validation_folds,
                scoring='accuracy' if self._is_classification_task(config.model_purpose) else 'r2',
                n_jobs=1,
                n_iter=50,
                random_state=config.random_state,
                verbose=1
            )
        
        # Perform search
        search.fit(X_train, y_train)
        
        best_pipeline = search.best_estimator_
        best_params = search.best_params_
        
        logger.info(f"✅ Hyperparameter tuning completed. Best score: {search.best_score_:.3f}")
        
        return best_pipeline, best_params
    
    def _get_hyperparameter_grids(self, config: TrainingConfig) -> Dict[str, Any]:
        """Get hyperparameter grids for tuning"""
        
        if config.model_type == ModelType.RANDOM_FOREST:
            return {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4]
            }
            
        elif config.model_type == ModelType.GRADIENT_BOOSTING:
            return {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.05, 0.1, 0.2],
                'model__max_depth': [3, 5, 7],
                'model__min_samples_split': [2, 5, 10]
            }
            
        elif config.model_type == ModelType.LOGISTIC_REGRESSION:
            return {
                'model__C': [0.1, 1.0, 10.0],
                'model__penalty': ['l1', 'l2'],
                'model__solver': ['liblinear', 'saga']
            }
            
        elif config.model_type == ModelType.SVM:
            return {
                'model__C': [0.1, 1, 10],
                'model__kernel': ['rbf', 'linear'],
                'model__gamma': ['scale', 'auto']
            }
            
        else:
            return {}
    
    # ============================
    # MODEL MANAGEMENT
    # ============================
    
    def _save_trained_model(self, result: TrainingResult, config: TrainingConfig):
        """Save trained model with metadata"""
        try:
            # Create model metadata
            # Create model metadata
            metadata = {
                'model_id': result.model_id,
                'model_type': result.model_type.value,
                'model_purpose': result.model_purpose.value,
                'version': '1.0.0',
                'created_at': result.start_time.isoformat(),
                'training_duration_seconds': result.training_duration_seconds,
                'performance_metrics': {
                    'accuracy': result.accuracy,
                    'precision': result.precision,
                    'recall': result.recall,
                    'f1_score': result.f1_score,
                    'auc_score': result.auc_score,
                    'r2_score': result.r2_score,
                    'mae': result.mae,
                    'rmse': result.rmse
                },
                'training_details': {
                    'training_samples': result.training_samples,
                    'validation_samples': result.validation_samples,
                    'feature_count': result.feature_count,
                    'cross_validation_scores': result.cross_validation_scores,
                    'best_hyperparameters': result.best_hyperparameters
                },
                'feature_importance': result.feature_importance,
                'model_path': result.model_path,
                'model_size_mb': result.model_size_mb
            }
            
            # Save metadata
            metadata_path = result.model_path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"💾 Model metadata saved: {metadata_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save model metadata: {e}")
    
    def _register_model(self, result: TrainingResult):
        """Register model in the model registry"""
        try:
            model_key = f"{result.model_purpose.value}_{result.model_type.value}"
            
            self.model_registry[model_key] = {
                'model_id': result.model_id,
                'model_path': result.model_path,
                'performance_metrics': {
                    'accuracy': result.accuracy,
                    'f1_score': result.f1_score,
                    'r2_score': result.r2_score
                },
                'created_at': result.start_time,
                'is_active': True,
                'feature_count': result.feature_count,
                'training_samples': result.training_samples
            }
            
            # Save registry to disk
            registry_path = os.path.join(self.models_path, 'model_registry.json')
            with open(registry_path, 'w') as f:
                json.dump(self.model_registry, f, indent=2, default=str)
            
            logger.info(f"📋 Model registered: {model_key}")
            
        except Exception as e:
            logger.error(f"❌ Failed to register model: {e}")
    
    def load_model_registry(self):
        """Load model registry from disk"""
        try:
            registry_path = os.path.join(self.models_path, 'model_registry.json')
            if os.path.exists(registry_path):
                with open(registry_path, 'r') as f:
                    self.model_registry = json.load(f)
                logger.info(f"📋 Model registry loaded: {len(self.model_registry)} models")
            else:
                logger.info("📋 No existing model registry found")
                
        except Exception as e:
            logger.error(f"❌ Failed to load model registry: {e}")
    
    def get_best_model(self, model_purpose: ModelPurpose) -> Optional[Dict[str, Any]]:
        """Get the best performing model for a specific purpose"""
        try:
            purpose_models = {
                k: v for k, v in self.model_registry.items() 
                if k.startswith(model_purpose.value) and v.get('is_active', True)
            }
            
            if not purpose_models:
                return None
            
            # Find best model based on performance metric
            if model_purpose == ModelPurpose.DATA_QUALITY:
                # For regression, use R² score
                best_model = max(purpose_models.items(), 
                               key=lambda x: x[1]['performance_metrics'].get('r2_score', 0))
            else:
                # For classification, use F1 score
                best_model = max(purpose_models.items(), 
                               key=lambda x: x[1]['performance_metrics'].get('f1_score', 0))
            
            return best_model[1]
            
        except Exception as e:
            logger.error(f"❌ Failed to get best model: {e}")
            return None
    
    # ============================
    # JOB STATUS AND MONITORING
    # ============================
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a training job"""
        # Check active jobs
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                'job_id': job.job_id,
                'status': job.status.value,
                'progress_percentage': job.progress_percentage,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'estimated_completion': self._estimate_completion_time(job),
                'config': {
                    'model_type': job.config.model_type.value,
                    'model_purpose': job.config.model_purpose.value
                }
            }
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
            return {
                'job_id': job.job_id,
                'status': job.status.value,
                'progress_percentage': 100.0 if job.status == TrainingStatus.COMPLETED else 0.0,
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.result.end_time.isoformat() if job.result and job.result.end_time else None,
                'result': self._format_training_result(job.result) if job.result else None
            }
        
        # Check queue
        for job in self.training_queue:
            if job.job_id == job_id:
                return {
                    'job_id': job.job_id,
                    'status': job.status.value,
                    'progress_percentage': 0.0,
                    'queue_position': self.training_queue.index(job) + 1,
                    'estimated_start_time': self._estimate_start_time(job)
                }
        
        return None
    
    def _estimate_completion_time(self, job: TrainingJob) -> Optional[str]:
        """Estimate job completion time"""
        if job.started_at and job.progress_percentage > 0:
            elapsed_minutes = (datetime.utcnow() - job.started_at).total_seconds() / 60
            estimated_total_minutes = elapsed_minutes / (job.progress_percentage / 100)
            remaining_minutes = estimated_total_minutes - elapsed_minutes
            
            completion_time = datetime.utcnow() + timedelta(minutes=remaining_minutes)
            return completion_time.isoformat()
        
        return None
    
    def _estimate_start_time(self, job: TrainingJob) -> str:
        """Estimate when a queued job will start"""
        queue_position = self.training_queue.index(job)
        
        # Calculate estimated wait time based on jobs ahead in queue
        wait_minutes = 0
        for i in range(queue_position):
            ahead_job = self.training_queue[i]
            wait_minutes += ahead_job.estimated_duration_minutes
        
        # Add current active job remaining time
        for active_job in self.active_jobs.values():
            if active_job.started_at:
                elapsed = (datetime.utcnow() - active_job.started_at).total_seconds() / 60
                remaining = max(0, active_job.estimated_duration_minutes - elapsed)
                wait_minutes += remaining / len(self.active_jobs)
        
        start_time = datetime.utcnow() + timedelta(minutes=wait_minutes)
        return start_time.isoformat()
    
    def _format_training_result(self, result: TrainingResult) -> Dict[str, Any]:
        """Format training result for API response"""
        return {
            'model_id': result.model_id,
            'model_type': result.model_type.value,
            'model_purpose': result.model_purpose.value,
            'training_status': result.training_status.value,
            'performance_metrics': {
                'accuracy': round(result.accuracy, 4),
                'precision': round(result.precision, 4),
                'recall': round(result.recall, 4),
                'f1_score': round(result.f1_score, 4),
                'auc_score': round(result.auc_score, 4),
                'r2_score': round(result.r2_score, 4),
                'mae': round(result.mae, 4),
                'rmse': round(result.rmse, 4)
            },
            'training_details': {
                'training_samples': result.training_samples,
                'validation_samples': result.validation_samples,
                'feature_count': result.feature_count,
                'training_duration_seconds': round(result.training_duration_seconds, 2),
                'cross_validation_mean': round(np.mean(result.cross_validation_scores), 4) if result.cross_validation_scores else 0,
                'cross_validation_std': round(np.std(result.cross_validation_scores), 4) if result.cross_validation_scores else 0
            },
            'model_artifacts': {
                'model_path': result.model_path,
                'model_size_mb': round(result.model_size_mb, 2)
            },
            'feature_importance': dict(sorted(
                result.feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]) if result.feature_importance else {},  # Top 10 features
            'error_message': result.error_message,
            'warnings': result.warnings
        }
    
    def list_all_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all training jobs (active, completed, and queued)"""
        all_jobs = []
        
        # Add completed jobs
        for job in list(self.completed_jobs.values())[-limit:]:
            all_jobs.append({
                'job_id': job.job_id,
                'status': job.status.value,
                'model_purpose': job.config.model_purpose.value,
                'model_type': job.config.model_type.value,
                'created_at': job.created_at.isoformat(),
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'completed_at': job.result.end_time.isoformat() if job.result and job.result.end_time else None,
                'user_id': job.user_id,
                'priority': job.priority
            })
        
        # Add active jobs
        for job in self.active_jobs.values():
            all_jobs.append({
                'job_id': job.job_id,
                'status': job.status.value,
                'model_purpose': job.config.model_purpose.value,
                'model_type': job.config.model_type.value,
                'created_at': job.created_at.isoformat(),
                'started_at': job.started_at.isoformat() if job.started_at else None,
                'progress_percentage': job.progress_percentage,
                'user_id': job.user_id,
                'priority': job.priority
            })
        
        # Add queued jobs
        for job in self.training_queue:
            all_jobs.append({
                'job_id': job.job_id,
                'status': job.status.value,
                'model_purpose': job.config.model_purpose.value,
                'model_type': job.config.model_type.value,
                'created_at': job.created_at.isoformat(),
                'queue_position': self.training_queue.index(job) + 1,
                'user_id': job.user_id,
                'priority': job.priority
            })
        
        # Sort by creation time (most recent first)
        all_jobs.sort(key=lambda x: x['created_at'], reverse=True)
        
        return all_jobs[:limit]
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job"""
        try:
            # Check if job is in queue
            for i, job in enumerate(self.training_queue):
                if job.job_id == job_id:
                    job.status = TrainingStatus.CANCELLED
                    self.training_queue.pop(i)
                    self.completed_jobs[job_id] = job
                    logger.info(f"🚫 Cancelled queued job: {job_id}")
                    return True
            
            # Note: Cannot cancel actively running jobs in this implementation
            # In a production system, you'd need more sophisticated job control
            
            logger.warning(f"⚠️ Cannot cancel job {job_id} - not found in queue or already running")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel job {job_id}: {e}")
            return False
    
    # ============================
    # BATCH AND ENSEMBLE TRAINING
    # ============================
    
    def train_ensemble_models(self, 
                             model_purpose: ModelPurpose,
                             training_data: List[Dict[str, Any]],
                             model_types: Optional[List[ModelType]] = None,
                             user_id: Optional[int] = None) -> str:
        """Train multiple models for ensemble"""
        
        if model_types is None:
            model_types = [ModelType.RANDOM_FOREST, ModelType.GRADIENT_BOOSTING, ModelType.LOGISTIC_REGRESSION]
        
        ensemble_job_ids = []
        
        for model_type in model_types:
            # Create configuration for each model type
            config = TrainingConfig(
                model_type=model_type,
                model_purpose=model_purpose,
                enable_hyperparameter_tuning=True
            )
            
            # Submit training job
            job_id = self.submit_training_job(
                model_purpose=model_purpose,
                training_data=training_data,
                config=config,
                user_id=user_id,
                priority=3  # Medium-high priority for ensemble
            )
            
            ensemble_job_ids.append(job_id)
        
        logger.info(f"🎭 Submitted ensemble training: {len(ensemble_job_ids)} models")
        
        # Return a combined job ID for tracking
        ensemble_id = f"ensemble_{model_purpose.value}_{int(time.time())}"
        
        # Store ensemble mapping
        self._store_ensemble_mapping(ensemble_id, ensemble_job_ids)
        
        return ensemble_id
    
    def _store_ensemble_mapping(self, ensemble_id: str, job_ids: List[str]):
        """Store mapping between ensemble ID and individual job IDs"""
        try:
            ensemble_path = os.path.join(self.models_path, 'ensemble_mappings.json')
            
            # Load existing mappings
            mappings = {}
            if os.path.exists(ensemble_path):
                with open(ensemble_path, 'r') as f:
                    mappings = json.load(f)
            
            # Add new mapping
            mappings[ensemble_id] = {
                'job_ids': job_ids,
                'created_at': datetime.utcnow().isoformat(),
                'status': 'training'
            }
            
            # Save mappings
            with open(ensemble_path, 'w') as f:
                json.dump(mappings, f, indent=2)
            
        except Exception as e:
            logger.error(f"❌ Failed to store ensemble mapping: {e}")
    
    def get_ensemble_status(self, ensemble_id: str) -> Optional[Dict[str, Any]]:
        """Get status of ensemble training"""
        try:
            ensemble_path = os.path.join(self.models_path, 'ensemble_mappings.json')
            
            if not os.path.exists(ensemble_path):
                return None
            
            with open(ensemble_path, 'r') as f:
                mappings = json.load(f)
            
            if ensemble_id not in mappings:
                return None
            
            ensemble_info = mappings[ensemble_id]
            job_ids = ensemble_info['job_ids']
            
            # Get status of all jobs
            job_statuses = []
            completed_count = 0
            failed_count = 0
            
            for job_id in job_ids:
                job_status = self.get_job_status(job_id)
                if job_status:
                    job_statuses.append(job_status)
                    if job_status['status'] == 'completed':
                        completed_count += 1
                    elif job_status['status'] == 'failed':
                        failed_count += 1
            
            # Calculate overall status
            if completed_count == len(job_ids):
                overall_status = 'completed'
            elif failed_count == len(job_ids):
                overall_status = 'failed'
            elif completed_count > 0 or failed_count > 0:
                overall_status = 'partially_completed'
            else:
                overall_status = 'training'
            
            return {
                'ensemble_id': ensemble_id,
                'overall_status': overall_status,
                'total_models': len(job_ids),
                'completed_models': completed_count,
                'failed_models': failed_count,
                'individual_jobs': job_statuses,
                'created_at': ensemble_info['created_at']
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get ensemble status: {e}")
            return None
    
    # ============================
    # SYNTHETIC DATA TRAINING
    # ============================
    
    def train_model_with_synthetic_data(self, 
                                       model_purpose: ModelPurpose,
                                       synthetic_samples: int = 1000) -> str:
        """Train model using synthetic data"""
        try:
            # Import synthetic data generator
            from ai.training.data_generator import SyntheticDataGenerator
            
            # Generate synthetic training data
            logger.info(f"🎲 Generating {synthetic_samples} synthetic samples for {model_purpose.value}")
            
            data_generator = SyntheticDataGenerator()
            
            # Map model purpose to data type
            data_type_mapping = {
                ModelPurpose.LEAD_PRIORITY: "lead_priority",
                ModelPurpose.DATA_QUALITY: "data_quality", 
                ModelPurpose.LEAD_SIMILARITY: "lead_similarity"
            }
            
            data_type = data_type_mapping.get(model_purpose, "lead_priority")
            
            # Generate synthetic data
            temp_file = f"ai/training_data/temp_{data_type}_{int(time.time())}.json"
            success = data_generator.generate_training_data(data_type, synthetic_samples, temp_file)
            
            if not success:
                raise Exception("Failed to generate synthetic training data")
            
            # Load generated data
            with open(temp_file, 'r') as f:
                synthetic_data = json.load(f)
            
            # Clean up temporary file
            os.remove(temp_file)
            
            # Convert synthetic data to training format
            training_data = []
            for sample in synthetic_data:
                features = sample['features']
                label = sample['label']
                
                # Create lead-like data structure
                lead_data = {
                    'first_name': 'John',
                    'last_name': 'Doe',
                    'email': 'john.doe@example.com',
                    'title': 'Manager',
                    'company': {'name': 'Company', 'industry': 'Technology'},
                    **features  # Add extracted features
                }
                
                # Add appropriate target
                if model_purpose == ModelPurpose.LEAD_PRIORITY:
                    lead_data['priority'] = label
                elif model_purpose == ModelPurpose.DATA_QUALITY:
                    lead_data['quality_score'] = float(label) if isinstance(label, str) else label
                elif model_purpose == ModelPurpose.LEAD_SIMILARITY:
                    lead_data['similarity_group'] = label
                
                training_data.append(lead_data)
            
            # Submit training job
            job_id = self.submit_training_job(
                model_purpose=model_purpose,
                training_data=training_data,
                priority=2  # Medium priority for synthetic training
            )
            
            logger.info(f"🎯 Submitted synthetic data training job: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"❌ Synthetic data training failed: {e}")
            raise
    
    # ============================
    # MODEL EVALUATION AND TESTING
    # ============================
    
    def evaluate_model(self, model_path: str, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a trained model on test data"""
        try:
            # Load model
            with open(model_path, 'rb') as f:
                model_pipeline = pickle.load(f)
            
            # Load model metadata
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            model_purpose = ModelPurpose(metadata['model_purpose'])
            
            # Prepare test data
            X_test, y_test = self._prepare_training_data(test_data, 
                                                       TrainingConfig(ModelType.RANDOM_FOREST, model_purpose))
            
            # Make predictions
            y_pred = model_pipeline.predict(X_test)
            
            # Calculate evaluation metrics
            if self._is_classification_task(model_purpose):
                evaluation_results = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                    'classification_report': classification_report(y_test, y_pred)
                }
                
                # Add AUC for binary classification
                if len(np.unique(y_test)) == 2 and hasattr(model_pipeline, 'predict_proba'):
                    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]
                    evaluation_results['auc_score'] = roc_auc_score(y_test, y_pred_proba)
                    
            else:  # Regression
                evaluation_results = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2_score': r2_score(y_test, y_pred)
                }
            
            evaluation_results.update({
                'test_samples': len(X_test),
                'model_metadata': metadata,
                'evaluation_date': datetime.utcnow().isoformat()
            })
            
            logger.info(f"📊 Model evaluation completed: {model_path}")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            return {'error': str(e)}
    
    def benchmark_models(self, model_purpose: ModelPurpose, 
                        test_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Benchmark all models for a specific purpose"""
        benchmark_results = {}
        
        # Find all models for the purpose
        purpose_models = {
            k: v for k, v in self.model_registry.items() 
            if k.startswith(model_purpose.value)
        }
        
        for model_key, model_info in purpose_models.items():
            try:
                model_path = model_info['model_path']
                if os.path.exists(model_path):
                    evaluation = self.evaluate_model(model_path, test_data)
                    benchmark_results[model_key] = evaluation
                    
            except Exception as e:
                logger.error(f"❌ Benchmark failed for {model_key}: {e}")
                benchmark_results[model_key] = {'error': str(e)}
        
        return benchmark_results
    
    # ============================
    # CLEANUP AND MAINTENANCE
    # ============================
    
    def cleanup_old_models(self, keep_latest: int = 5, days_old: int = 30):
        """Clean up old model files and jobs"""
        try:
            logger.info("🧹 Starting model cleanup...")
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            # Clean up old completed jobs
            old_jobs = {
                k: v for k, v in self.completed_jobs.items()
                if v.created_at < cutoff_date
            }
            
            for job_id in old_jobs.keys():
                del self.completed_jobs[job_id]
            
            # Clean up old model files
            model_files = []
            for filename in os.listdir(self.models_path):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.models_path, filename)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    model_files.append((file_path, file_time, filename))
            
            # Sort by modification time (newest first)
            model_files.sort(key=lambda x: x[1], reverse=True)
            
            # Group by model purpose and keep only latest N
            purpose_groups = {}
            for file_path, file_time, filename in model_files:
                purpose = filename.split('_')[1] if '_' in filename else 'unknown'
                if purpose not in purpose_groups:
                    purpose_groups[purpose] = []
                purpose_groups[purpose].append((file_path, file_time, filename))
            
            files_removed = 0
            for purpose, files in purpose_groups.items():
                # Remove old files beyond keep_latest limit
                for file_path, file_time, filename in files[keep_latest:]:
                    if file_time < cutoff_date:
                        try:
                            os.remove(file_path)
                            # Also remove metadata file
                            metadata_path = file_path.replace('.pkl', '_metadata.json')
                            if os.path.exists(metadata_path):
                                os.remove(metadata_path)
                            files_removed += 1
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to remove {file_path}: {e}")
            
            logger.info(f"🧹 Cleanup completed: removed {len(old_jobs)} old jobs and {files_removed} old model files")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'trainer_info': {
                'models_path': self.models_path,
                'max_concurrent_jobs': self.max_concurrent_jobs,
                'registered_models': len(self.model_registry)
            },
            'job_queue': {
                'queued_jobs': len(self.training_queue),
                'active_jobs': len(self.active_jobs),
                'completed_jobs': len(self.completed_jobs)
            },
            'model_registry': {
                'total_models': len(self.model_registry),
                'model_purposes': list(set(
                    info.get('model_id', '').split('_')[0] 
                    for info in self.model_registry.values()
                )),
                'average_accuracy': np.mean([
                    info['performance_metrics'].get('accuracy', 0)
                    for info in self.model_registry.values()
                ]) if self.model_registry else 0
            },
            'system_health': {
                'sklearn_available': SKLEARN_AVAILABLE,
                'xgboost_available': XGBOOST_AVAILABLE,
                'models_directory_exists': os.path.exists(self.models_path),
                'disk_space_mb': self._get_disk_usage()
            }
        }
    
    def _get_disk_usage(self) -> float:
        """Get disk usage of models directory in MB"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.models_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(file_path)
            return total_size / (1024 * 1024)  # Convert to MB
        except:
            return 0.0


# ============================
# CONVENIENCE FUNCTIONS
# ============================

def quick_train_model(model_purpose: str, training_data: List[Dict[str, Any]], 
                     model_type: str = "random_forest") -> str:
    """Quick training function for simple use cases"""
    trainer = ModelTrainer()
    
    purpose_enum = ModelPurpose(model_purpose)
    type_enum = ModelType(model_type)
    
    config = TrainingConfig(
        model_type=type_enum,
        model_purpose=purpose_enum,
        enable_hyperparameter_tuning=False  # Faster training
    )
    
    return trainer.submit_training_job(purpose_enum, training_data, config)


def train_all_model_types():
    """Train initial models for all purposes using synthetic data"""
    trainer = ModelTrainer()
    job_ids = []
    
    for purpose in ModelPurpose:
        try:
            job_id = trainer.train_model_with_synthetic_data(purpose, synthetic_samples=500)
            job_ids.append(job_id)
            logger.info(f"✅ Submitted training for {purpose.value}: {job_id}")
        except Exception as e:
            logger.error(f"❌ Failed to train {purpose.value}: {e}")
    
    return job_ids


# ============================
# TESTING AND VALIDATION
# ============================

def test_model_trainer():
    """Test the model trainer functionality"""
    print("🧪 Testing Model Trainer")
    print("=" * 50)
    
    # Sample training data
    sample_training_data = [
        {
            'id': 1,
            'first_name': 'John',
            'last_name': 'Smith',
            'email': 'john.smith@techcorp.com',
            'phone': '+1-555-123-4567',
            'title': 'Senior Software Engineer',
            'linkedin_url': 'https://www.linkedin.com/in/johnsmith',
            'company': {
                'name': 'TechCorp Inc',
                'domain': 'techcorp.com',
                'industry': 'Technology',
                'size': '201-500',
                'location': 'San Francisco, CA'
            },
            'source': 'linkedin',
            'created_at': '2024-01-15T10:30:00Z',
            'priority': 'high',
            'quality_score': 85.0,
            'converted': True
        },
        {
            'id': 2,
            'first_name': 'Jane',
            'last_name': 'Doe',
            'email': 'jane.doe@startup.com',
            'phone': '+1-555-987-6543',
            'title': 'Marketing Manager',
            'company': {
                'name': 'Startup Inc',
                'domain': 'startup.com',
                'industry': 'Marketing',
                'size': '11-50',
                'location': 'Austin, TX'
            },
            'source': 'referral',
            'created_at': '2024-02-10T14:20:00Z',
            'priority': 'medium',
            'quality_score': 72.0,
            'converted': False
        },
        {
            'id': 3,
            'first_name': 'Bob',
            'last_name': 'Johnson',
            'email': 'bob.johnson@enterprise.com',
            'phone': '+1-555-456-7890',
            'title': 'VP of Sales',
            'company': {
                'name': 'Enterprise Corp',
                'domain': 'enterprise.com',
                'industry': 'Finance',
                'size': '1001-5000',
                'location': 'New York, NY'
            },
            'source': 'website',
            'created_at': '2024-01-05T09:15:00Z',
            'priority': 'high',
            'quality_score': 91.0,
            'converted': True
        }
    ]
    
    # Multiply sample data for testing
    training_data = sample_training_data * 50  # 150 samples
    
    # Add some variation to the data
    for i, lead in enumerate(training_data):
        lead['id'] = i + 1
        # Add some randomness to simulate real data
        if i % 3 == 0:
            lead['priority'] = 'low'
            lead['quality_score'] = np.random.uniform(40, 60)
        elif i % 3 == 1:
            lead['priority'] = 'medium'
            lead['quality_score'] = np.random.uniform(60, 80)
        else:
            lead['priority'] = 'high'
            lead['quality_score'] = np.random.uniform(80, 95)
    
    # Initialize trainer
    print("\n🎓 Initializing Model Trainer:")
    print("-" * 30)
    
    trainer = ModelTrainer(max_concurrent_jobs=1)  # Single job for testing
    
    print(f"✅ Trainer initialized")
    print(f"Models path: {trainer.models_path}")
    print(f"Max concurrent jobs: {trainer.max_concurrent_jobs}")
    print(f"Default configs: {len(trainer.default_configs)}")
    
    # Test job submission
    print(f"\n📝 Testing Job Submission:")
    print("-" * 30)
    
    if SKLEARN_AVAILABLE:
        # Submit a priority classification job
        job_id = trainer.submit_training_job(
            model_purpose=ModelPurpose.LEAD_PRIORITY,
            training_data=training_data,
            priority=3
        )
        
        print(f"✅ Submitted training job: {job_id}")
        
        # Check job status
        status = trainer.get_job_status(job_id)
        if status:
            print(f"Job status: {status['status']}")
            print(f"Progress: {status.get('progress_percentage', 0):.1f}%")
        
        # Wait for completion (in real usage, you'd poll this)
        import time
        max_wait = 120  # 2 minutes max
        wait_time = 0
        
        while wait_time < max_wait:
            status = trainer.get_job_status(job_id)
            if status and status['status'] in ['completed', 'failed']:
                break
            
            time.sleep(5)
            wait_time += 5
            
            if wait_time % 20 == 0:
                print(f"  Still training... ({wait_time}s elapsed)")
        
        # Check final status
        final_status = trainer.get_job_status(job_id)
        if final_status:
            print(f"Final status: {final_status['status']}")
            
            if final_status['status'] == 'completed' and 'result' in final_status:
                result = final_status['result']
                print(f"✅ Training completed successfully!")
                print(f"Accuracy: {result['performance_metrics']['accuracy']:.3f}")
                print(f"F1 Score: {result['performance_metrics']['f1_score']:.3f}")
                print(f"Training samples: {result['training_details']['training_samples']}")
                print(f"Features: {result['training_details']['feature_count']}")
                
                # Test model evaluation
                print(f"\n📊 Testing Model Evaluation:")
                print("-" * 30)
                
                model_path = result['model_artifacts']['model_path']
                if os.path.exists(model_path):
                    # Use a subset of training data as test data
                    test_data = training_data[:20]
                    evaluation = trainer.evaluate_model(model_path, test_data)
                    
                    if 'error' not in evaluation:
                        print(f"✅ Model evaluation completed")
                        print(f"Test accuracy: {evaluation.get('accuracy', 0):.3f}")
                        print(f"Test samples: {evaluation.get('test_samples', 0)}")
                    else:
                        print(f"❌ Evaluation failed: {evaluation['error']}")
        
    else:
        print("⚠️ Scikit-learn not available - skipping ML training tests")
    
    # Test synthetic data training
    print(f"\n🎲 Testing Synthetic Data Training:")
    print("-" * 30)
    
    try:
        synthetic_job_id = trainer.train_model_with_synthetic_data(
            ModelPurpose.LEAD_PRIORITY,
            synthetic_samples=100
        )
        print(f"✅ Submitted synthetic data training: {synthetic_job_id}")
        
    except Exception as e:
        print(f"❌ Synthetic training failed: {e}")
    
    # Test ensemble training
    print(f"\n🎭 Testing Ensemble Training:")
    print("-" * 30)
    
    if SKLEARN_AVAILABLE:
        try:
            ensemble_id = trainer.train_ensemble_models(
                ModelPurpose.LEAD_PRIORITY,
                training_data[:50],  # Smaller dataset for ensemble
                model_types=[ModelType.RANDOM_FOREST, ModelType.LOGISTIC_REGRESSION]
            )
            print(f"✅ Submitted ensemble training: {ensemble_id}")
            
            # Check ensemble status
            ensemble_status = trainer.get_ensemble_status(ensemble_id)
            if ensemble_status:
                print(f"Ensemble status: {ensemble_status['overall_status']}")
                print(f"Total models: {ensemble_status['total_models']}")
                
        except Exception as e:
            print(f"❌ Ensemble training failed: {e}")
    
    # Test model registry
    print(f"\n📋 Testing Model Registry:")
    print("-" * 30)
    
    trainer.load_model_registry()
    
    if trainer.model_registry:
        print(f"✅ Model registry loaded: {len(trainer.model_registry)} models")
        
        for model_key, model_info in list(trainer.model_registry.items())[:3]:
            print(f"  Model: {model_key}")
            print(f"    Accuracy: {model_info['performance_metrics'].get('accuracy', 0):.3f}")
            print(f"    Created: {model_info.get('created_at', 'Unknown')}")
    else:
        print("📋 No models in registry yet")
    
    # Test job listing
    print(f"\n📜 Testing Job Listing:")
    print("-" * 30)
    
    all_jobs = trainer.list_all_jobs(limit=10)
    print(f"✅ Found {len(all_jobs)} jobs")
    
    for job in all_jobs[:3]:
        print(f"  Job: {job['job_id']}")
        print(f"    Status: {job['status']}")
        print(f"    Purpose: {job['model_purpose']}")
        print(f"    Type: {job['model_type']}")
    
    # Test system status
    print(f"\n🔧 Testing System Status:")
    print("-" * 30)
    
    system_status = trainer.get_system_status()
    print(f"✅ System status retrieved")
    print(f"Registered models: {system_status['model_registry']['total_models']}")
    print(f"Active jobs: {system_status['job_queue']['active_jobs']}")
    print(f"Queued jobs: {system_status['job_queue']['queued_jobs']}")
    print(f"Disk usage: {system_status['system_health']['disk_space_mb']:.2f} MB")
    print(f"Sklearn available: {system_status['system_health']['sklearn_available']}")
    
    # Test quick training function
    print(f"\n⚡ Testing Quick Training:")
    print("-" * 30)
    
    try:
        quick_job_id = quick_train_model(
            "lead_priority",
            training_data[:30],
            "random_forest"
        )
        print(f"✅ Quick training submitted: {quick_job_id}")
        
    except Exception as e:
        print(f"❌ Quick training failed: {e}")
    
    # Test cleanup
    print(f"\n🧹 Testing Cleanup:")
    print("-" * 30)
    
    try:
        trainer.cleanup_old_models(keep_latest=2, days_old=0)  # Aggressive cleanup for testing
        print(f"✅ Cleanup completed")
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
    
    print(f"\n✅ Model Trainer testing completed!")
    print(f"🎯 Ready for production use with comprehensive training capabilities")
    print(f"🚀 Features: Multi-model training, Ensemble methods, Job management, Synthetic data")


# ============================
# AUTOMATED TRAINING SCHEDULER
# ============================

class TrainingScheduler:
    """
    Automated training scheduler for periodic model retraining
    """
    
    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer
        self.scheduled_jobs = {}
        self.is_running = False
        
    def schedule_periodic_training(self, 
                                 model_purpose: ModelPurpose,
                                 training_data_source: callable,
                                 interval_hours: int = 24,
                                 min_data_threshold: int = 100):
        """Schedule periodic model retraining"""
        
        job_key = f"{model_purpose.value}_periodic"
        
        self.scheduled_jobs[job_key] = {
            'model_purpose': model_purpose,
            'data_source': training_data_source,
            'interval_hours': interval_hours,
            'min_data_threshold': min_data_threshold,
            'last_run': None,
            'next_run': datetime.utcnow() + timedelta(hours=interval_hours)
        }
        
        logger.info(f"📅 Scheduled periodic training for {model_purpose.value} every {interval_hours} hours")
    
    def start_scheduler(self):
        """Start the training scheduler"""
        self.is_running = True
        
        def scheduler_loop():
            while self.is_running:
                try:
                    self._check_scheduled_jobs()
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    logger.error(f"❌ Scheduler error: {e}")
                    time.sleep(60)  # Wait 1 minute on error
        
        # Run scheduler in background thread
        scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        scheduler_thread.start()
        
        logger.info("📅 Training scheduler started")
    
    def stop_scheduler(self):
        """Stop the training scheduler"""
        self.is_running = False
        logger.info("📅 Training scheduler stopped")
    
    def _check_scheduled_jobs(self):
        """Check and execute scheduled training jobs"""
        current_time = datetime.utcnow()
        
        for job_key, job_info in self.scheduled_jobs.items():
            if current_time >= job_info['next_run']:
                try:
                    # Get training data
                    training_data = job_info['data_source']()
                    
                    # Check if we have enough data
                    if len(training_data) >= job_info['min_data_threshold']:
                        # Submit training job
                        training_job_id = self.trainer.submit_training_job(
                            model_purpose=job_info['model_purpose'],
                            training_data=training_data,
                            priority=2  # Medium priority for scheduled jobs
                        )
                        
                        logger.info(f"📅 Scheduled training submitted: {training_job_id}")
                        
                        # Update schedule
                        job_info['last_run'] = current_time
                        job_info['next_run'] = current_time + timedelta(hours=job_info['interval_hours'])
                        
                    else:
                        logger.warning(f"⚠️ Insufficient data for {job_key}: {len(training_data)} < {job_info['min_data_threshold']}")
                        # Reschedule for later
                        job_info['next_run'] = current_time + timedelta(hours=1)
                        
                except Exception as e:
                    logger.error(f"❌ Scheduled training failed for {job_key}: {e}")
                    # Reschedule for later
                    job_info['next_run'] = current_time + timedelta(hours=1)


# ============================
# MODEL PERFORMANCE MONITORING
# ============================

class ModelPerformanceMonitor:
    """
    Monitor model performance and trigger retraining when needed
    """
    
    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer
        self.performance_history = {}
        self.performance_thresholds = {
            ModelPurpose.LEAD_PRIORITY: {'accuracy': 0.8, 'f1_score': 0.75},
            ModelPurpose.DATA_QUALITY: {'r2_score': 0.7, 'mae': 15.0},
            ModelPurpose.CONVERSION_PREDICTION: {'accuracy': 0.75, 'auc_score': 0.8}
        }
    
    def log_prediction_performance(self, 
                                 model_purpose: ModelPurpose,
                                 actual_values: List[Any],
                                 predicted_values: List[Any],
                                 prediction_confidence: List[float]):
        """Log real-world prediction performance"""
        
        # Calculate performance metrics
        if self.trainer._is_classification_task(model_purpose):
            accuracy = accuracy_score(actual_values, predicted_values)
            f1 = f1_score(actual_values, predicted_values, average='weighted', zero_division=0)
            
            performance_metrics = {
                'accuracy': accuracy,
                'f1_score': f1,
                'sample_count': len(actual_values),
                'avg_confidence': np.mean(prediction_confidence),
                'timestamp': datetime.utcnow()
            }
        else:
            mae = mean_absolute_error(actual_values, predicted_values)
            r2 = r2_score(actual_values, predicted_values)
            
            performance_metrics = {
                'mae': mae,
                'r2_score': r2,
                'sample_count': len(actual_values),
                'avg_confidence': np.mean(prediction_confidence),
                'timestamp': datetime.utcnow()
            }
        
        # Store performance history
        purpose_key = model_purpose.value
        if purpose_key not in self.performance_history:
            self.performance_history[purpose_key] = []
        
        self.performance_history[purpose_key].append(performance_metrics)
        
        # Keep only recent history (last 100 entries)
        self.performance_history[purpose_key] = self.performance_history[purpose_key][-100:]
        
        # Check if retraining is needed
        self._check_retraining_trigger(model_purpose, performance_metrics)
    
    def _check_retraining_trigger(self, model_purpose: ModelPurpose, latest_metrics: Dict[str, Any]):
        """Check if model performance has degraded and retraining is needed"""
        
        purpose_key = model_purpose.value
        thresholds = self.performance_thresholds.get(model_purpose, {})
        
        if not thresholds:
            return
        
        # Check current performance against thresholds
        needs_retraining = False
        reasons = []
        
        for metric, threshold in thresholds.items():
            current_value = latest_metrics.get(metric, 0)
            
            if metric in ['accuracy', 'f1_score', 'auc_score', 'r2_score']:
                # Higher is better
                if current_value < threshold:
                    needs_retraining = True
                    reasons.append(f"{metric} ({current_value:.3f}) below threshold ({threshold})")
            else:
                # Lower is better (MAE, RMSE)
                if current_value > threshold:
                    needs_retraining = True
                    reasons.append(f"{metric} ({current_value:.3f}) above threshold ({threshold})")
        
        # Check trend (performance degrading over time)
        if len(self.performance_history[purpose_key]) >= 5:
            recent_performance = self.performance_history[purpose_key][-5:]
            
            # Check if there's a declining trend
            for metric in thresholds.keys():
                if metric in ['accuracy', 'f1_score', 'auc_score', 'r2_score']:
                    # For metrics where higher is better
                    values = [p.get(metric, 0) for p in recent_performance]
                    if len(values) >= 3 and values[-1] < values[0] * 0.9:  # 10% decline
                        needs_retraining = True
                        reasons.append(f"{metric} showing declining trend")
        
        if needs_retraining:
            logger.warning(f"🚨 Model retraining triggered for {model_purpose.value}")
            logger.warning(f"Reasons: {', '.join(reasons)}")
            
            # You could automatically trigger retraining here
            # self._trigger_automatic_retraining(model_purpose, reasons)
    
    def get_performance_summary(self, model_purpose: ModelPurpose) -> Dict[str, Any]:
        """Get performance summary for a model"""
        
        purpose_key = model_purpose.value
        history = self.performance_history.get(purpose_key, [])
        
        if not history:
            return {'error': 'No performance history available'}
        
        recent_history = history[-10:]  # Last 10 entries
        
        summary = {
            'total_evaluations': len(history),
            'latest_evaluation': history[-1],
            'recent_average': {},
            'trend_analysis': {},
            'retraining_recommended': False
        }
        
        # Calculate recent averages
        for metric in ['accuracy', 'f1_score', 'r2_score', 'mae', 'auc_score']:
            values = [h.get(metric) for h in recent_history if h.get(metric) is not None]
            if values:
                summary['recent_average'][metric] = np.mean(values)
        
        # Trend analysis
        if len(history) >= 5:
            for metric in summary['recent_average'].keys():
                old_values = [h.get(metric) for h in history[-10:-5] if h.get(metric) is not None]
                new_values = [h.get(metric) for h in history[-5:] if h.get(metric) is not None]
                
                if old_values and new_values:
                    old_avg = np.mean(old_values)
                    new_avg = np.mean(new_values)
                    
                    if metric in ['accuracy', 'f1_score', 'auc_score', 'r2_score']:
                        trend = 'improving' if new_avg > old_avg else 'declining'
                        change_pct = ((new_avg - old_avg) / old_avg) * 100
                    else:
                        trend = 'improving' if new_avg < old_avg else 'declining'
                        change_pct = ((old_avg - new_avg) / old_avg) * 100
                    
                    summary['trend_analysis'][metric] = {
                        'trend': trend,
                        'change_percentage': change_pct
                    }
        
        return summary


# ============================
# MAIN EXECUTION
# ============================

if __name__ == "__main__":
    test_model_trainer()