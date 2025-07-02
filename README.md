# AI-Enhanced Lead Generation Tool

## Project Overview

An intelligent B2B lead generation and management platform that combines traditional lead scoring with advanced machine learning to provide smart lead prioritization, quality assessment, and conversion prediction.

## Features

### Core Features
- **Lead Management**: Create, update, and manage B2B leads with comprehensive contact information
- **Company Intelligence**: Store and enrich company data with industry, size, and location details
- **Campaign Management**: Organize leads into targeted campaigns with tracking capabilities
- **Data Export**: Export leads to CSV, CRM systems, and email marketing platforms

### AI-Enhanced Features
- **Smart Lead Scoring**: Hybrid scoring system combining rule-based logic with ML predictions
- **Priority Classification**: Random Forest classifier for lead priority prediction (High/Medium/Low)
- **Quality Assessment**: Automated data quality scoring and improvement suggestions
- **Conversion Prediction**: ML models to predict lead conversion likelihood
- **Duplicate Detection**: Intelligent deduplication using similarity matching
- **Batch Processing**: Bulk lead analysis and enrichment capabilities

## Tech Stack

### Backend
- **FastAPI**: Modern Python web framework for API development
- **SQLAlchemy**: Database ORM with SQLite backend
- **Pydantic**: Data validation and serialization
- **Scikit-learn**: Machine learning models and data preprocessing
- **Pandas/NumPy**: Data manipulation and analysis

### Frontend
- **HTML5/CSS3**: Responsive web interface
- **JavaScript**: Dynamic UI interactions and API communication
- **Chart.js**: Data visualization and analytics dashboard

### Machine Learning
- **Random Forest Classifier**: Primary model for lead prioritization
- **Feature Engineering**: 15+ engineered features from lead and company data
- **Cross-Validation**: 5-fold CV for model evaluation
- **Hyperparameter Tuning**: Grid search optimization

## Project Structure

```
lead-generation-tool/
├── main.py                      # FastAPI application entry point
├── database.py                  # Database configuration and models
├── models.py                    # SQLAlchemy database models
├── schemas.py                   # Pydantic response schemas
├── config.py                    # Application configuration
├── requirements.txt             # Python dependencies
├── init_sqlite_db.py           # Database initialization script
├── index.html                   # Frontend interface
├── synthetic_leads.csv          # Dataset (1000 leads)
│
├── routers/                     # API route handlers
│   ├── auth.py                 # Authentication endpoints
│   ├── leads.py                # Lead management with AI features
│   ├── companies.py            # Company management
│   ├── campaigns.py            # Campaign management
│   ├── analytics.py            # Analytics with AI insights
│   └── ai_insights.py          # AI-specific endpoints
│
├── utils/                       # Utility modules
│   ├── lead_scraper.py         # Web scraping utilities
│   ├── email_finder.py         # Email discovery tools
│   ├── lead_scorer.py          # Enhanced hybrid scoring system
│   ├── data_enrichment.py      # Data enrichment services
│   ├── export_service.py       # Export functionality
│   ├── ai_lead_analyzer.py     # AI intelligence engine
│   ├── data_quality_engine.py  # Quality assessment and deduplication
│   └── groq_client.py          # LLM integration
│
└── ai/                          # AI/ML components
    ├── models/
    │   ├── lead_classifier.py  # ML model implementations
    │   └── quality_scorer.py   # Data quality models
    ├── training/
    │   ├── data_generator.py   # Synthetic data generation
    │   └── model_trainer.py    # Model training pipeline
    └── saved_models/           # Trained model artifacts
```

## Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd lead-generation-tool
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the database and AI system**
   ```bash
   python init_sqlite_db.py
   ```
   This will:
   - Create SQLite database with all tables
   - Initialize AI model metadata
   - Create AI directory structure
   - Generate sample training data
   - Set up default ML models

4. **Start the application**
   ```bash
   uvicorn main:app --reload
   ```
   Or alternatively:
   ```bash
   python main.py
   ```

5. **Access the application**
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Alternative Docs: http://localhost:8000/redoc

### First Time Setup

**Create your account:**
1. Open http://localhost:8000 in your browser
2. Click "Create Account" 
3. Fill in email, username, and password
4. Login with your credentials

**Quick API Test (after login):**
```bash
# Test API health
curl http://localhost:8000/health

# Get auth token first
TOKEN=$(curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=your_username&password=your_password" | jq -r .access_token)

# Test lead creation with AI scoring (requires authentication)
curl -X POST "http://localhost:8000/leads/" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "first_name": "John",
    "last_name": "Doe",
    "email": "john.doe@example.com",
    "title": "CEO",
    "company": {
      "name": "Tech Corp",
      "industry": "Technology",
      "size": "101-500"
    }
  }'

# Check AI system status
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/ai/status
```

## API Endpoints

### Core Lead Management (requires authentication)
- `GET /leads/` - List leads with filtering
- `POST /leads/` - Create new lead with AI scoring
- `GET /leads/{id}` - Get lead details
- `PUT /leads/{id}` - Update lead
- `DELETE /leads/{id}` - Delete lead

### Authentication
- `POST /auth/register` - Create new user account
- `POST /auth/token` - Login and get access token
- `GET /auth/me` - Get current user info

### AI-Enhanced Features (requires authentication)
- `POST /ai/analyze-lead` - AI lead analysis and insights
- `POST /ai/batch-analyze` - Bulk lead analysis
- `GET /ai/dashboard` - AI insights dashboard
- `POST /ai/predict-priority` - Lead priority prediction
- `POST /ai/quality-check` - Data quality assessment
- `GET /ai/similar-leads/{id}` - Find similar leads
- `GET /ai/status` - AI system health check

### Analytics (requires authentication)
- `GET /analytics/dashboard` - Main analytics dashboard
- `GET /analytics/lead-performance` - Lead performance metrics
- `GET /analytics/quality-trends` - Data quality trends

## Machine Learning Model

### Model Details
- **Algorithm**: Random Forest Classifier
- **Training Data**: 1000 synthetic B2B leads
- **Features**: 15 engineered features including:
  - Contact completeness score
  - Title seniority level
  - Company industry relevance
  - Email quality indicators
  - Data freshness metrics

### Model Performance
- **Accuracy**: ~85% on test set
- **Cross-Validation**: 5-fold CV with consistent performance
- **Feature Importance**: Title and industry are top predictors
- **Inference Time**: <50ms per prediction

### Feature Engineering
```python
# Key features used in the model
features = [
    'email_quality_score',      # Email format and domain quality
    'contact_completeness',     # Completeness of contact info
    'title_seniority_level',    # Executive/Manager/Individual
    'industry_relevance',       # Target industry matching
    'company_size_score',       # Company size normalization
    'data_freshness',          # Recency of data updates
    'source_quality',          # Lead source reliability
    'engagement_indicators',    # Activity and interaction signals
    # ... additional features
]
```

## Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=sqlite:///./leadgen.db

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# External APIs (optional)
HUNTER_API_KEY=your-hunter-key
CLEARBIT_API_KEY=your-clearbit-key
GROQ_API_KEY=your-groq-key
```

## Development

### Running Tests
```bash
# Test database connection
python test_sqlite.py

# Test lead generation (requires running server and authentication)
python test_lead_generation.py

# Test AI models directly
python -m ai.models.lead_classifier

# Test model training
python -m ai.training.model_trainer
```

### Adding New Features
1. Create new route in appropriate router
2. Add database models if needed
3. Update schemas for API responses
4. Implement business logic in utils/
5. Add AI enhancement if applicable

## Deployment

### Docker Deployment
```bash
# Build image
docker build -t lead-generation-tool .

# Run container
docker run -p 8000:8000 lead-generation-tool
```

### Production Considerations
- **Database**: Use PostgreSQL instead of SQLite for production
- **Authentication**: Configure proper JWT secret keys and token expiration
- **Environment Variables**: Set up proper environment configuration
- **Rate Limiting**: Implement API rate limiting and request throttling
- **Monitoring**: Set up logging, metrics, and health checks
- **Model Retraining**: Implement periodic model retraining with new data
- **Caching**: Configure Redis for caching and background tasks
- **Security**: Enable HTTPS, CORS restrictions, and input validation

## Performance Optimization

### Model Optimization
- **Batch Processing**: Process multiple leads simultaneously
- **Caching**: Cache model predictions for similar leads
- **Model Versioning**: Track and compare model performance
- **Feature Selection**: Continuously evaluate feature importance

### Database Optimization
- **Indexing**: Optimized indexes for lead queries
- **Pagination**: Efficient pagination for large datasets
- **Connection Pooling**: Database connection management

## Monitoring and Maintenance

### Model Monitoring
- Track prediction accuracy over time
- Monitor feature drift and data quality
- Set up alerts for model performance degradation
- Regular model retraining with new data

### System Health
- API response times and error rates
- Database performance metrics
- Memory and CPU usage monitoring
- Lead processing throughput

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or support:
- Check the API documentation at `/docs`
- Review the codebase documentation
- Open an issue for bugs or feature requests

---

**Built with ❤️ for efficient B2B lead generation and AI-powered sales intelligence.**
