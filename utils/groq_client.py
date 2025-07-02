# utils/groq_client.py - Groq LLM Integration for AI-Enhanced Lead Analysis

# ============================
# IMPORTS
# ============================
import os
import json
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import hashlib

try:
    from groq import Groq, AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("Groq library not available. Install with: pip install groq")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# ENUMS AND DATA CLASSES
# ============================

class GroqModel(Enum):
    """Available Groq models"""
    LLAMA3_8B = "llama3-8b-8192"
    LLAMA3_70B = "llama3-70b-8192"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    GEMMA_7B = "gemma-7b-it"

class InsightType(Enum):
    """Types of insights to generate"""
    LEAD_ANALYSIS = "lead_analysis"
    PERSONALIZATION = "personalization"
    TIMING_OPTIMIZATION = "timing_optimization"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    MARKET_RESEARCH = "market_research"
    EMAIL_GENERATION = "email_generation"
    RISK_ASSESSMENT = "risk_assessment"

@dataclass
class GroqRequest:
    """Structured request for Groq API"""
    prompt: str
    model: GroqModel = GroqModel.LLAMA3_8B
    temperature: float = 0.3
    max_tokens: int = 1000
    system_prompt: Optional[str] = None
    json_mode: bool = False
    timeout: int = 30

@dataclass
class GroqResponse:
    """Structured response from Groq API"""
    content: str
    model: str
    tokens_used: int
    response_time: float
    success: bool
    error: Optional[str] = None
    parsed_json: Optional[Dict[str, Any]] = None
    cached: bool = False

@dataclass
class InsightRequest:
    """High-level insight request"""
    insight_type: InsightType
    lead_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    urgency: str = "normal"  # low, normal, high
    output_format: str = "json"  # json, text, structured

# ============================
# MAIN GROQ CLIENT CLASS
# ============================

class GroqClient:
    """
    Advanced Groq LLM client for lead generation and analysis
    with caching, rate limiting, and specialized prompts
    """
    
    def __init__(self, api_key: Optional[str] = None, enable_cache: bool = True):
        """
        Initialize Groq client
        
        Args:
            api_key: Groq API key (if None, will try to get from environment)
            enable_cache: Whether to cache responses for performance
        """
        self.api_key = "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"
        self.enable_cache = enable_cache
        
        # Initialize clients
        self.client = None
        self.async_client = None
        
        # Rate limiting
        self.requests_per_minute = 30  # Groq free tier limit
        self.request_timestamps = []
        
        # Response cache
        self.cache = {} if enable_cache else None
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.max_cache_size = 1000
        
        # Performance tracking
        self.total_requests = 0
        self.total_tokens_used = 0
        self.total_response_time = 0.0
        self.error_count = 0
        
        # Specialized prompts
        self.prompt_templates = self._load_prompt_templates()
        
        # Initialize if API key is available
        if self.api_key:
            self._initialize_clients()
        else:
            logger.warning("⚠️ No Groq API key provided. Set GROQ_API_KEY environment variable.")
    
    def _initialize_clients(self):
        """Initialize Groq clients"""
        if not GROQ_AVAILABLE:
            logger.error("❌ Groq library not available. Cannot initialize clients.")
            return
        
        try:
            self.client = Groq(api_key=self.api_key)
            self.async_client = AsyncGroq(api_key=self.api_key)
            logger.info("✅ Groq clients initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq clients: {e}")

    # ============================
    # CORE REQUEST METHODS
    # ============================
    
    def generate_response(self, request: GroqRequest) -> GroqResponse:
        """
        Generate response from Groq API with caching and error handling
        
        Args:
            request: Groq request configuration
            
        Returns:
            GroqResponse with content and metadata
        """
        if not self._is_available():
            return self._generate_fallback_response("Groq client not available")
        
        # Check cache first
        if self.enable_cache:
            cached_response = self._get_cached_response(request)
            if cached_response:
                return cached_response
        
        # Rate limiting check
        if not self._check_rate_limit():
            return self._generate_fallback_response("Rate limit exceeded")
        
        try:
            start_time = time.time()
            
            # Prepare messages
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            # API call parameters
            api_params = {
                "messages": messages,
                "model": request.model.value,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "timeout": request.timeout
            }
            
            # Add JSON mode if requested
            if request.json_mode:
                api_params["response_format"] = {"type": "json_object"}
            
            # Make API call
            response = self.client.chat.completions.create(**api_params)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Extract response data
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
            
            # Create response object
            groq_response = GroqResponse(
                content=content,
                model=request.model.value,
                tokens_used=tokens_used,
                response_time=response_time,
                success=True
            )
            
            # Try to parse JSON if requested
            if request.json_mode or self._is_json_content(content):
                try:
                    groq_response.parsed_json = json.loads(content)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON response")
            
            # Cache the response
            if self.enable_cache:
                self._cache_response(request, groq_response)
            
            # Update tracking
            self._update_metrics(tokens_used, response_time)
            
            return groq_response
            
        except Exception as e:
            logger.error(f"Groq API request failed: {e}")
            self.error_count += 1
            return self._generate_fallback_response(f"API error: {str(e)}")
    
    async def generate_response_async(self, request: GroqRequest) -> GroqResponse:
        """Async version of generate_response"""
        if not self._is_available():
            return self._generate_fallback_response("Groq client not available")
        
        # Check cache first
        if self.enable_cache:
            cached_response = self._get_cached_response(request)
            if cached_response:
                return cached_response
        
        # Rate limiting check
        if not self._check_rate_limit():
            return self._generate_fallback_response("Rate limit exceeded")
        
        try:
            start_time = time.time()
            
            # Prepare messages
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
            
            # API call parameters
            api_params = {
                "messages": messages,
                "model": request.model.value,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens
            }
            
            if request.json_mode:
                api_params["response_format"] = {"type": "json_object"}
            
            # Make async API call
            response = await self.async_client.chat.completions.create(**api_params)
            
            response_time = time.time() - start_time
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
            
            groq_response = GroqResponse(
                content=content,
                model=request.model.value,
                tokens_used=tokens_used,
                response_time=response_time,
                success=True
            )
            
            if request.json_mode or self._is_json_content(content):
                try:
                    groq_response.parsed_json = json.loads(content)
                except json.JSONDecodeError:
                    pass
            
            if self.enable_cache:
                self._cache_response(request, groq_response)
            
            self._update_metrics(tokens_used, response_time)
            
            return groq_response
            
        except Exception as e:
            logger.error(f"Async Groq API request failed: {e}")
            self.error_count += 1
            return self._generate_fallback_response(f"API error: {str(e)}")

    # ============================
    # SPECIALIZED INSIGHT METHODS
    # ============================
    
    def analyze_lead(self, lead_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> GroqResponse:
        """
        Analyze a lead using AI and provide strategic insights
        
        Args:
            lead_data: Lead information
            context: Additional context (campaign, market conditions, etc.)
            
        Returns:
            GroqResponse with lead analysis
        """
        prompt = self._build_lead_analysis_prompt(lead_data, context)
        
        request = GroqRequest(
            prompt=prompt,
            model=GroqModel.LLAMA3_8B,
            temperature=0.2,
            max_tokens=800,
            system_prompt=self.prompt_templates["lead_analysis_system"],
            json_mode=True
        )
        
        return self.generate_response(request)
    
    def generate_personalization_strategy(self, lead_data: Dict[str, Any], campaign_context: Optional[Dict[str, Any]] = None) -> GroqResponse:
        """
        Generate personalized outreach strategy for a lead
        
        Args:
            lead_data: Lead information
            campaign_context: Campaign-specific context
            
        Returns:
            GroqResponse with personalization strategy
        """
        prompt = self._build_personalization_prompt(lead_data, campaign_context)
        
        request = GroqRequest(
            prompt=prompt,
            model=GroqModel.LLAMA3_8B,
            temperature=0.4,
            max_tokens=600,
            system_prompt=self.prompt_templates["personalization_system"],
            json_mode=True
        )
        
        return self.generate_response(request)
    
    def optimize_timing(self, lead_data: Dict[str, Any], historical_data: Optional[Dict[str, Any]] = None) -> GroqResponse:
        """
        Optimize contact timing based on lead profile and historical data
        
        Args:
            lead_data: Lead information
            historical_data: Historical engagement data
            
        Returns:
            GroqResponse with timing recommendations
        """
        prompt = self._build_timing_optimization_prompt(lead_data, historical_data)
        
        request = GroqRequest(
            prompt=prompt,
            model=GroqModel.LLAMA3_8B,
            temperature=0.3,
            max_tokens=500,
            system_prompt=self.prompt_templates["timing_system"],
            json_mode=True
        )
        
        return self.generate_response(request)
    
    def generate_email_content(self, lead_data: Dict[str, Any], email_type: str = "cold_outreach", context: Optional[Dict[str, Any]] = None) -> GroqResponse:
        """
        Generate personalized email content for outreach
        
        Args:
            lead_data: Lead information
            email_type: Type of email (cold_outreach, follow_up, nurture)
            context: Additional context
            
        Returns:
            GroqResponse with email content
        """
        prompt = self._build_email_generation_prompt(lead_data, email_type, context)
        
        request = GroqRequest(
            prompt=prompt,
            model=GroqModel.LLAMA3_8B,
            temperature=0.5,
            max_tokens=800,
            system_prompt=self.prompt_templates["email_generation_system"],
            json_mode=True
        )
        
        return self.generate_response(request)
    
    def assess_competitive_landscape(self, lead_data: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> GroqResponse:
        """
        Assess competitive landscape for a lead's industry/company
        
        Args:
            lead_data: Lead information
            market_data: Market and competitive data
            
        Returns:
            GroqResponse with competitive analysis
        """
        prompt = self._build_competitive_analysis_prompt(lead_data, market_data)
        
        request = GroqRequest(
            prompt=prompt,
            model=GroqModel.LLAMA3_70B,  # Use larger model for complex analysis
            temperature=0.3,
            max_tokens=1000,
            system_prompt=self.prompt_templates["competitive_analysis_system"],
            json_mode=True
        )
        
        return self.generate_response(request)
    
    def generate_risk_assessment(self, lead_data: Dict[str, Any], business_context: Optional[Dict[str, Any]] = None) -> GroqResponse:
        """
        Generate risk assessment for lead engagement
        
        Args:
            lead_data: Lead information
            business_context: Business and market context
            
        Returns:
            GroqResponse with risk assessment
        """
        prompt = self._build_risk_assessment_prompt(lead_data, business_context)
        
        request = GroqRequest(
            prompt=prompt,
            model=GroqModel.LLAMA3_8B,
            temperature=0.2,
            max_tokens=600,
            system_prompt=self.prompt_templates["risk_assessment_system"],
            json_mode=True
        )
        
        return self.generate_response(request)

    # ============================
    # BATCH PROCESSING METHODS
    # ============================
    
    def batch_analyze_leads(self, leads: List[Dict[str, Any]], analysis_type: str = "standard") -> List[GroqResponse]:
        """
        Analyze multiple leads in batch with rate limiting
        
        Args:
            leads: List of lead data dictionaries
            analysis_type: Type of analysis (quick, standard, detailed)
            
        Returns:
            List of GroqResponse objects
        """
        results = []
        
        for i, lead in enumerate(leads):
            try:
                # Rate limiting between requests
                if i > 0:
                    time.sleep(1)  # Respect rate limits
                
                if analysis_type == "quick":
                    response = self._quick_lead_analysis(lead)
                elif analysis_type == "detailed":
                    response = self._detailed_lead_analysis(lead)
                else:  # standard
                    response = self.analyze_lead(lead)
                
                results.append(response)
                
            except Exception as e:
                logger.error(f"Failed to analyze lead {lead.get('id', i)}: {e}")
                results.append(self._generate_fallback_response(f"Analysis failed: {str(e)}"))
        
        return results
    
    async def batch_analyze_leads_async(self, leads: List[Dict[str, Any]], analysis_type: str = "standard", max_concurrent: int = 5) -> List[GroqResponse]:
        """
        Async batch analysis with concurrency control
        
        Args:
            leads: List of lead data dictionaries
            analysis_type: Type of analysis
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of GroqResponse objects
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_single_lead(lead_data):
            async with semaphore:
                try:
                    if analysis_type == "quick":
                        request = self._build_quick_analysis_request(lead_data)
                    elif analysis_type == "detailed":
                        request = self._build_detailed_analysis_request(lead_data)
                    else:
                        request = self._build_standard_analysis_request(lead_data)
                    
                    return await self.generate_response_async(request)
                    
                except Exception as e:
                    logger.error(f"Failed to analyze lead {lead_data.get('id')}: {e}")
                    return self._generate_fallback_response(f"Analysis failed: {str(e)}")
        
        # Execute all analyses concurrently
        tasks = [analyze_single_lead(lead) for lead in leads]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(self._generate_fallback_response(f"Exception: {str(result)}"))
            else:
                processed_results.append(result)
        
        return processed_results

    # ============================
    # PROMPT BUILDING METHODS
    # ============================
    
    def _build_lead_analysis_prompt(self, lead_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for lead analysis"""
        company = lead_data.get('company', {})
        
        prompt = f"""
        Analyze this B2B lead for sales potential and provide strategic insights:

        Lead Information:
        - Name: {lead_data.get('first_name', 'Unknown')} {lead_data.get('last_name', 'Unknown')}
        - Title: {lead_data.get('title', 'Unknown')}
        - Email: {lead_data.get('email', 'Not provided')}
        - Phone: {lead_data.get('phone', 'Not provided')}
        - LinkedIn: {lead_data.get('linkedin_url', 'Not provided')}

        Company Information:
        - Company: {company.get('name', 'Unknown')}
        - Industry: {company.get('industry', 'Unknown')}
        - Size: {company.get('size', 'Unknown')}
        - Location: {company.get('location', 'Unknown')}
        """
        
        if context:
            prompt += f"\n\nAdditional Context:\n"
            for key, value in context.items():
                prompt += f"- {key}: {value}\n"
        
        prompt += """
        
        Provide analysis in JSON format with these fields:
        {
            "priority_level": "high|medium|low",
            "decision_maker_likelihood": "high|medium|low",
            "industry_fit": "excellent|good|fair|poor",
            "company_size_fit": "ideal|good|acceptable|poor",
            "engagement_strategy": "detailed strategy recommendation",
            "key_talking_points": ["point1", "point2", "point3"],
            "potential_pain_points": ["pain1", "pain2", "pain3"],
            "timing_recommendation": "immediate|within_week|within_month|long_term",
            "success_probability": "percentage as integer",
            "next_actions": ["action1", "action2", "action3"],
            "risk_factors": ["risk1", "risk2"],
            "value_proposition_angle": "specific angle to emphasize"
        }
        """
        
        return prompt
    
    def _build_personalization_prompt(self, lead_data: Dict[str, Any], campaign_context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for personalization strategy"""
        return f"""
        Create a personalized outreach strategy for this lead:
        
        Lead: {lead_data.get('first_name')} {lead_data.get('last_name')}
        Title: {lead_data.get('title', 'Unknown')}
        Company: {lead_data.get('company', {}).get('name', 'Unknown')}
        Industry: {lead_data.get('company', {}).get('industry', 'Unknown')}
        
        Campaign Context: {json.dumps(campaign_context) if campaign_context else 'General outreach'}
        
        Provide personalization strategy in JSON format:
        {{
            "personal_connection_opportunities": ["opportunity1", "opportunity2"],
            "company_specific_research": ["insight1", "insight2"],
            "industry_trends_to_mention": ["trend1", "trend2"],
            "personalized_subject_lines": ["subject1", "subject2", "subject3"],
            "conversation_starters": ["starter1", "starter2"],
            "value_prop_customization": "customized value proposition",
            "social_proof_suggestions": ["proof1", "proof2"],
            "follow_up_personalization": "follow-up strategy"
        }}
        """
    
    def _build_timing_optimization_prompt(self, lead_data: Dict[str, Any], historical_data: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for timing optimization"""
        return f"""
        Optimize contact timing for this lead based on their profile and industry patterns:
        
        Lead Profile:
        - Title: {lead_data.get('title', 'Unknown')}
        - Company Size: {lead_data.get('company', {}).get('size', 'Unknown')}
        - Industry: {lead_data.get('company', {}).get('industry', 'Unknown')}
        - Location: {lead_data.get('company', {}).get('location', 'Unknown')}
        
        Historical Data: {json.dumps(historical_data) if historical_data else 'No historical data available'}
        
        Provide timing recommendations in JSON format:
        {{
            "optimal_day_of_week": "monday|tuesday|wednesday|thursday|friday",
            "best_time_range": "time range in local timezone",
            "timezone_considerations": "timezone analysis",
            "industry_timing_patterns": "industry-specific insights",
            "seniority_timing_factors": "seniority-based recommendations",
            "seasonal_considerations": "seasonal factors",
            "urgency_level": "low|medium|high",
            "follow_up_cadence": "recommended follow-up timing",
            "alternative_times": ["alt1", "alt2"]
        }}
        """
    
    def _build_email_generation_prompt(self, lead_data: Dict[str, Any], email_type: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for email generation"""
        return f"""
        Generate a personalized {email_type} email for this lead:
        
        Lead: {lead_data.get('first_name')} {lead_data.get('last_name')}
        Title: {lead_data.get('title', 'Unknown')}
        Company: {lead_data.get('company', {}).get('name', 'Unknown')}
        Industry: {lead_data.get('company', {}).get('industry', 'Unknown')}
        
        Context: {json.dumps(context) if context else 'Standard outreach'}
        
        Generate email content in JSON format:
        {{
            "subject_line": "compelling subject line",
            "email_body": "full email content with proper formatting",
            "call_to_action": "specific CTA",
            "tone": "professional|friendly|consultative",
            "personalization_elements": ["element1", "element2"],
            "follow_up_suggestions": "follow-up recommendations",
            "alternative_subject_lines": ["alt1", "alt2"],
            "key_message": "main message summary"
        }}
        
        Email should be:
        - Professional but personable
        - Concise (under 150 words)
        - Value-focused
        - Industry-appropriate
        - Action-oriented
        """
    
    def _build_competitive_analysis_prompt(self, lead_data: Dict[str, Any], market_data: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for competitive analysis"""
        return f"""
        Analyze the competitive landscape for this lead's industry and company:
        
        Lead Company: {lead_data.get('company', {}).get('name', 'Unknown')}
        Industry: {lead_data.get('company', {}).get('industry', 'Unknown')}
        Company Size: {lead_data.get('company', {}).get('size', 'Unknown')}
        Location: {lead_data.get('company', {}).get('location', 'Unknown')}
        
        Market Data: {json.dumps(market_data) if market_data else 'No specific market data provided'}
        
        Provide competitive analysis in JSON format:
        {{
            "market_maturity": "emerging|growing|mature|declining",
            "competition_level": "low|medium|high|very_high",
            "key_competitors": ["competitor1", "competitor2", "competitor3"],
            "market_trends": ["trend1", "trend2", "trend3"],
            "differentiation_opportunities": ["opportunity1", "opportunity2"],
            "buyer_behavior_patterns": "typical buying behavior in this industry",
            "decision_making_process": "typical B2B decision process",
            "budget_cycles": "typical budget timing",
            "pain_points": ["pain1", "pain2", "pain3"],
            "success_metrics": ["metric1", "metric2"],
            "competitive_advantages": ["advantage1", "advantage2"]
        }}
        """
    
    def _build_risk_assessment_prompt(self, lead_data: Dict[str, Any], business_context: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for risk assessment"""
        return f"""
        Assess the risks and opportunities for engaging with this lead:
        
        Lead Information:
        - Title: {lead_data.get('title', 'Unknown')}
        - Company: {lead_data.get('company', {}).get('name', 'Unknown')}
        - Industry: {lead_data.get('company', {}).get('industry', 'Unknown')}
        - Company Size: {lead_data.get('company', {}).get('size', 'Unknown')}
        
        Business Context: {json.dumps(business_context) if business_context else 'Standard B2B sales context'}
        
        Provide risk assessment in JSON format:
        {{
            "overall_risk_level": "low|medium|high",
            "deal_probability": "percentage as integer",
            "time_to_close_estimate": "estimated timeline",
            "budget_availability": "high|medium|low|unknown",
            "decision_complexity": "simple|moderate|complex",
            "stakeholder_involvement": "single|multiple|committee",
            "implementation_challenges": ["challenge1", "challenge2"],
            "competitive_threats": ["threat1", "threat2"],
            "economic_factors": "relevant economic considerations",
            "regulatory_considerations": "regulatory factors if any",
            "technology_fit": "excellent|good|fair|poor",
            "cultural_fit": "assessment of cultural alignment",
            "mitigation_strategies": ["strategy1", "strategy2"],
            "go_no_go_recommendation": "go|proceed_with_caution|no_go",
            "success_indicators": ["indicator1", "indicator2"]
        }}
        """

    # ============================
    # UTILITY METHODS
    # ============================
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load system prompt templates"""
        return {
            "lead_analysis_system": """You are an expert B2B sales analyst with deep knowledge of lead qualification, industry trends, and sales strategy. Your analysis should be data-driven, actionable, and focused on conversion probability. Always provide specific, tactical recommendations.""",
            
            "personalization_system": """You are a master of personalized B2B outreach with expertise in psychology, communication, and relationship building. Focus on authentic personalization that demonstrates genuine research and understanding of the prospect's situation.""",
            
            "timing_system": """You are a sales timing optimization expert with knowledge of industry patterns, executive schedules, and optimal contact strategies. Your recommendations should maximize response rates and respect the prospect's time.""",
            
            "email_generation_system": """You are a top-performing sales copywriter specializing in B2B cold outreach. Create emails that are professional, value-driven, and compelling while avoiding spam triggers. Focus on opening conversations, not making immediate sales.""",
            
            "competitive_analysis_system": """You are a market research expert with deep knowledge of competitive landscapes, industry dynamics, and market positioning. Provide insights that give sales teams strategic advantages in their approach.""",
            
            "risk_assessment_system": """You are a business risk analyst specializing in B2B sales opportunities. Evaluate deals objectively, considering all factors that could impact success. Your assessments help sales teams prioritize efforts and avoid wasted resources."""
        }
    
    def _is_available(self) -> bool:
        """Check if Groq client is available and properly configured"""
        return GROQ_AVAILABLE and self.client is not None and self.api_key is not None
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = time.time()
        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
        
        if len(self.request_timestamps) >= self.requests_per_minute:
            return False
        
        self.request_timestamps.append(now)
        return True
    
    def _generate_cache_key(self, request: GroqRequest) -> str:
        """Generate cache key for request"""
        cache_data = {
            'prompt': request.prompt,
            'model': request.model.value,
            'temperature': request.temperature,
            'system_prompt': request.system_prompt
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cached_response(self, request: GroqRequest) -> Optional[GroqResponse]:
        """Get cached response if available and valid"""
        if not self.cache:
            return None
        
        cache_key = self._generate_cache_key(request)
        if cache_key not in self.cache:
            return None
        
        cached_entry = self.cache[cache_key]
        
        # Check if cache entry is still valid
        if time.time() - cached_entry['timestamp'] > self.cache_ttl:
            del self.cache[cache_key]
            return None
        
        # Return cached response with cache flag
        response = cached_entry['response']
        response.cached = True
        return response
    
    def _cache_response(self, request: GroqRequest, response: GroqResponse):
        """Cache response for future use"""
        if not self.cache:
            return
        
        # Clean old entries if cache is full
        if len(self.cache) >= self.max_cache_size:
            self._clean_cache()
        
        cache_key = self._generate_cache_key(request)
        self.cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
    
    def _clean_cache(self):
        """Remove oldest cache entries"""
        if not self.cache:
            return
        
        # Sort by timestamp and remove oldest 20%
        sorted_items = sorted(self.cache.items(), key=lambda x: x[1]['timestamp'])
        num_to_remove = max(1, len(sorted_items) // 5)
        
        for i in range(num_to_remove):
            key = sorted_items[i][0]
            del self.cache[key]
    
    def _is_json_content(self, content: str) -> bool:
        """Check if content appears to be JSON"""
        content = content.strip()
        return (content.startswith('{') and content.endswith('}')) or \
               (content.startswith('[') and content.endswith(']'))
    
    def _generate_fallback_response(self, error_message: str) -> GroqResponse:
        """Generate fallback response when API is unavailable"""
        return GroqResponse(
            content=f"Fallback response: {error_message}",
            model="fallback",
            tokens_used=0,
            response_time=0.0,
            success=False,
            error=error_message
        )
    
    def _update_metrics(self, tokens_used: int, response_time: float):
        """Update performance metrics"""
        self.total_requests += 1
        self.total_tokens_used += tokens_used
        self.total_response_time += response_time
    
    def _quick_lead_analysis(self, lead_data: Dict[str, Any]) -> GroqResponse:
        """Quick lead analysis with reduced token usage"""
        prompt = f"""
        Quick analysis for lead: {lead_data.get('first_name')} {lead_data.get('last_name')}
        Title: {lead_data.get('title', 'Unknown')}
        Company: {lead_data.get('company', {}).get('name', 'Unknown')}
        
        Provide brief JSON analysis:
        {{
            "priority": "high|medium|low",
            "approach": "brief strategy",
            "timing": "when to contact"
        }}
        """
        
        request = GroqRequest(
            prompt=prompt,
            model=GroqModel.LLAMA3_8B,
            temperature=0.3,
            max_tokens=200,
            json_mode=True
        )
        
        return self.generate_response(request)
    
    def _detailed_lead_analysis(self, lead_data: Dict[str, Any]) -> GroqResponse:
        """Detailed lead analysis with comprehensive insights"""
        # Use the full analysis method but with more tokens
        response = self.analyze_lead(lead_data)
        return response
    
    def _build_quick_analysis_request(self, lead_data: Dict[str, Any]) -> GroqRequest:
        """Build request for quick analysis"""
        prompt = f"""
        Quick B2B lead assessment:
        {lead_data.get('first_name')} {lead_data.get('last_name')} - {lead_data.get('title', 'Unknown')}
        Company: {lead_data.get('company', {}).get('name', 'Unknown')}
        
        JSON response: {{"priority": "high|medium|low", "next_action": "specific action"}}
        """
        
        return GroqRequest(
            prompt=prompt,
            model=GroqModel.LLAMA3_8B,
            temperature=0.2,
            max_tokens=150,
            json_mode=True
        )
    
    def _build_standard_analysis_request(self, lead_data: Dict[str, Any]) -> GroqRequest:
        """Build request for standard analysis"""
        prompt = self._build_lead_analysis_prompt(lead_data)
        
        return GroqRequest(
            prompt=prompt,
            model=GroqModel.LLAMA3_8B,
            temperature=0.3,
            max_tokens=600,
            system_prompt=self.prompt_templates["lead_analysis_system"],
            json_mode=True
        )
    
    def _build_detailed_analysis_request(self, lead_data: Dict[str, Any]) -> GroqRequest:
        """Build request for detailed analysis"""
        prompt = self._build_lead_analysis_prompt(lead_data)
        
        return GroqRequest(
            prompt=prompt,
            model=GroqModel.LLAMA3_70B,  # Use larger model for detailed analysis
            temperature=0.2,
            max_tokens=1200,
            system_prompt=self.prompt_templates["lead_analysis_system"],
            json_mode=True
        )

    # ============================
    # STREAMING METHODS
    # ============================
    
    def generate_streaming_response(self, request: GroqRequest) -> AsyncGenerator[str, None]:
        """
        Generate streaming response for real-time insights
        Note: This is a placeholder as Groq doesn't support streaming in the current API
        """
        async def stream():
            response = await self.generate_response_async(request)
            if response.success:
                # Simulate streaming by yielding content in chunks
                content = response.content
                chunk_size = 50
                for i in range(0, len(content), chunk_size):
                    yield content[i:i+chunk_size]
                    await asyncio.sleep(0.1)  # Simulate streaming delay
            else:
                yield f"Error: {response.error}"
        
        return stream()

    # ============================
    # ANALYTICS AND MONITORING
    # ============================
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get detailed usage statistics"""
        avg_response_time = (self.total_response_time / max(self.total_requests, 1))
        avg_tokens_per_request = (self.total_tokens_used / max(self.total_requests, 1))
        
        return {
            'total_requests': self.total_requests,
            'total_tokens_used': self.total_tokens_used,
            'total_response_time': round(self.total_response_time, 2),
            'average_response_time': round(avg_response_time, 2),
            'average_tokens_per_request': round(avg_tokens_per_request, 1),
            'error_count': self.error_count,
            'error_rate': round((self.error_count / max(self.total_requests, 1)) * 100, 2),
            'cache_size': len(self.cache) if self.cache else 0,
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'requests_per_minute_limit': self.requests_per_minute,
            'current_minute_requests': len(self.request_timestamps)
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if not self.cache or self.total_requests == 0:
            return 0.0
        
        # This is a simplified calculation
        # In production, you'd track hits vs misses more precisely
        cache_usage = min(len(self.cache), self.total_requests)
        return round((cache_usage / self.total_requests) * 100, 2)
    
    def reset_statistics(self):
        """Reset all usage statistics"""
        self.total_requests = 0
        self.total_tokens_used = 0
        self.total_response_time = 0.0
        self.error_count = 0
        self.request_timestamps = []
    
    def clear_cache(self):
        """Clear all cached responses"""
        if self.cache:
            self.cache.clear()

    # ============================
    # HIGH-LEVEL INSIGHT GENERATION
    # ============================
    
    def generate_insight(self, insight_request: InsightRequest) -> GroqResponse:
        """
        High-level method to generate insights based on request type
        
        Args:
            insight_request: Structured insight request
            
        Returns:
            GroqResponse with requested insights
        """
        if insight_request.insight_type == InsightType.LEAD_ANALYSIS:
            return self.analyze_lead(insight_request.lead_data, insight_request.context)
        
        elif insight_request.insight_type == InsightType.PERSONALIZATION:
            return self.generate_personalization_strategy(insight_request.lead_data, insight_request.context)
        
        elif insight_request.insight_type == InsightType.TIMING_OPTIMIZATION:
            return self.optimize_timing(insight_request.lead_data, insight_request.context)
        
        elif insight_request.insight_type == InsightType.EMAIL_GENERATION:
            email_type = insight_request.context.get('email_type', 'cold_outreach') if insight_request.context else 'cold_outreach'
            return self.generate_email_content(insight_request.lead_data, email_type, insight_request.context)
        
        elif insight_request.insight_type == InsightType.COMPETITIVE_ANALYSIS:
            return self.assess_competitive_landscape(insight_request.lead_data, insight_request.context)
        
        elif insight_request.insight_type == InsightType.RISK_ASSESSMENT:
            return self.generate_risk_assessment(insight_request.lead_data, insight_request.context)
        
        else:
            return self._generate_fallback_response(f"Unsupported insight type: {insight_request.insight_type}")
    
    def generate_comprehensive_lead_report(self, lead_data: Dict[str, Any], include_sections: Optional[List[str]] = None) -> Dict[str, GroqResponse]:
        """
        Generate comprehensive lead report with multiple analysis sections
        
        Args:
            lead_data: Lead information
            include_sections: Specific sections to include (default: all)
            
        Returns:
            Dictionary with section names as keys and GroqResponse as values
        """
        default_sections = [
            'lead_analysis',
            'personalization',
            'timing_optimization', 
            'risk_assessment',
            'email_generation'
        ]
        
        sections = include_sections or default_sections
        report = {}
        
        for section in sections:
            try:
                if section == 'lead_analysis':
                    report[section] = self.analyze_lead(lead_data)
                
                elif section == 'personalization':
                    report[section] = self.generate_personalization_strategy(lead_data)
                
                elif section == 'timing_optimization':
                    report[section] = self.optimize_timing(lead_data)
                
                elif section == 'risk_assessment':
                    report[section] = self.generate_risk_assessment(lead_data)
                
                elif section == 'email_generation':
                    report[section] = self.generate_email_content(lead_data)
                
                elif section == 'competitive_analysis':
                    report[section] = self.assess_competitive_landscape(lead_data)
                
                # Add delay between requests to respect rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to generate {section}: {e}")
                report[section] = self._generate_fallback_response(f"Failed to generate {section}: {str(e)}")
        
        return report


# ============================
# SPECIALIZED GROQ MANAGERS
# ============================

class GroqInsightManager:
    """High-level manager for AI insights with caching and optimization"""
    
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client
        self.insight_cache = {}
        self.cache_ttl = 1800  # 30 minutes for insights
    
    def get_lead_insights(self, lead_data: Dict[str, Any], insight_types: List[InsightType]) -> Dict[str, Any]:
        """Get multiple types of insights for a lead"""
        insights = {}
        
        for insight_type in insight_types:
            cache_key = f"{lead_data.get('id', 'unknown')}_{insight_type.value}"
            
            # Check cache first
            if cache_key in self.insight_cache:
                cached_entry = self.insight_cache[cache_key]
                if time.time() - cached_entry['timestamp'] < self.cache_ttl:
                    insights[insight_type.value] = cached_entry['data']
                    continue
            
            # Generate new insight
            request = InsightRequest(
                insight_type=insight_type,
                lead_data=lead_data
            )
            
            response = self.groq_client.generate_insight(request)
            
            if response.success:
                insight_data = response.parsed_json or {'content': response.content}
                insights[insight_type.value] = insight_data
                
                # Cache the result
                self.insight_cache[cache_key] = {
                    'data': insight_data,
                    'timestamp': time.time()
                }
            else:
                insights[insight_type.value] = {'error': response.error}
        
        return insights
    
    def get_campaign_insights(self, leads: List[Dict[str, Any]], campaign_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate campaign-level insights across multiple leads"""
        # Analyze lead patterns
        industries = {}
        titles = {}
        company_sizes = {}
        
        for lead in leads:
            company = lead.get('company', {})
            industry = company.get('industry', 'Unknown')
            title = lead.get('title', 'Unknown')
            size = company.get('size', 'Unknown')
            
            industries[industry] = industries.get(industry, 0) + 1
            titles[title] = titles.get(title, 0) + 1
            company_sizes[size] = company_sizes.get(size, 0) + 1
        
        # Generate campaign strategy prompt
        prompt = f"""
        Analyze this B2B campaign targeting {len(leads)} leads:
        
        Campaign Context: {json.dumps(campaign_context)}
        
        Lead Distribution:
        - Industries: {json.dumps(industries)}
        - Job Titles: {json.dumps(titles)}
        - Company Sizes: {json.dumps(company_sizes)}
        
        Provide campaign strategy in JSON format:
        {{
            "overall_strategy": "campaign approach",
            "messaging_themes": ["theme1", "theme2", "theme3"],
            "segmentation_recommendations": ["segment1", "segment2"],
            "timing_strategy": "optimal timing approach",
            "success_metrics": ["metric1", "metric2"],
            "potential_challenges": ["challenge1", "challenge2"],
            "optimization_opportunities": ["opportunity1", "opportunity2"]
        }}
        """
        
        request = GroqRequest(
            prompt=prompt,
            model=GroqModel.LLAMA3_8B,
            temperature=0.3,
            max_tokens=800,
            json_mode=True
        )
        
        response = self.groq_client.generate_response(request)
        
        return {
            'campaign_insights': response.parsed_json if response.success else {'error': response.error},
            'lead_distribution': {
                'industries': industries,
                'titles': titles,
                'company_sizes': company_sizes
            },
            'total_leads': len(leads)
        }


# ============================
# USAGE EXAMPLES AND TESTING
# ============================

if __name__ == "__main__":
    # Initialize Groq client
    groq_client = GroqClient()
    
    # Test lead data
    test_lead = {
        'id': 1,
        'first_name': 'Sarah',
        'last_name': 'Johnson',
        'email': 'sarah.johnson@techstart.ai',
        'phone': '+1-555-987-6543',
        'title': 'VP of Engineering',
        'linkedin_url': 'https://linkedin.com/in/sarah-johnson-vp',
        'company': {
            'name': 'TechStart AI',
            'industry': 'Artificial Intelligence',
            'size': '51-200',
            'location': 'Austin, TX'
        }
    }
    
    print("🧪 Testing Groq Client")
    print("=" * 50)
    
    if not groq_client._is_available():
        print("⚠️ Groq client not available. Set GROQ_API_KEY environment variable.")
        print("📝 Showing example usage without actual API calls...")
        
        # Show example request/response structure
        example_request = GroqRequest(
            prompt="Analyze this lead...",
            model=GroqModel.LLAMA3_8B,
            temperature=0.3,
            max_tokens=600,
            json_mode=True
        )
        
        print(f"\n📋 Example Request Structure:")
        print(f"  Model: {example_request.model.value}")
        print(f"  Temperature: {example_request.temperature}")
        print(f"  Max Tokens: {example_request.max_tokens}")
        print(f"  JSON Mode: {example_request.json_mode}")
        
    else:
        # Test lead analysis
        print(f"\n🔍 Lead Analysis Test:")
        print("-" * 30)
        
        analysis_response = groq_client.analyze_lead(test_lead)
        
        if analysis_response.success:
            print(f"✅ Analysis completed in {analysis_response.response_time:.2f}s")
            print(f"📊 Tokens used: {analysis_response.tokens_used}")
            
            if analysis_response.parsed_json:
                analysis = analysis_response.parsed_json
                print(f"🎯 Priority Level: {analysis.get('priority_level', 'Unknown')}")
                print(f"📈 Success Probability: {analysis.get('success_probability', 'Unknown')}%")
                print(f"⏰ Timing: {analysis.get('timing_recommendation', 'Unknown')}")
            else:
                print(f"📄 Response: {analysis_response.content[:200]}...")
        else:
            print(f"❌ Analysis failed: {analysis_response.error}")
        
        # Test personalization
        print(f"\n🎨 Personalization Test:")
        print("-" * 30)
        
        personalization_response = groq_client.generate_personalization_strategy(test_lead)
        
        if personalization_response.success:
            print(f"✅ Personalization completed in {personalization_response.response_time:.2f}s")
            
            if personalization_response.parsed_json:
                strategy = personalization_response.parsed_json
                subject_lines = strategy.get('personalized_subject_lines', [])
                print(f"📧 Suggested Subject Lines:")
                for i, subject in enumerate(subject_lines[:2], 1):
                    print(f"  {i}. {subject}")
        else:
            print(f"❌ Personalization failed: {personalization_response.error}")
        
        # Test email generation
        print(f"\n📧 Email Generation Test:")
        print("-" * 30)
        
        email_response = groq_client.generate_email_content(test_lead, "cold_outreach")
        
        if email_response.success:
            print(f"✅ Email generated in {email_response.response_time:.2f}s")
            
            if email_response.parsed_json:
                email_data = email_response.parsed_json
                print(f"📝 Subject: {email_data.get('subject_line', 'N/A')}")
                email_body = email_data.get('email_body', '')
                print(f"📄 Body Preview: {email_body[:150]}...")
        else:
            print(f"❌ Email generation failed: {email_response.error}")
        
        # Test batch analysis
        print(f"\n📊 Batch Analysis Test:")
        print("-" * 30)
        
        test_leads = [test_lead]  # Small batch for testing
        batch_results = groq_client.batch_analyze_leads(test_leads, "quick")
        
        print(f"📈 Batch processed: {len(batch_results)} leads")
        for i, result in enumerate(batch_results, 1):
            if result.success:
                print(f"  Lead {i}: ✅ Success ({result.response_time:.2f}s)")
            else:
                print(f"  Lead {i}: ❌ Failed - {result.error}")
    
    # Test insight manager
    print(f"\n🧠 Insight Manager Test:")
    print("-" * 30)
    
    insight_manager = GroqInsightManager(groq_client)
    
    if groq_client._is_available():
        insights = insight_manager.get_lead_insights(
            test_lead, 
            [InsightType.LEAD_ANALYSIS, InsightType.TIMING_OPTIMIZATION]
        )
        
        print(f"🎯 Generated insights:")
        for insight_type, data in insights.items():
            if 'error' not in data:
                print(f"  ✅ {insight_type}: Success")
            else:
                print(f"  ❌ {insight_type}: {data['error']}")
    else:
        print("⚠️ Insight manager requires valid Groq API key")
    
    # Show usage statistics
    print(f"\n📊 Usage Statistics:")
    print("-" * 30)
    
    stats = groq_client.get_usage_statistics()
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Total Tokens: {stats['total_tokens_used']}")
    print(f"Average Response Time: {stats['average_response_time']}s")
    print(f"Error Rate: {stats['error_rate']}%")
    print(f"Cache Size: {stats['cache_size']}")
    
    print(f"\n✅ Groq Client testing completed!")
    print(f"🚀 Ready for production use with comprehensive LLM integration")
    print(f"🎯 Features: Lead Analysis, Personalization, Email Generation, Batch Processing, Caching")