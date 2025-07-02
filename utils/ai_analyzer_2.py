# Enhanced AI Lead Analyzer with Groq LLM Integration - FIXED VERSION
# File: utils/groq_enhanced_analyzer.py

import json
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import existing Groq client
from .groq_client import GroqClient, GroqRequest, GroqModel

logger = logging.getLogger(__name__)

class GroqEnhancedAnalyzer:
    """Enhanced AI analyzer that uses Groq LLM to generate human-readable explanations for lead priorities"""
    
    def __init__(self):
        self.groq_client = GroqClient()
        self.scoring_weights = {
            "contact_quality": 0.25,
            "company_profile": 0.35,
            "role_seniority": 0.25,
            "timing_factors": 0.15
        }
    
    def analyze_lead(self, lead_data: Dict[str, Any], context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        COMPATIBILITY METHOD: Main analysis function that maintains compatibility with existing code
        This method matches the interface expected by your existing code
        """
        # Convert lead object to dict if needed
        if hasattr(lead_data, '__dict__'):
            lead_dict = lead_data.__dict__
        else:
            lead_dict = lead_data
        
        # Use the enhanced analysis
        enhanced_analysis = self.analyze_lead_with_groq_explanation(lead_dict)
        
        # Convert to the format expected by existing code
        return {
            'priority_score': enhanced_analysis.get('priority_score', 50),
            'confidence': enhanced_analysis.get('confidence', 0.5),
            'reason': enhanced_analysis.get('ai_explanation', 'AI analysis completed'),
            'actions': self._generate_suggested_actions(enhanced_analysis),
            'urgency_score': enhanced_analysis.get('timing_analysis', {}).get('score', 50),
            'response_rate': self._estimate_response_rate(enhanced_analysis),
            'optimal_timing': 'Business hours',
            'detailed_analysis': enhanced_analysis  # Include full analysis for reference
        }
    
    def analyze_lead_with_groq_explanation(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main analysis function that combines scoring with Groq-generated explanations"""
        
        # Step 1: Calculate priority score with detailed factors
        priority_analysis = self._calculate_priority_score(lead_data)
        
        # Step 2: Generate Groq explanation for high priority leads
        if priority_analysis["priority_score"] >= 70:  # Only use Groq for high-priority leads
            groq_explanation = self._generate_groq_explanation(lead_data, priority_analysis)
            priority_analysis["ai_explanation"] = groq_explanation
        else:
            # Use rule-based explanation for lower priority leads to save API calls
            priority_analysis["ai_explanation"] = self._generate_rule_based_explanation(priority_analysis)
        
        return priority_analysis
    
    def _calculate_priority_score(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate priority score with detailed factor breakdown"""
        
        # Analyze contact quality
        contact_analysis = self._analyze_contact_quality(lead_data)
        
        # Analyze company profile  
        company_analysis = self._analyze_company_profile(lead_data)
        
        # Analyze role seniority
        role_analysis = self._analyze_role_seniority(lead_data)
        
        # Analyze timing factors
        timing_analysis = self._analyze_timing_factors(lead_data)
        
        # Calculate weighted score
        priority_score = (
            contact_analysis["score"] * self.scoring_weights["contact_quality"] +
            company_analysis["score"] * self.scoring_weights["company_profile"] +
            role_analysis["score"] * self.scoring_weights["role_seniority"] +
            timing_analysis["score"] * self.scoring_weights["timing_factors"]
        )
        
        # Determine priority level
        if priority_score >= 85:
            priority_level = "critical"
        elif priority_score >= 70:
            priority_level = "high" 
        elif priority_score >= 50:
            priority_level = "medium"
        else:
            priority_level = "low"
        
        return {
            "priority_score": round(priority_score, 1),
            "priority_level": priority_level,
            "contact_analysis": contact_analysis,
            "company_analysis": company_analysis,
            "role_analysis": role_analysis,
            "timing_analysis": timing_analysis,
            "confidence": self._calculate_confidence(contact_analysis, company_analysis, role_analysis),
            "key_factors": self._extract_key_factors(contact_analysis, company_analysis, role_analysis, timing_analysis)
        }
    
    def _generate_groq_explanation(self, lead_data: Dict[str, Any], priority_analysis: Dict[str, Any]) -> str:
        """Generate AI explanation using Groq LLM for high priority leads"""
        
        if not self.groq_client._is_available():
            logger.warning("Groq not available, falling back to rule-based explanation")
            return self._generate_rule_based_explanation(priority_analysis)
        
        try:
            # Prepare context for Groq
            lead_summary = self._prepare_lead_summary(lead_data, priority_analysis)
            
            # Create the prompt
            prompt = f"""Analyze this high-priority lead and explain in 2-3 sentences why they are valuable and what makes them worth immediate attention.

Lead Information:
{lead_summary}

Priority Score: {priority_analysis['priority_score']}/100 ({priority_analysis['priority_level']} priority)

Key Strengths:
{self._format_key_factors(priority_analysis['key_factors'])}

Instructions:
- Write a compelling explanation in 2-3 sentences
- Focus on the specific factors that make this lead valuable
- Mention concrete details (role, company, industry) when relevant
- Sound professional but enthusiastic
- Avoid generic language
- Don't repeat the priority score

Example format: "This lead represents exceptional value due to [specific factor]. As a [role] at [company type], they have [decision-making power/budget authority/strategic influence]. The [timing/company situation/role level] makes this an ideal opportunity for immediate outreach."
"""

            # Make Groq request
            groq_request = GroqRequest(
                prompt=prompt,
                model=GroqModel.LLAMA3_8B,
                temperature=0.3,
                max_tokens=200,
                system_prompt="You are an expert sales analyst who creates compelling, specific explanations for why high-value leads should be prioritized. Focus on concrete details and business value."
            )
            
            response = self.groq_client.generate_response(groq_request)
            
            if response.success and response.content:
                # Clean up the response
                explanation = response.content.strip()
                
                # Remove quotes if present
                if explanation.startswith('"') and explanation.endswith('"'):
                    explanation = explanation[1:-1]
                
                logger.info(f"✅ Generated Groq explanation for lead {lead_data.get('id', 'unknown')}")
                return explanation
            else:
                logger.error(f"Groq response failed: {response.error}")
                return self._generate_rule_based_explanation(priority_analysis)
                
        except Exception as e:
            logger.error(f"Error generating Groq explanation: {e}")
            return self._generate_rule_based_explanation(priority_analysis)
    
    def _prepare_lead_summary(self, lead_data: Dict[str, Any], priority_analysis: Dict[str, Any]) -> str:
        """Prepare a structured summary of lead data for Groq"""
        
        name = f"{lead_data.get('first_name', '')} {lead_data.get('last_name', '')}".strip()
        title = lead_data.get('title', 'Unknown Role')
        email = lead_data.get('email', 'No email')
        phone = lead_data.get('phone', 'No phone')
        
        # Handle company data (could be dict or string)
        company_data = lead_data.get('company', {})
        if isinstance(company_data, dict):
            company_name = company_data.get('name', 'Unknown Company')
            industry = company_data.get('industry', 'Unknown Industry')
            company_size = company_data.get('size', 'Unknown Size')
        else:
            company_name = str(company_data) if company_data else 'Unknown Company'
            industry = 'Unknown Industry'
            company_size = 'Unknown Size'
        
        summary = f"""
Name: {name}
Title: {title}
Company: {company_name}
Industry: {industry}
Company Size: {company_size}
Email: {email}
Phone: {phone}
Contact Quality: {priority_analysis['contact_analysis']['score']}/100
Company Score: {priority_analysis['company_analysis']['score']}/100  
Role Seniority: {priority_analysis['role_analysis']['score']}/100
"""
        return summary.strip()
    
    def _format_key_factors(self, key_factors: List[str]) -> str:
        """Format key factors for the prompt"""
        if not key_factors:
            return "- General lead qualification factors"
        
        return "\n".join([f"- {factor}" for factor in key_factors])
    
    def _generate_rule_based_explanation(self, priority_analysis: Dict[str, Any]) -> str:
        """Generate rule-based explanation for when Groq is not available"""
        
        score = priority_analysis["priority_score"]
        key_factors = priority_analysis.get("key_factors", [])
        
        if score >= 85:
            base_text = "Exceptional lead quality with multiple strong indicators."
        elif score >= 70:
            base_text = "High-priority lead with strong conversion potential."
        elif score >= 50:
            base_text = "Solid lead with moderate conversion potential."
        else:
            base_text = "Lead requires qualification and data enrichment."
        
        if key_factors:
            factor_text = f" Key strengths include {', '.join(key_factors[:2])}."
            if len(key_factors) > 2:
                factor_text += f" Additional positive indicators: {', '.join(key_factors[2:4])}."
        else:
            factor_text = " Multiple factors contribute to this assessment."
        
        return base_text + factor_text
    
    def _analyze_contact_quality(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze contact information quality"""
        score = 0
        factors = []
        issues = []
        
        email = lead_data.get('email', '')
        phone = lead_data.get('phone', '')
        first_name = lead_data.get('first_name', '')
        last_name = lead_data.get('last_name', '')
        
        # Email analysis
        if email:
            if '@' in email and '.' in email.split('@')[1]:
                score += 30
                if not any(generic in email.lower() for generic in ['info@', 'contact@', 'sales@']):
                    score += 15
                    factors.append("Personal email address")
                
                # Professional domain check
                domain = email.split('@')[1].lower()
                if domain not in ['gmail.com', 'yahoo.com', 'hotmail.com']:
                    score += 10
                    factors.append("Business email domain")
            else:
                issues.append("Invalid email format")
        else:
            issues.append("No email address")
        
        # Phone analysis
        if phone:
            score += 20
            factors.append("Phone number available")
        else:
            issues.append("No phone number")
        
        # Name completeness
        if first_name and last_name:
            score += 15
            factors.append("Complete name information")
        elif first_name or last_name:
            score += 5
        else:
            issues.append("Incomplete name information")
        
        return {
            "score": min(100, score),
            "factors": factors,
            "issues": issues
        }
    
    def _analyze_company_profile(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze company profile strength"""
        score = 0
        factors = []
        
        company_data = lead_data.get('company', {})
        if isinstance(company_data, dict):
            company_name = company_data.get('name', '')
            industry = company_data.get('industry', '')
            company_size = company_data.get('size', '')
            website = company_data.get('website', '')
        else:
            company_name = str(company_data) if company_data else ''
            industry = ''
            company_size = ''
            website = ''
        
        # Company name
        if company_name:
            score += 20
            factors.append("Company identified")
            
            # Look for indicators of established business
            if any(indicator in company_name.lower() for indicator in ['inc', 'corp', 'ltd', 'llc']):
                score += 10
                factors.append("Established business structure")
        
        # Industry analysis
        if industry:
            score += 20
            factors.append(f"Industry: {industry}")
            
            # High-value industries
            high_value = ['technology', 'software', 'saas', 'fintech', 'healthcare', 'consulting']
            if any(hv in industry.lower() for hv in high_value):
                score += 15
                factors.append("High-value industry")
        
        # Company size
        if company_size:
            score += 15
            factors.append(f"Company size: {company_size}")
            
            # Optimal company sizes (have budget but not too complex)
            optimal_sizes = ['51-200', '201-500', '101-250', '251-500']
            if company_size in optimal_sizes:
                score += 10
                factors.append("Optimal company size")
        
        # Website
        if website:
            score += 10
            factors.append("Website available")
        
        return {
            "score": min(100, score),
            "factors": factors
        }
    
    def _analyze_role_seniority(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze role seniority and decision-making power"""
        score = 0
        factors = []
        
        title = lead_data.get('title', '').lower()
        
        if not title:
            return {"score": 20, "factors": ["Role title not specified"]}
        
        # Executive level
        executive_keywords = ['ceo', 'cto', 'cfo', 'founder', 'president', 'owner']
        if any(keyword in title for keyword in executive_keywords):
            score = 95
            factors.append("C-level executive")
            factors.append("Ultimate decision-making authority")
        
        # VP/SVP level
        elif any(vp in title for vp in ['vp', 'vice president', 'svp']):
            score = 85
            factors.append("VP-level executive")
            factors.append("High decision-making influence")
        
        # Director level
        elif 'director' in title or 'head of' in title:
            score = 75
            factors.append("Director-level role")
            factors.append("Department decision-making authority")
        
        # Manager level
        elif any(mgr in title for mgr in ['manager', 'lead', 'principal']):
            score = 60
            factors.append("Management role")
            factors.append("Team leadership position")
        
        # Senior individual contributor
        elif any(senior in title for senior in ['senior', 'sr.', 'architect', 'specialist']):
            score = 45
            factors.append("Senior individual contributor")
        
        # Entry/mid level
        else:
            score = 30
            factors.append("Individual contributor role")
        
        return {
            "score": score,
            "factors": factors
        }
    
    def _analyze_timing_factors(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze timing-related factors"""
        score = 50  # Base score
        factors = []
        
        # Check when lead was created/updated
        created_at = lead_data.get('created_at')
        if created_at:
            # Recent leads get higher urgency
            score += 20
            factors.append("Fresh lead - optimal timing window")
        
        # Check lead source for urgency indicators
        source = lead_data.get('source', '').lower()
        if source in ['demo_request', 'pricing_inquiry', 'contact_form']:
            score += 25
            factors.append("High-intent source")
        elif source in ['linkedin', 'referral']:
            score += 10
            factors.append("Warm source")
        
        # Current month/quarter factors (you could expand this)
        current_month = datetime.now().month
        if current_month in [3, 6, 9, 12]:  # Quarter end
            score += 15
            factors.append("Quarter-end timing advantage")
        
        return {
            "score": min(100, score),
            "factors": factors
        }
    
    def _calculate_confidence(self, contact_analysis, company_analysis, role_analysis) -> float:
        """Calculate confidence in the analysis based on data completeness"""
        
        # Base confidence on data completeness
        contact_completeness = len(contact_analysis["factors"]) / 4  # Max 4 contact factors
        company_completeness = len(company_analysis["factors"]) / 5  # Max 5 company factors  
        role_completeness = 1.0 if role_analysis["score"] > 30 else 0.5  # Role specified or not
        
        overall_completeness = (contact_completeness + company_completeness + role_completeness) / 3
        
        # Convert to confidence score (0.3 to 0.95 range)
        confidence = 0.3 + (overall_completeness * 0.65)
        
        return round(confidence, 2)
    
    def _extract_key_factors(self, contact_analysis, company_analysis, role_analysis, timing_analysis) -> List[str]:
        """Extract the most important factors driving the score"""
        
        key_factors = []
        
        # Add top factors from each category
        if role_analysis["score"] >= 75:
            key_factors.extend(role_analysis["factors"][:1])
        
        if company_analysis["score"] >= 70:
            key_factors.extend([f for f in company_analysis["factors"] if "high-value" in f.lower() or "optimal" in f.lower()][:1])
        
        if contact_analysis["score"] >= 70:
            key_factors.extend([f for f in contact_analysis["factors"] if "business" in f.lower() or "personal" in f.lower()][:1])
        
        if timing_analysis["score"] >= 70:
            key_factors.extend(timing_analysis["factors"][:1])
        
        return key_factors[:4]  # Limit to top 4 factors
    
    def _generate_suggested_actions(self, insights: Dict[str, Any]) -> List[str]:
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

    def _estimate_response_rate(self, insights: Dict[str, Any]) -> float:
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