# utils/enhanced_lead_scorer.py
import re
import random
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LeadScorer:
    """
    Advanced lead scoring system with realistic variation, comprehensive analysis,
    and AI integration capabilities
    """
    
    def __init__(self, use_ai_enhancement: bool = True):
        self.use_ai_enhancement = use_ai_enhancement
        
        # Scoring weights for different factors
        self.scoring_weights = {
            'email_available': 25,
            'phone_available': 15,
            'linkedin_profile': 20,
            'senior_title': 35,
            'company_size': 15,
            'industry_match': 20,
            'location_match': 10,
            'domain_quality': 15,
            'name_completeness': 10,
            'email_quality': 10,
            'data_freshness': 5,
            'social_presence': 8,
            'company_maturity': 12,
            'lead_source_quality': 8
        }
        
        # Title classifications for scoring
        self.executive_titles = [
            'ceo', 'cto', 'cfo', 'president', 'founder', 'chief executive',
            'chief technology', 'chief financial', 'chief marketing', 'chief operating',
            'chief product', 'chief data', 'chief innovation', 'chief revenue',
            'managing partner', 'executive chairman', 'board member'
        ]
        
        self.senior_titles = [
            'vp', 'vice president', 'director', 'head of', 'senior director',
            'executive director', 'managing director', 'general manager',
            'senior vice president', 'svp', 'principal', 'senior principal'
        ]
        
        self.manager_titles = [
            'manager', 'senior manager', 'lead', 'principal', 'team lead',
            'project manager', 'product manager', 'program manager',
            'engineering manager', 'marketing manager', 'sales manager',
            'operations manager', 'development manager'
        ]
        
        # High-value industries for scoring (updated for 2024/2025)
        self.target_industries = {
            'tier_1': [  # Highest value - Hot sectors
                'artificial intelligence', 'machine learning', 'ai', 'ml',
                'saas', 'fintech', 'cybersecurity', 'blockchain', 'cryptocurrency',
                'quantum computing', 'robotics', 'autonomous vehicles', 'space technology'
            ],
            'tier_2': [  # High value - Established tech
                'technology', 'software', 'cloud computing', 'data science',
                'biotechnology', 'medtech', 'healthtech', 'edtech', 'proptech',
                'clean energy', 'renewable energy', 'electric vehicles'
            ],
            'tier_3': [  # Medium value - Traditional sectors
                'e-commerce', 'digital marketing', 'consulting', 'finance',
                'healthcare', 'telecommunications', 'automotive', 'manufacturing',
                'logistics', 'supply chain', 'agriculture tech'
            ]
        }
        
        # Premium domain extensions
        self.premium_domains = {
            'tier_1': ['.ai', '.io', '.tech', '.dev', '.app'],
            'tier_2': ['.com', '.co', '.inc'],
            'tier_3': ['.net', '.org', '.biz', '.info']
        }
        
        # Major tech hubs and business centers for location scoring
        self.tech_hubs = {
            'tier_1': [  # Top global tech cities
                'san francisco', 'palo alto', 'mountain view', 'cupertino',
                'seattle', 'new york', 'boston', 'austin', 'los angeles',
                'london', 'berlin', 'tel aviv', 'singapore', 'shenzhen',
                'bangalore', 'toronto', 'vancouver'
            ],
            'tier_2': [  # Major business centers
                'chicago', 'denver', 'atlanta', 'dallas', 'miami',
                'amsterdam', 'paris', 'munich', 'zurich', 'stockholm',
                'copenhagen', 'sydney', 'melbourne', 'tokyo', 'seoul',
                'hong kong', 'dublin', 'barcelona', 'milan'
            ],
            'tier_3': [  # Other developed markets
                'phoenix', 'philadelphia', 'washington dc', 'portland',
                'edinburgh', 'manchester', 'warsaw', 'prague', 'vienna',
                'helsinki', 'oslo', 'lisbon', 'madrid', 'rome'
            ]
        }
        
        # Email pattern quality indicators
        self.professional_email_patterns = [
            r'^[a-z]+\.[a-z]+@',  # firstname.lastname
            r'^[a-z]+[a-z]+@',    # firstnamelastname
            r'^[a-z]\.[a-z]+@',   # f.lastname
            r'^[a-z]+\.[a-z]@'    # firstname.l
        ]
        
        self.generic_email_patterns = [
            'info@', 'contact@', 'hello@', 'support@', 'admin@',
            'sales@', 'marketing@', 'hr@', 'jobs@', 'team@'
        ]
        
        # Company size value mapping
        self.company_size_values = {
            '1-10': 0.4,
            '11-50': 0.6,
            '51-200': 0.8,
            '201-500': 1.0,
            '501-1000': 1.2,
            '1000+': 1.4,
            '5000+': 1.6,
            '10000+': 1.8
        }
        
        # Source quality multipliers
        self.source_quality_multipliers = {
            'linkedin': 1.3,
            'hunter': 1.2,
            'website': 1.1,
            'referral': 1.4,
            'event': 1.2,
            'webinar': 1.1,
            'directory': 1.0,
            'google': 0.9,
            'manual': 1.0,
            'import': 0.8,
            'purchased': 0.6
        }
    
    def calculate_score(self, lead_data: Dict[str, Any], context: Optional[Dict] = None) -> float:
        """
        Calculate comprehensive lead score with realistic variation and context awareness
        
        Args:
            lead_data: Dictionary containing lead information
            context: Optional context for scoring adjustments
            
        Returns:
            Float score between 5-95
        """
        try:
            score = 0.0
            scoring_details = {}
            
            # Base randomization for natural variation (±3 points)
            base_variation = random.uniform(-3, 3)
            score += base_variation
            scoring_details['base_variation'] = base_variation
            
            # Core scoring components
            email_score = self._score_email(lead_data)
            contact_score = self._score_contact_info(lead_data)
            title_score = self._score_title(lead_data)
            company_score = self._score_company(lead_data)
            name_score = self._score_name_completeness(lead_data)
            source_score = self._score_source_quality(lead_data)
            
            # Advanced scoring components
            freshness_score = self._score_data_freshness(lead_data)
            social_score = self._score_social_presence(lead_data)
            engagement_score = self._score_engagement_potential(lead_data)
            
            # Aggregate scores
            component_scores = {
                'email': email_score,
                'contact': contact_score,
                'title': title_score,
                'company': company_score,
                'name': name_score,
                'source': source_score,
                'freshness': freshness_score,
                'social': social_score,
                'engagement': engagement_score
            }
            
            total_component_score = sum(component_scores.values())
            score += total_component_score
            
            # Context-aware adjustments
            if context:
                context_adjustment = self._apply_context_adjustments(lead_data, context)
                score += context_adjustment
                scoring_details['context_adjustment'] = context_adjustment
            
            # AI enhancement if enabled
            if self.use_ai_enhancement:
                ai_adjustment = self._apply_ai_enhancement(lead_data, component_scores)
                score += ai_adjustment
                scoring_details['ai_enhancement'] = ai_adjustment
            
            # Final randomization for realism (±2 points)
            final_variation = random.uniform(-2, 2)
            score += final_variation
            
            # Industry and timing bonuses
            industry_bonus = self._calculate_industry_bonus(lead_data)
            timing_bonus = self._calculate_timing_bonus(lead_data)
            score += industry_bonus + timing_bonus
            
            # Normalize and bound score
            max_possible_score = sum(self.scoring_weights.values()) + 50  # Account for bonuses
            normalized_score = (score / max_possible_score) * 100
            
            # Ensure score is within realistic bounds (5-95)
            final_score = max(5, min(95, normalized_score))
            
            # Store detailed scoring breakdown
            scoring_details.update({
                'component_scores': component_scores,
                'industry_bonus': industry_bonus,
                'timing_bonus': timing_bonus,
                'final_score': final_score,
                'max_possible': max_possible_score,
                'normalization_factor': normalized_score / final_score if final_score > 0 else 1
            })
            
            # Cache scoring details for explanation
            self._last_scoring_details = scoring_details
            
            return round(final_score, 1)
            
        except Exception as e:
            logger.error(f"Error calculating lead score: {e}")
            return 50.0  # Return neutral score on error
    
    def _score_email(self, lead_data: Dict[str, Any]) -> float:
        """Enhanced email scoring with deliverability and pattern analysis"""
        email = lead_data.get('email', '')
        if not email:
            return 0
        
        email_score = self.scoring_weights['email_available']
        email_lower = email.lower()
        
        # Professional email patterns get bonus
        for pattern in self.professional_email_patterns:
            if re.match(pattern, email_lower):
                if 'firstname.lastname' in pattern:
                    email_score += 12  # Highest bonus for full name format
                else:
                    email_score += 8
                break
        
        # Generic emails get penalty
        for generic_pattern in self.generic_email_patterns:
            if generic_pattern in email_lower:
                email_score -= 15 if generic_pattern in ['info@', 'contact@'] else 10
                break
        
        # Domain-based scoring
        domain = email.split('@')[1] if '@' in email else ''
        if domain:
            domain_score = self._score_email_domain(domain)
            email_score += domain_score
        
        # Email length and complexity
        if len(email) > 50:
            email_score -= 3  # Very long emails might be fake
        elif len(email) < 10:
            email_score -= 5  # Very short emails might be incomplete
        
        # Check for numbers in email (often indicates lower quality)
        if re.search(r'\d{3,}', email):  # 3+ consecutive digits
            email_score -= 3
        
        return max(0, email_score)
    
    def _score_email_domain(self, domain: str) -> float:
        """Score email domain quality"""
        domain_score = 0
        
        # Free email providers (lower score)
        free_providers = [
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
            'aol.com', 'icloud.com', 'protonmail.com', 'mail.com',
            'yandex.com', 'qq.com', '163.com'
        ]
        
        if domain.lower() in free_providers:
            domain_score -= 8  # Penalty for free email
        else:
            domain_score += 5  # Bonus for corporate email
        
        # Premium domain extensions
        for tier, extensions in self.premium_domains.items():
            if any(domain.endswith(ext) for ext in extensions):
                if tier == 'tier_1':
                    domain_score += 8
                elif tier == 'tier_2':
                    domain_score += 5
                else:
                    domain_score += 2
                break
        
        # Domain age estimation (longer domains often more established)
        if len(domain.split('.')[0]) >= 8:  # Main part is 8+ chars
            domain_score += 2
        
        return domain_score
    
    def _score_contact_info(self, lead_data: Dict[str, Any]) -> float:
        """Enhanced contact information scoring"""
        score = 0
        
        # Phone availability with format validation
        phone = lead_data.get('phone', '')
        if phone:
            phone_score = self.scoring_weights['phone_available']
            
            # Validate phone format
            phone_clean = re.sub(r'[^\d+]', '', phone)
            if len(phone_clean) >= 10:  # Valid phone length
                phone_score += 3
            if phone.startswith('+'):  # International format
                phone_score += 2
            
            score += phone_score
        
        # LinkedIn profile with URL validation
        linkedin_url = lead_data.get('linkedin_url', '')
        if linkedin_url:
            linkedin_score = self.scoring_weights['linkedin_profile']
            
            # Quality check for LinkedIn URL
            if 'linkedin.com/in/' in linkedin_url.lower():
                linkedin_score += 8  # Personal profile
            elif 'linkedin.com/company/' in linkedin_url.lower():
                linkedin_score += 5  # Company page
            elif 'linkedin.com' in linkedin_url.lower():
                linkedin_score += 3  # Other LinkedIn URL
            
            # Check for custom LinkedIn URL (indicates active user)
            if '/in/' in linkedin_url and not re.search(r'/in/[a-z]+-[a-z]+-\w{8}', linkedin_url):
                linkedin_score += 4  # Custom URL bonus
            
            score += linkedin_score
        
        # Additional social media presence
        social_urls = lead_data.get('social_media', {})
        if isinstance(social_urls, dict):
            for platform, url in social_urls.items():
                if url and platform.lower() in ['twitter', 'github', 'facebook']:
                    score += 2  # Small bonus per additional platform
        
        return score
    
    def _score_title(self, lead_data: Dict[str, Any]) -> float:
        """Enhanced job title and seniority scoring"""
        title = lead_data.get('title', '').lower()
        if not title:
            return 0
        
        title_score = 0
        seniority_level = 'junior'
        
        # Executive level (highest scoring)
        if any(exec_title in title for exec_title in self.executive_titles):
            title_score = self.scoring_weights['senior_title'] + 25
            seniority_level = 'executive'
            
            # CEO/Founder get extra bonus
            if any(top_title in title for top_title in ['ceo', 'founder', 'president', 'managing partner']):
                title_score += 15
        
        # Senior level
        elif any(senior_title in title for senior_title in self.senior_titles):
            title_score = self.scoring_weights['senior_title'] + 10
            seniority_level = 'senior'
            
            # VP gets extra bonus
            if 'vp' in title or 'vice president' in title:
                title_score += 8
        
        # Manager level
        elif any(manager_title in title for manager_title in self.manager_titles):
            title_score = self.scoring_weights['senior_title'] * 0.8
            seniority_level = 'manager'
            
            # Senior managers get bonus
            if 'senior' in title or 'lead' in title or 'principal' in title:
                title_score += 6
        
        # Individual contributor
        else:
            title_score = self.scoring_weights['senior_title'] * 0.4
            seniority_level = 'individual'
            
            # Senior ICs get some bonus
            if 'senior' in title or 'principal' in title or 'lead' in title:
                title_score += 10
        
        # Department/function bonuses
        high_value_functions = [
            'engineering', 'product', 'data', 'ai', 'ml', 'software', 'technical',
            'revenue', 'growth', 'business development', 'strategy', 'innovation'
        ]
        
        if any(func in title for func in high_value_functions):
            title_score += random.uniform(3, 8)
        
        # Title specificity bonus (longer, more detailed titles)
        title_words = len(title.split())
        if title_words >= 4:
            title_score += random.uniform(3, 7)
        elif title_words >= 3:
            title_score += random.uniform(1, 4)
        
        # Store seniority level for later use
        lead_data['_seniority_level'] = seniority_level
        
        return title_score
    
    def _score_company(self, lead_data: Dict[str, Any]) -> float:
        """Enhanced company-related scoring"""
        company = lead_data.get('company', {})
        if not company:
            return 0
        
        score = 0
        
        # Industry scoring
        score += self._score_industry(company)
        
        # Company size scoring with enhanced logic
        score += self._score_company_size(company)
        
        # Location scoring
        score += self._score_location(company)
        
        # Domain quality scoring
        score += self._score_domain_quality(company)
        
        # Company maturity indicators
        score += self._score_company_maturity(company)
        
        # Funding and growth indicators
        score += self._score_company_growth(company)
        
        return score
    
    def _score_industry(self, company: Dict[str, Any]) -> float:
        """Enhanced industry scoring with trend awareness"""
        industry = company.get('industry', '').lower()
        if not industry:
            return 0
        
        # Check tier 1 industries (highest value - trending sectors)
        if any(target in industry for target in self.target_industries['tier_1']):
            industry_score = self.scoring_weights['industry_match'] + 20
            
            # Extra bonus for hottest AI/ML industries
            if any(hot in industry for hot in ['ai', 'artificial intelligence', 'machine learning', 'ml']):
                industry_score += 12
            
            # Fintech bonus
            if 'fintech' in industry or 'financial technology' in industry:
                industry_score += 10
        
        # Check tier 2 industries
        elif any(target in industry for target in self.target_industries['tier_2']):
            industry_score = self.scoring_weights['industry_match'] + 10
            
            # SaaS bonus
            if 'saas' in industry or 'software as a service' in industry:
                industry_score += 8
        
        # Check tier 3 industries
        elif any(target in industry for target in self.target_industries['tier_3']):
            industry_score = self.scoring_weights['industry_match'] + 5
        
        # Other industries
        else:
            industry_score = self.scoring_weights['industry_match'] * 0.5
            
            # Small bonus for traditional but stable industries
            stable_industries = ['banking', 'insurance', 'pharmaceuticals', 'utilities']
            if any(stable in industry for stable in stable_industries):
                industry_score += 3
        
        return industry_score
    
    def _score_company_size(self, company: Dict[str, Any]) -> float:
        """Enhanced company size scoring"""
        company_size = company.get('size', '')
        if not company_size:
            return 0
        
        base_score = self.scoring_weights['company_size']
        
        # Use mapping for consistent scoring
        for size_range, multiplier in self.company_size_values.items():
            if size_range in company_size:
                size_score = base_score * multiplier
                break
        else:
            # Default for unknown sizes
            size_score = base_score * 0.6
        
        # Special handling for startups in hot industries
        if ('1-10' in company_size or '11-50' in company_size):
            industry = company.get('industry', '').lower()
            if any(hot in industry for hot in ['ai', 'fintech', 'saas', 'blockchain']):
                size_score += 8  # Startup bonus in hot sectors
        
        return size_score
    
    def _score_location(self, company: Dict[str, Any]) -> float:
        """Enhanced location scoring with global awareness"""
        location = company.get('location', '').lower()
        if not location:
            return 0
        
        location_score = 0
        
        # Tier 1 tech hubs and global business centers
        if any(hub in location for hub in self.tech_hubs['tier_1']):
            location_score = self.scoring_weights['location_match'] + 12
        
        # Tier 2 major cities
        elif any(hub in location for hub in self.tech_hubs['tier_2']):
            location_score = self.scoring_weights['location_match'] + 6
        
        # Tier 3 developed markets
        elif any(hub in location for hub in self.tech_hubs['tier_3']):
            location_score = self.scoring_weights['location_match'] + 2
        
        # English-speaking countries bonus
        elif any(country in location for country in ['usa', 'canada', 'uk', 'australia', 'new zealand']):
            location_score = self.scoring_weights['location_match'] * 0.8
        
        # EU countries
        elif any(country in location for country in ['germany', 'france', 'netherlands', 'sweden', 'denmark']):
            location_score = self.scoring_weights['location_match'] * 0.7
        
        # Other locations
        else:
            location_score = self.scoring_weights['location_match'] * 0.4
        
        # Remote work bonus
        if 'remote' in location or 'distributed' in location:
            location_score += 5
        
        return location_score
    
    def _score_domain_quality(self, company: Dict[str, Any]) -> float:
        """Enhanced domain quality assessment"""
        domain = company.get('domain', '').lower()
        if not domain:
            return 0
        
        domain_score = 0
        
        # Premium TLD scoring
        for tier, extensions in self.premium_domains.items():
            if any(domain.endswith(ext) for ext in extensions):
                if tier == 'tier_1':
                    domain_score = self.scoring_weights['domain_quality'] + 8
                elif tier == 'tier_2':
                    domain_score = self.scoring_weights['domain_quality'] + 3
                else:
                    domain_score = self.scoring_weights['domain_quality']
                break
        else:
            domain_score = self.scoring_weights['domain_quality'] * 0.4
        
        # Domain name quality analysis
        domain_name = domain.split('.')[0]
        
        # Optimal domain length (brand recognition sweet spot)
        if 3 <= len(domain_name) <= 10:
            domain_score += 6
        elif 11 <= len(domain_name) <= 15:
            domain_score += 3
        elif len(domain_name) > 25:
            domain_score -= 4
        
        # Penalize domains with numbers or hyphens (less professional)
        if re.search(r'\d', domain_name):
            domain_score -= 3
        if '-' in domain_name:
            domain_score -= 2
        
        # Bonus for dictionary words or recognizable terms
        if len(domain_name) <= 12 and domain_name.isalpha():
            domain_score += 4
        
        return domain_score
    
    def _score_company_maturity(self, company: Dict[str, Any]) -> float:
        """Score company maturity indicators"""
        maturity_score = 0
        
        # Website quality indicators
        website = company.get('website', '')
        if website:
            maturity_score += 3
            
            # HTTPS bonus
            if website.startswith('https://'):
                maturity_score += 2
        
        # Description quality
        description = company.get('description', '')
        if description and len(description) > 50:
            maturity_score += 4
            
            # Professional language indicators
            professional_terms = ['leading', 'innovative', 'established', 'founded', 'expertise']
            if any(term in description.lower() for term in professional_terms):
                maturity_score += 2
        
        # Social media presence
        social_media = company.get('social_media', {})
        if isinstance(social_media, dict) and social_media:
            maturity_score += len(social_media.keys()) * 1.5  # Bonus per platform
        
        return min(maturity_score, 15)  # Cap maturity bonus
    
    def _score_company_growth(self, company: Dict[str, Any]) -> float:
        """Score company growth and funding indicators"""
        growth_score = 0
        
        # Revenue indicators
        revenue = company.get('revenue', '')
        if revenue:
            growth_score += 5
            
            # High revenue bonus
            if any(indicator in revenue.lower() for indicator in ['million', 'billion', 'm+', 'b+']):
                growth_score += 8
        
        # Funding stage indicators
        description = company.get('description', '').lower()
        funding_indicators = ['series a', 'series b', 'series c', 'funded', 'investment', 'venture']
        if any(indicator in description for indicator in funding_indicators):
            growth_score += 6
        
        # Technology stack (if available)
        technologies = company.get('technologies', [])
        if technologies:
            tech_count = len(technologies) if isinstance(technologies, list) else 1
            growth_score += min(tech_count * 0.5, 5)  # Bonus for tech diversity
        
        return growth_score
    
    def _score_name_completeness(self, lead_data: Dict[str, Any]) -> float:
        """Enhanced name completeness and quality scoring"""
        first_name = lead_data.get('first_name', '')
        last_name = lead_data.get('last_name', '')
        
        if not first_name and not last_name:
            return 0
        
        name_score = 0
        
        # Both names available
        if first_name and last_name:
            name_score = self.scoring_weights['name_completeness']
            
            # Quality checks
            if (2 <= len(first_name) <= 20 and 2 <= len(last_name) <= 25 and
                first_name.replace('-', '').replace("'", "").isalpha() and 
                last_name.replace('-', '').replace("'", "").isalpha()):
                name_score += 6
                
                # Bonus for reasonable name lengths
                if 3 <= len(first_name) <= 12 and 3 <= len(last_name) <= 15:
                    name_score += 3
        
        # Only one name
        elif first_name or last_name:
            name_score = self.scoring_weights['name_completeness'] * 0.6
            
            single_name = first_name or last_name
            if len(single_name) >= 2 and single_name.replace('-', '').replace("'", "").isalpha():
                name_score += 2
        
        # Penalty for obviously fake names
        full_name = f"{first_name} {last_name}".lower()
        fake_indicators = ['test', 'example', 'demo', 'sample', 'admin', 'user']
        if any(fake in full_name for fake in fake_indicators):
            name_score -= 10
        
        return max(0, name_score)
    
    def _score_source_quality(self, lead_data: Dict[str, Any]) -> float:
        """Enhanced source quality scoring"""
        source = lead_data.get('source', '').lower()
        
        # Base score from source multiplier
        multiplier = self.source_quality_multipliers.get(source, 1.0)
        base_score = self.scoring_weights['lead_source_quality'] * multiplier
        
        # Additional source-specific bonuses
        if source == 'referral':
            base_score += 5  # Referrals are high quality
        elif source == 'event':
            base_score += 3  # Event leads show intent
        elif source == 'webinar':
            base_score += 3  # Educational engagement
        elif source == 'organic':
            base_score += 4  # Found you naturally
        
        return base_score
    
    def _score_data_freshness(self, lead_data: Dict[str, Any]) -> float:
        """Score data freshness and recency"""
        freshness_score = self.scoring_weights['data_freshness']
        
        # Check for creation date
        created_at = lead_data.get('created_at')
        if created_at:
            try:
                if isinstance(created_at, str):
                    created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                else:
                    created_date = created_at
                
                days_old = (datetime.utcnow() - created_date.replace(tzinfo=None)).days
                
                if days_old <= 7:
                    freshness_score += 5  # Very fresh
                elif days_old <= 30:
                    freshness_score += 3  # Fresh
                elif days_old <= 90:
                    freshness_score += 1  # Moderately fresh
                else:
                    freshness_score -= 2  # Old data
                    
            except Exception:
                pass  # Invalid date format
        
        return freshness_score
    
    def _score_social_presence(self, lead_data: Dict[str, Any]) -> float:
        """Score social media presence and digital footprint"""
        social_score = 0
        
        # LinkedIn presence (already scored in contact info)
        if lead_data.get('linkedin_url'):
            social_score += 3
        
        # Additional social platforms
        social_platforms = ['twitter_url', 'github_url', 'facebook_url']
        for platform in social_platforms:
            if lead_data.get(platform):
                social_score += 2
        
        # Social media data in company info
        company = lead_data.get('company', {})
        company_social = company.get('social_media', {})
        if isinstance(company_social, dict):
            social_score += len(company_social.keys()) * 1.5
        
        return min(social_score, self.scoring_weights['social_presence'])
    
    def _score_engagement_potential(self, lead_data: Dict[str, Any]) -> float:
        """Score potential for engagement based on available data"""
        engagement_score = 0
        
        # Multiple contact methods increase engagement potential
        contact_methods = 0
        if lead_data.get('email'):
            contact_methods += 1
        if lead_data.get('phone'):
            contact_methods += 1
        if lead_data.get('linkedin_url'):
            contact_methods += 1
        
        engagement_score += contact_methods * 3
        
        # Professional title suggests business engagement
        title = lead_data.get('title', '').lower()
        if title:
            decision_maker_keywords = [
                'director', 'manager', 'head', 'lead', 'chief', 'vp', 'president',
                'owner', 'founder', 'partner', 'executive'
            ]
            if any(keyword in title for keyword in decision_maker_keywords):
                engagement_score += 5
        
        # Company size affects engagement likelihood
        company = lead_data.get('company', {})
        company_size = company.get('size', '')
        if company_size:
            if '1-10' in company_size or '11-50' in company_size:
                engagement_score += 4  # Smaller companies more accessible
            elif '51-200' in company_size:
                engagement_score += 6  # Sweet spot for B2B
        
        return min(engagement_score, 20)  # Cap engagement score
    
    def _apply_context_adjustments(self, lead_data: Dict[str, Any], context: Dict) -> float:
        """Apply context-aware scoring adjustments"""
        adjustment = 0
        
        # Campaign context
        campaign_type = context.get('campaign_type')
        if campaign_type:
            if campaign_type == 'enterprise' and self._is_enterprise_lead(lead_data):
                adjustment += 10
            elif campaign_type == 'smb' and self._is_smb_lead(lead_data):
                adjustment += 8
        
        # Geographic context
        target_regions = context.get('target_regions', [])
        if target_regions:
            lead_location = lead_data.get('company', {}).get('location', '').lower()
            if any(region.lower() in lead_location for region in target_regions):
                adjustment += 5
        
        # Industry focus
        target_industries = context.get('target_industries', [])
        if target_industries:
            lead_industry = lead_data.get('company', {}).get('industry', '').lower()
            if any(industry.lower() in lead_industry for industry in target_industries):
                adjustment += 8
        
        # Timing context
        if context.get('urgent_campaign'):
            # Boost leads with immediate contact potential
            if lead_data.get('phone') and lead_data.get('email'):
                adjustment += 6
        
        return adjustment
    
    def _apply_ai_enhancement(self, lead_data: Dict[str, Any], component_scores: Dict) -> float:
        """Apply AI-based scoring enhancement"""
        try:
            # Simple AI enhancement - could be replaced with actual ML model
            ai_adjustment = 0
            
            # Pattern recognition adjustments
            if self._has_high_intent_signals(lead_data):
                ai_adjustment += 8
            
            if self._has_quality_indicators(lead_data, component_scores):
                ai_adjustment += 6
            
            # Behavioral pattern simulation
            if self._matches_successful_lead_pattern(lead_data):
                ai_adjustment += 10
            
            # Risk factor detection
            if self._has_risk_factors(lead_data):
                ai_adjustment -= 5
            
            return ai_adjustment
            
        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
            return 0
    
    def _calculate_industry_bonus(self, lead_data: Dict[str, Any]) -> float:
        """Calculate time-sensitive industry bonuses"""
        bonus = 0
        company = lead_data.get('company', {})
        industry = company.get('industry', '').lower()
        
        # 2024/2025 hot industries
        hot_trends = {
            'artificial intelligence': 15,
            'machine learning': 12,
            'generative ai': 18,
            'quantum computing': 20,
            'cybersecurity': 10,
            'fintech': 12,
            'climate tech': 14,
            'space technology': 16
        }
        
        for trend, bonus_value in hot_trends.items():
            if trend in industry:
                bonus += bonus_value
                break
        
        return bonus
    
    def _calculate_timing_bonus(self, lead_data: Dict[str, Any]) -> float:
        """Calculate timing-based bonuses"""
        bonus = 0
        current_time = datetime.utcnow()
        
        # Quarter-end bonus (higher urgency)
        month = current_time.month
        if month in [3, 6, 9, 12]:  # Quarter end months
            if current_time.day >= 25:  # Last week of quarter
                bonus += 5
        
        # Year-end bonus
        if month == 12 and current_time.day >= 15:
            bonus += 8
        
        # New year bonus (fresh budget)
        if month == 1 and current_time.day <= 31:
            bonus += 6
        
        # Business hours bonus for immediate follow-up
        if 9 <= current_time.hour <= 17:  # Business hours UTC
            bonus += 2
        
        return bonus
    
    def _is_enterprise_lead(self, lead_data: Dict[str, Any]) -> bool:
        """Determine if lead is enterprise-level"""
        company = lead_data.get('company', {})
        size = company.get('size', '')
        
        # Large company size
        if any(indicator in size for indicator in ['1000+', '5000+', '10000+']):
            return True
        
        # Enterprise title indicators
        title = lead_data.get('title', '').lower()
        enterprise_titles = ['chief', 'vp', 'vice president', 'director', 'head of']
        if any(et in title for et in enterprise_titles):
            return True
        
        return False
    
    def _is_smb_lead(self, lead_data: Dict[str, Any]) -> bool:
        """Determine if lead is small-medium business"""
        company = lead_data.get('company', {})
        size = company.get('size', '')
        
        return any(indicator in size for indicator in ['1-10', '11-50', '51-200'])
    
    def _has_high_intent_signals(self, lead_data: Dict[str, Any]) -> bool:
        """Detect high purchase intent signals"""
        # Title indicates decision-making authority
        title = lead_data.get('title', '').lower()
        decision_titles = ['ceo', 'cto', 'founder', 'owner', 'president', 'vp']
        
        if any(dt in title for dt in decision_titles):
            return True
        
        # Recent company growth signals
        company = lead_data.get('company', {})
        description = company.get('description', '').lower()
        growth_signals = ['growing', 'expanding', 'scaling', 'hiring', 'funding']
        
        if any(signal in description for signal in growth_signals):
            return True
        
        return False
    
    def _has_quality_indicators(self, lead_data: Dict[str, Any], component_scores: Dict) -> bool:
        """Detect overall quality indicators"""
        # High email and contact scores
        if (component_scores.get('email', 0) > 20 and 
            component_scores.get('contact', 0) > 15):
            return True
        
        # Professional domain and complete profile
        email = lead_data.get('email', '')
        if email and not any(free in email for free in ['gmail', 'yahoo', 'hotmail']):
            if lead_data.get('linkedin_url') and lead_data.get('phone'):
                return True
        
        return False
    
    def _matches_successful_lead_pattern(self, lead_data: Dict[str, Any]) -> bool:
        """Check if lead matches historically successful patterns"""
        # This would typically use ML model predictions
        # For now, using rule-based pattern matching
        
        success_indicators = 0
        
        # Tech industry + senior title
        company = lead_data.get('company', {})
        industry = company.get('industry', '').lower()
        title = lead_data.get('title', '').lower()
        
        if any(tech in industry for tech in ['technology', 'software', 'saas']):
            success_indicators += 1
        
        if any(senior in title for senior in ['director', 'vp', 'head', 'chief']):
            success_indicators += 1
        
        # Complete contact information
        if (lead_data.get('email') and lead_data.get('phone') and 
            lead_data.get('linkedin_url')):
            success_indicators += 1
        
        # Mid-size company (sweet spot)
        size = company.get('size', '')
        if any(size_range in size for size_range in ['51-200', '201-500']):
            success_indicators += 1
        
        return success_indicators >= 3
    
    def _has_risk_factors(self, lead_data: Dict[str, Any]) -> bool:
        """Detect potential risk factors"""
        risk_factors = 0
        
        # Generic email
        email = lead_data.get('email', '')
        if any(generic in email for generic in ['info@', 'contact@', 'admin@']):
            risk_factors += 1
        
        # Incomplete name information
        if not lead_data.get('first_name') or not lead_data.get('last_name'):
            risk_factors += 1
        
        # Very old or suspicious domain
        company = lead_data.get('company', {})
        domain = company.get('domain', '')
        if domain and (len(domain) > 30 or any(char in domain for char in ['_', '%'])):
            risk_factors += 1
        
        return risk_factors >= 2
    
    def get_score_explanation(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed explanation of score calculation with actionable insights
        """
        score = self.calculate_score(lead_data)
        explanation = {
            'total_score': score,
            'quality_tier': self._determine_quality_tier(score),
            'factors': [],
            'recommendations': [],
            'score_breakdown': getattr(self, '_last_scoring_details', {}),
            'improvement_potential': 0
        }
        
        # Analyze positive factors
        factors, recommendations = self._analyze_scoring_factors(lead_data, score)
        explanation['factors'] = factors
        explanation['recommendations'] = recommendations
        
        # Calculate improvement potential
        explanation['improvement_potential'] = self._calculate_improvement_potential(lead_data)
        
        return explanation
    
    def _determine_quality_tier(self, score: float) -> str:
        """Determine quality tier with more granular levels"""
        if score >= 90:
            return 'Exceptional'
        elif score >= 80:
            return 'Excellent'
        elif score >= 70:
            return 'High'
        elif score >= 55:
            return 'Medium'
        elif score >= 40:
            return 'Low'
        else:
            return 'Poor'
    
    def _analyze_scoring_factors(self, lead_data: Dict[str, Any], score: float) -> Tuple[List[str], List[str]]:
        """Analyze factors contributing to score and generate recommendations"""
        factors = []
        recommendations = []
        
        # Email analysis
        email = lead_data.get('email', '')
        if email:
            if any(pattern in email for pattern in ['.', '_']) and '@' in email:
                factors.append('✅ Professional email format')
            else:
                recommendations.append('🎯 Verify email format and deliverability')
        else:
            recommendations.append('🔍 Find email address to increase engagement potential')
        
        # Contact completeness
        contact_methods = sum([
            bool(lead_data.get('email')),
            bool(lead_data.get('phone')),
            bool(lead_data.get('linkedin_url'))
        ])
        
        if contact_methods >= 2:
            factors.append(f'✅ Multiple contact methods available ({contact_methods}/3)')
        else:
            recommendations.append('📞 Gather additional contact information')
        
        # Title analysis
        title = lead_data.get('title', '').lower()
        if any(exec in title for exec in ['ceo', 'cto', 'founder', 'president']):
            factors.append('🌟 Executive-level contact')
        elif any(senior in title for senior in ['director', 'vp', 'head']):
            factors.append('✅ Senior decision maker')
        elif not title:
            recommendations.append('💼 Identify job title for better targeting')
        
        # Company analysis
        company = lead_data.get('company', {})
        industry = company.get('industry', '').lower()
        
        if any(hot in industry for hot in ['ai', 'saas', 'fintech', 'technology']):
            factors.append('🚀 High-value industry sector')
        
        size = company.get('size', '')
        if any(good_size in size for good_size in ['51-200', '201-500', '501-1000']):
            factors.append('🎯 Optimal company size for B2B')
        elif '1-10' in size:
            factors.append('⚡ Startup - potentially agile decision making')
        
        # Location analysis
        location = company.get('location', '').lower()
        if any(hub in location for hub in ['san francisco', 'new york', 'london', 'seattle']):
            factors.append('🌆 Located in major business hub')
        
        # Data completeness
        completeness = self._calculate_data_completeness(lead_data)
        if completeness > 80:
            factors.append(f'📋 High data completeness ({completeness}%)')
        elif completeness < 60:
            recommendations.append('📊 Enrich lead data for better insights')
        
        # Source quality
        source = lead_data.get('source', '')
        if source in ['linkedin', 'referral', 'event']:
            factors.append(f'✅ High-quality source: {source}')
        elif source in ['purchased', 'import']:
            recommendations.append('🔍 Verify lead authenticity and interest')
        
        # AI-specific recommendations
        if self.use_ai_enhancement:
            ai_recommendations = self._generate_ai_recommendations(lead_data)
            recommendations.extend(ai_recommendations)
        
        return factors, recommendations
    
    def _calculate_improvement_potential(self, lead_data: Dict[str, Any]) -> float:
        """Calculate how much the score could potentially improve"""
        current_score = self.calculate_score(lead_data)
        
        # Simulate perfect data completion
        improved_data = lead_data.copy()
        
        # Add missing email
        if not improved_data.get('email'):
            improved_data['email'] = 'contact@company.com'
        
        # Add missing phone
        if not improved_data.get('phone'):
            improved_data['phone'] = '+1-555-123-4567'
        
        # Add missing LinkedIn
        if not improved_data.get('linkedin_url'):
            improved_data['linkedin_url'] = 'https://linkedin.com/in/contact'
        
        # Improve title if missing
        if not improved_data.get('title'):
            improved_data['title'] = 'Director'
        
        # Calculate potential score
        potential_score = self.calculate_score(improved_data)
        
        return round(potential_score - current_score, 1)
    
    def _calculate_data_completeness(self, lead_data: Dict[str, Any]) -> float:
        """Calculate data completeness percentage"""
        total_fields = 0
        completed_fields = 0
        
        # Core lead fields
        lead_fields = ['first_name', 'last_name', 'email', 'phone', 'title', 'linkedin_url']
        for field in lead_fields:
            total_fields += 1
            if lead_data.get(field):
                completed_fields += 1
        
        # Company fields
        company = lead_data.get('company', {})
        company_fields = ['name', 'domain', 'industry', 'size', 'location']
        for field in company_fields:
            total_fields += 1
            if company.get(field):
                completed_fields += 1
        
        return round((completed_fields / total_fields) * 100, 1) if total_fields > 0 else 0
    
    def _generate_ai_recommendations(self, lead_data: Dict[str, Any]) -> List[str]:
        """Generate AI-specific recommendations"""
        recommendations = []
        
        # Pattern-based recommendations
        if self._has_high_intent_signals(lead_data):
            recommendations.append('🤖 AI detected high purchase intent - prioritize for immediate contact')
        
        if not self._has_quality_indicators(lead_data, {}):
            recommendations.append('🤖 AI suggests data enrichment to improve lead quality')
        
        # Timing recommendations
        current_hour = datetime.utcnow().hour
        if current_hour < 9 or current_hour > 17:
            recommendations.append('🤖 AI recommends contacting during business hours (9-17 UTC)')
        
        # Industry-specific recommendations
        company = lead_data.get('company', {})
        industry = company.get('industry', '').lower()
        
        if 'ai' in industry or 'machine learning' in industry:
            recommendations.append('🤖 AI industry lead - emphasize technical capabilities')
        elif 'fintech' in industry:
            recommendations.append('🤖 Fintech lead - highlight security and compliance features')
        
        return recommendations
    
    def score_batch(self, leads: List[Dict[str, Any]], context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Score multiple leads efficiently with batch optimizations
        """
        scored_leads = []
        
        # Batch processing optimizations
        batch_context = context or {}
        batch_context['batch_processing'] = True
        
        for i, lead in enumerate(leads):
            try:
                lead_copy = lead.copy()
                
                # Calculate score with batch context
                score = self.calculate_score(lead, batch_context)
                lead_copy['score'] = score
                
                # Add batch-specific metadata
                lead_copy['batch_position'] = i
                lead_copy['scored_at'] = datetime.utcnow().isoformat()
                
                # Get explanation for high-value leads only (optimization)
                if score >= 70:
                    lead_copy['score_explanation'] = self.get_score_explanation(lead)
                
                scored_leads.append(lead_copy)
                
            except Exception as e:
                logger.error(f"Error scoring lead {i}: {e}")
                # Add lead with default score on error
                lead_copy = lead.copy()
                lead_copy['score'] = 50.0
                lead_copy['error'] = str(e)
                scored_leads.append(lead_copy)
        
        # Sort by score (highest first)
        scored_leads.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Add batch rankings
        for i, lead in enumerate(scored_leads):
            lead['batch_rank'] = i + 1
            lead['percentile'] = round((1 - i / len(scored_leads)) * 100, 1)
        
        return scored_leads
    
    def get_score_statistics(self, scores: List[float]) -> Dict[str, Any]:
        """
        Get comprehensive statistics about score distribution with insights
        """
        if not scores:
            return {'error': 'No scores provided'}
        
        scores_sorted = sorted(scores)
        n = len(scores)
        
        stats = {
            'total_leads': n,
            'average_score': round(sum(scores) / n, 1),
            'median_score': round(scores_sorted[n // 2], 1),
            'min_score': min(scores),
            'max_score': max(scores),
            'standard_deviation': round(self._calculate_std_dev(scores), 2),
            'score_distribution': {
                'exceptional': len([s for s in scores if s >= 90]),
                'excellent': len([s for s in scores if 80 <= s < 90]),
                'high': len([s for s in scores if 70 <= s < 80]),
                'medium': len([s for s in scores if 55 <= s < 70]),
                'low': len([s for s in scores if 40 <= s < 55]),
                'poor': len([s for s in scores if s < 40])
            },
            'percentiles': {
                '95th': round(scores_sorted[int(n * 0.95)], 1),
                '90th': round(scores_sorted[int(n * 0.9)], 1),
                '75th': round(scores_sorted[int(n * 0.75)], 1),
                '50th': round(scores_sorted[int(n * 0.5)], 1),
                '25th': round(scores_sorted[int(n * 0.25)], 1),
                '10th': round(scores_sorted[int(n * 0.1)], 1)
            }
        }
        
        # Add quality insights
        excellent_rate = ((stats['score_distribution']['exceptional'] + 
                          stats['score_distribution']['excellent']) / n) * 100
        high_quality_rate = excellent_rate + (stats['score_distribution']['high'] / n) * 100
        
        stats['quality_insights'] = {
            'excellent_rate': round(excellent_rate, 1),
            'high_quality_rate': round(high_quality_rate, 1),
            'overall_quality': self._determine_batch_quality(stats['average_score'], excellent_rate),
            'improvement_opportunities': self._identify_improvement_opportunities(stats)
        }
        
        # Performance benchmarks
        stats['benchmarks'] = {
            'industry_average': 65.0,  # Simulated benchmark
            'top_quartile_threshold': stats['percentiles']['75th'],
            'performance_vs_benchmark': round(stats['average_score'] - 65.0, 1)
        }
        
        return stats
    
    def _calculate_std_dev(self, scores: List[float]) -> float:
        """Calculate standard deviation of scores"""
        if len(scores) <= 1:
            return 0.0
        
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)
        return variance ** 0.5
    
    def _determine_batch_quality(self, avg_score: float, excellent_rate: float) -> str:
        """Determine overall batch quality"""
        if excellent_rate >= 30 and avg_score >= 80:
            return 'Exceptional'
        elif excellent_rate >= 20 and avg_score >= 75:
            return 'Excellent'
        elif excellent_rate >= 10 and avg_score >= 65:
            return 'High'
        elif avg_score >= 55:
            return 'Medium'
        else:
            return 'Low'
    
    def _identify_improvement_opportunities(self, stats: Dict) -> List[str]:
        """Identify opportunities for batch improvement"""
        opportunities = []
        
        poor_rate = (stats['score_distribution']['poor'] / stats['total_leads']) * 100
        low_rate = (stats['score_distribution']['low'] / stats['total_leads']) * 100
        
        if poor_rate > 20:
            opportunities.append('High percentage of poor quality leads - review data sources')
        
        if low_rate > 30:
            opportunities.append('Consider data enrichment to improve lead quality')
        
        if stats['average_score'] < 60:
            opportunities.append('Overall lead quality below average - implement qualification criteria')
        
        excellent_rate = stats['quality_insights']['excellent_rate']
        if excellent_rate < 10:
            opportunities.append('Low percentage of excellent leads - refine targeting criteria')
        
        return opportunities
    
    def benchmark_lead(self, lead_data: Dict[str, Any], comparison_leads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Benchmark a lead against a set of comparison leads with detailed analysis
        """
        lead_score = self.calculate_score(lead_data)
        comparison_scores = [self.calculate_score(comp_lead) for comp_lead in comparison_leads]
        
        if not comparison_scores:
            return {'error': 'No comparison leads provided'}
        
        # Calculate percentile rank
        better_count = len([score for score in comparison_scores if lead_score > score])
        percentile_rank = (better_count / len(comparison_scores)) * 100
        
        # Detailed ranking analysis
        ranking_tier = self._determine_ranking_tier(percentile_rank)
        
        # Comparative analysis
        avg_comparison_score = sum(comparison_scores) / len(comparison_scores)
        score_gap = lead_score - avg_comparison_score
        
        return {
            'lead_score': lead_score,
            'percentile_rank': round(percentile_rank, 1),
            'ranking_tier': ranking_tier,
            'better_than': f"{better_count} out of {len(comparison_scores)} leads",
            'score_vs_average': round(score_gap, 1),
            'competitive_advantage': self._analyze_competitive_advantage(lead_data, comparison_leads),
            'comparison_stats': self.get_score_statistics(comparison_scores),
            'improvement_recommendations': self._get_benchmark_recommendations(lead_score, percentile_rank)
        }
    
    def _determine_ranking_tier(self, percentile_rank: float) -> str:
        """Determine ranking tier based on percentile"""
        if percentile_rank >= 95:
            return 'Top 5%'
        elif percentile_rank >= 90:
            return 'Top 10%'
        elif percentile_rank >= 75:
            return 'Top 25%'
        elif percentile_rank >= 50:
            return 'Top 50%'
        else:
            return 'Bottom 50%'
    
    def _analyze_competitive_advantage(self, lead_data: Dict, comparison_leads: List[Dict]) -> List[str]:
        """Analyze what makes this lead stand out"""
        advantages = []
        
        # Title comparison
        lead_title = lead_data.get('title', '').lower()
        if any(exec in lead_title for exec in self.executive_titles):
            exec_count = sum(1 for comp in comparison_leads 
                           if any(exec in comp.get('title', '').lower() for exec in self.executive_titles))
            if exec_count / len(comparison_leads) < 0.3:
                advantages.append('Executive-level title (rare in comparison set)')
        
        # Industry advantage
        lead_industry = lead_data.get('company', {}).get('industry', '').lower()
        if any(hot in lead_industry for hot in self.target_industries['tier_1']):
            advantages.append('High-value industry sector')
        
        # Contact completeness
        lead_contacts = sum([bool(lead_data.get(field)) for field in ['email', 'phone', 'linkedin_url']])
        avg_contacts = sum(sum([bool(comp.get(field)) for field in ['email', 'phone', 'linkedin_url']]) 
                          for comp in comparison_leads) / len(comparison_leads)
        
        if lead_contacts > avg_contacts:
            advantages.append('Above-average contact information completeness')
        
        return advantages
    
    def _get_benchmark_recommendations(self, score: float, percentile: float) -> List[str]:
        """Get recommendations based on benchmark position"""
        recommendations = []
        
        if percentile >= 90:
            recommendations.append('Excellent lead - prioritize for immediate outreach')
        elif percentile >= 75:
            recommendations.append('High-quality lead - include in priority campaign')
        elif percentile >= 50:
            recommendations.append('Good lead - suitable for standard outreach')
        else:
            recommendations.append('Consider data enrichment before outreach')
        
        if score < 60:
            recommendations.append('Below average score - verify lead quality before investment')
        
        return recommendations
    
    def export_scoring_model(self) -> Dict[str, Any]:
        """Export the scoring model configuration for reproducibility"""
        return {
            'model_version': '2.0.0',
            'weights': self.scoring_weights,
            'title_classifications': {
                'executive': self.executive_titles,
                'senior': self.senior_titles,
                'manager': self.manager_titles
            },
            'industry_tiers': self.target_industries,
            'domain_tiers': self.premium_domains,
            'location_tiers': self.tech_hubs,
            'source_multipliers': self.source_quality_multipliers,
            'ai_enhancement_enabled': self.use_ai_enhancement,
            'export_timestamp': datetime.utcnow().isoformat()
        }
    
    def import_scoring_model(self, model_config: Dict[str, Any]) -> bool:
        """Import a scoring model configuration"""
        try:
            if 'weights' in model_config:
                self.scoring_weights.update(model_config['weights'])
            
            if 'source_multipliers' in model_config:
                self.source_quality_multipliers.update(model_config['source_multipliers'])
            
            if 'ai_enhancement_enabled' in model_config:
                self.use_ai_enhancement = model_config['ai_enhancement_enabled']
            
            logger.info(f"Imported scoring model version {model_config.get('model_version', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import scoring model: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    scorer = LeadScorer(use_ai_enhancement=True)
    
    # Test lead data
    test_lead = {
        'first_name': 'Sarah',
        'last_name': 'Chen',
        'email': 'sarah.chen@techcorp.ai',
        'phone': '+1-555-987-6543',
        'title': 'Chief Technology Officer',
        'linkedin_url': 'https://linkedin.com/in/sarah-chen-cto',
        'source': 'linkedin',
        'created_at': datetime.utcnow().isoformat(),
        'company': {
            'name': 'TechCorp AI',
            'domain': 'techcorp.ai',
            'industry': 'Artificial Intelligence',
            'size': '201-500',
            'location': 'San Francisco, CA',
            'website': 'https://techcorp.ai',
            'description': 'Leading AI company focused on enterprise solutions',
            'social_media': {
                'twitter': 'https://twitter.com/techcorpai',
                'linkedin': 'https://linkedin.com/company/techcorpai'
            }
        }
    }
    