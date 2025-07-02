# ai/training/data_generator.py - Synthetic Lead Data Generator for ML Training

# ============================
# IMPORTS
# ============================
import random
import json
import csv
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================
# ENUMS AND DATA CLASSES
# ============================

class IndustryType(Enum):
    """Industry categories"""
    TECHNOLOGY = "technology"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    CONSULTING = "consulting"
    REAL_ESTATE = "real_estate"
    MEDIA = "media"
    GOVERNMENT = "government"

class CompanySize(Enum):
    """Company size categories"""
    STARTUP = "1-10"
    SMALL = "11-50"
    MEDIUM = "51-200"
    LARGE = "201-500"
    ENTERPRISE = "501-1000"
    MEGA = "1001-5000"
    GIANT = "5000+"

class LeadSource(Enum):
    """Lead acquisition sources"""
    LINKEDIN = "linkedin"
    WEBSITE = "website"
    REFERRAL = "referral"
    EVENT = "event"
    WEBINAR = "webinar"
    EMAIL_CAMPAIGN = "email_campaign"
    COLD_OUTREACH = "cold_outreach"
    DIRECTORY = "directory"
    PURCHASED = "purchased"
    ORGANIC = "organic"

@dataclass
class GenerationConfig:
    """Configuration for data generation"""
    num_leads: int = 1000
    priority_distribution: Dict[str, float] = None
    quality_distribution: Dict[str, float] = None
    industry_weights: Dict[str, float] = None
    source_weights: Dict[str, float] = None
    output_format: str = "json"  # json, csv, pandas
    include_labels: bool = True
    noise_level: float = 0.1  # Amount of realistic noise/errors
    
    def __post_init__(self):
        if self.priority_distribution is None:
            self.priority_distribution = {"high": 0.2, "medium": 0.5, "low": 0.3}
        if self.quality_distribution is None:
            self.quality_distribution = {"excellent": 0.15, "good": 0.35, "fair": 0.35, "poor": 0.15}
        if self.industry_weights is None:
            self.industry_weights = {
                "technology": 0.3, "finance": 0.15, "healthcare": 0.12,
                "consulting": 0.1, "manufacturing": 0.08, "retail": 0.08,
                "education": 0.07, "media": 0.05, "real_estate": 0.03, "government": 0.02
            }
        if self.source_weights is None:
            self.source_weights = {
                "linkedin": 0.25, "website": 0.2, "referral": 0.15, "event": 0.1,
                "webinar": 0.08, "directory": 0.07, "cold_outreach": 0.06,
                "email_campaign": 0.05, "purchased": 0.03, "organic": 0.01
            }

# ============================
# MAIN DATA GENERATOR CLASS
# ============================

class LeadDataGenerator:
    """
    Generates realistic synthetic lead data for ML training and testing
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """
        Initialize the data generator
        
        Args:
            config: Generation configuration
        """
        self.config = config or GenerationConfig()
        self._load_data_sources()
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        logger.info("🎲 Lead Data Generator initialized")
    
    def _load_data_sources(self):
        """Load realistic data sources for generation"""
        
        # First names
        self.first_names = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
            "David", "Elizabeth", "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
            "Thomas", "Sarah", "Christopher", "Karen", "Charles", "Helen", "Daniel", "Nancy",
            "Matthew", "Betty", "Anthony", "Dorothy", "Mark", "Lisa", "Donald", "Anna",
            "Steven", "Kimberly", "Paul", "Rebecca", "Andrew", "Sharon", "Kenneth", "Michelle",
            "Alexander", "Laura", "Brian", "Emily", "George", "Donna", "Timothy", "Carol",
            "Ronald", "Ruth", "Jason", "Sandra", "Edward", "Maria", "Jeffrey", "Kate",
            "Ryan", "Heather", "Jacob", "Amy", "Gary", "Angela", "Nicholas", "Nicole",
            "Eric", "Brenda", "Jonathan", "Emma", "Stephen", "Catherine", "Larry", "Frances",
            "Justin", "Christine", "Scott", "Samantha", "Brandon", "Deborah", "Benjamin", "Rachel",
            "Samuel", "Carolyn", "Frank", "Janet", "Gregory", "Virginia", "Raymond", "Madison",
            "Jack", "Sophie", "Dennis", "Hannah", "Jerry", "Grace", "Tyler", "Olivia"
        ]
        
        # Last names
        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
            "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
            "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
            "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
            "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
            "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
            "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
            "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
            "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey",
            "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson",
            "Watson", "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza"
        ]
        
        # Job titles by seniority level
        self.job_titles = {
            "executive": [
                "Chief Executive Officer", "Chief Technology Officer", "Chief Financial Officer",
                "Chief Marketing Officer", "Chief Operating Officer", "President", "Vice President",
                "Founder", "Co-Founder", "Managing Director", "Executive Director"
            ],
            "senior": [
                "Senior Vice President", "Senior Director", "Director of Engineering",
                "Director of Sales", "Director of Marketing", "Director of Operations",
                "Senior Manager", "Principal Engineer", "Senior Architect", "Head of Product",
                "Head of Sales", "Head of Marketing", "Head of HR", "Senior Consultant"
            ],
            "manager": [
                "Engineering Manager", "Sales Manager", "Marketing Manager", "Product Manager",
                "Project Manager", "Operations Manager", "Team Lead", "Technical Lead",
                "Business Development Manager", "Account Manager", "Regional Manager",
                "Program Manager", "Quality Manager", "Finance Manager"
            ],
            "individual": [
                "Software Engineer", "Senior Software Engineer", "Data Scientist", "Sales Representative",
                "Marketing Specialist", "Business Analyst", "Account Executive", "Consultant",
                "Architect", "Developer", "Designer", "Researcher", "Analyst", "Specialist",
                "Coordinator", "Associate", "Administrator", "Technician"
            ]
        }
        
        # Company names by industry
        self.company_names = {
            "technology": [
                "TechCorp Solutions", "DataFlow Systems", "CloudVision Technologies", "NextGen Software",
                "InnovateAI", "CyberShield Security", "QuantumCode Labs", "SmartCloud Inc",
                "DevOps Dynamics", "AlgoTech Innovations", "FutureStack Solutions", "CodeCraft Studios",
                "ByteStream Technologies", "DigitalForge Corp", "TechNova Systems", "DataLink Solutions"
            ],
            "finance": [
                "Capital Ventures Group", "Financial Solutions Inc", "InvestSmart Partners", "WealthTech Corp",
                "FinanceFlow Systems", "SecureBank Technologies", "MoneyTree Advisors", "CreditCore Solutions",
                "PaymentStream Inc", "RiskGuard Financial", "AssetMax Partners", "FinTech Innovations"
            ],
            "healthcare": [
                "MedTech Solutions", "HealthCare Innovations", "BioLife Systems", "WellCare Technologies",
                "MedData Corp", "HealthStream Solutions", "CarePlus Systems", "MedFlow Technologies",
                "BioTech Innovations", "HealthGuard Solutions", "MedCore Systems", "VitalCare Corp"
            ],
            "consulting": [
                "Strategic Consulting Group", "Business Solutions Partners", "Elite Advisors", "ProConsult Inc",
                "Excellence Consulting", "Growth Partners LLC", "Advisory Solutions Group", "Transform Consulting",
                "Strategy First Partners", "Insight Consulting Corp", "Peak Performance Advisors", "Success Partners"
            ],
            "manufacturing": [
                "Industrial Solutions Corp", "Manufacturing Excellence Inc", "Production Systems Ltd",
                "Quality Manufacturing Co", "AutoTech Industries", "Precision Manufacturing", "BuildTech Corp",
                "Assembly Line Solutions", "Industrial Innovations", "Manufacturing Plus", "Production Pro"
            ]
        }
        
        # Email domains by type
        self.email_domains = {
            "corporate": [".com", ".org", ".net", ".ai", ".io", ".tech", ".co"],
            "free": ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"]
        }
        
        # Cities and locations
        self.locations = [
            "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX", "Boston, MA",
            "Chicago, IL", "Denver, CO", "Atlanta, GA", "Los Angeles, CA", "Miami, FL",
            "Portland, OR", "Dallas, TX", "Phoenix, AZ", "Philadelphia, PA", "Detroit, MI",
            "Toronto, ON", "Vancouver, BC", "London, UK", "Berlin, Germany", "Amsterdam, Netherlands",
            "Singapore", "Sydney, Australia", "Tel Aviv, Israel", "Barcelona, Spain", "Dublin, Ireland"
        ]
        
        # Phone area codes
        self.area_codes = [
            "415", "650", "212", "646", "917", "206", "425", "512", "737", "617", "857",
            "312", "773", "303", "720", "404", "678", "213", "310", "323", "305", "786",
            "503", "971", "214", "469", "972", "602", "623", "215", "267", "313", "248"
        ]
    
    def generate_leads(self, num_leads: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate synthetic lead data
        
        Args:
            num_leads: Number of leads to generate (uses config default if None)
            
        Returns:
            List of lead dictionaries
        """
        num_leads = num_leads or self.config.num_leads
        leads = []
        
        logger.info(f"🎲 Generating {num_leads} synthetic leads...")
        
        for i in range(num_leads):
            lead = self._generate_single_lead(i + 1)
            leads.append(lead)
            
            # Progress logging
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_leads} leads")
        
        logger.info(f"✅ Generated {len(leads)} leads successfully")
        return leads
    
    def _generate_single_lead(self, lead_id: int) -> Dict[str, Any]:
        """Generate a single realistic lead"""
        
        # Basic demographics
        first_name = random.choice(self.first_names)
        last_name = random.choice(self.last_names)
        
        # Company and industry
        industry = self._weighted_choice(self.config.industry_weights)
        company_data = self._generate_company_data(industry)
        
        # Job title based on seniority
        seniority_level = self._generate_seniority_level()
        title = random.choice(self.job_titles[seniority_level])
        
        # Contact information
        email = self._generate_email(first_name, last_name, company_data["name"], seniority_level)
        phone = self._generate_phone()
        linkedin_url = self._generate_linkedin_url(first_name, last_name)
        
        # Source information
        source = self._weighted_choice(self.config.source_weights)
        
        # Timestamps
        created_at = self._generate_timestamp()
        
        # Generate labels based on features
        priority = self._calculate_priority(seniority_level, industry, company_data, source)
        quality = self._calculate_quality(email, phone, linkedin_url, company_data)
        
        # Apply noise if configured
        if self.config.noise_level > 0:
            first_name, last_name, email, phone = self._apply_data_noise(
                first_name, last_name, email, phone, self.config.noise_level
            )
        
        lead = {
            "id": lead_id,
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "phone": phone,
            "title": title,
            "linkedin_url": linkedin_url,
            "source": source,
            "created_at": created_at,
            "updated_at": created_at,
            "company": company_data,
            "notes": self._generate_notes(seniority_level, industry),
            "tags": self._generate_tags(industry, seniority_level, source)
        }
        
        # Add labels if configured
        if self.config.include_labels:
            lead.update({
                "priority": priority,
                "quality": quality,
                "conversion_likely": self._calculate_conversion_likelihood(priority, quality),
                "score": self._calculate_lead_score(priority, quality, seniority_level)
            })
        
        return lead
    
    def _generate_company_data(self, industry: str) -> Dict[str, Any]:
        """Generate company information"""
        
        # Company name
        if industry in self.company_names:
            name = random.choice(self.company_names[industry])
        else:
            name = f"{random.choice(['Tech', 'Pro', 'Smart', 'Digital', 'Advanced'])} {random.choice(['Solutions', 'Systems', 'Corp', 'Inc', 'Group'])}"
        
        # Company size
        size_weights = {
            CompanySize.STARTUP.value: 0.15,
            CompanySize.SMALL.value: 0.25,
            CompanySize.MEDIUM.value: 0.25,
            CompanySize.LARGE.value: 0.20,
            CompanySize.ENTERPRISE.value: 0.10,
            CompanySize.MEGA.value: 0.04,
            CompanySize.GIANT.value: 0.01
        }
        size = self._weighted_choice(size_weights)
        
        # Generate domain from company name
        domain = self._generate_company_domain(name)
        
        return {
            "name": name,
            "industry": industry.replace("_", " ").title(),
            "size": size,
            "location": random.choice(self.locations),
            "website": f"https://www.{domain}",
            "domain": domain,
            "founded_year": random.randint(1990, 2023),
            "employee_count": self._estimate_employee_count(size)
        }
    
    def _generate_company_domain(self, company_name: str) -> str:
        """Generate company domain from name"""
        # Clean company name
        clean_name = company_name.lower()
        clean_name = clean_name.replace(" ", "").replace(",", "").replace(".", "")
        clean_name = clean_name.replace("inc", "").replace("corp", "").replace("llc", "")
        clean_name = clean_name.replace("solutions", "").replace("systems", "")
        
        # Add domain extension
        extension = random.choice(self.email_domains["corporate"])
        
        return f"{clean_name}{extension}"
    
    def _estimate_employee_count(self, size_range: str) -> int:
        """Estimate actual employee count from size range"""
        size_mapping = {
            "1-10": (1, 10),
            "11-50": (11, 50),
            "51-200": (51, 200),
            "201-500": (201, 500),
            "501-1000": (501, 1000),
            "1001-5000": (1001, 5000),
            "5000+": (5000, 20000)
        }
        
        min_count, max_count = size_mapping.get(size_range, (10, 100))
        return random.randint(min_count, max_count)
    
    def _generate_seniority_level(self) -> str:
        """Generate seniority level with realistic distribution"""
        seniority_weights = {
            "executive": 0.05,
            "senior": 0.15,
            "manager": 0.25,
            "individual": 0.55
        }
        return self._weighted_choice(seniority_weights)
    
    def _generate_email(self, first_name: str, last_name: str, company_name: str, seniority: str) -> str:
        """Generate realistic email address"""
        
        # Determine if corporate or personal email
        corporate_probability = {
            "executive": 0.95,
            "senior": 0.90,
            "manager": 0.85,
            "individual": 0.70
        }
        
        is_corporate = random.random() < corporate_probability[seniority]
        
        # Generate email local part
        patterns = [
            f"{first_name.lower()}.{last_name.lower()}",
            f"{first_name.lower()}{last_name.lower()}",
            f"{first_name[0].lower()}{last_name.lower()}",
            f"{first_name.lower()}{last_name[0].lower()}",
            f"{first_name.lower()}.{last_name[0].lower()}",
        ]
        
        local_part = random.choice(patterns)
        
        if is_corporate:
            # Use company domain
            domain = self._generate_company_domain(company_name)
        else:
            # Use free email domain
            domain = random.choice(self.email_domains["free"])
        
        return f"{local_part}@{domain}"
    
    def _generate_phone(self) -> str:
        """Generate realistic phone number"""
        if random.random() < 0.85:  # 85% have phone numbers
            area_code = random.choice(self.area_codes)
            exchange = random.randint(200, 999)
            number = random.randint(1000, 9999)
            
            # Various formats
            formats = [
                f"+1-{area_code}-{exchange}-{number}",
                f"({area_code}) {exchange}-{number}",
                f"{area_code}.{exchange}.{number}",
                f"{area_code}-{exchange}-{number}",
                f"+1 {area_code} {exchange} {number}"
            ]
            
            return random.choice(formats)
        else:
            return ""  # No phone number
    
    def _generate_linkedin_url(self, first_name: str, last_name: str) -> str:
        """Generate LinkedIn URL"""
        if random.random() < 0.65:  # 65% have LinkedIn
            # Various LinkedIn username patterns
            patterns = [
                f"{first_name.lower()}-{last_name.lower()}",
                f"{first_name.lower()}{last_name.lower()}",
                f"{first_name[0].lower()}{last_name.lower()}",
                f"{first_name.lower()}-{last_name.lower()}-{random.randint(1, 999)}",
            ]
            
            username = random.choice(patterns)
            return f"https://www.linkedin.com/in/{username}"
        else:
            return ""  # No LinkedIn
    
    def _generate_timestamp(self) -> str:
        """Generate realistic creation timestamp"""
        # Random time in the last 2 years
        start_date = datetime.now() - timedelta(days=730)
        end_date = datetime.now()
        
        # Weight more recent dates higher
        days_back = random.expovariate(1/180)  # Average 180 days back
        days_back = min(days_back, 730)  # Cap at 2 years
        
        timestamp = end_date - timedelta(days=days_back)
        
        # Add some business hours weighting
        if random.random() < 0.7:  # 70% during business hours
            hour = random.randint(9, 17)
            minute = random.randint(0, 59)
            timestamp = timestamp.replace(hour=hour, minute=minute)
        
        return timestamp.isoformat() + "Z"
    
    def _generate_notes(self, seniority: str, industry: str) -> str:
        """Generate realistic notes"""
        if random.random() < 0.3:  # 30% have notes
            note_templates = [
                f"Interested in {industry} solutions",
                f"Met at {random.choice(['conference', 'webinar', 'networking event'])}",
                f"Referred by {random.choice(['colleague', 'partner', 'client'])}",
                f"Downloaded {random.choice(['whitepaper', 'case study', 'ebook'])}",
                f"Attended {random.choice(['demo', 'presentation', 'workshop'])}",
                "Follow up in Q2",
                "Budget approved for new solutions",
                "Currently evaluating vendors",
                "Decision maker for technology purchases"
            ]
            return random.choice(note_templates)
        else:
            return ""
    
    def _generate_tags(self, industry: str, seniority: str, source: str) -> List[str]:
        """Generate relevant tags"""
        tags = []
        
        # Industry tags
        tags.append(industry.replace("_", "-"))
        
        # Seniority tags
        if seniority == "executive":
            tags.append("decision-maker")
        elif seniority == "senior":
            tags.append("influencer")
        
        # Source tags
        if source in ["referral", "event", "webinar"]:
            tags.append("warm-lead")
        elif source == "linkedin":
            tags.append("social")
        
        # Random additional tags
        additional_tags = [
            "budget-approved", "immediate-need", "research-phase", "comparison-shopping",
            "enterprise", "growth-company", "early-adopter", "price-sensitive"
        ]
        
        if random.random() < 0.4:  # 40% get additional tags
            tags.extend(random.sample(additional_tags, random.randint(1, 2)))
        
        return tags
    
    def _calculate_priority(self, seniority: str, industry: str, company: Dict, source: str) -> str:
        """Calculate priority based on lead characteristics"""
        score = 0
        
        # Seniority scoring
        seniority_scores = {"executive": 0.4, "senior": 0.3, "manager": 0.2, "individual": 0.1}
        score += seniority_scores[seniority]
        
        # Industry scoring
        industry_scores = {
            "technology": 0.3, "finance": 0.25, "healthcare": 0.2,
            "consulting": 0.15, "manufacturing": 0.1
        }
        score += industry_scores.get(industry, 0.05)
        
        # Company size scoring
        size_scores = {
            "201-500": 0.2, "501-1000": 0.15, "51-200": 0.15,
            "1001-5000": 0.1, "11-50": 0.1, "1-10": 0.05, "5000+": 0.05
        }
        score += size_scores.get(company["size"], 0.05)
        
        # Source scoring
        source_scores = {
            "referral": 0.1, "linkedin": 0.08, "event": 0.08, "website": 0.06,
            "webinar": 0.05, "directory": 0.03, "purchased": 0.01
        }
        score += source_scores.get(source, 0.02)
        
        # Convert to priority level
        if score >= 0.7:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _calculate_quality(self, email: str, phone: str, linkedin: str, company: Dict) -> str:
        """Calculate quality based on data completeness and accuracy"""
        score = 0
        
        # Contact completeness
        if email:
            score += 0.3
            if "@" in email and "." in email:
                score += 0.1
        
        if phone:
            score += 0.2
        
        if linkedin:
            score += 0.1
        
        # Company data quality
        if company.get("domain"):
            score += 0.1
        
        if company.get("employee_count", 0) > 0:
            score += 0.1
        
        # Data consistency
        if email and company.get("domain"):
            email_domain = email.split("@")[1]
            if email_domain == company["domain"]:
                score += 0.1
        
        # Convert to quality level
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _calculate_conversion_likelihood(self, priority: str, quality: str) -> str:
        """Calculate conversion likelihood"""
        priority_scores = {"high": 0.6, "medium": 0.4, "low": 0.2}
        quality_scores = {"excellent": 0.4, "good": 0.3, "fair": 0.2, "poor": 0.1}
        
        total_score = priority_scores[priority] + quality_scores[quality]
        
        if total_score >= 0.8:
            return "high"
        elif total_score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _calculate_lead_score(self, priority: str, quality: str, seniority: str) -> int:
        """Calculate numerical lead score (0-100)"""
        priority_scores = {"high": 40, "medium": 25, "low": 10}
        quality_scores = {"excellent": 30, "good": 20, "fair": 15, "poor": 5}
        seniority_scores = {"executive": 30, "senior": 20, "manager": 15, "individual": 10}
        
        score = (
            priority_scores[priority] +
            quality_scores[quality] +
            seniority_scores[seniority]
        )
        
        # Add some randomness
        score += random.randint(-5, 5)
        
        return max(0, min(100, score))
    
    def _apply_data_noise(self, first_name: str, last_name: str, email: str, phone: str, noise_level: float) -> Tuple[str, str, str, str]:
        """Apply realistic data noise/errors"""
        
        # Name noise
        if random.random() < noise_level:
            if random.random() < 0.5:
                first_name = first_name.lower()  # Case error
            else:
                first_name = first_name.upper()
        
        if random.random() < noise_level:
            if random.random() < 0.5:
                last_name = last_name.lower()
            else:
                last_name = last_name.upper()
        
        # Email noise
        if email and random.random() < noise_level * 0.5:  # Less email corruption
            # Add extra dots or characters
            if random.random() < 0.5:
                email = email.replace("@", "@@", 1)  # Double @
            else:
                email = email.replace(".", "..", 1)  # Double dots
        
        # Phone noise
        if phone and random.random() < noise_level:
            # Remove some formatting
            phone = phone.replace("-", "").replace("(", "").replace(")", "").replace(" ", "")
            # Sometimes add back partial formatting
            if random.random() < 0.5:
                phone = phone[:3] + "-" + phone[3:6] + "-" + phone[6:]
        
        return first_name, last_name, email, phone
    
    def _weighted_choice(self, weights: Dict[str, float]) -> str:
        """Make weighted random choice"""
        choices = list(weights.keys())
        probabilities = list(weights.values())
        return np.random.choice(choices, p=probabilities)
    
    def save_to_file(self, leads: List[Dict[str, Any]], filepath: str, format: str = "json"):
        """Save generated leads to file"""
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if format.lower() == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(leads, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == "csv":
            if leads:
                # Flatten nested company data
                flattened_leads = []
                for lead in leads:
                    flat_lead = lead.copy()
                    company = flat_lead.pop("company", {})
                    
                    # Add company fields with prefix
                    for key, value in company.items():
                        flat_lead[f"company_{key}"] = value
                    
                    # Convert lists to strings
                    if "tags" in flat_lead:
                        flat_lead["tags"] = ", ".join(flat_lead["tags"])
                    
                    flattened_leads.append(flat_lead)
                
                df = pd.DataFrame(flattened_leads)
                df.to_csv(filepath, index=False)
        
        elif format.lower() == "pandas":
            df = pd.DataFrame(leads)
            df.to_pickle(filepath)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"💾 Saved {len(leads)} leads to {filepath} in {format} format")
    
    def export_to_dataframe(self, leads: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert leads to pandas DataFrame"""
        if not leads:
            return pd.DataFrame()
        
        # Flatten nested company data
        flattened_leads = []
        for lead in leads:
            flat_lead = lead.copy()
            company = flat_lead.pop("company", {})
            
            # Add company fields with prefix
            for key, value in company.items():
                flat_lead[f"company_{key}"] = value
            
            # Convert lists to strings
            if "tags" in flat_lead:
                flat_lead["tags"] = ", ".join(flat_lead["tags"])
            
            flattened_leads.append(flat_lead)
        
        return pd.DataFrame(flattened_leads)
    
    def generate_statistics(self, leads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about the generated leads"""
        if not leads:
            return {}
        
        df = self.export_to_dataframe(leads)
        
        stats = {
            "total_leads": len(leads),
            "generation_date": datetime.now().isoformat(),
            "distribution": {
                "by_priority": df["priority"].value_counts().to_dict() if "priority" in df.columns else {},
                "by_quality": df["quality"].value_counts().to_dict() if "quality" in df.columns else {},
                "by_source": df["source"].value_counts().to_dict() if "source" in df.columns else {},
                "by_industry": df["company_industry"].value_counts().to_dict() if "company_industry" in df.columns else {},
                "by_company_size": df["company_size"].value_counts().to_dict() if "company_size" in df.columns else {}
            },
            "completion_rates": {
                "email": len([l for l in leads if l.get("email")]) / len(leads) * 100,
                "phone": len([l for l in leads if l.get("phone")]) / len(leads) * 100,
                "linkedin": len([l for l in leads if l.get("linkedin_url")]) / len(leads) * 100,
                "company_info": len([l for l in leads if l.get("company", {}).get("name")]) / len(leads) * 100
            }
        }
        
        if "score" in df.columns:
            stats["score_statistics"] = {
                "mean": float(df["score"].mean()),
                "median": float(df["score"].median()),
                "std": float(df["score"].std()),
                "min": float(df["score"].min()),
                "max": float(df["score"].max())
            }
        
        return stats
    
    def validate_generated_data(self, leads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the quality of generated data"""
        validation_results = {
            "total_leads": len(leads),
            "validation_passed": True,
            "issues": [],
            "warnings": []
        }
        
        for i, lead in enumerate(leads):
            lead_id = lead.get("id", i + 1)
            
            # Required field validation
            required_fields = ["first_name", "last_name", "title", "source"]
            for field in required_fields:
                if not lead.get(field):
                    validation_results["issues"].append(f"Lead {lead_id}: Missing required field '{field}'")
                    validation_results["validation_passed"] = False
            
            # Email validation
            email = lead.get("email")
            if email and "@" not in email:
                validation_results["issues"].append(f"Lead {lead_id}: Invalid email format")
                validation_results["validation_passed"] = False
            
            # Phone validation
            phone = lead.get("phone")
            if phone and len(phone.replace("-", "").replace("(", "").replace(")", "").replace(" ", "").replace("+", "").replace(".", "")) < 10:
                validation_results["warnings"].append(f"Lead {lead_id}: Phone number seems too short")
            
            # Company validation
            company = lead.get("company", {})
            if not company.get("name"):
                validation_results["warnings"].append(f"Lead {lead_id}: Missing company name")
            
            # Score validation (if present)
            if "score" in lead:
                score = lead["score"]
                if not isinstance(score, (int, float)) or score < 0 or score > 100:
                    validation_results["issues"].append(f"Lead {lead_id}: Invalid score value")
                    validation_results["validation_passed"] = False
        
        validation_results["issue_count"] = len(validation_results["issues"])
        validation_results["warning_count"] = len(validation_results["warnings"])
        
        return validation_results


# ============================
# SYNTHETIC TRAINING DATA GENERATOR
# ============================

class SyntheticDataGenerator:
    """
    Generates synthetic training data for machine learning models
    """
    
    def __init__(self):
        self.lead_generator = LeadDataGenerator()
        logger.info("🎯 Synthetic Training Data Generator initialized")
    
    def generate_training_data(self, dataset_type: str, sample_count: int, output_file: str) -> bool:
        """Generate training data for specific ML tasks"""
        try:
            if dataset_type == "lead_priority":
                return self._generate_priority_training_data(sample_count, output_file)
            elif dataset_type == "data_quality":
                return self._generate_quality_training_data(sample_count, output_file)
            elif dataset_type == "lead_similarity":
                return self._generate_similarity_training_data(sample_count, output_file)
            else:
                logger.error(f"Unknown dataset type: {dataset_type}")
                return False
                
        except Exception as e:
            logger.error(f"Training data generation failed: {e}")
            return False
    
    def _generate_priority_training_data(self, sample_count: int, output_file: str) -> bool:
        """Generate training data for lead priority classification"""
        config = GenerationConfig(
            num_leads=sample_count,
            include_labels=True,
            noise_level=0.15
        )
        
        generator = LeadDataGenerator(config)
        leads = generator.generate_leads()
        
        # Convert to ML training format
        training_data = []
        for lead in leads:
            features = {
                "seniority_level": self._get_seniority_from_title(lead["title"]),
                "industry": lead["company"]["industry"],
                "company_size": lead["company"]["size"],
                "source": lead["source"],
                "email_domain_type": self._classify_email_domain(lead["email"]),
                "has_phone": bool(lead.get("phone")),
                "has_linkedin": bool(lead.get("linkedin_url")),
                "company_employee_count": lead["company"].get("employee_count", 0),
                "data_completeness_score": self._calculate_completeness(lead)
            }
            
            training_sample = {
                "features": features,
                "label": lead["priority"],
                "lead_id": lead["id"]
            }
            training_data.append(training_sample)
        
        # Save training data
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Generated {len(training_data)} priority training samples")
        return True
    
    def _generate_quality_training_data(self, sample_count: int, output_file: str) -> bool:
        """Generate training data for data quality assessment"""
        config = GenerationConfig(
            num_leads=sample_count,
            include_labels=True,
            noise_level=0.2  # Higher noise for quality assessment
        )
        
        generator = LeadDataGenerator(config)
        leads = generator.generate_leads()
        
        training_data = []
        for lead in leads:
            features = {
                "email_format_valid": self._validate_email_format(lead.get("email", "")),
                "phone_format_valid": self._validate_phone_format(lead.get("phone", "")),
                "name_format_valid": self._validate_name_format(lead.get("first_name", ""), lead.get("last_name", "")),
                "linkedin_url_valid": self._validate_linkedin_url(lead.get("linkedin_url", "")),
                "company_data_complete": self._assess_company_completeness(lead.get("company", {})),
                "field_completion_rate": self._calculate_completeness(lead),
                "data_consistency_score": self._check_data_consistency(lead),
                "has_corporate_email": self._has_corporate_email(lead.get("email", ""), lead.get("company", {}))
            }
            
            training_sample = {
                "features": features,
                "label": lead["quality"],
                "lead_id": lead["id"]
            }
            training_data.append(training_sample)
        
        # Save training data
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Generated {len(training_data)} quality training samples")
        return True
    
    def _generate_similarity_training_data(self, sample_count: int, output_file: str) -> bool:
        """Generate training data for lead similarity detection"""
        config = GenerationConfig(
            num_leads=sample_count,
            include_labels=True,
            noise_level=0.1
        )
        
        generator = LeadDataGenerator(config)
        leads = generator.generate_leads()
        
        training_data = []
        
        # Generate pairs of leads for similarity comparison
        for i in range(0, len(leads), 2):
            if i + 1 < len(leads):
                lead1 = leads[i]
                lead2 = leads[i + 1]
                
                # Calculate similarity features
                features = {
                    "name_similarity": self._calculate_name_similarity(lead1, lead2),
                    "email_similarity": self._calculate_email_similarity(lead1, lead2),
                    "company_similarity": self._calculate_company_similarity(lead1, lead2),
                    "title_similarity": self._calculate_title_similarity(lead1, lead2),
                    "industry_match": lead1["company"]["industry"] == lead2["company"]["industry"],
                    "same_company_domain": self._same_company_domain(lead1, lead2),
                    "location_similarity": self._calculate_location_similarity(lead1, lead2)
                }
                
                # Determine if leads are similar (threshold-based)
                similarity_score = sum([
                    features["name_similarity"] * 0.3,
                    features["email_similarity"] * 0.2,
                    features["company_similarity"] * 0.3,
                    features["title_similarity"] * 0.2
                ])
                
                is_similar = similarity_score > 0.7
                
                training_sample = {
                    "features": features,
                    "label": "similar" if is_similar else "different",
                    "similarity_score": similarity_score,
                    "lead1_id": lead1["id"],
                    "lead2_id": lead2["id"]
                }
                training_data.append(training_sample)
        
        # Save training data
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        logger.info(f"Generated {len(training_data)} similarity training samples")
        return True
    
    # Helper methods for training data generation
    def _get_seniority_from_title(self, title: str) -> str:
        """Extract seniority level from job title"""
        title_lower = title.lower()
        if any(word in title_lower for word in ["ceo", "cto", "cfo", "president", "founder", "executive"]):
            return "executive"
        elif any(word in title_lower for word in ["senior", "director", "head", "principal"]):
            return "senior"
        elif any(word in title_lower for word in ["manager", "lead"]):
            return "manager"
        else:
            return "individual"
    
    def _classify_email_domain(self, email: str) -> str:
        """Classify email domain type"""
        if not email or "@" not in email:
            return "unknown"
        
        domain = email.split("@")[1]
        free_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"]
        
        return "free" if domain in free_domains else "corporate"
    
    def _calculate_completeness(self, lead: Dict[str, Any]) -> float:
        """Calculate data completeness score"""
        total_fields = 10
        completed_fields = 0
        
        fields_to_check = ["first_name", "last_name", "email", "phone", "title", "linkedin_url"]
        for field in fields_to_check:
            if lead.get(field):
                completed_fields += 1
        
        company = lead.get("company", {})
        company_fields = ["name", "industry", "size", "location"]
        for field in company_fields:
            if company.get(field):
                completed_fields += 1
        
        return completed_fields / total_fields
    
    def _validate_email_format(self, email: str) -> bool:
        """Validate email format"""
        return bool(email and "@" in email and "." in email.split("@")[1])
    
    def _validate_phone_format(self, phone: str) -> bool:
        """Validate phone format"""
        if not phone:
            return False
        
        # Remove formatting
        clean_phone = phone.replace("-", "").replace("(", "").replace(")", "").replace(" ", "").replace("+", "").replace(".", "")
        return len(clean_phone) >= 10 and clean_phone.isdigit()
    
    def _validate_name_format(self, first_name: str, last_name: str) -> bool:
        """Validate name format"""
        return bool(first_name and last_name and first_name.replace(" ", "").isalpha() and last_name.replace(" ", "").isalpha())
    
    def _validate_linkedin_url(self, url: str) -> bool:
        """Validate LinkedIn URL format"""
        return bool(url and "linkedin.com/in/" in url)
    
    def _assess_company_completeness(self, company: Dict[str, Any]) -> float:
        """Assess company data completeness"""
        if not company:
            return 0.0
        
        required_fields = ["name", "industry", "size", "location"]
        completed = sum(1 for field in required_fields if company.get(field))
        return completed / len(required_fields)
    
    def _check_data_consistency(self, lead: Dict[str, Any]) -> float:
        """Check data consistency across fields"""
        consistency_score = 1.0
        
        # Check email-company domain consistency
        email = lead.get("email", "")
        company = lead.get("company", {})
        
        if email and "@" in email and company.get("domain"):
            email_domain = email.split("@")[1]
            if email_domain != company["domain"]:
                consistency_score -= 0.2
        
        # Check name consistency (no numbers in names)
        first_name = lead.get("first_name", "")
        last_name = lead.get("last_name", "")
        
        if any(char.isdigit() for char in first_name + last_name):
            consistency_score -= 0.3
        
        return max(0.0, consistency_score)
    
    def _has_corporate_email(self, email: str, company: Dict[str, Any]) -> bool:
        """Check if email is corporate"""
        if not email or "@" not in email:
            return False
        
        domain = email.split("@")[1]
        free_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"]
        
        return domain not in free_domains
    
    def _calculate_name_similarity(self, lead1: Dict, lead2: Dict) -> float:
        """Calculate name similarity between two leads"""
        from difflib import SequenceMatcher
        
        name1 = f"{lead1.get('first_name', '')} {lead1.get('last_name', '')}".strip()
        name2 = f"{lead2.get('first_name', '')} {lead2.get('last_name', '')}".strip()
        
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    
    def _calculate_email_similarity(self, lead1: Dict, lead2: Dict) -> float:
        """Calculate email similarity between two leads"""
        from difflib import SequenceMatcher
        
        email1 = lead1.get("email", "")
        email2 = lead2.get("email", "")
        
        if not email1 or not email2:
            return 0.0
        
        return SequenceMatcher(None, email1.lower(), email2.lower()).ratio()
    
    def _calculate_company_similarity(self, lead1: Dict, lead2: Dict) -> float:
        """Calculate company similarity between two leads"""
        from difflib import SequenceMatcher
        
        company1 = lead1.get("company", {}).get("name", "")
        company2 = lead2.get("company", {}).get("name", "")
        
        if not company1 or not company2:
            return 0.0
        
        return SequenceMatcher(None, company1.lower(), company2.lower()).ratio()
    
    def _calculate_title_similarity(self, lead1: Dict, lead2: Dict) -> float:
        """Calculate title similarity between two leads"""
        from difflib import SequenceMatcher
        
        title1 = lead1.get("title", "")
        title2 = lead2.get("title", "")
        
        if not title1 or not title2:
            return 0.0
        
        return SequenceMatcher(None, title1.lower(), title2.lower()).ratio()
    
    def _same_company_domain(self, lead1: Dict, lead2: Dict) -> bool:
        """Check if leads have the same company domain"""
        domain1 = lead1.get("company", {}).get("domain", "")
        domain2 = lead2.get("company", {}).get("domain", "")
        
        return bool(domain1 and domain2 and domain1 == domain2)
    
    def _calculate_location_similarity(self, lead1: Dict, lead2: Dict) -> float:
        """Calculate location similarity between two leads"""
        from difflib import SequenceMatcher
        
        location1 = lead1.get("company", {}).get("location", "")
        location2 = lead2.get("company", {}).get("location", "")
        
        if not location1 or not location2:
            return 0.0
        
        return SequenceMatcher(None, location1.lower(), location2.lower()).ratio()


# ============================
# USAGE EXAMPLES AND MAIN FUNCTION
# ============================

def main():
    """Main function for running the data generator"""
    
    print("🎲 AI Lead Data Generator")
    print("=" * 50)
    
    # Configuration for generation
    config = GenerationConfig(
        num_leads=1000,
        include_labels=True,
        noise_level=0.1,
        output_format="json"
    )
    
    # Initialize generator
    generator = LeadDataGenerator(config)
    
    # Generate leads
    print(f"🎯 Generating {config.num_leads} synthetic leads...")
    leads = generator.generate_leads()
    
    # Validate data
    print("🔍 Validating generated data...")
    validation_results = generator.validate_generated_data(leads)
    
    if validation_results["validation_passed"]:
        print("✅ Data validation passed!")
    else:
        print(f"❌ Data validation found {validation_results['issue_count']} issues")
        for issue in validation_results["issues"][:5]:  # Show first 5 issues
            print(f"  - {issue}")
    
    if validation_results["warning_count"] > 0:
        print(f"⚠️  {validation_results['warning_count']} warnings found")
    
    # Generate statistics
    print("📊 Generating statistics...")
    stats = generator.generate_statistics(leads)
    
    print(f"📈 Statistics Summary:")
    print(f"  Total leads: {stats['total_leads']}")
    print(f"  Email completion: {stats['completion_rates']['email']:.1f}%")
    print(f"  Phone completion: {stats['completion_rates']['phone']:.1f}%")
    print(f"  LinkedIn completion: {stats['completion_rates']['linkedin']:.1f}%")
    
    if "score_statistics" in stats:
        score_stats = stats["score_statistics"]
        print(f"  Average score: {score_stats['mean']:.1f}")
        print(f"  Score range: {score_stats['min']:.0f} - {score_stats['max']:.0f}")
    
    # Save files
    print("💾 Saving generated data...")
    
    # Save main dataset
    generator.save_to_file(leads, "data/synthetic_leads.json", "json")
    generator.save_to_file(leads, "data/synthetic_leads.csv", "csv")
    
    # Save statistics
    with open("data/generation_statistics.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    
    # Save validation results
    with open("data/validation_results.json", "w") as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    print("✅ Data generation completed successfully!")
    print(f"📁 Files saved:")
    print(f"  - data/synthetic_leads.json")
    print(f"  - data/synthetic_leads.csv")
    print(f"  - data/generation_statistics.json")
    print(f"  - data/validation_results.json")
    
    # Generate training data for ML
    print("\n🤖 Generating training data for ML models...")
    training_generator = SyntheticDataGenerator()
    
    training_datasets = {
        "lead_priority": 800,
        "data_quality": 600,
        "lead_similarity": 400
    }
    
    for dataset_type, sample_count in training_datasets.items():
        output_file = f"ai/training_data/{dataset_type}_training_data.json"
        success = training_generator.generate_training_data(dataset_type, sample_count, output_file)
        
        if success:
            print(f"  ✅ Generated {dataset_type} training data ({sample_count} samples)")
        else:
            print(f"  ❌ Failed to generate {dataset_type} training data")
    
    print("\n🎉 All data generation tasks completed!")


if __name__ == "__main__":
    main()