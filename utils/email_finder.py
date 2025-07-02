import httpx
import re
import dns.resolver
import socket
import smtplib
from typing import Optional, List, Dict, Tuple
import asyncio
import logging
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
import random
from email.mime.text import MIMEText
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailFinder:
    def __init__(self):
        self.session = httpx.Client(
            timeout=30.0,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        
        # Extended email patterns for better coverage
        self.email_patterns = [
            "{first}.{last}@{domain}",
            "{first}{last}@{domain}",
            "{first}@{domain}",
            "{last}@{domain}",
            "{first_initial}{last}@{domain}",
            "{first}{last_initial}@{domain}",
            "{first_initial}.{last}@{domain}",
            "{first}.{last_initial}@{domain}",
            "{first}-{last}@{domain}",
            "{first}_{last}@{domain}",
            "{last}.{first}@{domain}",
            "{last}{first}@{domain}",
            "{last}-{first}@{domain}",
            "{last}_{first}@{domain}",
            # Common role-based emails
            "contact@{domain}",
            "info@{domain}",
            "hello@{domain}",
            "sales@{domain}",
            "support@{domain}",
            "admin@{domain}",
            "team@{domain}",
            "office@{domain}"
        ]
        
        # Hunter.io API configuration
        self.hunter_api_key = os.getenv("HUNTER_API_KEY")
        self.hunter_api_url = "https://api.hunter.io/v2"
        
        # Email verification services
        self.verification_services = {
            'zerobounce': os.getenv("ZEROBOUNCE_API_KEY"),
            'neverbounce': os.getenv("NEVERBOUNCE_API_KEY"),
            'emaillistverify': os.getenv("EMAILLISTVERIFY_API_KEY")
        }
    
    def find_email(self, first_name: str, last_name: str, domain: str, 
                   verify: bool = True, timeout: int = 30) -> Optional[Dict]:
        """
        Comprehensive email finding with multiple strategies
        Returns detailed result with confidence score
        """
        logger.info(f"🔍 Finding email for {first_name} {last_name} at {domain}")
        
        # Clean and normalize inputs
        first_name = self._clean_name(first_name)
        last_name = self._clean_name(last_name)
        domain = self._clean_domain(domain)
        
        result = {
            'email': None,
            'confidence': 0,
            'source': None,
            'verified': False,
            'risk_level': 'unknown',
            'attempts': []
        }
        
        # Strategy 1: Try Hunter.io API (highest accuracy)
        hunter_result = self._try_hunter_api(first_name, last_name, domain)
        if hunter_result:
            result.update(hunter_result)
            result['source'] = 'hunter_api'
            if result['confidence'] > 80:
                return result
        
        # Strategy 2: Website scraping for existing emails
        scraped_result = self._scrape_website_emails(domain, first_name, last_name)
        if scraped_result and scraped_result['confidence'] > result['confidence']:
            result.update(scraped_result)
            result['source'] = 'website_scraping'
        
        # Strategy 3: Social media and professional networks
        social_result = self._search_social_media(first_name, last_name, domain)
        if social_result and social_result['confidence'] > result['confidence']:
            result.update(social_result)
            result['source'] = 'social_media'
        
        # Strategy 4: Email pattern testing with verification
        pattern_result = self._test_email_patterns(first_name, last_name, domain, verify)
        if pattern_result and pattern_result['confidence'] > result['confidence']:
            result.update(pattern_result)
            result['source'] = 'pattern_matching'
        
        # Strategy 5: Company directory mining
        directory_result = self._mine_company_directory(domain, first_name, last_name)
        if directory_result and directory_result['confidence'] > result['confidence']:
            result.update(directory_result)
            result['source'] = 'directory_mining'
        
        # Final verification if email found
        if result['email'] and verify:
            verification = self._verify_email_deliverability(result['email'])
            result.update(verification)
        
        return result if result['email'] else None
    
    def _clean_name(self, name: str) -> str:
        """Clean and normalize name input"""
        if not name:
            return ""
        return re.sub(r'[^a-zA-Z]', '', name.lower().strip())
    
    def _clean_domain(self, domain: str) -> str:
        """Clean and normalize domain input"""
        domain = domain.lower().strip()
        domain = re.sub(r'^https?://', '', domain)
        domain = re.sub(r'^www\.', '', domain)
        domain = domain.split('/')[0]
        return domain
    
    def _try_hunter_api(self, first_name: str, last_name: str, domain: str) -> Optional[Dict]:
        """
        Use Hunter.io API for professional email finding
        """
        if not self.hunter_api_key:
            logger.info("Hunter.io API key not configured")
            return None
        
        try:
            # Email finder endpoint
            url = f"{self.hunter_api_url}/email-finder"
            params = {
                'domain': domain,
                'first_name': first_name,
                'last_name': last_name,
                'api_key': self.hunter_api_key
            }
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('data', {}).get('email'):
                    email_data = data['data']
                    return {
                        'email': email_data['email'],
                        'confidence': email_data.get('confidence', 50),
                        'verified': True,
                        'risk_level': 'low',
                        'attempts': [f"hunter_api: {email_data['email']}"]
                    }
            
            # Try domain search as fallback
            return self._hunter_domain_search(domain, first_name, last_name)
            
        except Exception as e:
            logger.warning(f"Hunter.io API failed: {e}")
            return None
    
    def _hunter_domain_search(self, domain: str, first_name: str, last_name: str) -> Optional[Dict]:
        """
        Search Hunter.io domain database for email patterns
        """
        try:
            url = f"{self.hunter_api_url}/domain-search"
            params = {
                'domain': domain,
                'api_key': self.hunter_api_key,
                'limit': 100
            }
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                emails = data.get('data', {}).get('emails', [])
                
                # Look for matching names
                for email_data in emails:
                    email = email_data.get('value', '')
                    first = email_data.get('first_name', '').lower()
                    last = email_data.get('last_name', '').lower()
                    
                    if (first == first_name and last == last_name) or \
                       (first_name in first and last_name in last):
                        return {
                            'email': email,
                            'confidence': 85,
                            'verified': True,
                            'risk_level': 'low',
                            'attempts': [f"hunter_domain: {email}"]
                        }
                
                # Extract email patterns for later use
                patterns = self._extract_email_patterns(emails)
                best_pattern = self._find_best_pattern(patterns, first_name, last_name, domain)
                
                if best_pattern:
                    return {
                        'email': best_pattern,
                        'confidence': 70,
                        'verified': False,
                        'risk_level': 'medium',
                        'attempts': [f"hunter_pattern: {best_pattern}"]
                    }
            
        except Exception as e:
            logger.warning(f"Hunter domain search failed: {e}")
        
        return None
    
    def _scrape_website_emails(self, domain: str, first_name: str, last_name: str) -> Optional[Dict]:
        """
        Advanced website scraping for email addresses
        """
        try:
            base_url = f"https://{domain}"
            
            # Get main page
            response = self.session.get(base_url)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all potential contact pages
            contact_pages = self._find_contact_pages(soup, base_url)
            all_pages = [base_url] + contact_pages[:5]  # Limit to avoid excessive requests
            
            all_emails = []
            attempts = []
            
            for page_url in all_pages:
                try:
                    page_response = self.session.get(page_url)
                    if page_response.status_code == 200:
                        emails = self._extract_emails_from_content(page_response.text)
                        all_emails.extend(emails)
                        attempts.append(f"scraped_page: {page_url}")
                    time.sleep(1)  # Be respectful
                except:
                    continue
            
            # Filter emails from the target domain
            domain_emails = [email for email in all_emails if domain in email.lower()]
            
            if not domain_emails:
                return None
            
            # Look for name-specific emails
            name_variations = self._generate_name_variations(first_name, last_name)
            
            for email in domain_emails:
                email_prefix = email.split('@')[0].lower()
                for variation in name_variations:
                    if variation in email_prefix:
                        return {
                            'email': email,
                            'confidence': 90,
                            'verified': False,
                            'risk_level': 'low',
                            'attempts': attempts
                        }
            
            # Return best general email if no name match
            priority_emails = [email for email in domain_emails if any(
                keyword in email.lower() for keyword in ['contact', 'info', 'hello', 'sales']
            )]
            
            if priority_emails:
                return {
                    'email': priority_emails[0],
                    'confidence': 40,
                    'verified': False,
                    'risk_level': 'medium',
                    'attempts': attempts
                }
            
        except Exception as e:
            logger.warning(f"Website scraping failed for {domain}: {e}")
        
        return None
    
    def _search_social_media(self, first_name: str, last_name: str, domain: str) -> Optional[Dict]:
        """
        Search social media and professional networks for email
        """
        try:
            # LinkedIn search (basic approach - LinkedIn heavily restricts scraping)
            linkedin_email = self._search_linkedin(first_name, last_name, domain)
            if linkedin_email:
                return linkedin_email
            
            # Twitter/X search for email mentions
            twitter_email = self._search_twitter(first_name, last_name, domain)
            if twitter_email:
                return twitter_email
            
            # GitHub search for developer emails
            github_email = self._search_github(first_name, last_name, domain)
            if github_email:
                return github_email
            
        except Exception as e:
            logger.warning(f"Social media search failed: {e}")
        
        return None
    
    def _test_email_patterns(self, first_name: str, last_name: str, domain: str, verify: bool = True) -> Optional[Dict]:
        """
        Test common email patterns with verification
        """
        attempts = []
        best_result = None
        
        for pattern in self.email_patterns:
            try:
                email = pattern.format(
                    first=first_name,
                    last=last_name,
                    first_initial=first_name[0] if first_name else "",
                    last_initial=last_name[0] if last_name else "",
                    domain=domain
                )
                
                attempts.append(f"pattern_test: {email}")
                
                if not self._is_valid_email_format(email):
                    continue
                
                # Basic domain validation
                if not self._verify_domain_mx(domain):
                    continue
                
                # SMTP verification if requested
                verification_score = 50  # Base score for valid format
                
                if verify:
                    smtp_result = self._verify_email_smtp(email)
                    if smtp_result['deliverable']:
                        verification_score = 85
                    elif smtp_result['exists']:
                        verification_score = 70
                
                # Prioritize name-based patterns
                if first_name in email and last_name in email:
                    verification_score += 15
                elif first_name in email or last_name in email:
                    verification_score += 10
                
                if not best_result or verification_score > best_result['confidence']:
                    best_result = {
                        'email': email,
                        'confidence': min(verification_score, 95),
                        'verified': verify,
                        'risk_level': 'low' if verification_score > 70 else 'medium',
                        'attempts': attempts.copy()
                    }
                
                # Stop if we found a high-confidence match
                if verification_score > 80:
                    break
                    
            except Exception as e:
                logger.debug(f"Pattern test failed for {email}: {e}")
                continue
        
        return best_result
    
    def _mine_company_directory(self, domain: str, first_name: str, last_name: str) -> Optional[Dict]:
        """
        Mine company directory pages for employee emails
        """
        try:
            base_url = f"https://{domain}"
            response = self.session.get(base_url)
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for directory/team/staff pages
            directory_keywords = ['directory', 'team', 'staff', 'employees', 'people', 'leadership', 'management']
            directory_links = []
            
            for link in soup.find_all('a', href=True):
                href = link.get('href', '').lower()
                text = link.get_text().lower()
                
                if any(keyword in href or keyword in text for keyword in directory_keywords):
                    full_url = urljoin(base_url, link['href'])
                    directory_links.append(full_url)
            
            # Search directory pages
            for dir_url in directory_links[:3]:  # Limit to 3 directory pages
                try:
                    dir_response = self.session.get(dir_url)
                    if dir_response.status_code == 200:
                        result = self._extract_person_from_directory(
                            dir_response.text, first_name, last_name, domain
                        )
                        if result:
                            return result
                    time.sleep(1)
                except:
                    continue
            
        except Exception as e:
            logger.warning(f"Directory mining failed for {domain}: {e}")
        
        return None
    
    def _verify_email_deliverability(self, email: str) -> Dict:
        """
        Comprehensive email deliverability verification
        """
        result = {
            'verified': False,
            'deliverable': False,
            'risk_level': 'unknown'
        }
        
        try:
            # Try professional verification services first
            for service, api_key in self.verification_services.items():
                if api_key:
                    service_result = self._verify_with_service(email, service, api_key)
                    if service_result:
                        return service_result
            
            # Fallback to SMTP verification
            smtp_result = self._verify_email_smtp(email)
            result.update(smtp_result)
            
        except Exception as e:
            logger.warning(f"Email verification failed for {email}: {e}")
        
        return result
    
    def _verify_email_smtp(self, email: str) -> Dict:
        """
        SMTP-based email verification
        """
        result = {
            'verified': True,
            'deliverable': False,
            'exists': False,
            'risk_level': 'medium'
        }
        
        try:
            domain = email.split('@')[1]
            
            # Get MX records
            mx_records = dns.resolver.resolve(domain, 'MX')
            mx_record = str(mx_records[0].exchange)
            
            # Connect to SMTP server
            server = smtplib.SMTP(timeout=10)
            server.connect(mx_record, 25)
            server.helo('gmail.com')
            server.mail('test@gmail.com')
            
            # Test recipient
            code, message = server.rcpt(email)
            server.quit()
            
            if code == 250:
                result.update({
                    'deliverable': True,
                    'exists': True,
                    'risk_level': 'low'
                })
            elif code in [251, 252]:  # Uncertain but likely exists
                result.update({
                    'exists': True,
                    'risk_level': 'medium'
                })
            
        except Exception as e:
            logger.debug(f"SMTP verification failed for {email}: {e}")
        
        return result
    
    def _verify_domain_mx(self, domain: str) -> bool:
        """
        Verify domain has MX records
        """
        try:
            mx_records = dns.resolver.resolve(domain, 'MX')
            return len(mx_records) > 0
        except:
            return False
    
    def _is_valid_email_format(self, email: str) -> bool:
        """
        Validate email format
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def _generate_name_variations(self, first_name: str, last_name: str) -> List[str]:
        """
        Generate name variations for matching
        """
        variations = []
        if first_name and last_name:
            variations.extend([
                f"{first_name}.{last_name}",
                f"{first_name}{last_name}",
                f"{first_name[0]}{last_name}",
                f"{first_name}{last_name[0]}",
                f"{first_name}-{last_name}",
                f"{first_name}_{last_name}",
                f"{last_name}.{first_name}",
                f"{last_name}{first_name}"
            ])
        return variations
    
    def _find_contact_pages(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Find contact-related pages on website
        """
        contact_keywords = ['contact', 'about', 'team', 'staff', 'directory', 'people']
        contact_pages = []
        
        for link in soup.find_all('a', href=True):
            href = link.get('href', '').lower()
            text = link.get_text().lower()
            
            if any(keyword in href or keyword in text for keyword in contact_keywords):
                full_url = urljoin(base_url, link['href'])
                contact_pages.append(full_url)
        
        return list(set(contact_pages))  # Remove duplicates
    
    def _extract_emails_from_content(self, content: str) -> List[str]:
        """
        Extract email addresses from text content
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, content)
        
        # Filter out common false positives
        filtered_emails = []
        skip_patterns = ['@example.com', '@domain.com', 'noreply@', 'no-reply@']
        
        for email in emails:
            if not any(skip in email.lower() for skip in skip_patterns):
                filtered_emails.append(email)
        
        return list(set(filtered_emails))  # Remove duplicates
    
    def _extract_email_patterns(self, emails: List[Dict]) -> Dict[str, int]:
        """
        Extract common email patterns from a list of emails
        """
        patterns = {}
        
        for email_data in emails:
            email = email_data.get('value', '')
            first = email_data.get('first_name', '')
            last = email_data.get('last_name', '')
            
            if email and first and last:
                prefix = email.split('@')[0]
                
                # Determine pattern
                if f"{first.lower()}.{last.lower()}" == prefix.lower():
                    patterns['first.last'] = patterns.get('first.last', 0) + 1
                elif f"{first.lower()}{last.lower()}" == prefix.lower():
                    patterns['firstlast'] = patterns.get('firstlast', 0) + 1
                elif f"{first.lower()[0]}{last.lower()}" == prefix.lower():
                    patterns['flast'] = patterns.get('flast', 0) + 1
        
        return patterns
    
    def _find_best_pattern(self, patterns: Dict[str, int], first_name: str, last_name: str, domain: str) -> Optional[str]:
        """
        Find the most common email pattern and apply it
        """
        if not patterns:
            return None
        
        # Get most common pattern
        best_pattern = max(patterns, key=patterns.get)
        
        # Apply pattern
        if best_pattern == 'first.last':
            return f"{first_name}.{last_name}@{domain}"
        elif best_pattern == 'firstlast':
            return f"{first_name}{last_name}@{domain}"
        elif best_pattern == 'flast':
            return f"{first_name[0]}{last_name}@{domain}"
        
        return None
    
    def _search_linkedin(self, first_name: str, last_name: str, domain: str) -> Optional[Dict]:
        """
        Search LinkedIn for professional email (limited due to restrictions)
        """
        # LinkedIn heavily restricts scraping, so this is a basic implementation
        # In production, you'd use LinkedIn's official API
        return None
    
    def _search_twitter(self, first_name: str, last_name: str, domain: str) -> Optional[Dict]:
        """
        Search Twitter/X for email mentions
        """
        # Basic Twitter search implementation
        # In production, you'd use Twitter API
        return None
    
    def _search_github(self, first_name: str, last_name: str, domain: str) -> Optional[Dict]:
        """
        Search GitHub for developer emails
        """
        try:
            # GitHub search API (public repositories)
            search_query = f"{first_name} {last_name} {domain}"
            github_url = f"https://api.github.com/search/users?q={search_query}"
            
            response = self.session.get(github_url)
            if response.status_code == 200:
                data = response.json()
                users = data.get('items', [])
                
                for user in users[:3]:  # Check first 3 results
                    user_url = f"https://api.github.com/users/{user['login']}"
                    user_response = self.session.get(user_url)
                    
                    if user_response.status_code == 200:
                        user_data = user_response.json()
                        email = user_data.get('email')
                        
                        if email and domain in email:
                            return {
                                'email': email,
                                'confidence': 75,
                                'verified': False,
                                'risk_level': 'low',
                                'attempts': [f"github: {email}"]
                            }
            
        except Exception as e:
            logger.debug(f"GitHub search failed: {e}")
        
        return None
    
    def _extract_person_from_directory(self, content: str, first_name: str, last_name: str, domain: str) -> Optional[Dict]:
        """
        Extract person's email from directory page content
        """
        # Look for the person's name and nearby email
        name_patterns = [
            f"{first_name}\\s+{last_name}",
            f"{first_name}\\s*{last_name}",
            f"{last_name},\\s*{first_name}",
        ]
        
        for pattern in name_patterns:
            name_matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in name_matches:
                # Look for email within 500 characters of the name
                start = max(0, match.start() - 250)
                end = min(len(content), match.end() + 250)
                context = content[start:end]
                
                emails = self._extract_emails_from_content(context)
                domain_emails = [email for email in emails if domain in email]
                
                if domain_emails:
                    return {
                        'email': domain_emails[0],
                        'confidence': 85,
                        'verified': False,
                        'risk_level': 'low',
                        'attempts': [f"directory: {domain_emails[0]}"]
                    }
        
        return None
    
    def _verify_with_service(self, email: str, service: str, api_key: str) -> Optional[Dict]:
        """
        Verify email using professional verification services
        """
        try:
            if service == 'zerobounce':
                return self._verify_zerobounce(email, api_key)
            elif service == 'neverbounce':
                return self._verify_neverbounce(email, api_key)
            elif service == 'emaillistverify':
                return self._verify_emaillistverify(email, api_key)
        except Exception as e:
            logger.warning(f"{service} verification failed: {e}")
        
        return None
    
    def _verify_zerobounce(self, email: str, api_key: str) -> Optional[Dict]:
        """
        Verify email using ZeroBounce API
        """
        url = f"https://api.zerobounce.net/v2/validate"
        params = {
            'api_key': api_key,
            'email': email
        }
        
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            status = data.get('status', '')
            
            return {
                'verified': True,
                'deliverable': status == 'valid',
                'risk_level': 'low' if status == 'valid' else 'high'
            }
        
        return None
    
    def bulk_find_emails(self, contacts: List[Dict], verify: bool = True) -> List[Dict]:
        """
        Find emails for multiple contacts efficiently
        """
        results = []
        
        for contact in contacts:
            first_name = contact.get('first_name', '')
            last_name = contact.get('last_name', '')
            domain = contact.get('domain', '')
            
            if first_name and last_name and domain:
                result = self.find_email(first_name, last_name, domain, verify)
                contact['email_result'] = result
            
            results.append(contact)
            
            # Rate limiting
            time.sleep(0.5)
        
        return results
    
    def find_all_company_emails(self, domain: str, limit: int = 50) -> List[Dict]:
        """
        Find all possible emails for a company domain
        """
        all_emails = []
        
        try:
            # Website scraping
            website_emails = self._scrape_all_website_emails(domain)
            all_emails.extend(website_emails)
            
            # Hunter.io domain search
            if self.hunter_api_key:
                hunter_emails = self._get_hunter_domain_emails(domain)
                all_emails.extend(hunter_emails)
            
            # Remove duplicates and limit results
            unique_emails = []
            seen_emails = set()
            
            for email_data in all_emails:
                email = email_data.get('email', '')
                if email not in seen_emails:
                    seen_emails.add(email)
                    unique_emails.append(email_data)
                    
                    if len(unique_emails) >= limit:
                        break
            
            return unique_emails
            
        except Exception as e:
            logger.error(f"Company email search failed for {domain}: {e}")
            return []
    
    def _scrape_all_website_emails(self, domain: str) -> List[Dict]:
        """
        Comprehensively scrape all emails from a website
        """
        all_emails = []
        
        try:
            base_url = f"https://{domain}"
            response = self.session.get(base_url)
            
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get all internal links
            all_links = set([base_url])
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/'):
                    all_links.add(urljoin(base_url, href))
                elif domain in href:
                    all_links.add(href)
            
            # Scrape emails from each page (limit to avoid excessive requests)
            for page_url in list(all_links)[:10]:
                try:
                    page_response = self.session.get(page_url)
                    if page_response.status_code == 200:
                        emails = self._extract_emails_from_content(page_response.text)
                        for email in emails:
                            if domain in email:
                                all_emails.append({
                                    'email': email,
                                    'source': 'website_scraping',
                                    'page': page_url,
                                    'confidence': 60
                                })
                    time.sleep(1)  # Be respectful
                except:
                    continue
            
        except Exception as e:
            logger.warning(f"Website email scraping failed for {domain}: {e}")
        
        return all_emails
    
    def _get_hunter_domain_emails(self, domain: str) -> List[Dict]:
        """
        Get all emails from Hunter.io domain search
        """
        emails = []
        
        try:
            url = f"{self.hunter_api_url}/domain-search"
            params = {
                'domain': domain,
                'api_key': self.hunter_api_key,
                'limit': 100
            }
            
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                hunter_emails = data.get('data', {}).get('emails', [])
                
                for email_data in hunter_emails:
                    emails.append({
                        'email': email_data.get('value', ''),
                        'first_name': email_data.get('first_name', ''),
                        'last_name': email_data.get('last_name', ''),
                        'position': email_data.get('position', ''),
                        'source': 'hunter_api',
                        'confidence': email_data.get('confidence', 50),
                        'verified': True
                    })
            
        except Exception as e:
            logger.warning(f"Hunter.io domain search failed: {e}")
        
        return emails
    
    def validate_email_list(self, emails: List[str]) -> List[Dict]:
        """
        Validate and score a list of email addresses
        """
        results = []
        
        for email in emails:
            result = {
                'email': email,
                'valid_format': self._is_valid_email_format(email),
                'domain_exists': False,
                'deliverable': False,
                'risk_score': 100,  # 0 = low risk, 100 = high risk
                'confidence': 0
            }
            
            if result['valid_format']:
                result['confidence'] += 20
                
                # Check domain MX records
                domain = email.split('@')[1]
                if self._verify_domain_mx(domain):
                    result['domain_exists'] = True
                    result['confidence'] += 30
                    
                    # SMTP verification
                    smtp_result = self._verify_email_smtp(email)
                    if smtp_result.get('deliverable'):
                        result['deliverable'] = True
                        result['confidence'] += 50
                        result['risk_score'] = 10
                    elif smtp_result.get('exists'):
                        result['confidence'] += 30
                        result['risk_score'] = 30
                    else:
                        result['risk_score'] = 80
                else:
                    result['risk_score'] = 90
            else:
                result['risk_score'] = 100
            
            results.append(result)
        
        return results
    
    def get_email_insights(self, email: str) -> Dict:
        """
        Get comprehensive insights about an email address
        """
        insights = {
            'email': email,
            'domain_info': {},
            'social_presence': {},
            'professional_networks': {},
            'risk_assessment': {},
            'deliverability': {}
        }
        
        try:
            domain = email.split('@')[1]
            
            # Domain information
            insights['domain_info'] = self._analyze_domain(domain)
            
            # Social media presence
            insights['social_presence'] = self._check_social_presence(email)
            
            # Professional networks
            insights['professional_networks'] = self._check_professional_presence(email)
            
            # Risk assessment
            insights['risk_assessment'] = self._assess_email_risk(email)
            
            # Deliverability check
            insights['deliverability'] = self._verify_email_deliverability(email)
            
        except Exception as e:
            logger.error(f"Email insights failed for {email}: {e}")
        
        return insights
    
    def _analyze_domain(self, domain: str) -> Dict:
        """
        Analyze domain characteristics
        """
        info = {
            'domain': domain,
            'mx_records': [],
            'domain_age': None,
            'company_domain': False,
            'free_email': False
        }
        
        try:
            # Check if it's a free email provider
            free_providers = [
                'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
                'aol.com', 'icloud.com', 'protonmail.com', 'mail.com'
            ]
            info['free_email'] = domain.lower() in free_providers
            info['company_domain'] = not info['free_email']
            
            # Get MX records
            try:
                mx_records = dns.resolver.resolve(domain, 'MX')
                info['mx_records'] = [str(mx.exchange) for mx in mx_records]
            except:
                pass
            
        except Exception as e:
            logger.debug(f"Domain analysis failed for {domain}: {e}")
        
        return info
    
    def _check_social_presence(self, email: str) -> Dict:
        """
        Check social media presence for email
        """
        presence = {
            'gravatar': False,
            'twitter': False,
            'facebook': False,
            'linkedin': False
        }
        
        try:
            # Check Gravatar
            import hashlib
            email_hash = hashlib.md5(email.lower().encode()).hexdigest()
            gravatar_url = f"https://www.gravatar.com/{email_hash}.json"
            
            try:
                response = self.session.get(gravatar_url)
                if response.status_code == 200:
                    presence['gravatar'] = True
            except:
                pass
            
            # Note: Social media checks would require API access
            # This is a basic framework for social presence checking
            
        except Exception as e:
            logger.debug(f"Social presence check failed for {email}: {e}")
        
        return presence
    
    def _check_professional_presence(self, email: str) -> Dict:
        """
        Check professional network presence
        """
        presence = {
            'linkedin': False,
            'github': False,
            'stackoverflow': False,
            'company_directory': False
        }
        
        # Basic implementation - would require API access for full functionality
        return presence
    
    def _assess_email_risk(self, email: str) -> Dict:
        """
        Assess risk factors for email address
        """
        risk = {
            'overall_risk': 'medium',
            'factors': [],
            'score': 50  # 0-100, lower is better
        }
        
        try:
            domain = email.split('@')[1]
            prefix = email.split('@')[0]
            
            # Check for suspicious patterns
            if len(prefix) < 3:
                risk['factors'].append('Very short email prefix')
                risk['score'] += 20
            
            if any(char in prefix for char in ['..', '--', '__']):
                risk['factors'].append('Suspicious characters in prefix')
                risk['score'] += 15
            
            # Check domain reputation
            suspicious_domains = ['temp-mail.org', '10minutemail.com', 'guerrillamail.com']
            if any(suspicious in domain for suspicious in suspicious_domains):
                risk['factors'].append('Temporary email domain')
                risk['score'] += 50
            
            # Determine overall risk
            if risk['score'] < 30:
                risk['overall_risk'] = 'low'
            elif risk['score'] > 70:
                risk['overall_risk'] = 'high'
            
        except Exception as e:
            logger.debug(f"Risk assessment failed for {email}: {e}")
        
        return risk
    
    def find_email_variations(self, first_name: str, last_name: str, domain: str) -> List[Dict]:
        """
        Generate and test all possible email variations
        """
        variations = []
        
        # Generate all possible patterns
        patterns = [
            f"{first_name}.{last_name}@{domain}",
            f"{first_name}{last_name}@{domain}",
            f"{first_name}@{domain}",
            f"{last_name}@{domain}",
            f"{first_name[0]}.{last_name}@{domain}",
            f"{first_name}.{last_name[0]}@{domain}",
            f"{first_name[0]}{last_name}@{domain}",
            f"{first_name}{last_name[0]}@{domain}",
            f"{first_name}-{last_name}@{domain}",
            f"{first_name}_{last_name}@{domain}",
            f"{last_name}.{first_name}@{domain}",
            f"{last_name}{first_name}@{domain}",
            f"{last_name}-{first_name}@{domain}",
            f"{last_name}_{first_name}@{domain}",
            f"{first_name[0]}.{last_name}@{domain}",
        ]
        
        for pattern in patterns:
            if self._is_valid_email_format(pattern):
                # Quick validation
                confidence = 30  # Base confidence
                
                # Boost confidence for common patterns
                if f"{first_name}.{last_name}" in pattern:
                    confidence += 40
                elif f"{first_name}{last_name}" in pattern:
                    confidence += 35
                elif first_name in pattern and last_name in pattern:
                    confidence += 25
                
                # Basic domain check
                if self._verify_domain_mx(domain):
                    confidence += 20
                
                variations.append({
                    'email': pattern,
                    'confidence': min(confidence, 95),
                    'pattern_type': self._classify_pattern(pattern, first_name, last_name)
                })
        
        # Sort by confidence
        variations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return variations
    
    def _classify_pattern(self, email: str, first_name: str, last_name: str) -> str:
        """
        Classify the email pattern type
        """
        prefix = email.split('@')[0]
        
        if f"{first_name}.{last_name}" == prefix:
            return "first.last"
        elif f"{first_name}{last_name}" == prefix:
            return "firstlast"
        elif f"{first_name[0]}{last_name}" == prefix:
            return "flast"
        elif f"{first_name}{last_name[0]}" == prefix:
            return "firstl"
        elif first_name == prefix:
            return "first"
        elif last_name == prefix:
            return "last"
        else:
            return "other"
    
    def export_results(self, results: List[Dict], format: str = 'csv') -> str:
        """
        Export email finding results in various formats
        """
        if format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            
            if results:
                fieldnames = results[0].keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            return output.getvalue()
        
        elif format == 'json':
            import json
            return json.dumps(results, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        Get statistics from email finding results
        """
        stats = {
            'total_searches': len(results),
            'emails_found': 0,
            'verified_emails': 0,
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'source_distribution': {},
            'average_confidence': 0
        }
        
        total_confidence = 0
        
        for result in results:
            if result.get('email'):
                stats['emails_found'] += 1
                
                confidence = result.get('confidence', 0)
                total_confidence += confidence
                
                if confidence > 80:
                    stats['confidence_distribution']['high'] += 1
                elif confidence > 50:
                    stats['confidence_distribution']['medium'] += 1
                else:
                    stats['confidence_distribution']['low'] += 1
                
                if result.get('verified'):
                    stats['verified_emails'] += 1
                
                source = result.get('source', 'unknown')
                stats['source_distribution'][source] = stats['source_distribution'].get(source, 0) + 1
        
        if stats['emails_found'] > 0:
            stats['average_confidence'] = round(total_confidence / stats['emails_found'], 2)
            stats['success_rate'] = round((stats['emails_found'] / stats['total_searches']) * 100, 2)
        else:
            stats['success_rate'] = 0
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    finder = EmailFinder()
    
    # Test single email finding
    result = finder.find_email("john", "doe", "example.com")
    if result:
        print(f"Found: {result['email']} (Confidence: {result['confidence']}%)")
    
    # Test bulk email finding
    contacts = [
        {"first_name": "jane", "last_name": "smith", "domain": "company.com"},
        {"first_name": "bob", "last_name": "johnson", "domain": "business.org"}
    ]
    
    bulk_results = finder.bulk_find_emails(contacts)
    print(f"Bulk search completed: {len(bulk_results)} contacts processed")
    
    # Get statistics
    stats = finder.get_statistics(bulk_results)
    print(f"Success rate: {stats['success_rate']}%")