import pandas as pd
import json
from typing import List, Dict, Optional
from io import BytesIO
import csv

class ExportService:
    def __init__(self):
        self.supported_formats = ['csv', 'xlsx', 'json']
    
    def export_leads(self, leads: List[Dict], format: str = 'csv') -> BytesIO:
        """
        Export leads data in specified format
        """
        if format not in self.supported_formats:
            raise ValueError(f"Unsupported format. Choose from: {self.supported_formats}")
        
        if format == 'csv':
            return self._export_csv(leads)
        elif format == 'xlsx':
            return self._export_xlsx(leads)
        elif format == 'json':
            return self._export_json(leads)
    
    def _export_csv(self, leads: List[Dict]) -> BytesIO:
        """Export to CSV format"""
        # Flatten the data structure
        flattened_leads = []
        for lead in leads:
            flat_lead = {
                'first_name': lead.get('first_name', ''),
                'last_name': lead.get('last_name', ''),
                'email': lead.get('email', ''),
                'phone': lead.get('phone', ''),
                'title': lead.get('title', ''),
                'linkedin_url': lead.get('linkedin_url', ''),
                'source': lead.get('source', ''),
                'status': lead.get('status', ''),
                'score': lead.get('score', 0),
                'company_name': lead.get('company', {}).get('name', ''),
                'company_domain': lead.get('company', {}).get('domain', ''),
                'company_industry': lead.get('company', {}).get('industry', ''),
                'company_size': lead.get('company', {}).get('size', ''),
                'company_location': lead.get('company', {}).get('location', ''),
                'created_at': lead.get('created_at', '')
            }
            flattened_leads.append(flat_lead)
        
        # Convert to DataFrame
        df = pd.DataFrame(flattened_leads)
        
        # Create BytesIO object
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return output
    
    def _export_xlsx(self, leads: List[Dict]) -> BytesIO:
        """Export to Excel format"""
        # Flatten the data structure (same as CSV)
        flattened_leads = []
        for lead in leads:
            flat_lead = {
                'First Name': lead.get('first_name', ''),
                'Last Name': lead.get('last_name', ''),
                'Email': lead.get('email', ''),
                'Phone': lead.get('phone', ''),
                'Title': lead.get('title', ''),
                'LinkedIn URL': lead.get('linkedin_url', ''),
                'Source': lead.get('source', ''),
                'Status': lead.get('status', ''),
                'Score': lead.get('score', 0),
                'Company Name': lead.get('company', {}).get('name', ''),
                'Company Domain': lead.get('company', {}).get('domain', ''),
                'Company Industry': lead.get('company', {}).get('industry', ''),
                'Company Size': lead.get('company', {}).get('size', ''),
                'Company Location': lead.get('company', {}).get('location', ''),
                'Created At': lead.get('created_at', '')
            }
            flattened_leads.append(flat_lead)
        
        # Convert to DataFrame
        df = pd.DataFrame(flattened_leads)
        
        # Create BytesIO object
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Leads', index=False)
        output.seek(0)
        
        return output
    
    def _export_json(self, leads: List[Dict]) -> BytesIO:
        """Export to JSON format"""
        output = BytesIO()
        json_data = json.dumps(leads, indent=2, default=str)
        output.write(json_data.encode('utf-8'))
        output.seek(0)
        
        return output