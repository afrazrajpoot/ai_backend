from typing import List
from fastapi import UploadFile
from services.parse_companies.parse_companies import parse_file, process_text_with_langchain
from services.database_services import DatabaseService
import json

class ParseCompaniesController:
    """
    Controller to handle Excel file parsing and processing.
    """
    def __init__(self):
        self.db_service = DatabaseService()

    async def parse_files(self, files: List[UploadFile]):
        results = []
        
        try:
            # Connect to database
            await self.db_service.connect()
            
            for file in files:
                try:
                    # Parse the file
                    text = await parse_file(file)
                    print(text,'text')
                    processed_json = await process_text_with_langchain(text)
                    print(processed_json,'company detail')
                    # Save to database
                    # saved_company = await self.db_service.save_company_data(processed_json)
                    
                    results.append({
                        "filename": file.filename,
                        "parsed_json": processed_json,
                     
                    })
                    
                except Exception as e:
                    results.append({
                        "filename": file.filename,
                        "error": str(e)
                    })
                    
        finally:
            # Always disconnect from database
            await self.db_service.disconnect()
            
        return results

    async def get_company(self, company_id: str):
        """Get a specific company by ID"""
        try:
            await self.db_service.connect()
            company = await self.db_service.get_company_by_id(company_id)
            if not company:
                return None
            return company
        finally:
            await self.db_service.disconnect()

    async def get_all_companies(self, skip: int = 0, take: int = 100):
        """Get all companies with pagination"""
        try:
            await self.db_service.connect()
            companies = await self.db_service.get_all_companies(skip, take)
            return companies
        finally:
            await self.db_service.disconnect()