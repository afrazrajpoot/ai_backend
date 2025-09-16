from prisma import Prisma
import json
from typing import Dict, Any, Optional

class DatabaseService:
    def __init__(self):
        self.prisma = Prisma()

    async def connect(self):
        """Connect to the database"""
        await self.prisma.connect()

    async def disconnect(self):
        """Disconnect from the database"""
        await self.prisma.disconnect()

    async def save_company_data(self, company_detail: Any) -> Dict[str, Any]:
        """
        Save company data to the database
        Args:
            company_detail: The parsed JSON data from the file
        Returns:
            The created company record
        """
        try:
            # Ensure we're connected
            if not self.prisma.is_connected():
                await self.connect()

            # Convert to appropriate format for Prisma
            json_data = None
            
            if isinstance(company_detail, dict):
                # Convert dict to JSON string
                json_data = json.dumps(company_detail)
            elif isinstance(company_detail, str):
                # If it's already a string, use as-is (assuming it's valid JSON)
                json_data = company_detail
            elif company_detail is None:
                # Handle None case
                json_data = json.dumps({"message": "No data provided"})
            else:
                # For any other type, convert to string and wrap
                json_data = json.dumps({"raw_content": str(company_detail)})

            # print(f"Final JSON data to save: {json_data}")
            # print(f"Final JSON data type: {type(json_data)}")

            # Create company record with JSON string
            company = await self.prisma.company.create(
                data={
                    "companyDetail": json_data
                }
            )
            
            return {
                "id": company.id,
                "companyDetail": company.companyDetail,
                "createdAt": company.createdAt,
                "updatedAt": company.updatedAt
            }
        except Exception as e:
            # print(f"Database error details: {str(e)}")
            # print(f"Error type: {type(e)}")
            raise Exception(f"Error saving company data: {str(e)}")
    async def get_company_by_id(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Get a company by ID"""
        try:
            if not self.prisma.is_connected():
                await self.connect()

            company = await self.prisma.company.find_unique(
                where={"id": company_id}
            )
            
            if company:
                return {
                    "id": company.id,
                    "companyDetail": company.companyDetail,
                    "createdAt": company.createdAt,
                    "updatedAt": company.updatedAt
                }
            return None
        except Exception as e:
            raise Exception(f"Error fetching company: {str(e)}")

    async def get_all_companies(self, skip: int = 0, take: int = 100) -> list:
        """Get all companies with pagination"""
        try:
            if not self.prisma.is_connected():
                await self.connect()

            companies = await self.prisma.company.find_many(
                skip=skip,
                take=take,
                order_by={"createdAt": "desc"}
            )
            
            return [
                {
                    "id": company.id,
                    "companyDetail": company.companyDetail,
                    "createdAt": company.createdAt,
                    "updatedAt": company.updatedAt
                }
                for company in companies
            ]
        except Exception as e:
            raise Exception(f"Error fetching companies: {str(e)}")