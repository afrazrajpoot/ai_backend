#!/usr/bin/env python3
"""
Test script for database save functionality
Run this script to test saving assessment reports to the database
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from controllers.assessment_analyze import AssessmentController
from utils.logger import logger

# Sample test data that mimics the structure from your actual assessment
SAMPLE_REPORT_DATA = {
    "userId": "4481f0cc-d7f7-438f-89b0-4e9dde358c56",
    "report": {
        "executive_summary": "This is a test executive summary for database testing.",
        "genius_factor_profile": {
            "primary_genius_factor": "Tech Genius",
            "description": "Test description for tech genius",
            "key_strengths": ["Problem solving", "Technical skills"],
            "secondary_genius_factor": "Tech Genius",
            "energy_sources": ["Coding", "Learning new tech"]
        },
        "current_role_alignment_analysis": {
            "alignment_score": "85",
            "assessment": "Test assessment",
            "strengths_utilized": ["Technical problem-solving"],
            "underutilized_talents": ["Leadership opportunities"],
            "retention_risk_level": "Moderate"
        },
        "internal_career_opportunities": {
            "primary_industry": "Technology",
            "secondary_industry": "Financial Services",
            "recommended_departments": ["Software Development", "Data Analytics"],
            "specific_role_suggestions": ["Technical Lead", "Data Scientist"],
            "career_pathways": {
                "Development Track": "Software Engineer → Senior Developer → Technical Lead",
                "Security Track": "Security Analyst → Security Engineer → CISO",
                "AI Track": "Data Scientist → ML Engineer → AI Director"
            },
            "progress_transition_timeline": {
                "6_months": "Transition to Technical Lead role",
                "1_year": "Move to CTO position",
                "2_years": "Establish thought leadership"
            },
            "required_skill_development": ["Python", "AWS", "Machine Learning"]
        },
        "retention_and_mobility_strategies": {
            "retention_strategies": ["Mentorship programs", "Clear advancement paths"],
            "internal_mobility_recommendations": ["Lateral moves", "Talent marketplace"],
            "development_support": ["Training programs", "Networking opportunities"]
        },
        "development_action_plan": {
            "thirty_day_goals": ["Enroll in leadership training"],
            "ninety_day_goals": ["Lead a small project"],
            "six_month_goals": ["Take on Technical Lead role"],
            "networking_strategy": ["Attend conferences", "Join professional associations"]
        },
        "personalized_resources": {
            "affirmations": ["I am a natural problem-solver"],
            "mindfulness_practices": ["Daily meditation", "Journaling"],
            "reflection_questions": ["What challenges have I overcome?"],
            "learning_resources": ["Online courses", "Technical books"]
        },
        "data_sources_and_methodology": {
            "data_sources": ["Genius Factor Assessment", "Industry Mapping"],
            "methodology": "Structured framework analysis"
        },
        "genius_factor_score": 85
    },
    "risk_analysis": {
        "analysis_summary": "Test risk analysis summary",
        "scores": {
            "genius_factor_score": 90,
            "retention_risk_score": 30,
            "mobility_opportunity_score": 95
        },
        "trends": {
            "retention_trends": "Focus on employee engagement",
            "mobility_trends": "Internal mobility programs gaining traction"
        },
        "recommendations": ["Implement mentorship", "Create advancement pathways"],
        "genius_factors": ["Tech Genius"],
        "company": "Fortune 1000 Test Company"
    },
    "metadata": {
        "processingTimestamp": "2025-09-13T15:43:31.000000",
        "modelUsed": "gpt-4o-mini",
        "dataSourcesUsed": ["test-source-1", "test-source-2"]
    }
}

async def test_database_save():
    """Test the database save functionality"""
    try:
        print("🧪 Starting database save test...")
        print(f"📊 Test data structure: {list(SAMPLE_REPORT_DATA.keys())}")
        
        # Test the save functionality
        result = await AssessmentController.save_to_database(SAMPLE_REPORT_DATA)
        
        print("✅ Database save test completed!")
        print(f"📋 Result: {json.dumps(result, indent=2)}")
        
        if result.get("status") == "success":
            print(f"🎉 SUCCESS: Report saved with ID: {result.get('report_id')}")
            return True
        else:
            print(f"❌ FAILED: {result.get('message')}")
            return False
            
    except Exception as e:
        print(f"💥 ERROR: {str(e)}")
        logger.error(f"Test database save failed: {str(e)}")
        return False

async def test_database_connection():
    """Test basic database connection"""
    try:
        print("🔌 Testing database connection...")
        
        from prisma import Prisma
        prisma = Prisma()
        await prisma.connect()
        
        # Try to count users
        user_count = await prisma.user.count()
        print(f"👥 Found {user_count} users in database")
        
        await prisma.disconnect()
        print("✅ Database connection test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Database Save Tests")
    print("=" * 50)
    
    # Test 1: Database connection
    print("\n📝 Test 1: Database Connection")
    conn_success = await test_database_connection()
    
    if not conn_success:
        print("🛑 Database connection failed. Exiting tests.")
        return
    
    # Test 2: Database save
    print("\n📝 Test 2: Database Save")
    save_success = await test_database_save()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print(f"Database Connection: {'✅ PASS' if conn_success else '❌ FAIL'}")
    print(f"Database Save: {'✅ PASS' if save_success else '❌ FAIL'}")
    
    if conn_success and save_success:
        print("🎉 All tests passed! Database functionality is working.")
    else:
        print("⚠️  Some tests failed. Check the logs for details.")

if __name__ == "__main__":
    print("Running database tests...")
    asyncio.run(main())
