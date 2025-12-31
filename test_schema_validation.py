#!/usr/bin/env python3
"""
Test script to verify AI service report generation and database save
Tests that reports match the IndividualEmployeeReport schema
"""

import asyncio
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from services.ai_service import AIService
from controllers.assessment_analyze import AssessmentController
from utils.logger import logger


async def test_report_validation():
    """Test that report structure matches schema"""
    print("\n" + "="*80)
    print("TEST: Report Structure Validation")
    print("="*80)
    
    # Sample assessment data
    sample_data = {
        "userId": "test-user-123",
        "hrId": "test-hr-456",
        "employeeName": "Test Employee",
        "employeeEmail": "test@example.com",
        "departement": "Engineering",
        "data": [
            {
                "part": "Self-Awareness Part 1",
                "optionCounts": {"A": 5, "B": 3, "C": 2, "D": 0}
            },
            {
                "part": "Talent Part 1",
                "optionCounts": {"A": 4, "B": 4, "C": 1, "D": 1}
            },
            {
                "part": "Passion Part 1",
                "optionCounts": {"A": 6, "B": 2, "C": 1, "D": 1}
            }
        ]
    }
    
    try:
        # Test step 1: Generate report
        print("\n📊 Step 1: Generating report with AI service...")
        result = await AIService.generate_career_recommendation(
            "Test analysis",
            [],
            user_id=None,  # Skip user data fetch for test
            payload_data=sample_data
        )
        
        if result.get("status") != "success":
            print(f"❌ FAILED: Report generation failed - {result.get('message', 'Unknown error')}")
            return False
        
        report = result.get("report", {})
        print(f"✅ Report generated successfully")
        
        # Test step 2: Validate required sections
        print("\n🔍 Step 2: Validating report sections...")
        required_sections = [
            "executive_summary",
            "genius_factor_profile",
            "current_role_alignment_analysis",
            "internal_career_opportunities",
            "retention_and_mobility_strategies",
            "development_action_plan",
            "personalized_resources",
            "data_sources_and_methodology",
            "genius_factor_score",
            "retention_risk_score",
            "mobility_opportunity_score"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in report:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ FAILED: Missing sections: {', '.join(missing_sections)}")
            return False
        else:
            print(f"✅ All {len(required_sections)} required sections present")
        
        # Test step 3: Validate score types and ranges
        print("\n🔢 Step 3: Validating scores...")
        scores_valid = True
        for score_field in ["genius_factor_score", "retention_risk_score", "mobility_opportunity_score"]:
            score = report.get(score_field)
            if not isinstance(score, int):
                print(f"❌ FAILED: {score_field} is not an integer: {type(score)}")
                scores_valid = False
            elif score < 0 or score > 100:
                print(f"❌ FAILED: {score_field} out of range (0-100): {score}")
                scores_valid = False
            else:
                print(f"✅ {score_field}: {score} (valid)")
        
        if not scores_valid:
            return False
        
        # Test step 4: Validate executive summary
        print("\n📝 Step 4: Validating executive summary...")
        exec_summary = report.get("executive_summary")
        if not isinstance(exec_summary, str):
            print(f"❌ FAILED: executive_summary is not a string: {type(exec_summary)}")
            return False
        elif len(exec_summary) < 10:
            print(f"❌ FAILED: executive_summary too short: {len(exec_summary)} chars")
            return False
        else:
            print(f"✅ executive_summary valid ({len(exec_summary)} chars)")
        
        # Test step 5: Validate JSON structures
        print("\n🗂️  Step 5: Validating JSON field structures...")
        json_fields = [
            "genius_factor_profile",
            "current_role_alignment_analysis",
            "internal_career_opportunities",
            "retention_and_mobility_strategies",
            "development_action_plan",
            "personalized_resources",
            "data_sources_and_methodology"
        ]
        
        json_valid = True
        for field in json_fields:
            value = report.get(field)
            if not isinstance(value, (dict, list)):
                print(f"❌ FAILED: {field} is not a dict/list: {type(value)}")
                json_valid = False
            else:
                print(f"✅ {field}: {type(value).__name__}")
        
        if not json_valid:
            return False
        
        # Test step 6: Test controller validation
        print("\n✔️  Step 6: Testing controller validation...")
        final_result = {
            "userId": sample_data["userId"],
            "hrId": sample_data["hrId"],
            "departement": sample_data["departement"],
            "report": report,
            "risk_analysis": result.get("risk_analysis", {})
        }
        
        validated = AssessmentController._validate_report_for_db(final_result)
        print(f"✅ Controller validation passed")
        
        # Print summary
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        print(f"\nReport Summary:")
        print(f"  - Genius Factor Score: {report['genius_factor_score']}")
        print(f"  - Retention Risk Score: {report['retention_risk_score']}")
        print(f"  - Mobility Opportunity Score: {report['mobility_opportunity_score']}")
        print(f"  - Executive Summary: {exec_summary[:100]}...")
        print(f"  - Primary Genius Factor: {result.get('context_analysis', {}).get('primary_genius_factor', 'N/A')}")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run tests"""
    print("\n" + "🧪"*40)
    print("AI SERVICE SCHEMA VALIDATION TEST")
    print("🧪"*40)
    
    success = await test_report_validation()
    
    if success:
        print("\n✅ All validation tests passed!")
        print("The AI service generates reports that match the database schema.\n")
        return 0
    else:
        print("\n❌ Some tests failed!")
        print("Please check the errors above.\n")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
