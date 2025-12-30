import asyncio
import json
import os
from services.ai_service import AIService
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_generation():
    print("Starting AI Service Verification...")
    
    # Mock Payload Data
    payload_data = {
        "part1": {"A": 5, "B": 2, "C": 1, "D": 2},
        "part2": {"A": 10, "B": 3, "C": 2, "D": 0},
        "part3": {"A": 8, "B": 4, "C": 3, "D": 0},
        "part4": {"A": 10, "B": 0, "C": 0, "D": 0, "E": 0, "F": 2, "G": 0, "H": 0, "I": 0}
    }
    
    try:
        result = await AIService.generate_career_recommendation(
            analysis_result="Test Analysis",
            all_answers={},
            user_id=None,
            payload_data=payload_data
        )
        
        if result.get("status") == "success":
            print("\nSUCCESS: Report Generated!")
            print("-" * 50)
            scores = result.get("risk_analysis", {}).get("scores", {})
            print(f"Genius Factor Score: {scores.get('genius_factor_score')}")
            print(f"Retention Risk Score: {scores.get('retention_risk_score')}")
            print(f"Mobility Opportunity Score: {scores.get('mobility_opportunity_score')}")
            
            # Validation
            if scores.get('genius_factor_score') == 75:
                print("\nWARNING: Genius Factor Score is exactly 75. Check if dynamic scoring is working.")
            else:
                print("\nVALIDATION PASSED: Genius Factor Score is dynamic.")
                
            print("-" * 50)
        else:
            print(f"\nFAILED: {result.get('error')}")
            
    except Exception as e:
        print(f"\nEXCEPTION: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_generation())
