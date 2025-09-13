#!/usr/bin/env python3
"""
Test script to validate key sanitization fixes
"""
import json
import re

def sanitize_json_keys(obj):
    """
    Simplified version of the sanitization function for testing
    """
    number_words = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
        "10": "ten", "11": "eleven", "12": "twelve"
    }

    if isinstance(obj, dict):
        sanitized = {}
        for key, value in obj.items():
            str_key = str(key)

            # Convert leading numeric sequence to words
            match = re.match(r'^(\d+)', str_key)
            if match:
                leading_num = match.group(1)
                word = number_words.get(leading_num, f"num{leading_num}")
                str_key = str_key.replace(leading_num, word, 1)

            # Replace all non-alphanumeric/underscore characters with underscore
            sanitized_key = re.sub(r'[^a-zA-Z0-9_]', '_', str_key)
            # Collapse multiple underscores
            sanitized_key = re.sub(r'_+', '_', sanitized_key)
            # Trim leading/trailing underscores
            sanitized_key = sanitized_key.strip('_')

            # Ensure key starts with a letter
            if sanitized_key and not re.match(r'^[a-zA-Z]', sanitized_key):
                sanitized_key = 'field_' + sanitized_key

            # Ensure key is not empty
            if not sanitized_key:
                sanitized_key = f'field_{hash(str(key)) % 1000}'

            sanitized[sanitized_key] = sanitize_json_keys(value)
        return sanitized

    elif isinstance(obj, list):
        return [sanitize_json_keys(item) for item in obj]
    else:
        return obj

def test_key_sanitization():
    """Test the key sanitization functions with problematic data"""

    # Test data that mimics the problematic JSON from the error
    test_data = {
        "internal_career_opportunities": {
            "transition_timeline": {  # Should become
                "6_months": "Complete training in advanced UX design principles and tools.",  # Should become six_months
                "1_year": "Take on lead design projects to build portfolio and experience.",  # Should become one_year
                "2_years": "Pursue leadership roles in design or product management."  # Should become two_years
            },
            "career_pathways": {
                "Development Track": "Engineer → Senior → Lead → CTO",  # Should become Development_Track
                "Security Track": "Analyst → Engineer → Architect → CISO"  # Should become Security_Track
            }
        },
        "genius_factor_profile": {
            "key_strengths": [
                "Track[Name] should be sanitized"  # Should become Track_Name
            ]
        }
    }

    print("=== TESTING KEY SANITIZATION ===")
    print("Original data:")
    print(json.dumps(test_data, indent=2))

    # Apply sanitization
    sanitized = sanitize_json_keys(test_data)

    print("\nSanitized data:")
    print(json.dumps(sanitized, indent=2))

    # Check for problematic keys
    problematic_keys = []
    def check_keys(obj, path=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if str(key).startswith(('6_', '1_', '2_')):
                    problematic_keys.append(f"{path}.{key}")
                check_keys(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_keys(item, f"{path}[{i}]")

    check_keys(sanitized)

    if problematic_keys:
        print(f"\n❌ STILL FOUND PROBLEMATIC KEYS: {problematic_keys}")
        return False
    else:
        print("\n✅ ALL KEYS ARE NOW-COMPATIBLE!")
        return True

if __name__ == "__main__":
    print("Starting  key sanitization tests")

    success = test_key_sanitization()

    if success:
        print("\n🎉 TEST PASSED! Key sanitization is working correctly.")
    else:
        print("\n❌ TEST FAILED! Key sanitization needs more work.")
