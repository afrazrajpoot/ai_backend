# Updated generate_career_recommendation method in AIService class


@classmethod
async def _calculate_paid_genius_score(cls, parsed_metrics: Dict, analysis_result: str, all_answers: Any) -> int:
    """
    Calculate enhanced genius factor score for paid users
    """
    try:
        # Extract primary and secondary percentages from parsed_metrics
        primary_name = parsed_metrics.get("primary_genius", "")
        secondary_name = parsed_metrics.get("secondary_genius", "")
        
        # Parse analysis result for percentages
        import re
        
        # Try to extract primary percentage
        primary_percentage = 65  # Default
        primary_match = re.search(r"Primary.*?(\d+\.?\d*)%", analysis_result)
        if primary_match:
            primary_percentage = float(primary_match.group(1))
        
        # Try to extract secondary percentage
        secondary_percentage = 35  # Default
        secondary_match = re.search(r"Secondary.*?(\d+\.?\d*)%", analysis_result)
        if secondary_match:
            secondary_percentage = float(secondary_match.group(1))
        
        # Enhanced calculation based on algorithm
        base_score = (primary_percentage * 0.40) + (secondary_percentage * 0.25)
        
        # Add response consistency bonus (simplified)
        if isinstance(all_answers, dict):
            total_answers = sum(len(v) for v in all_answers.values() if isinstance(v, list))
            if total_answers > 0:
                consistency = min(100, (total_answers / 68) * 100)  # 68 total questions
                base_score += (consistency * 0.20)
        
        # Factor synergy (simplified based on common combinations)
        synergy_map = {
            ("Tech Genius", "Number Genius"): 85,
            ("Business Genius", "Strategic Genius"): 80,
            ("Creative Genius", "Galvanizing Genius"): 75,
            ("Service Genius", "Enablement Genius"): 70,
        }
        
        synergy = synergy_map.get((primary_name, secondary_name), 50)
        base_score += (synergy * 0.15)
        
        # Apply adjustments
        variance = abs(primary_percentage - secondary_percentage)
        if variance < 15:
            base_score -= 15  # Low variance penalty
        elif primary_percentage > 70:
            base_score += 10  # High concentration bonus
        
        if secondary_percentage < 15:
            base_score -= 8  # Weak secondary penalty
        
        # Clamp and round
        final_score = max(35, min(95, base_score))
        
        return round(final_score)
        
    except Exception as e:
        logger.error(f"Error calculating paid genius score: {str(e)}")
        return 65  # Default fallback

@classmethod
def _calculate_free_genius_score(cls, parsed_metrics: Dict) -> int:
    """
    Calculate basic genius factor score for free users
    """
    # Simple calculation for free users
    primary_name = parsed_metrics.get("primary_genius", "")
    
    # Map genius names to base scores
    base_scores = {
        "Tech Genius": 75,
        "Business Genius": 72,
        "Creative Genius": 68,
        "Service Genius": 65,
        "Strategic Genius": 70,
        "Number Genius": 73,
        "Galvanizing Genius": 67,
        "Enablement Genius": 64,
        "Orchestrating Genius": 71,
        "Tenacity Genius": 69,
    }
    
    base_score = base_scores.get(primary_name, 65)
    
    # Add small random variation
    import random
    variation = random.randint(-5, 5)
    
    final_score = max(40, min(100, base_score + variation))
    
    return final_score