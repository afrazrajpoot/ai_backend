

from typing import List, Optional
from schemas.assessment import AssessmentPart
def analyze_assessment_data(data: List[AssessmentPart]):
    results = []
    for part_data in data:
        option_counts = part_data.optionCounts.dict(exclude_none=True)
        if not option_counts:
            result = {"part": part_data.part, "majorityOptions": None, "maxCount": 0}
            results.append(result)
            continue
        
        # Find maximum count and majority options
        max_count = max(option_counts.values(), default=0)
        majority_options = [
            option for option, count in option_counts.items() if count == max_count and count > 0
        ]
        majority_options.sort()  # Sort for consistent display
        
        result = {
            "part": part_data.part,
            "majorityOptions": majority_options if majority_options else None,
            "maxCount": max_count
        }
        results.append(result)
    
    return results