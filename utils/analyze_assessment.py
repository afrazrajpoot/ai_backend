

from typing import List, Optional, Dict, Any
from schemas.assessment import AssessmentPart
from utils.logger import logger
def analyze_assessment_data(data: List[AssessmentPart]) -> List[Dict[str, Any]]:
    """
    For each AssessmentPart, return a dict:
      {
        "part": <part name>,
        "optionCounts": {"A":7,"B":4,...},
        "total": 12,
        "percentages": {"A":58.3,...},
        "majorityOptions": ["A"],  # existing shape for backward compatibility
        "maxCount": 7
      }
    """
    results = []
    for part_data in data:
        try:
            option_counts = part_data.optionCounts.dict(exclude_none=True)
        except Exception:
            # If optionCounts is already a dict or unexpected, try to use it directly
            option_counts = getattr(part_data, "optionCounts", {}) or {}
            if hasattr(option_counts, "dict"):
                option_counts = option_counts.dict(exclude_none=True)
        # ensure counts are ints
        counts_clean = {}
        for k, v in (option_counts or {}).items():
            try:
                counts_clean[str(k)] = int(v)
            except Exception:
                logger.warning(f"Non-int count for {k} in part {part_data.part}: {v}. Coercing to 0")
                counts_clean[str(k)] = 0

        total = sum(counts_clean.values())
        percentages = {k: (v/total*100) if total else 0.0 for k,v in counts_clean.items()}
        max_count = max(counts_clean.values()) if counts_clean else 0
        majority_options = sorted([opt for opt, cnt in counts_clean.items() if cnt == max_count and cnt > 0])

        results.append({
            "part": part_data.part,
            "optionCounts": counts_clean,
            "total": total,
            "percentages": percentages,
            "majorityOptions": majority_options if majority_options else None,
            "maxCount": max_count
        })
    logger.info("analyze_assessment_data output prepared")
    logger.debug(results)
    return results