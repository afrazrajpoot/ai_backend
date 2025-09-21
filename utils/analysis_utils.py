# analysis_utils.py
import logging
from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)

# === FACTOR DICTS (complete as per your document) ===
# Self-awareness / Talent / Passion: A-D
SELF_AWARENESS_FACTORS = {
    "A": {"name": "Number/Tech Genius", "qualities": [
        "Natural analytical and systematic thinking",
        "Preference for logical problem-solving",
        "Comfort with data and technical challenges"
    ]},
    "B": {"name": "Social Genius", "qualities": [
        "Natural relationship-building abilities",
        "Preference for collaborative environments",
        "Comfort with interpersonal dynamics"
    ]},
    "C": {"name": "Visual/Athletic Genius", "qualities": [
        "Natural creative and innovative thinking",
        "Preference for hands-on and visual approaches",
        "Comfort with design and spatial challenges"
    ]},
    "D": {"name": "Word/Spiritual Genius", "qualities": [
        "Natural meaning-making and purpose-driven thinking",
        "Preference for values-based decisions",
        "Comfort with philosophical and communication challenges"
    ]}
}
# reuse for talent/passion (A-D)
TALENT_FACTORS = SELF_AWARENESS_FACTORS
PASSION_FACTORS = SELF_AWARENESS_FACTORS

# Mapping factors (Part IV + Goals) A-I -> full set
MAPPING_FACTORS = {
    "A": {"name": "Tech Genius", "qualities": [
        "Affinity for technology, systems and logical problem-solving",
        "Thrives in technical roles requiring programming or engineering"
    ]},
    "B": {"name": "Social Genius", "qualities": [
        "Ability to build relationships and facilitate human connections",
        "Thrives in people-centered roles such as HR, Sales"
    ]},
    "C": {"name": "Visual Genius", "qualities": [
        "Talent for design, spatial reasoning, and creative expression",
        "Thrives in product design, UX, creative roles"
    ]},
    "D": {"name": "Word Genius", "qualities": [
        "Ability to communicate effectively through written and spoken language",
        "Thrives in communications, PR, content strategy"
    ]},
    "E": {"name": "Athletic Genius", "qualities": [
        "Physical coordination and preference for hands-on activities",
        "Thrives in operations, implementation, physical roles"
    ]},
    "F": {"name": "Number Genius", "qualities": [
        "Analytical thinking and comfort with quantitative problem solving",
        "Thrives in finance, analytics, business intelligence"
    ]},
    "G": {"name": "Eco Genius", "qualities": [
        "Connection to environmental systems and sustainability thinking",
        "Thrives in sustainability, CSR, environmental roles"
    ]},
    "H": {"name": "Word Genius (Communication Focus)", "qualities": [
        "Clear written communication and documentation",
        "Thrives in technical writing, training, knowledge management"
    ]},
    "I": {"name": "Spiritual Genius", "qualities": [
        "Wisdom, empathy, and purpose-driven leadership",
        "Thrives in leadership development, coaching, culture roles"
    ]}
}

# Hybrid mapping examples (expand as you like)
HYBRID_ROLE_MAP = {
    frozenset({"A","F"}): "Data Science / Business Intelligence (Tech + Number)",
    frozenset({"B","D"}): "Communications / Organizational Development (Social + Word)",
    frozenset({"C","A"}): "UX/Product Design (Visual + Tech)",
    frozenset({"E","B"}): "Operations Leadership (Athletic + Social)",
    frozenset({"G","F"}): "Sustainability Analytics (Eco + Number)",
    frozenset({"I","B"}): "Executive Coaching / Culture (Spiritual + Social)"
}

@dataclass
class SectionResult:
    name: str
    counts: Dict[str,int]
    total: int
    percentages: Dict[str,float]
    dominant: List[str]
    secondary: List[str]
    qualities: Dict[str,List[str]]

def _compute_percentages(counts: Dict[str,int], total:int) -> Dict[str,float]:
    return {k: (v/total*100) if total > 0 else 0.0 for k,v in counts.items()}

def analyze_section_counts(section_name: str,
                           counts: Dict[str,int],
                           factor_dict: Dict[str,Dict]) -> SectionResult:
    """
    Analyze raw counts (already aggregated per section).
    Returns dominant and secondary letters and attached qualities.
    """
    total = sum(counts.values())
    if total == 0:
        logger.info(f"[{section_name}] no responses")
        return SectionResult(section_name, counts, 0, {}, [], [], {})

    max_count = max(counts.values())
    # Primary: all letters that equal max_count (handles ties)
    dominant = [k for k,v in counts.items() if v == max_count and v > 0]

    # Secondary: letters within 20% of max (but not in dominant)
    secondary = [k for k,v in counts.items()
                 if k not in dominant and (max_count - v) <= max(1, max_count * 0.2)]

    percentages = _compute_percentages(counts, total)
    qualities = {k: factor_dict.get(k, {}).get("qualities", []) for k in (dominant + secondary) if k in factor_dict}

    logger.info(f"[{section_name}] counts={counts}, total={total}, perc={percentages}, dominant={dominant}, secondary={secondary}")
    return SectionResult(section_name, counts, total, percentages, dominant, secondary, qualities)


def categorize_part_name(part_name: str) -> str:
    """
    Robust classifier for assessment 'part' names.
    Returns one of: SelfAwareness, Talent, Passion, Mapping, Goals.
    Handles case differences, Roman numerals, and descriptive suffixes.
    """
    pn = (part_name or "").strip().lower()

    # Use more specific Roman numeral patterns with word boundaries
    if ("self" in pn or "self-awareness" in pn or "self awareness" in pn
            or "part i:" in pn or pn.startswith("part i ")):
        return "SelfAwareness"

    if ("talent" in pn or "talent audit" in pn
            or "part ii:" in pn or pn.startswith("part ii ") or "part 2" in pn):
        return "Talent"

    if ("passion" in pn or "passion audit" in pn
            or "part iii:" in pn or pn.startswith("part iii ") or "part 3" in pn):
        return "Passion"

    if ("mapping" in pn or "genius factor mapping" in pn
            or "genius mapping" in pn
            or "part iv:" in pn or pn.startswith("part iv ") or "part 4" in pn):
        return "Mapping"

    if ("goal" in pn or "career vision" in pn or "goal setting" in pn
            or "part v:" in pn or pn.startswith("part v ") or "part 5" in pn):
        return "Goals"

    # Fallback: Mapping is the most detailed section; default there.
    logger.debug(f"categorize_part_name fallback for '{part_name}' -> Mapping")
    return "Mapping"



def aggregate_parts_to_sections(parts_results: List[Dict[str,Any]]) -> Dict[str,Dict[str,int]]:
    """
    Input: parts_results: list of dicts like {"part": "Part I: Self-Awareness", "optionCounts": {"A":7,"B":4}, ...}
    Returns: section_counts = { "SelfAwareness": {"A":x,"B":y,...}, ... }
    """
    section_counts: Dict[str, Dict[str,int]] = {
        "SelfAwareness": Counter(),
        "Talent": Counter(),
        "Passion": Counter(),
        "Mapping": Counter(),
        "Goals": Counter()
    }
    # ensure all present keys
    for pr in parts_results:
        # Handle both dict and string cases
        if isinstance(pr, dict):
            part_name = pr.get("part", "")
            # optionCounts may be a dict-like
            counts = pr.get("optionCounts") or {}
        elif isinstance(pr, str):
            # If pr is a string, skip it or handle appropriately
            logger.warning(f"Skipping string result in parts_results: {pr}")
            continue
        else:
            logger.warning(f"Unexpected type in parts_results: {type(pr)}")
            continue
            
        # If the optionCounts are not a dict, skip
        if not isinstance(counts, dict):
            logger.warning(f"Unexpected optionCounts type for part '{part_name}': {type(counts)}")
            continue
        section_key = categorize_part_name(part_name)
        # Add counts to section
        for letter, cnt in counts.items():
            try:
                cnt_i = int(cnt)
            except Exception:
                logger.warning(f"Non-int count for {letter} in part '{part_name}': {cnt}")
                continue
            section_counts[section_key][letter] = section_counts[section_key].get(letter, 0) + cnt_i

    # normalize to plain dicts
    return {k: dict(v) for k,v in section_counts.items()}


def determine_primary_secondary_from_mapping(mapping_counts: Dict[str,int]) -> Tuple[List[Dict], List[Dict]]:
    """
    Apply thresholds from scoring guide:
    - Primary Genius Factor = Highest scoring factor from Q51-62
    - Secondary Genius Factor = Second highest scoring factor (if within 20% of primary)
    """
    # convert to items sorted desc
    items = sorted(mapping_counts.items(), key=lambda x: x[1], reverse=True)
    primary = []
    secondary = []

    if not items or items[0][1] == 0:
        logger.info("[GenieusMapping] No valid mapping responses found")
        return primary, secondary

    total_responses = sum(mapping_counts.values())
    logger.info(f"[GeniusMapping] Total mapping responses: {total_responses}, sorted items: {items}")

    top_letter, top_count = items[0]
    top_percentage = (top_count / total_responses) * 100 if total_responses > 0 else 0

    # determine strength based on percentage and absolute count
    def strength_label(cnt: int, total: int) -> str:
        pct = (cnt / total) * 100 if total > 0 else 0
        if pct >= 60:
            return "Strong"
        elif pct >= 40:
            return "Moderate"
        elif pct >= 20:
            return "Secondary"
        else:
            return "Weak"

    primary.append({
        "letter": top_letter,
        "name": MAPPING_FACTORS.get(top_letter, {}).get("name", f"Factor {top_letter}"),
        "count": top_count,
        "percentage": round(top_percentage, 2),
        "strength": strength_label(top_count, total_responses)
    })
    logger.info(f"[GeniusMapping] Primary genius: {primary[0]}")

    # Secondary factor: within 20% of primary (scoring guide requirement)
    if len(items) > 1:
        second_letter, second_count = items[1]
        second_percentage = (second_count / total_responses) * 100 if total_responses > 0 else 0
        
        # Check if second is within 20% of primary
        percentage_diff = top_percentage - second_percentage
        if percentage_diff <= 20 and second_count > 0:
            secondary.append({
                "letter": second_letter,
                "name": MAPPING_FACTORS.get(second_letter, {}).get("name", f"Factor {second_letter}"),
                "count": second_count,
                "percentage": round(second_percentage, 2),
                "strength": strength_label(second_count, total_responses)
            })
            logger.info(f"[GeniusMapping] Secondary genius: {secondary[0]} (within 20% threshold)")
        else:
            logger.info(f"[GeniusMapping] No secondary genius (second factor {second_percentage:.2f}% vs primary {top_percentage:.2f}%, diff={percentage_diff:.2f}%)")

    return primary, secondary


def compute_talent_passion_overlap(talent_counts: Dict[str,int], passion_counts: Dict[str,int]) -> float:
    """
    Compute overlap percentage using min-sum approach:
    overlap = sum( min(talent[l], passion[l]) for all letters ) / total_talent * 100
    This tells how much of talent responses fall into same letters as passion.
    """
    total_talent = sum(talent_counts.values()) or 1
    overlap_sum = 0
    for l, t_cnt in talent_counts.items():
        p_cnt = passion_counts.get(l, 0)
        overlap_sum += min(t_cnt, p_cnt)
    return (overlap_sum / total_talent) * 100 if total_talent else 0.0


def detect_misalignment(talent_counts: Dict[str,int], passion_counts: Dict[str,int]) -> Dict[str,str]:
    """
    For each domain letter, classify alignment:
      - "High Talent, Low Passion"
      - "High Passion, Low Talent"
      - "Aligned"
      - "Low Both"
    Thresholds: 'High' = top 33% of counts for that section or absolute proportion >= 40%
    """
    res = {}
    t_total = sum(talent_counts.values()) or 1
    p_total = sum(passion_counts.values()) or 1
    for l in set(list(talent_counts.keys()) + list(passion_counts.keys())):
        t_pct = (talent_counts.get(l, 0) / t_total) * 100
        p_pct = (passion_counts.get(l, 0) / p_total) * 100
        if t_pct >= 40 and p_pct < 20:
            res[l] = "High Talent, Low Passion"
        elif p_pct >= 40 and t_pct < 20:
            res[l] = "High Passion, Low Talent"
        elif t_pct < 15 and p_pct < 15:
            res[l] = "Low Both"
        else:
            res[l] = "Aligned"
    return res


def analyze_full_from_parts(parts_results: List[Dict[str,Any]]) -> Dict[str,Any]:
    """
    Top-level function: take the output of analyze_assessment_data (list of part dicts),
    aggregate counts into sections, run per-section analysis, and compute final discovery metrics.
    """
    logger.info("Starting aggregation of parts into sections")
    print(f'{parts_results} parts found')
    section_counts = aggregate_parts_to_sections(parts_results)

    # Per-section analysis
    sections_analysis = {}
    sections_analysis["SelfAwareness"] = analyze_section_counts("Self-Awareness", section_counts.get("SelfAwareness", {}), SELF_AWARENESS_FACTORS)
    sections_analysis["Talent"] = analyze_section_counts("Talent", section_counts.get("Talent", {}), TALENT_FACTORS)
    sections_analysis["Passion"] = analyze_section_counts("Passion", section_counts.get("Passion", {}), PASSION_FACTORS)
    sections_analysis["Mapping"] = analyze_section_counts("Genius Mapping", section_counts.get("Mapping", {}), MAPPING_FACTORS)
    sections_analysis["Goals"] = analyze_section_counts("Goals", section_counts.get("Goals", {}), MAPPING_FACTORS)

    # Determine primary and secondary genius from mapping section
    mapping_counts = section_counts.get("Mapping", {})
    logger.info(f"[DeepAnalysis] Mapping section counts: {mapping_counts}")
    primary, secondary = determine_primary_secondary_from_mapping(mapping_counts)

    # Attach qualities to primary and secondary genius
    def get_qualities(letter):
        # Prefer mapping factors, fallback to self-awareness
        return MAPPING_FACTORS.get(letter, {}).get("qualities") or SELF_AWARENESS_FACTORS.get(letter, {}).get("qualities", [])

    for p in primary:
        p["qualities"] = get_qualities(p["letter"])
    for s in secondary:
        s["qualities"] = get_qualities(s["letter"])

    mapping_total = sum(mapping_counts.values())
    logger.info(f"[DeepAnalysis] Mapping section total responses: {mapping_total}")
    
    # Only fallback if mapping section is completely empty
    if mapping_total == 0:
        logger.info("[DeepAnalysis] No mapping responses found; falling back to Self-Awareness for primary detection")
        sa_counts = section_counts.get("SelfAwareness", {})
        logger.info(f"[DeepAnalysis] SelfAwareness fallback counts: {sa_counts}")
        if sa_counts:
            primary_letters = sorted(sa_counts.items(), key=lambda x: x[1], reverse=True)[:2]
            primary = [{
                "letter": l, "name": SELF_AWARENESS_FACTORS.get(l, {}).get("name", f"Factor {l}"),
                "count": c, "strength": "Fallback",
                "qualities": get_qualities(l)
            } for l,c in primary_letters if c > 0]
            logger.info(f"[DeepAnalysis] Fallback primary from Self-Awareness: {primary}")
        else:
            logger.warning("[DeepAnalysis] No Self-Awareness data available for fallback")
    else:
        logger.info(f"[DeepAnalysis] Using Mapping section for genius detection: Primary={primary}, Secondary={secondary}")
        # Add qualities to primary and secondary from mapping
        for p in primary:
            p["qualities"] = get_qualities(p["letter"])
        for s in secondary:
            s["qualities"] = get_qualities(s["letter"])

    # Talent-Passion alignment (numeric overlap)
    talent_counts = section_counts.get("Talent", {})
    passion_counts = section_counts.get("Passion", {})
    talent_passion_overlap_pct = compute_talent_passion_overlap(talent_counts, passion_counts)
    misalignment_map = detect_misalignment(talent_counts, passion_counts)

    # Hybrid detection (15% rule from scoring guide)
    hybrid_key = None
    hybrid_classification = None
    hybrid_qualities = []
    if primary and secondary and len(primary) > 0 and len(secondary) > 0:
        pk = primary[0]["letter"]
        sk = secondary[0]["letter"]
        key = frozenset({pk, sk})
        hybrid_key = HYBRID_ROLE_MAP.get(key)
        
        # Check if factors are within 15% of each other for hybrid classification
        p_pct = primary[0].get("percentage", 0)
        s_pct = secondary[0].get("percentage", 0)
        percentage_diff = p_pct - s_pct
        
        if percentage_diff <= 15:
            hybrid_classification = f"Hybrid: {primary[0]['name']} + {secondary[0]['name']}"
            hybrid_qualities = primary[0].get("qualities", []) + secondary[0].get("qualities", [])
            logger.info(f"[DeepAnalysis] Hybrid detected: {hybrid_classification} (diff: {percentage_diff:.2f}%)")
        else:
            logger.info(f"[DeepAnalysis] No hybrid: percentage difference {percentage_diff:.2f}% > 15% threshold")

    # Confidence level calculation (from scoring guide)
    confidence_level = "Unknown"
    if mapping_total > 0 and len(primary) > 0:
        top_pct = primary[0].get("percentage", 0)
        logger.info(f"[DeepAnalysis] Confidence calc: primary percentage={top_pct:.2f}%")
        
        if top_pct >= 85:
            confidence_level = "High"
        elif top_pct >= 70:
            confidence_level = "Moderate" 
        elif top_pct >= 50:
            confidence_level = "Low"
        else:
            confidence_level = "Very Low"
        
        logger.info(f"[DeepAnalysis] Confidence level: {confidence_level} (based on {top_pct:.2f}%)")
    else:
        logger.info(f"[DeepAnalysis] Cannot calculate confidence: mapping_total={mapping_total}, primary_count={len(primary)}")

    # Talent-Passion alignment label
    alignment_label = "Low"
    if talent_passion_overlap_pct >= 80:
        alignment_label = "High"
    elif talent_passion_overlap_pct >= 60:
        alignment_label = "Moderate"

    # Current role alignment calculation (improved from stub)
    current_role_alignment_label = "Unknown"
    current_role_alignment_pct = None
    
    # For now, keep as Unknown since we don't have direct access to role information
    # in this function. The AI service will handle role alignment calculation
    # when it has access to assessment data and user profile information
    logger.info(f"[DeepAnalysis] Role alignment calculation deferred to AI service layer")

    # Build final output
    final = {
        "sections": {k: asdict(v) for k,v in sections_analysis.items()},
        "primary_genius": primary,
        "secondary_genius": secondary,
        "talent_passion_overlap_pct": talent_passion_overlap_pct,
        "talent_passion_alignment_label": alignment_label,
        "misalignment_map": misalignment_map,
        "hybrid_role_suggestion": hybrid_key,
        "hybrid_classification": hybrid_classification,
        "hybrid_qualities": hybrid_qualities,
        "confidence_level": confidence_level,
        "current_role_alignment_label": current_role_alignment_label,
        "section_counts": section_counts  # raw counts for easier debugging
    }

    logger.info("Deep analysis complete")
    logger.debug(final)
    return final
