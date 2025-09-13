from fastapi import HTTPException
from schemas.assessment import AssessmentData
from services.ai_service import AIService
from utils.analyze_assessment import analyze_assessment_data
from utils.logger import logger
from services.notification_service import NotificationService
from typing import Dict, Any
from prisma import Prisma
from prisma import fields
import json
import re


class AssessmentController:

    # Expanded and more readable number mapping (use underscores for multi-word)
    number_words = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "10": "ten",
        "11": "eleven",
        "12": "twelve",
        "18": "eighteen",
        "24": "twenty_four",
        "30": "thirty",
        "36": "thirty_six",
        "48": "forty_eight",
        "60": "sixty",
        "72": "seventy_two",
        "90": "ninety",
    }

    @staticmethod
    def sanitize_json_keys(obj):
        """
        Recursively sanitize JSON keys to be valid for GraphQL/Prisma:
         - convert leading numbers to words using number_words
         - replace invalid characters (spaces, brackets, punctuation) with underscores
         - collapse multiple underscores
         - trim leading/trailing underscores
         - ensure key starts with a letter by prefixing 'field_' if necessary
        """
        if isinstance(obj, dict):
            sanitized = {}
            for key, value in obj.items():
                str_key = str(key)

                # Convert leading numeric sequence to words (e.g. '6_month' -> 'six_month')
                match = re.match(r'^(\d+)', str_key)
                if match:
                    leading_num = match.group(1)
                    word = AssessmentController.number_words.get(leading_num, f"num{leading_num}")
                    str_key = str_key.replace(leading_num, word, 1)
                    logger.debug(f"Converted numeric key: '{key}' -> '{str_key}' (replaced '{leading_num}' with '{word}')")

                # Replace all non-alphanumeric/underscore characters with underscore
                sanitized_key = re.sub(r'[^a-zA-Z0-9_]', '_', str_key)
                # Collapse multiple underscores
                sanitized_key = re.sub(r'_+', '_', sanitized_key)
                # Trim leading/trailing underscores
                sanitized_key = sanitized_key.strip('_')

                # Ensure key starts with a letter (GraphQL requirement)
                if sanitized_key and not re.match(r'^[a-zA-Z]', sanitized_key):
                    sanitized_key = 'field_' + sanitized_key

                # Ensure key is not empty
                if not sanitized_key:
                    sanitized_key = f'field_{hash(str(key)) % 1000}'

                if sanitized_key != key:
                    logger.debug(f"Sanitized key: '{key}' -> '{sanitized_key}'")

                sanitized[sanitized_key] = AssessmentController.sanitize_json_keys(value)
            return sanitized

        elif isinstance(obj, list):
            return [AssessmentController.sanitize_json_keys(item) for item in obj]
        else:
            return obj

    @staticmethod
    def normalize_timeline_keys(timeline_obj: dict) -> dict:
        """
        Specifically normalize keys in progress_transition_timeline objects.
        Accepts keys like '6_month', '6-month', '6 month', '6months', '6_months'
        and returns keys like 'six_month'.

        Rules:
        - Extract leading digits (if any), convert to words using number_words.
        - Keep rest of the key (e.g. 'month', 'months', 'year', etc.), normalize to safe chars.
        - Output: <number_word>_<rest> or just <number_word> if no rest.
        """
        if not isinstance(timeline_obj, dict):
            return timeline_obj

        normalized = {}
        for raw_k, v in timeline_obj.items():
            k = str(raw_k).strip()

            # Try to match leading digits plus optional separators and the rest
            m = re.match(r'^(\d+)[\s_\-]*([A-Za-z0-9_]*)', k)
            if m:
                leading = m.group(1)
                rest = m.group(2) or ''
                word = AssessmentController.number_words.get(leading, f"num{leading}")
                if rest:
                    # normalize rest portion (remove invalid chars)
                    rest_norm = re.sub(r'[^a-zA-Z0-9_]', '_', rest)
                    rest_norm = re.sub(r'_+', '_', rest_norm).strip('_')
                    new_key = f"{word}_{rest_norm}" if rest_norm else word
                else:
                    new_key = word
            else:
                # no leading digits, sanitize generically
                new_key = re.sub(r'[^a-zA-Z0-9_]', '_', k)
                new_key = re.sub(r'_+', '_', new_key).strip('_')
                if new_key and not re.match(r'^[a-zA-Z]', new_key):
                    new_key = 'field_' + new_key
                if not new_key:
                    new_key = f'field_{hash(k) % 1000}'

            # final safety: ensure it starts with a letter
            if new_key and not re.match(r'^[a-zA-Z]', new_key):
                new_key = 'field_' + new_key

            # avoid accidental collisions: if key exists already, append suffix
            if new_key in normalized:
                suffix = 1
                base = new_key
                while f"{base}_{suffix}" in normalized:
                    suffix += 1
                new_key = f"{base}_{suffix}"

            normalized[new_key] = v

            logger.debug(f"Normalized timeline key: '{raw_k}' -> '{new_key}'")

        return normalized

    @staticmethod
    async def save_to_database(assessment_data: dict):
        """
        Save assessment data to the database using Prisma.
        Sanitizes entire report first, then specifically normalizes progress_transition_timeline.
        """
        prisma = Prisma()
        await prisma.connect()

        try:
            logger.info("Starting database save process for assessment data")
            logger.info("Database connection established")

            # Extract the raw report and sanitize the entire report at once
            report_raw = assessment_data.get("report", {}) or {}
            logger.info(f"Extracted report data: {bool(report_raw)}")

            # 1) Sanitize whole report keys with enhanced sanitization
            sanitized_report = AssessmentController.sanitize_json_keys(report_raw)
            
            # 2) Apply additional aggressive sanitization for problematic keys
            def extra_sanitize_keys(obj, depth=0):
                if isinstance(obj, dict):
                    result = {}
                    for key, value in obj.items():
                        # Convert key to string and apply comprehensive fixes
                        str_key = str(key)
                        
                        # Fix common problematic patterns
                        fixed_key = str_key
                        
                        # Replace spaces with underscores
                        if " " in fixed_key:
                            fixed_key = fixed_key.replace(" ", "_")
                            logger.warning(f"Fixed space in key: '{key}' -> '{fixed_key}'")
                        
                        # Replace other problematic characters
                        original_key = fixed_key
                        fixed_key = re.sub(r'[^a-zA-Z0-9_]', '_', fixed_key)
                        if fixed_key != original_key:
                            logger.warning(f"Fixed special chars in key: '{original_key}' -> '{fixed_key}'")
                        
                        # Remove consecutive underscores
                        fixed_key = re.sub(r'_+', '_', fixed_key).strip('_')
                        
                        # Ensure starts with letter
                        if fixed_key and not re.match(r'^[a-zA-Z]', fixed_key):
                            fixed_key = 'field_' + fixed_key
                        
                        # Ensure not empty
                        if not fixed_key:
                            fixed_key = f'field_{hash(str(key)) % 1000}'
                        
                        # Recursively process the value
                        result[fixed_key] = extra_sanitize_keys(value, depth + 1)
                    return result
                elif isinstance(obj, list):
                    return [extra_sanitize_keys(item, depth + 1) for item in obj]
                else:
                    return obj
            
            # Apply extra sanitization
            sanitized_report = extra_sanitize_keys(sanitized_report)
            
            # ADDITIONAL FIX: Specific handling for "6_months" key that might slip through
            def fix_six_months_key(obj):
                """Recursively fix any remaining '6_months' keys to 'six_months'"""
                if isinstance(obj, dict):
                    result = {}
                    for key, value in obj.items():
                        fixed_key = key
                        if str(key) == "6_months":
                            fixed_key = "six_months"
                            logger.warning(f"Fixed remaining '6_months' key to 'six_months'")
                        result[fixed_key] = fix_six_months_key(value)
                    return result
                elif isinstance(obj, list):
                    return [fix_six_months_key(item) for item in obj]
                else:
                    return obj
            
            sanitized_report = fix_six_months_key(sanitized_report)
            
            # FINAL VALIDATION: Ensure all keys are GraphQL-compatible
            def validate_graphql_keys(obj, path="root"):
                """Validate that all keys are GraphQL-compatible"""
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        str_key = str(key)
                        # Check for GraphQL-invalid patterns
                        if re.match(r'^\d', str_key):  # Starts with number
                            logger.error(f"INVALID KEY FOUND at {path}: '{str_key}' starts with number")
                            raise ValueError(f"GraphQL key validation failed: '{str_key}' at {path} starts with number")
                        if re.search(r'[^a-zA-Z0-9_]', str_key):  # Contains invalid characters
                            logger.error(f"INVALID KEY FOUND at {path}: '{str_key}' contains invalid characters")
                            raise ValueError(f"GraphQL key validation failed: '{str_key}' at {path} contains invalid characters")
                        # Recursively validate nested objects
                        validate_graphql_keys(value, f"{path}.{str_key}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        validate_graphql_keys(item, f"{path}[{i}]")
            
            try:
                validate_graphql_keys(sanitized_report)
                logger.info("✅ All keys passed GraphQL validation")
            except ValueError as ve:
                logger.error(f"❌ GraphQL validation failed: {ve}")
                raise ve

            # ========== DEEP SANITIZATION FINAL PASS (canonical) ==========
            def graphql_safe_key(raw: str) -> str:
                s = str(raw).strip()
                # Replace brackets and other punctuation with underscores
                s = re.sub(r'[^a-zA-Z0-9_ ]', '_', s)
                # Spaces -> underscores
                s = s.replace(' ', '_')
                # Collapse repeats
                s = re.sub(r'_+', '_', s)
                s = s.strip('_')
                # If starts with digit, map leading number sequence
                m = re.match(r'^(\d+)(.*)$', s)
                if m:
                    num, rest = m.group(1), m.group(2)
                    word = AssessmentController.number_words.get(num, f"num{num}")
                    rest = rest.lstrip('_')
                    s = f"{word}_{rest}" if rest else word
                # Ensure starts with letter
                if not re.match(r'^[A-Za-z]', s):
                    s = f"field_{s}" if s else "field_key"
                return s or "field_key"

            def deep_sanitize(obj):
                if isinstance(obj, dict):
                    cleaned = {}
                    for k, v in obj.items():
                        new_k = graphql_safe_key(k)
                        if new_k != k:
                            logger.debug(f"DeepSanitize key: '{k}' -> '{new_k}'")
                        cleaned[new_k] = deep_sanitize(v)
                    return cleaned
                if isinstance(obj, list):
                    return [deep_sanitize(i) for i in obj]
                return obj

            sanitized_report = deep_sanitize(sanitized_report)

            # Remap any lingering transition_timeline key after deep sanitize
            ico = sanitized_report.get('internal_career_opportunities') or sanitized_report.get('internal_career_opportunities'.replace('-', '_'))
            if isinstance(ico, dict):
                if 'transition_timeline' in ico and 'progress_transition_timeline' not in ico:
                    ico['progress_transition_timeline'] = ico.pop('transition_timeline')
                    logger.warning("Final pass remap: transition_timeline -> progress_transition_timeline")
                # Normalize timeline keys again robustly
                tl = ico.get('progress_transition_timeline')
                if isinstance(tl, dict):
                    normalized = {}
                    for k, v in tl.items():
                        # unify known variants
                        lower = k.lower()
                        if lower in ('6_months','six_months','6_month','six_month'):
                            nk = 'six_months'
                        elif lower in ('1_year','one_year','1year','oneyear'):
                            nk = 'one_year'
                        elif lower in ('2_years','two_years','2_year','two_year'):
                            nk = 'two_years'
                        else:
                            nk = graphql_safe_key(k)
                        if nk in normalized:
                            # avoid collision
                            idx = 1
                            base = nk
                            while f"{base}_{idx}" in normalized:
                                idx += 1
                            nk = f"{base}_{idx}"
                        if nk != k:
                            logger.debug(f"Timeline key normalized: '{k}' -> '{nk}'")
                        normalized[nk] = v
                    ico['progress_transition_timeline'] = normalized
                    logger.info(f"Final timeline keys: {list(normalized.keys())}")

            # Log any residual bracketed keys (should be none)
            def find_bracket_keys(obj, path='root'):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if '[' in k or ']' in k:
                            logger.error(f"Residual bracket key at {path}: {k}")
                        find_bracket_keys(v, f"{path}.{k}")
                elif isinstance(obj, list):
                    for i, it in enumerate(obj):
                        find_bracket_keys(it, f"{path}[{i}]")

            find_bracket_keys(sanitized_report)

            # ---------- COERCE / VALIDATE JSON FIELD VALUES (Prisma expects pure JSON-compatible types) ----------
            def coerce_json(value, path="root"):
                """Ensure value is composed only of JSON-serializable primitives (dict/list/str/int/float/bool/None)."""
                import datetime
                import decimal
                if isinstance(value, dict):
                    coerced = {}
                    for k, v in value.items():
                        coerced[k] = coerce_json(v, f"{path}.{k}")
                    return coerced
                if isinstance(value, list):
                    return [coerce_json(v, f"{path}[]") for v in value]
                if isinstance(value, (str, int, float, bool)) or value is None:
                    return value
                if isinstance(value, (datetime.date, datetime.datetime)):
                    return value.isoformat()
                if isinstance(value, decimal.Decimal):
                    return float(value)
                # Fallback: stringify any unsupported type
                logger.warning(f"Coercing non-JSON-primitive at {path}: {type(value).__name__}")
                return str(value)

            # Coerce each JSON section pre-extraction
            sanitized_report = coerce_json(sanitized_report, 'report')

            # 2) If internal_career_opportunities.progress_transition_timeline exists, normalize its keys
            if isinstance(sanitized_report.get("internal_career_opportunities"), dict):
                internal_career = sanitized_report["internal_career_opportunities"]
                if isinstance(internal_career.get("progress_transition_timeline"), dict):
                    normalized_timeline = AssessmentController.normalize_timeline_keys(internal_career["progress_transition_timeline"])
                    sanitized_report["internal_career_opportunities"]["progress_transition_timeline"] = normalized_timeline
                    logger.info(f"Normalized progress_transition_timeline keys: {list(normalized_timeline.keys())}")
                else:
                    logger.debug("No progress_transition_timeline dict found to normalize.")
            else:
                logger.debug("No internal_career_opportunities present or not a dict.")

            # Get user details for hrId and department
            user_id = assessment_data["userId"]
            logger.info(f"Looking up user with ID: {user_id}")
            user = await prisma.user.find_unique(where={"id": user_id})

            if not user:
                logger.error(f"User not found with ID: {user_id}")
                raise ValueError("User not found")

            logger.info(f"User found: {user.id}, hrId: {user.hrId}, department: {user.department}")

            # Extract sanitized subfields (these are already sanitized)
            genius_factor_json = sanitized_report.get("genius_factor_profile", {})
            current_role_json = sanitized_report.get("current_role_alignment_analysis", {})
            internal_career_json = sanitized_report.get("internal_career_opportunities", {})
            retention_strategies_json = sanitized_report.get("retention_and_mobility_strategies", {})
            development_plan_json = sanitized_report.get("development_action_plan", {})
            personalized_resources_json = sanitized_report.get("personalized_resources", {})
            data_sources_json = sanitized_report.get("data_sources_and_methodology", {})

            # Sanitize risk_analysis separately (it may be outside report)
            risk_analysis_raw = assessment_data.get("risk_analysis", {}) or {}
            risk_analysis_json = coerce_json(AssessmentController.sanitize_json_keys(risk_analysis_raw), 'risk_analysis')

            # Final coercion for each JSON field (safety)
            json_sections = {
                'geniusFactorProfileJson': genius_factor_json,
                'currentRoleAlignmentAnalysisJson': current_role_json,
                'internalCareerOpportunitiesJson': internal_career_json,
                'retentionAndMobilityStrategiesJson': retention_strategies_json,
                'developmentActionPlanJson': development_plan_json,
                'personalizedResourcesJson': personalized_resources_json,
                'dataSourcesAndMethodologyJson': data_sources_json,
                'risk_analysis': risk_analysis_json,
            }
            for label, section in json_sections.items():
                try:
                    import json as _json
                    _json.dumps(section)  # validate serializable
                except Exception as ser_e:
                    logger.error(f"JSON serialization failed for {label}: {ser_e}. Attempting coercion.")
                    json_sections[label] = coerce_json(section, label)
                else:
                    logger.debug(f"{label} JSON-serializable OK (type={type(section).__name__})")

            genius_factor_json = json_sections['geniusFactorProfileJson']
            current_role_json = json_sections['currentRoleAlignmentAnalysisJson']
            internal_career_json = json_sections['internalCareerOpportunitiesJson']
            retention_strategies_json = json_sections['retentionAndMobilityStrategiesJson']
            development_plan_json = json_sections['developmentActionPlanJson']
            personalized_resources_json = json_sections['personalizedResourcesJson']
            data_sources_json = json_sections['dataSourcesAndMethodologyJson']
            risk_analysis_json = json_sections['risk_analysis']

            # SECONDARY HARD COERCION: forcefully round-trip through json.dumps/loads to guarantee pristine types
            import json as _json_final
            def hard_json(value, label):
                try:
                    packed = _json_final.dumps(value, ensure_ascii=False)
                    return _json_final.loads(packed)
                except Exception as e:
                    logger.error(f"Hard JSON conversion failed for {label}: {e}; coercing generically.")
                    coerced = coerce_json(value, f"hard.{label}")
                    return _json_final.loads(_json_final.dumps(coerced))

            genius_factor_json = hard_json(genius_factor_json, 'geniusFactorProfileJson')
            current_role_json = hard_json(current_role_json, 'currentRoleAlignmentAnalysisJson')
            internal_career_json = hard_json(internal_career_json, 'internalCareerOpportunitiesJson')
            retention_strategies_json = hard_json(retention_strategies_json, 'retentionAndMobilityStrategiesJson')
            development_plan_json = hard_json(development_plan_json, 'developmentActionPlanJson')
            personalized_resources_json = hard_json(personalized_resources_json, 'personalizedResourcesJson')
            data_sources_json = hard_json(data_sources_json, 'dataSourcesAndMethodologyJson')
            risk_analysis_json = hard_json(risk_analysis_json, 'risk_analysis')

            # FINAL LOGGING of representative structures (sizes only to avoid clutter)
            def structure_info(obj):
                if isinstance(obj, dict):
                    return { 'type': 'dict', 'keys': list(obj.keys())[:10], 'len': len(obj) }
                if isinstance(obj, list):
                    return { 'type': 'list', 'len': len(obj) }
                return { 'type': type(obj).__name__ }
            logger.info(f"JSON field summary before create: geniusFactorProfileJson={structure_info(genius_factor_json)} internalCareerOpportunitiesJson={structure_info(internal_career_json)}")

            # Debug output right before insert
            logger.info(f"Prepared sanitized keys for insert. Internal career transition keys: "
                        f"{list(internal_career_json.get('progress_transition_timeline', {}).keys()) if isinstance(internal_career_json, dict) else 'n/a'}")

            # Prepare data for report creation (use sanitized_report for string fields too)
            report_data = {
                "userId": user_id,
                "hrId": user.hrId,
                "departement": user.department[-1] if user.department else "General",
                "executiveSummary": str(sanitized_report.get("executive_summary", "")),
                "geniusFactorScore": int(sanitized_report.get("genius_factor_score", 0)),
                "geniusFactorProfileJson": fields.Json(genius_factor_json),
                "currentRoleAlignmentAnalysisJson": fields.Json(current_role_json),
                "internalCareerOpportunitiesJson": fields.Json(internal_career_json),
                "retentionAndMobilityStrategiesJson": fields.Json(retention_strategies_json),
                "developmentActionPlanJson": fields.Json(development_plan_json),
                "personalizedResourcesJson": fields.Json(personalized_resources_json),
                "dataSourcesAndMethodologyJson": fields.Json(data_sources_json),
                "risk_analysis": fields.Json(risk_analysis_json),
            }

            logger.info("Prepared report data for creation")

            # Create the report
            logger.info("Creating individual employee report in database...")
            saved_report = await prisma.individualemployeereport.create(data=report_data)

            logger.info(f"Successfully saved report to database with ID: {saved_report.id}")

            # Build verification payload (non-sensitive summary)
            verification = {
                "report_id": saved_report.id,
                "timeline_keys": list(internal_career_json.get('progress_transition_timeline', {}).keys()) if isinstance(internal_career_json, dict) else [],
                "has_transition_timeline": 'transition_timeline' in internal_career_json if isinstance(internal_career_json, dict) else False,
                "genius_factor_profile_keys": list(genius_factor_json.keys())[:12] if isinstance(genius_factor_json, dict) else [],
                "internal_career_keys": list(internal_career_json.keys())[:12] if isinstance(internal_career_json, dict) else [],
                "risk_analysis_keys": list(risk_analysis_json.keys()) if isinstance(risk_analysis_json, dict) else [],
            }
            await prisma.disconnect()
            return {"status": "success", **verification}

        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            logger.error(f"Full exception details: {type(e).__name__}: {e}")
            await prisma.disconnect()
            return {"status": "error", "message": str(e)}

    @staticmethod
    async def analyze_assessment(input_data: AssessmentData) -> Dict[str, Any]:
        """
        Endpoint for assessment analysis with real-time notifications and Next.js integration
        """
        try:
            logger.info("Starting assessment analysis")
            basic_results = analyze_assessment_data(input_data.data)
            rag_results = await AIService.analyze_majority_answers(basic_results)

            recommendations = await AIService.generate_career_recommendation(rag_results)
            print(f"Recommendations generated: {recommendations}")

            if recommendations.get("status") != "success":
                error_msg = f"Failed to generate recommendations: {recommendations.get('message', 'Unknown error')}"
                logger.error(error_msg)

                await NotificationService.send_user_notification(
                    input_data.userId,
                    {
                        "message": "Analysis failed",
                        "progress": 100,
                        "status": "error",
                        "error": error_msg,
                    },
                )

                raise HTTPException(status_code=500, detail="Failed to generate career recommendations")

            final_result = {
                "status": "success",
                "userId": input_data.userId,
                "report": recommendations.get("report"),
                "risk_analysis": recommendations.get("risk_analysis"),
                "metadata": recommendations.get("metadata"),
            }
            print(f"Assessment analysis and recommendation generation completed: {final_result}")

            db_response = await AssessmentController.save_to_database(final_result)

            if db_response.get("status") == "error":
                logger.warning(f"Database save failed but proceeding: {db_response.get('message')}")

            await NotificationService.send_user_notification(
                input_data.userId,
                {
                    "message": "Assessment analysis completed successfully!",
                    "progress": 100,
                    "status": "completed",
                    "report_id": db_response.get("report_id"),
                    "db_response": db_response,
                },
            )

            logger.info("Assessment analysis, report generation, and Next.js integration completed successfully")

            return final_result

        except Exception as e:
            logger.error(f"Error in analyze_assessment: {str(e)}")

            await NotificationService.send_user_notification(
                input_data.userId,
                {
                    "message": "Assessment analysis failed",
                    "progress": 100,
                    "status": "error",
                    "error": str(e),
                },
            )

            raise HTTPException(status_code=500, detail=str(e))
