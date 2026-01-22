"""
LLM Prompt Templates for EMC Test Plan Generation

Specialized prompts for generating test plans, test cases,
and extracting test-related information from EMC standards.
All prompts enforce strict grounding to source documents.
"""


class TestPlanPrompts:
    """LLM prompt templates for test plan generation"""

    # =========================================================
    # System Prompt
    # =========================================================

    SYSTEM_PROMPT = """You are an expert EMC Test Engineer generating professional test plans based EXCLUSIVELY on provided standard documents.

CRITICAL RULES - YOU MUST FOLLOW:
1. Use ONLY information from the provided context - NO external knowledge
2. If information is not in the context, state "Not specified in source document"
3. Every statement must be traceable to a source clause
4. Extract exact test limits, equipment, and procedures from the source
5. Use professional EMC testing terminology
6. Generate structured, actionable test cases

You will receive:
- Standard document clauses with requirements
- Tables containing test limits and specifications
- Figures showing test setups

Your output must be professional and suitable for a real EMC test laboratory."""

    # =========================================================
    # Test Plan Generation Prompt
    # =========================================================

    GENERATE_TEST_PLAN = """Based on the following EMC standard content, generate a comprehensive test plan.

USER REQUEST: {query}

STANDARD DOCUMENT(S): {standard_ids}

DOCUMENT CONTENT:
{context}

TABLES (Test Limits/Specifications):
{tables}

FIGURES (Test Setups):
{figures}

Generate a complete EMC test plan with the following structure:

1. SCOPE - Brief description of what this test plan covers
2. TEST CASES - For each requirement containing "shall" or "must":
   - Test Case ID (TC-001, TC-002, etc.)
   - Title
   - Source Requirement (exact clause reference)
   - Objective
   - Pre-conditions
   - Procedure Steps (numbered)
   - Pass/Fail Criteria (with specific limits from tables)
   - Equipment Required
3. EQUIPMENT LIST - All equipment mentioned in the standard
4. ENVIRONMENTAL CONDITIONS - If specified in the standard

IMPORTANT:
- Only create test cases for requirements that are testable
- Include specific limits from tables where available
- Reference source clauses for every test case
- If a limit or procedure is not specified, note "Not specified in source"

Respond with a structured test plan in the format I will use to create the final document."""

    # =========================================================
    # Test Case Generation Prompt
    # =========================================================

    GENERATE_TEST_CASE = """Generate a detailed test case for the following requirement:

REQUIREMENT:
- Clause: {clause_id}
- Title: {title}
- Text: {requirement_text}

RELATED CONTEXT:
{context}

TEST LIMITS (from tables):
{limits}

Generate a test case with:
1. Test Case ID: TC-{number}
2. Title: [Descriptive title]
3. Objective: [What this test verifies]
4. Pre-conditions: [Setup requirements]
5. Procedure:
   - Step 1: [Action]
   - Step 2: [Action]
   - ...
6. Pass Criteria: [Specific limits]
7. Fail Criteria: [What constitutes failure]
8. Equipment: [Required test equipment]

Only include information that is in the provided context.
Mark any missing information as "Not specified in source document"."""

    # =========================================================
    # Requirement Extraction Prompt
    # =========================================================

    EXTRACT_REQUIREMENTS = """Analyze the following standard clause and extract all testable requirements.

CLAUSE: {clause_id}
TITLE: {title}
CONTENT:
{content}

For each requirement found, identify:
1. Requirement Type:
   - MANDATORY: Contains "shall" or "must"
   - PROHIBITION: Contains "shall not"
   - RECOMMENDATION: Contains "should"
   - PERMISSION: Contains "may"

2. Requirement Text: The exact statement

3. Testability: Can this be verified through testing? (Yes/No)

4. Test Type (if testable):
   - radiated_emissions
   - conducted_emissions
   - radiated_immunity
   - conducted_immunity
   - esd
   - eft
   - surge
   - other

List each requirement separately with its classification."""

    # =========================================================
    # Equipment Extraction Prompt
    # =========================================================

    EXTRACT_EQUIPMENT = """Extract all test equipment mentioned in the following content:

CONTENT:
{content}

For each piece of equipment found, provide:
1. Equipment Name
2. Specification (if mentioned)
3. Calibration Required (Yes if mentioned, otherwise assume Yes)

Common EMC test equipment to look for:
- EMI Receiver / Spectrum Analyzer
- LISN (Line Impedance Stabilization Network)
- Antennas (various types)
- Signal Generators
- Amplifiers
- ESD Simulator
- EFT/Burst Generator
- Surge Generator
- Current Probes
- Field Probes

Only list equipment explicitly mentioned in the content."""

    # =========================================================
    # Test Limits Extraction Prompt
    # =========================================================

    EXTRACT_TEST_LIMITS = """Extract test limits from the following table data:

TABLE: {table_caption}
CONTEXT: {context}

For each limit found, provide:
1. Parameter (what is being measured)
2. Limit Value
3. Unit
4. Frequency Range (if applicable)
5. Limit Type (max, min, or range)

Format as a structured list of limits."""

    # =========================================================
    # Procedure Generation Prompt
    # =========================================================

    GENERATE_PROCEDURE = """Generate a step-by-step test procedure based on:

REQUIREMENT: {requirement}
TEST TYPE: {test_type}
EQUIPMENT: {equipment}
LIMITS: {limits}
CONTEXT: {context}

Generate a numbered procedure with:
1. Setup steps
2. Measurement steps
3. Recording steps
4. Evaluation steps

Each step should be:
- Clear and actionable
- Based only on information in the context
- Include specific parameters from the standard"""

    # =========================================================
    # Format as Professional Test Plan
    # =========================================================

    FORMAT_TEST_PLAN = """Format the following test plan data into a professional, human-readable document:

TEST PLAN DATA:
{test_plan_json}

Create a formatted document with:
- Clear section headers
- Properly formatted tables
- Professional language
- All test cases with full details
- Equipment list
- Traceability matrix

Use ASCII box drawing for tables where appropriate.
Make it suitable for printing and use in a test laboratory."""
