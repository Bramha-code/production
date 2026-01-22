"""
EMC Test Plan Data Models

Pydantic models for generating professional, human-readable test plans
with full source traceability and requirement coverage tracking.
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
import hashlib


# =========================================================
# Enums
# =========================================================

class TestType(str, Enum):
    """EMC Test Types"""
    RADIATED_EMISSIONS = "radiated_emissions"
    CONDUCTED_EMISSIONS = "conducted_emissions"
    RADIATED_IMMUNITY = "radiated_immunity"
    CONDUCTED_IMMUNITY = "conducted_immunity"
    ESD = "electrostatic_discharge"
    EFT = "electrical_fast_transient"
    SURGE = "surge"
    POWER_FREQUENCY_MAGNETIC_FIELD = "power_frequency_magnetic"
    VOLTAGE_DIP = "voltage_dip_interruption"
    HARMONICS = "harmonics"
    FLICKER = "flicker"
    GENERAL_EMC = "general_emc"
    OTHER = "other"


class TestPhase(str, Enum):
    """Test execution phases"""
    PRE_TEST = "pre_test"
    SETUP = "setup"
    EXECUTION = "execution"
    POST_TEST = "post_test"
    VERIFICATION = "verification"


class TestPriority(str, Enum):
    """Test priority levels based on requirement type"""
    CRITICAL = "critical"      # From mandatory requirements (shall)
    HIGH = "high"              # From prohibition requirements (shall not)
    MEDIUM = "medium"          # From recommendation requirements (should)
    LOW = "low"                # From permission requirements (may)


class RequirementType(str, Enum):
    """Requirement classification"""
    MANDATORY = "mandatory"        # shall, must
    PROHIBITION = "prohibition"    # shall not
    RECOMMENDATION = "recommendation"  # should
    PERMISSION = "permission"      # may


# =========================================================
# Source Traceability Models
# =========================================================

class SourceReference(BaseModel):
    """Source traceability for any generated content"""
    chunk_id: str
    document_id: str
    clause_id: str
    title: Optional[str] = None
    excerpt: Optional[str] = Field(None, max_length=500)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# =========================================================
# Equipment & Environment Models
# =========================================================

class EquipmentItem(BaseModel):
    """Test equipment specification"""
    name: str
    specification: Optional[str] = None
    calibration_required: bool = True
    source_ref: Optional[SourceReference] = None


class EnvironmentalCondition(BaseModel):
    """Test environmental requirements"""
    parameter: str  # e.g., "temperature", "humidity"
    value: str      # e.g., "23 +/- 5 C", "45-75% RH"
    source_ref: Optional[SourceReference] = None


class TestLimit(BaseModel):
    """Test limit/threshold specification"""
    parameter: str
    limit_value: str
    unit: str
    frequency_range: Optional[str] = None
    limit_type: str = "max"  # max, min, range
    source_ref: Optional[SourceReference] = None


# =========================================================
# Test Procedure Models
# =========================================================

class TestStep(BaseModel):
    """Single step in a test procedure"""
    step_number: int
    action: str
    expected_result: Optional[str] = None
    notes: Optional[str] = None


class PassFailCriterion(BaseModel):
    """Test pass/fail criterion"""
    description: str
    limit: Optional[TestLimit] = None
    source_ref: Optional[SourceReference] = None


# =========================================================
# Test Case Model
# =========================================================

class TestCase(BaseModel):
    """Complete test case definition"""
    test_case_id: str
    title: str

    # Requirement linkage
    requirement_id: str
    requirement_text: str
    requirement_type: RequirementType
    source_clause: str

    # Test classification
    test_type: TestType = TestType.GENERAL_EMC
    priority: TestPriority = TestPriority.CRITICAL

    # Test details
    objective: str
    pre_conditions: List[str] = Field(default_factory=list)
    procedure_steps: List[TestStep] = Field(default_factory=list)
    pass_fail_criteria: List[PassFailCriterion] = Field(default_factory=list)

    # Resources
    equipment_required: List[EquipmentItem] = Field(default_factory=list)
    environmental_conditions: List[EnvironmentalCondition] = Field(default_factory=list)
    test_limits: List[TestLimit] = Field(default_factory=list)

    # Traceability
    source_chunks: List[str] = Field(default_factory=list)


# =========================================================
# Coverage Models
# =========================================================

class RequirementCoverageItem(BaseModel):
    """Single requirement coverage entry"""
    requirement_id: str
    requirement_text: str
    requirement_type: str
    source_clause: str
    covered_by_tests: List[str]  # Test case IDs
    coverage_status: str  # "covered", "not_covered"


class RequirementCoverageMatrix(BaseModel):
    """Complete requirement-to-test coverage matrix"""
    total_requirements: int
    covered_requirements: int
    not_covered: int
    coverage_percentage: float
    items: List[RequirementCoverageItem] = Field(default_factory=list)


# =========================================================
# Validation Model
# =========================================================

class TestPlanValidation(BaseModel):
    """Validation results for test plan"""
    is_valid: bool
    groundedness_score: float
    warnings: List[str] = Field(default_factory=list)
    missing_requirements: List[str] = Field(default_factory=list)


# =========================================================
# Complete Test Plan Model
# =========================================================

class TestPlan(BaseModel):
    """Complete EMC Test Plan"""
    # Header
    test_plan_id: str
    document_number: str
    revision: str = "1.0"
    date: str
    title: str

    # Scope
    scope: str
    applicable_standards: List[str]

    # EUT (Equipment Under Test)
    eut_description: Optional[str] = None
    operating_conditions: Optional[str] = None

    # Test Cases
    test_cases: List[TestCase]
    total_test_cases: int

    # Resources Summary
    all_equipment: List[EquipmentItem] = Field(default_factory=list)
    environmental_conditions: List[EnvironmentalCondition] = Field(default_factory=list)

    # Coverage & Validation
    coverage_matrix: RequirementCoverageMatrix
    validation: TestPlanValidation

    # Traceability
    sources_used: List[str] = Field(default_factory=list)
    query: str

    # Timestamps
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @staticmethod
    def generate_id(query: str) -> str:
        """Generate unique test plan ID"""
        hash_input = f"{query}{datetime.utcnow().isoformat()}"
        return f"TP-{datetime.utcnow().strftime('%Y%m%d')}-{hashlib.md5(hash_input.encode()).hexdigest()[:6].upper()}"

    @staticmethod
    def generate_doc_number() -> str:
        """Generate document number"""
        return f"TP-{datetime.utcnow().strftime('%Y')}-{hashlib.md5(datetime.utcnow().isoformat().encode()).hexdigest()[:4].upper()}"


# =========================================================
# Request/Response Models for API
# =========================================================

class TestPlanGenerateRequest(BaseModel):
    """Request to generate a test plan"""
    query: str
    standard_ids: Optional[List[str]] = None
    test_types: Optional[List[str]] = None
    include_recommendations: bool = True


class TestPlanResponse(BaseModel):
    """API response containing test plan"""
    success: bool
    test_plan: Optional[TestPlan] = None
    formatted_output: Optional[str] = None  # Human-readable text
    error: Optional[str] = None
