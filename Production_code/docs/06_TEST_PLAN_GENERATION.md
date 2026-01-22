# Test Plan Generation Guide

This document explains how to generate professional EMC test plans using the chatbot.

---

## Overview

The system generates structured, audit-ready test plans based on EMC standards in the knowledge graph. All content is grounded in source documents with full traceability.

---

## Test Plan Structure

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                           EMC TEST PLAN                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Document No: TP-2024-XXXX     Rev: 1.0     Date: 2024-12-28                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. SCOPE
   └── Description of test coverage

2. APPLICABLE STANDARDS
   └── List of referenced standards

3. TEST EQUIPMENT
   └── Required equipment with specifications

4. ENVIRONMENTAL CONDITIONS
   └── Temperature, humidity, pressure requirements

5. TEST CASES
   └── TC-001: [Title]
       ├── Test Type
       ├── Priority
       ├── Source Clause
       ├── Objective
       ├── Pre-conditions
       ├── Procedure Steps
       ├── Test Limits
       └── Pass/Fail Criteria

6. REQUIREMENTS TRACEABILITY MATRIX
   └── Coverage summary and mapping

7. VALIDATION STATUS
   └── Groundedness score and warnings
```

---

## How to Generate Test Plans

### Method 1: Chat Interface

Simply ask in the chat:

```
"Generate a test plan for ESD testing per IEC 61000-4-2"

"Create conducted emissions test cases for Class B equipment"

"Generate radiated immunity test procedure for IEC 61000-4-3"
```

### Method 2: API Call

```http
POST /api/v2/test-plan
Content-Type: application/json

{
  "query": "Generate ESD test plan for Class B device",
  "standard_ids": ["IEC 61000-4-2"],
  "include_recommendations": true
}
```

---

## Supported Test Types

| Test Type | Standard | Description |
|-----------|----------|-------------|
| Radiated Emissions | CISPR 32 | RF emissions measurement |
| Conducted Emissions | CISPR 32 | Conducted RF on power lines |
| Radiated Immunity | IEC 61000-4-3 | RF immunity testing |
| Conducted Immunity | IEC 61000-4-6 | Conducted RF immunity |
| ESD | IEC 61000-4-2 | Electrostatic discharge |
| EFT/Burst | IEC 61000-4-4 | Fast transients |
| Surge | IEC 61000-4-5 | Surge immunity |
| Voltage Dips | IEC 61000-4-11 | Power quality |
| Harmonics | IEC 61000-3-2 | Harmonic emissions |
| Flicker | IEC 61000-3-3 | Voltage fluctuations |

---

## Test Case Details

Each test case includes:

### Header Information
```
┌─────────────────┬────────────────────────────────────────────────┐
│ Test Type       │ ELECTROSTATIC DISCHARGE                        │
│ Priority        │ CRITICAL                                       │
│ Source Clause   │ IEC 61000-4-2, Clause 5.1                      │
│ Requirement     │ MANDATORY                                       │
└─────────────────┴────────────────────────────────────────────────┘
```

### Objective
> Verify EUT immunity to electrostatic discharge per IEC 61000-4-2 requirements

### Pre-conditions
- EUT powered and operating in normal mode
- Test environment meets ambient requirements
- All test equipment calibrated

### Test Procedure
| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Configure ESD generator to contact discharge | Generator ready |
| 2 | Apply discharge to coupling plane | EUT continues operation |
| 3 | Record observations | Log any anomalies |
| 4 | Repeat for air discharge | Complete test matrix |

### Test Limits
| Parameter | Limit | Unit | Test Level |
|-----------|-------|------|------------|
| Contact Discharge | ±4 | kV | Level 2 |
| Air Discharge | ±8 | kV | Level 3 |
| Indirect Discharge | ±4 | kV | Level 2 |

### Pass/Fail Criteria
- ✓ EUT shall continue normal operation during and after discharge
- ✓ No loss of data or function
- ✓ Self-recovery within specified time

---

## Priority Levels

Test cases are prioritized based on requirement type:

| Priority | Source | Icon | Description |
|----------|--------|------|-------------|
| CRITICAL | "shall" | 🔴 | Mandatory requirements |
| HIGH | "shall not" | 🟠 | Prohibited conditions |
| MEDIUM | "should" | 🟡 | Recommendations |
| LOW | "may" | 🟢 | Permissions |

---

## Requirements Traceability

The system generates a coverage matrix:

```
┌────────────────────────────────────────────────────────────┐
│                  COVERAGE SUMMARY                           │
├─────────────────────────┬──────────────────────────────────┤
│ Total Requirements      │ 24                               │
│ Covered Requirements    │ 22                               │
│ Not Covered             │ 2                                │
│ Coverage Percentage     │ 91.7%                            │
└─────────────────────────┴──────────────────────────────────┘
```

Individual requirement tracing:
| Requirement ID | Source Clause | Type | Status | Covered By |
|----------------|---------------|------|--------|------------|
| REQ-001 | 5.1 | mandatory | ✅ | TC-001 |
| REQ-002 | 5.2 | mandatory | ✅ | TC-002 |
| REQ-003 | 6.1 | recommend | ❌ | — |

---

## Validation & Groundedness

Every test plan is validated:

### Passed
```
╔══════════════════════════════════════════════════════════════╗
║  ✅ TEST PLAN VALIDATION: PASSED                              ║
║  Groundedness Score: 95.0%                                    ║
╚══════════════════════════════════════════════════════════════╝
```

### With Warnings
```
╔══════════════════════════════════════════════════════════════╗
║  ⚠️ TEST PLAN VALIDATION: WARNINGS                            ║
║  Groundedness Score: 78.0%                                    ║
╚══════════════════════════════════════════════════════════════╝

Warnings:
- ⚠️ Some test limits not found in source documents
- ⚠️ Equipment specifications may need verification
```

---

## Export Options

### PDF Export
```http
POST /api/v2/test-plan/export
Content-Type: application/json

{
  "query": "Generate ESD test plan",
  "format": "pdf"
}
```

### Download Existing
```http
GET /api/v2/test-plan/{test_plan_id}/pdf
```

---

## Example Queries

### ESD Testing
```
"Generate a complete ESD test plan for a handheld consumer device
per IEC 61000-4-2. Include contact and air discharge tests at
Level 3 severity."
```

### Emissions Testing
```
"Create a radiated emissions test plan for a Class B Information
Technology Equipment per CISPR 32 / FCC Part 15."
```

### Immunity Suite
```
"Generate a full immunity test suite for industrial equipment
including ESD, EFT, surge, and conducted immunity per
IEC 61000-6-2."
```

### Specific Standard
```
"Create test cases for MIL-STD-461G RE102 radiated emissions
testing from 10 kHz to 18 GHz."
```

---

## Best Practices

### Query Tips
1. **Be specific** about the product type
2. **Reference standards** explicitly
3. **Specify test levels** when known
4. **Mention equipment class** (Class A/B, Group 1/2)

### Verification
1. Review all test limits against standards
2. Verify equipment specifications
3. Check environmental conditions
4. Validate pass/fail criteria

### Customization
1. Add site-specific procedures
2. Include additional test cases
3. Modify limits for specific applications
4. Add company-specific formatting

---

## Troubleshooting

### "Not enough context"
- Ensure relevant standards are in the knowledge graph
- Try more specific queries
- Reference exact clause numbers

### "Low groundedness score"
- Some content may be from general knowledge
- Verify critical limits manually
- Upload additional source documents

### "Missing test limits"
- Tables may not be fully extracted
- Check source document quality
- Upload clearer PDF if needed
