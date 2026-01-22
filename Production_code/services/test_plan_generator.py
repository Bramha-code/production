"""
EMC Test Plan Generator Service

Main orchestration service for generating comprehensive EMC test plans
using the Knowledge Graph, Grounding Engine, and LLM.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from models.test_plan_models import (
    TestPlan, TestCase, TestStep, TestLimit, EquipmentItem,
    EnvironmentalCondition, PassFailCriterion, SourceReference,
    RequirementCoverageMatrix, RequirementCoverageItem,
    TestPlanValidation, TestType, TestPriority, RequirementType,
    TestPlanGenerateRequest, TestPlanResponse
)
from services.test_plan_queries import TestPlanCypherQueries
from services.test_plan_prompts import TestPlanPrompts


class TestPlanGenerator:
    """
    Generates comprehensive EMC test plans from knowledge graph data.

    Flow:
    1. Extract requirements from Neo4j based on query
    2. Retrieve test limits from tables
    3. Retrieve test setups from figures
    4. Generate test cases using LLM with grounding
    5. Build complete test plan with coverage matrix
    6. Format for human-readable output
    """

    def __init__(
        self,
        neo4j_driver,
        qdrant_client,
        embedding_model,
        llm_client,
        llm_model: str = "qwen2.5:7b",
        collection_name: str = "emc_embeddings"
    ):
        self.neo4j_driver = neo4j_driver
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.collection_name = collection_name
        self.queries = TestPlanCypherQueries()
        self.prompts = TestPlanPrompts()

    # =========================================================
    # Main Generation Method
    # =========================================================

    def generate_test_plan(self, request: TestPlanGenerateRequest) -> TestPlanResponse:
        """
        Generate a complete test plan based on the request.
        """
        try:
            # Step 1: Determine which standards to use
            standard_ids = request.standard_ids
            if not standard_ids:
                # Get standards from semantic search
                standard_ids = self._find_relevant_standards(request.query)

            if not standard_ids:
                return TestPlanResponse(
                    success=False,
                    error="No relevant standards found for the query"
                )

            # Step 2: Extract requirements from knowledge graph
            requirements = []
            for std_id in standard_ids:
                reqs = self._extract_requirements(std_id)
                requirements.extend(reqs)

            if not requirements:
                return TestPlanResponse(
                    success=False,
                    error="No testable requirements found in the specified standards"
                )

            # Step 2.5: Filter requirements based on query relevance
            requirements = self._filter_requirements_by_query(request.query, requirements)

            # Step 3: Get tables and figures for context
            tables = []
            figures = []
            for std_id in standard_ids:
                tables.extend(self._get_tables(std_id))
                figures.extend(self._get_figures(std_id))

            # Step 4: Generate test cases using LLM
            test_cases = self._generate_test_cases(
                query=request.query,
                requirements=requirements,
                tables=tables,
                figures=figures,
                include_recommendations=request.include_recommendations
            )

            # Step 5: Extract equipment and conditions
            equipment = self._extract_equipment(requirements)
            conditions = self._extract_environmental_conditions(requirements)

            # Step 6: Build coverage matrix
            coverage_matrix = self._build_coverage_matrix(requirements, test_cases)

            # Step 7: Validate the test plan
            validation = self._validate_test_plan(test_cases, requirements)

            # Step 8: Assemble complete test plan
            test_plan = TestPlan(
                test_plan_id=TestPlan.generate_id(request.query),
                document_number=TestPlan.generate_doc_number(),
                revision="1.0",
                date=datetime.utcnow().strftime("%Y-%m-%d"),
                title=self._generate_title(request.query, standard_ids),
                scope=self._generate_scope(request.query, standard_ids),
                applicable_standards=standard_ids,
                eut_description=self._extract_eut_description(request.query),
                test_cases=test_cases,
                total_test_cases=len(test_cases),
                all_equipment=equipment,
                environmental_conditions=conditions,
                coverage_matrix=coverage_matrix,
                validation=validation,
                sources_used=[r["chunk_id"] for r in requirements],
                query=request.query
            )

            # Step 9: Format for human-readable output
            formatted_output = self._format_test_plan(test_plan)

            return TestPlanResponse(
                success=True,
                test_plan=test_plan,
                formatted_output=formatted_output
            )

        except Exception as e:
            return TestPlanResponse(
                success=False,
                error=f"Test plan generation failed: {str(e)}"
            )

    # =========================================================
    # Requirement Extraction
    # =========================================================

    def _extract_requirements(self, standard_id: str) -> List[Dict]:
        """Extract all testable requirements from a standard."""
        requirements = []

        if not self.neo4j_driver:
            return requirements

        try:
            with self.neo4j_driver.session() as session:
                # Try to get clauses with content containing requirements
                result = session.run("""
                    MATCH (c:Clause)
                    WHERE c.document_id CONTAINS $doc_id
                      AND c.content_text IS NOT NULL
                      AND length(c.content_text) > 20
                    RETURN c.uid as chunk_id,
                           c.document_id as document_id,
                           c.clause_id as clause_id,
                           c.title as title,
                           c.content_text as content
                    ORDER BY c.clause_id
                    LIMIT 50
                """, doc_id=standard_id)

                for record in result:
                    content = record.get("content", "")
                    if not content:
                        continue

                    # Classify requirement type
                    req_type = self._classify_requirement(content)
                    if req_type:
                        requirements.append({
                            "chunk_id": record.get("chunk_id", "") or f"clause:{standard_id}:{record.get('clause_id', '')}",
                            "document_id": record.get("document_id", "") or standard_id,
                            "clause_id": record.get("clause_id", ""),
                            "title": record.get("title", "") or f"Clause {record.get('clause_id', '')}",
                            "content": content,
                            "requirement_type": req_type,
                            "table_count": 0,
                            "figure_count": 0
                        })

        except Exception as e:
            print(f"Error extracting requirements: {e}")
            # Fallback: try to get any clauses
            try:
                with self.neo4j_driver.session() as session:
                    result = session.run("""
                        MATCH (c:Clause)
                        WHERE c.content_text IS NOT NULL
                        RETURN c.uid as chunk_id,
                               c.document_id as document_id,
                               c.clause_id as clause_id,
                               c.title as title,
                               c.content_text as content
                        LIMIT 20
                    """)
                    for record in result:
                        content = record.get("content", "")
                        req_type = self._classify_requirement(content)
                        if req_type:
                            requirements.append({
                                "chunk_id": record.get("chunk_id", ""),
                                "document_id": record.get("document_id", ""),
                                "clause_id": record.get("clause_id", ""),
                                "title": record.get("title", ""),
                                "content": content,
                                "requirement_type": req_type,
                                "table_count": 0,
                                "figure_count": 0
                            })
            except:
                pass

        return requirements

    def _classify_requirement(self, content: str) -> Optional[str]:
        """Classify requirement type based on keywords."""
        content_lower = content.lower()

        if "shall not" in content_lower:
            return RequirementType.PROHIBITION.value
        elif "shall" in content_lower or "must" in content_lower:
            return RequirementType.MANDATORY.value
        elif "should" in content_lower:
            return RequirementType.RECOMMENDATION.value
        elif "may" in content_lower:
            return RequirementType.PERMISSION.value

        return None

    def _filter_requirements_by_query(self, query: str, requirements: List[Dict]) -> List[Dict]:
        """
        Filter requirements based on semantic similarity to the query.
        Returns unique requirements sorted by relevance.
        Also applies keyword filtering to ensure relevance.
        """
        if not requirements or not self.embedding_model:
            return requirements

        try:
            import numpy as np
            from numpy.linalg import norm

            # Extract key terms from query for keyword filtering
            query_lower = query.lower()
            query_keywords = self._extract_query_keywords(query_lower)

            # De-duplicate requirements by clause_id and content
            seen_keys = set()
            unique_requirements = []
            for req in requirements:
                clause_id = req.get("clause_id", "")
                content_start = req.get("content", "")[:100]
                unique_key = f"{clause_id}:{hash(content_start)}"
                if unique_key not in seen_keys:
                    seen_keys.add(unique_key)
                    unique_requirements.append(req)

            # Get query embedding
            query_embedding = self.embedding_model.encode(query)

            # Score each requirement by cosine similarity + keyword bonus
            scored_requirements = []
            for req in unique_requirements:
                content = req.get("content", "")
                title = req.get("title", "")
                content_lower = content.lower()
                title_lower = title.lower()
                text = f"{title} {content}"

                # Get requirement embedding
                req_embedding = self.embedding_model.encode(text)

                # Calculate cosine similarity
                similarity = np.dot(query_embedding, req_embedding) / (
                    norm(query_embedding) * norm(req_embedding)
                )

                # Keyword bonus: boost score if content contains query keywords
                keyword_bonus = 0.0
                keyword_matches = 0
                for kw in query_keywords:
                    if kw in content_lower or kw in title_lower:
                        keyword_matches += 1
                        keyword_bonus += 0.1

                # Penalty for obviously irrelevant content
                irrelevant_penalty = 0.0
                irrelevant_terms = ['fire load', 'flame', 'burning', 'combustion heater',
                                   'smoke', 'extinguisher', 'ignition source']
                if not any(term in query_lower for term in ['fire', 'flame', 'burn']):
                    for term in irrelevant_terms:
                        if term in content_lower:
                            irrelevant_penalty += 0.15

                final_score = similarity + keyword_bonus - irrelevant_penalty

                # Only include if has at least one keyword match OR high similarity
                if keyword_matches > 0 or similarity > 0.35:
                    scored_requirements.append((req, final_score, keyword_matches))

            # Sort by score (highest first)
            scored_requirements.sort(key=lambda x: x[1], reverse=True)

            # Filter: prioritize keyword matches, then by score
            filtered = []
            for req, score, kw_count in scored_requirements:
                if score >= 0.25 or len(filtered) < 10:
                    req["relevance_score"] = float(score)
                    req["keyword_matches"] = kw_count
                    filtered.append(req)

            # Limit to top 15 most relevant unique requirements
            return filtered[:15]

        except Exception as e:
            print(f"Error filtering requirements: {e}")
            seen = set()
            unique = []
            for req in requirements:
                key = f"{req.get('clause_id', '')}:{hash(req.get('content', '')[:100])}"
                if key not in seen:
                    seen.add(key)
                    unique.append(req)
                if len(unique) >= 10:
                    break
            return unique

    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract EMC-relevant keywords from query."""
        # EMC test type keywords
        emc_keywords = {
            'radiated': ['radiated', 'radiation', 'antenna', 'field strength', 'rf'],
            'conducted': ['conducted', 'power line', 'mains', 'lisn'],
            'emissions': ['emissions', 'emission', 'emi', 'interference', 'disturbance'],
            'immunity': ['immunity', 'susceptibility', 'ems', 'withstand'],
            'esd': ['esd', 'electrostatic', 'discharge', 'static'],
            'eft': ['eft', 'fast transient', 'burst'],
            'surge': ['surge', 'transient', 'overvoltage'],
            'harmonics': ['harmonics', 'harmonic', 'thd', 'distortion'],
            'flicker': ['flicker', 'voltage fluctuation'],
            'magnetic': ['magnetic field', 'power frequency'],
            'automotive': ['vehicle', 'automotive', 'car', 'ece', 'cispr 25'],
            'consumer': ['consumer', 'household', 'cispr 32'],
            'medical': ['medical', 'iec 60601'],
            'industrial': ['industrial', 'iec 61000']
        }

        keywords = set()
        for category, terms in emc_keywords.items():
            for term in terms:
                if term in query:
                    keywords.update(terms)
                    break

        # Add general EMC terms if query mentions EMC/EMI
        if 'emc' in query or 'emi' in query or 'electromagnetic' in query:
            keywords.update(['emc', 'emi', 'electromagnetic', 'compatibility', 'test', 'limit'])

        return list(keywords) if keywords else ['test', 'requirement', 'limit', 'measurement']

    # =========================================================
    # Table & Figure Extraction
    # =========================================================

    def _get_tables(self, standard_id: str) -> List[Dict]:
        """Get all tables from a standard."""
        tables = []

        if not self.neo4j_driver:
            return tables

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    self.queries.GET_TABLES_FOR_DOCUMENT,
                    doc_id=standard_id
                )
                tables = [dict(record) for record in result]
        except Exception as e:
            print(f"Error getting tables: {e}")

        return tables

    def _get_figures(self, standard_id: str) -> List[Dict]:
        """Get all figures from a standard."""
        figures = []

        if not self.neo4j_driver:
            return figures

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    self.queries.GET_FIGURES_FOR_DOCUMENT,
                    doc_id=standard_id
                )
                figures = [dict(record) for record in result]
        except Exception as e:
            print(f"Error getting figures: {e}")

        return figures

    # =========================================================
    # Test Case Generation
    # =========================================================

    def _generate_test_cases(
        self,
        query: str,
        requirements: List[Dict],
        tables: List[Dict],
        figures: List[Dict],
        include_recommendations: bool = True
    ) -> List[TestCase]:
        """Generate test cases from requirements using LLM."""
        test_cases = []
        tc_number = 1

        # Filter requirements based on type
        filtered_reqs = [
            r for r in requirements
            if r["requirement_type"] in [
                RequirementType.MANDATORY.value,
                RequirementType.PROHIBITION.value
            ] or (include_recommendations and r["requirement_type"] == RequirementType.RECOMMENDATION.value)
        ]

        # Build context for LLM
        context = self._build_context(filtered_reqs, tables, figures)

        # Generate test plan using LLM
        if self.llm_client:
            try:
                llm_response = self._call_llm_for_test_plan(query, context, filtered_reqs)
                test_cases = self._parse_llm_test_cases(llm_response, filtered_reqs)
            except Exception as e:
                print(f"LLM generation failed: {e}")
                # Fallback to template-based generation
                test_cases = self._generate_template_test_cases(filtered_reqs, tc_number)
        else:
            # Template-based generation
            test_cases = self._generate_template_test_cases(filtered_reqs, tc_number)

        return test_cases

    def _call_llm_for_test_plan(
        self,
        query: str,
        context: str,
        requirements: List[Dict]
    ) -> str:
        """Call LLM to generate test plan content."""
        # Build table and figure summaries
        tables_text = "No tables available"
        figures_text = "No figures available"

        prompt = self.prompts.GENERATE_TEST_PLAN.format(
            query=query,
            standard_ids=", ".join(set(r["document_id"] for r in requirements)),
            context=context,
            tables=tables_text,
            figures=figures_text
        )

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": self.prompts.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4000
        )

        return response.choices[0].message.content

    def _parse_llm_test_cases(
        self,
        llm_response: str,
        requirements: List[Dict]
    ) -> List[TestCase]:
        """Parse LLM response into TestCase objects."""
        test_cases = []
        tc_number = 1

        # Create a test case for each requirement mentioned in response
        for req in requirements:
            # Detect test type from content
            test_type = self._detect_test_type(req["content"])
            priority = self._get_priority(req["requirement_type"])

            # Extract procedure from LLM response or use default
            procedure = self._extract_procedure_from_response(
                llm_response, req["clause_id"]
            )

            test_case = TestCase(
                test_case_id=f"TC-{tc_number:03d}",
                title=f"{req['title']}",
                requirement_id=f"REQ-{tc_number:03d}",
                requirement_text=req["content"][:500],
                requirement_type=RequirementType(req["requirement_type"]),
                source_clause=req["clause_id"],
                test_type=test_type,
                priority=priority,
                objective=f"Verify compliance with {req['title']}",
                pre_conditions=["EUT powered and in normal operating mode"],
                procedure_steps=procedure,
                pass_fail_criteria=[
                    PassFailCriterion(
                        description="Meet requirements as specified in source clause",
                        source_ref=SourceReference(
                            chunk_id=req["chunk_id"],
                            document_id=req["document_id"],
                            clause_id=req["clause_id"],
                            title=req["title"]
                        )
                    )
                ],
                source_chunks=[req["chunk_id"]]
            )

            test_cases.append(test_case)
            tc_number += 1

        return test_cases

    def _generate_template_test_cases(
        self,
        requirements: List[Dict],
        start_number: int = 1
    ) -> List[TestCase]:
        """Generate test cases using templates (fallback)."""
        test_cases = []
        tc_number = start_number

        for req in requirements:
            test_type = self._detect_test_type(req["content"])
            priority = self._get_priority(req["requirement_type"])

            procedure = [
                TestStep(step_number=1, action="Review test requirements from source clause"),
                TestStep(step_number=2, action="Prepare EUT according to standard requirements"),
                TestStep(step_number=3, action="Configure test equipment as specified"),
                TestStep(step_number=4, action="Perform test measurement/verification"),
                TestStep(step_number=5, action="Record results and compare with limits"),
                TestStep(step_number=6, action="Document pass/fail status")
            ]

            test_case = TestCase(
                test_case_id=f"TC-{tc_number:03d}",
                title=req["title"] or f"Test Case {tc_number}",
                requirement_id=f"REQ-{tc_number:03d}",
                requirement_text=req["content"][:500],
                requirement_type=RequirementType(req["requirement_type"]),
                source_clause=req["clause_id"],
                test_type=test_type,
                priority=priority,
                objective=f"Verify compliance with clause {req['clause_id']}: {req['title']}",
                pre_conditions=["EUT in normal operating condition"],
                procedure_steps=procedure,
                pass_fail_criteria=[
                    PassFailCriterion(
                        description=f"Compliance with {req['clause_id']}",
                        source_ref=SourceReference(
                            chunk_id=req["chunk_id"],
                            document_id=req["document_id"],
                            clause_id=req["clause_id"],
                            title=req["title"]
                        )
                    )
                ],
                source_chunks=[req["chunk_id"]]
            )

            test_cases.append(test_case)
            tc_number += 1

        return test_cases

    def _extract_procedure_from_response(
        self,
        response: str,
        clause_id: str
    ) -> List[TestStep]:
        """Extract procedure steps from LLM response."""
        steps = []

        # Look for numbered steps in response
        step_pattern = r'(?:Step\s*)?(\d+)[.:]\s*(.+?)(?=(?:Step\s*)?\d+[.:]|$)'
        matches = re.findall(step_pattern, response, re.IGNORECASE | re.DOTALL)

        if matches:
            for num, action in matches[:10]:  # Limit to 10 steps
                steps.append(TestStep(
                    step_number=int(num),
                    action=action.strip()[:200]
                ))
        else:
            # Default steps
            steps = [
                TestStep(step_number=1, action="Prepare test setup according to standard"),
                TestStep(step_number=2, action="Configure EUT in specified operating mode"),
                TestStep(step_number=3, action="Perform required measurements"),
                TestStep(step_number=4, action="Record and evaluate results")
            ]

        return steps

    # =========================================================
    # Helper Methods
    # =========================================================

    def _find_relevant_standards(self, query: str) -> List[str]:
        """Find relevant standards using semantic search."""
        if not self.qdrant_client or not self.embedding_model:
            # Return all available standards
            return self._get_all_standard_ids()

        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=10
            )

            # Extract unique document IDs
            doc_ids = list(set(
                r.payload.get("document_id", "") for r in results
                if r.payload.get("document_id")
            ))
            return doc_ids[:5]  # Limit to 5 standards

        except Exception as e:
            print(f"Semantic search failed: {e}")
            return self._get_all_standard_ids()

    def _get_all_standard_ids(self) -> List[str]:
        """Get all standard IDs from Neo4j."""
        if not self.neo4j_driver:
            return []

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(self.queries.GET_ALL_DOCUMENTS)
                return [record["document_id"] for record in result][:5]
        except:
            return []

    def _detect_test_type(self, content: str) -> TestType:
        """Detect test type from requirement content."""
        content_lower = content.lower()

        if "radiated emission" in content_lower or "radiated disturbance" in content_lower:
            return TestType.RADIATED_EMISSIONS
        elif "conducted emission" in content_lower or "conducted disturbance" in content_lower:
            return TestType.CONDUCTED_EMISSIONS
        elif "radiated immunity" in content_lower or "radiated susceptibility" in content_lower:
            return TestType.RADIATED_IMMUNITY
        elif "conducted immunity" in content_lower:
            return TestType.CONDUCTED_IMMUNITY
        elif "esd" in content_lower or "electrostatic" in content_lower:
            return TestType.ESD
        elif "eft" in content_lower or "fast transient" in content_lower:
            return TestType.EFT
        elif "surge" in content_lower:
            return TestType.SURGE
        elif "emc" in content_lower or "electromagnetic" in content_lower:
            return TestType.GENERAL_EMC

        return TestType.OTHER

    def _get_priority(self, requirement_type: str) -> TestPriority:
        """Map requirement type to test priority."""
        mapping = {
            RequirementType.MANDATORY.value: TestPriority.CRITICAL,
            RequirementType.PROHIBITION.value: TestPriority.HIGH,
            RequirementType.RECOMMENDATION.value: TestPriority.MEDIUM,
            RequirementType.PERMISSION.value: TestPriority.LOW
        }
        return mapping.get(requirement_type, TestPriority.MEDIUM)

    def _build_context(
        self,
        requirements: List[Dict],
        tables: List[Dict],
        figures: List[Dict]
    ) -> str:
        """Build context string for LLM."""
        context_parts = []

        for req in requirements[:20]:  # Limit context size
            context_parts.append(
                f"[Clause {req['clause_id']}] {req['title']}\n{req['content']}\n"
            )

        return "\n---\n".join(context_parts)

    def _extract_equipment(self, requirements: List[Dict]) -> List[EquipmentItem]:
        """Extract equipment from requirements."""
        equipment = []
        equipment_keywords = [
            "receiver", "antenna", "LISN", "generator", "amplifier",
            "probe", "meter", "analyzer", "oscilloscope", "simulator"
        ]

        seen = set()
        for req in requirements:
            content = req.get("content", "").lower()
            for keyword in equipment_keywords:
                if keyword.lower() in content and keyword not in seen:
                    equipment.append(EquipmentItem(
                        name=keyword.title(),
                        calibration_required=True,
                        source_ref=SourceReference(
                            chunk_id=req["chunk_id"],
                            document_id=req["document_id"],
                            clause_id=req["clause_id"]
                        )
                    ))
                    seen.add(keyword)

        return equipment

    def _extract_environmental_conditions(
        self,
        requirements: List[Dict]
    ) -> List[EnvironmentalCondition]:
        """Extract environmental conditions from requirements."""
        conditions = []

        # Default EMC test conditions
        conditions.append(EnvironmentalCondition(
            parameter="Temperature",
            value="23 +/- 5 C"
        ))
        conditions.append(EnvironmentalCondition(
            parameter="Humidity",
            value="45-75% RH"
        ))

        return conditions

    def _build_coverage_matrix(
        self,
        requirements: List[Dict],
        test_cases: List[TestCase]
    ) -> RequirementCoverageMatrix:
        """Build requirement coverage matrix."""
        items = []
        covered = 0

        # Map test cases to requirements
        tc_by_clause = {}
        for tc in test_cases:
            tc_by_clause[tc.source_clause] = tc.test_case_id

        for i, req in enumerate(requirements):
            clause_id = req["clause_id"]
            tc_id = tc_by_clause.get(clause_id)
            is_covered = tc_id is not None

            if is_covered:
                covered += 1

            items.append(RequirementCoverageItem(
                requirement_id=f"REQ-{i+1:03d}",
                requirement_text=req["content"][:200],
                requirement_type=req["requirement_type"],
                source_clause=clause_id,
                covered_by_tests=[tc_id] if tc_id else [],
                coverage_status="covered" if is_covered else "not_covered"
            ))

        total = len(requirements) if requirements else 1
        return RequirementCoverageMatrix(
            total_requirements=len(requirements),
            covered_requirements=covered,
            not_covered=len(requirements) - covered,
            coverage_percentage=round(covered / total * 100, 2),
            items=items
        )

    def _validate_test_plan(
        self,
        test_cases: List[TestCase],
        requirements: List[Dict]
    ) -> TestPlanValidation:
        """Validate the generated test plan."""
        warnings = []
        missing = []

        # Check mandatory requirement coverage
        mandatory_reqs = [
            r for r in requirements
            if r["requirement_type"] == RequirementType.MANDATORY.value
        ]
        covered_clauses = {tc.source_clause for tc in test_cases}

        for req in mandatory_reqs:
            if req["clause_id"] not in covered_clauses:
                missing.append(req["clause_id"])
                warnings.append(
                    f"Mandatory requirement {req['clause_id']} has no test case"
                )

        is_valid = len(missing) == 0
        groundedness = 1.0 if is_valid else max(0.5, 1 - (len(missing) / len(mandatory_reqs)))

        return TestPlanValidation(
            is_valid=is_valid,
            groundedness_score=groundedness,
            warnings=warnings,
            missing_requirements=missing
        )

    def _generate_title(self, query: str, standard_ids: List[str]) -> str:
        """Generate test plan title."""
        standards = ", ".join(standard_ids[:3])
        return f"EMC Test Plan - {standards}"

    def _generate_scope(self, query: str, standard_ids: List[str]) -> str:
        """Generate test plan scope."""
        return f"This test plan covers EMC compliance testing based on {', '.join(standard_ids)}."

    def _extract_eut_description(self, query: str) -> str:
        """Extract EUT description from query."""
        return "Equipment Under Test as specified in test request"

    def _format_test_plan(self, test_plan: TestPlan) -> str:
        """Format test plan for human-readable output."""
        # Import the exporter for formatting
        from services.test_plan_exporter import TestPlanExporter
        exporter = TestPlanExporter()
        return exporter.export_to_text(test_plan)
