"""
EMC Test Plan Exporter

Exports test plans to human-readable formats:
- Professional text format for display/printing
- PDF format for formal documentation
- Markdown format for web display
"""

from typing import List, Optional
from datetime import datetime
from io import BytesIO

from models.test_plan_models import (
    TestPlan, TestCase, TestStep, EquipmentItem,
    RequirementCoverageMatrix, TestPlanValidation
)

# PDF generation imports
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, KeepTogether, HRFlowable
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class TestPlanExporter:
    """
    Exports test plans to various human-readable formats.
    """

    def __init__(self):
        self.line_width = 80
        self.box_char = "═"
        self.h_line = "─"
        self.v_line = "│"

    # =========================================================
    # Text Export (Professional Format)
    # =========================================================

    def export_to_text(self, test_plan: TestPlan) -> str:
        """
        Export test plan to professional text format.
        """
        lines = []

        # Header
        lines.append(self._make_header())
        lines.append(self._center_text("EMC TEST PLAN"))
        lines.append(self._make_header())
        lines.append("")

        # Document Info
        lines.append(f"Document No  : {test_plan.document_number}")
        lines.append(f"Revision     : {test_plan.revision}")
        lines.append(f"Date         : {test_plan.date}")
        lines.append(f"Title        : {test_plan.title}")
        lines.append("")

        # Section 1: Scope
        lines.append(self._section_header("1. SCOPE"))
        lines.append(self._wrap_text(test_plan.scope))
        lines.append("")

        # Section 2: Applicable Standards
        lines.append(self._section_header("2. APPLICABLE STANDARDS"))
        for std in test_plan.applicable_standards:
            lines.append(f"   - {std}")
        lines.append("")

        # Section 3: Equipment Under Test
        lines.append(self._section_header("3. EQUIPMENT UNDER TEST (EUT)"))
        lines.append(f"   Description: {test_plan.eut_description or 'As specified in test request'}")
        lines.append(f"   Operating Conditions: Normal operation")
        lines.append("")

        # Section 4: Test Equipment Required
        lines.append(self._section_header("4. TEST EQUIPMENT REQUIRED"))
        lines.append(self._make_equipment_table(test_plan.all_equipment))
        lines.append("")

        # Section 5: Environmental Conditions
        lines.append(self._section_header("5. ENVIRONMENTAL CONDITIONS"))
        for cond in test_plan.environmental_conditions:
            lines.append(f"   - {cond.parameter}: {cond.value}")
        if not test_plan.environmental_conditions:
            lines.append("   - Temperature: 23 +/- 5 C")
            lines.append("   - Humidity: 45-75% RH")
        lines.append("")

        # Section 6: Test Cases
        lines.append(self._section_header("6. TEST CASES"))
        lines.append(f"   Total Test Cases: {test_plan.total_test_cases}")
        lines.append("")

        for tc in test_plan.test_cases:
            lines.append(self._format_test_case(tc))
            lines.append("")

        # Section 7: Traceability Matrix
        lines.append(self._section_header("7. REQUIREMENT TRACEABILITY MATRIX"))
        lines.append(self._make_traceability_table(test_plan.coverage_matrix))
        lines.append("")

        # Section 8: Summary
        lines.append(self._section_header("8. SUMMARY"))
        lines.append(f"   Total Requirements : {test_plan.coverage_matrix.total_requirements}")
        lines.append(f"   Covered           : {test_plan.coverage_matrix.covered_requirements}")
        lines.append(f"   Coverage          : {test_plan.coverage_matrix.coverage_percentage}%")
        lines.append(f"   Validation Status : {'PASSED' if test_plan.validation.is_valid else 'WARNINGS'}")
        lines.append("")

        if test_plan.validation.warnings:
            lines.append("   Warnings:")
            for warning in test_plan.validation.warnings[:5]:
                lines.append(f"   - {warning}")
            lines.append("")

        # Section 9: Approval
        lines.append(self._section_header("9. APPROVAL"))
        lines.append("")
        lines.append("   Prepared by : ________________________  Date: ____________")
        lines.append("")
        lines.append("   Reviewed by : ________________________  Date: ____________")
        lines.append("")
        lines.append("   Approved by : ________________________  Date: ____________")
        lines.append("")

        # Footer
        lines.append(self._make_header())
        lines.append(self._center_text("END OF DOCUMENT"))
        lines.append(self._make_header())

        return "\n".join(lines)

    # =========================================================
    # Formatting Helpers
    # =========================================================

    def _make_header(self) -> str:
        """Create header/footer line."""
        return self.box_char * self.line_width

    def _center_text(self, text: str) -> str:
        """Center text within line width."""
        padding = (self.line_width - len(text)) // 2
        return " " * padding + text

    def _section_header(self, title: str) -> str:
        """Create section header."""
        return f"\n{title}\n{'─' * len(title)}"

    def _wrap_text(self, text: str, indent: int = 3) -> str:
        """Wrap text to line width with indent."""
        words = text.split()
        lines = []
        current_line = " " * indent

        for word in words:
            if len(current_line) + len(word) + 1 <= self.line_width:
                current_line += word + " "
            else:
                lines.append(current_line.rstrip())
                current_line = " " * indent + word + " "

        if current_line.strip():
            lines.append(current_line.rstrip())

        return "\n".join(lines)

    def _format_test_case(self, tc: TestCase) -> str:
        """Format a single test case."""
        lines = []

        # Test case header box
        header = f" TEST CASE {tc.test_case_id}: {tc.title[:50]} "
        lines.append("   ┌" + "─" * 72 + "┐")
        lines.append(f"   │{header:<72}│")
        lines.append("   ├" + "─" * 72 + "┤")

        # Requirement
        lines.append(f"   │ Requirement: {tc.source_clause:<58}│")
        req_text = tc.requirement_text[:65]
        lines.append(f"   │ \"{req_text}...\"" + " " * max(0, 70 - len(req_text) - 5) + "│")
        lines.append("   │" + " " * 72 + "│")

        # Objective
        lines.append(f"   │ OBJECTIVE:{' ' * 61}│")
        obj_text = tc.objective[:68]
        lines.append(f"   │ {obj_text:<70}│")
        lines.append("   │" + " " * 72 + "│")

        # Pre-conditions
        lines.append(f"   │ PRE-CONDITIONS:{' ' * 56}│")
        for pre in tc.pre_conditions[:3]:
            lines.append(f"   │ - {pre[:67]:<69}│")
        lines.append("   │" + " " * 72 + "│")

        # Procedure
        lines.append(f"   │ PROCEDURE:{' ' * 61}│")
        for step in tc.procedure_steps[:8]:
            step_text = f"Step {step.step_number}: {step.action[:58]}"
            lines.append(f"   │ {step_text:<70}│")
        lines.append("   │" + " " * 72 + "│")

        # Pass Criteria
        lines.append(f"   │ PASS CRITERIA:{' ' * 57}│")
        for criteria in tc.pass_fail_criteria[:3]:
            crit_text = criteria.description[:68]
            lines.append(f"   │ - {crit_text:<69}│")
        lines.append("   │" + " " * 72 + "│")

        # Equipment
        if tc.equipment_required:
            lines.append(f"   │ EQUIPMENT REQUIRED:{' ' * 52}│")
            for eq in tc.equipment_required[:5]:
                eq_text = f"{eq.name}"
                if eq.specification:
                    eq_text += f" ({eq.specification})"
                lines.append(f"   │ - {eq_text[:69]:<69}│")
            lines.append("   │" + " " * 72 + "│")

        # Source
        source = tc.source_chunks[0] if tc.source_chunks else "N/A"
        lines.append(f"   │ Source: [{source[:60]}]" + " " * max(0, 60 - len(source)) + "│")

        # Close box
        lines.append("   └" + "─" * 72 + "┘")

        return "\n".join(lines)

    def _make_equipment_table(self, equipment: List[EquipmentItem]) -> str:
        """Create equipment table."""
        lines = []

        lines.append("   ┌" + "─" * 30 + "┬" + "─" * 25 + "┬" + "─" * 15 + "┐")
        lines.append(f"   │{'Equipment':<30}│{'Specification':<25}│{'Calibration':<15}│")
        lines.append("   ├" + "─" * 30 + "┼" + "─" * 25 + "┼" + "─" * 15 + "┤")

        if equipment:
            for eq in equipment[:10]:
                name = eq.name[:28]
                spec = (eq.specification or "As required")[:23]
                cal = "Yes" if eq.calibration_required else "No"
                lines.append(f"   │{name:<30}│{spec:<25}│{cal:<15}│")
        else:
            lines.append(f"   │{'EMI Receiver':<30}│{'Per standard':<25}│{'Yes':<15}│")
            lines.append(f"   │{'Test Antenna':<30}│{'Per standard':<25}│{'Yes':<15}│")
            lines.append(f"   │{'LISN':<30}│{'50uH/50 Ohm':<25}│{'Yes':<15}│")

        lines.append("   └" + "─" * 30 + "┴" + "─" * 25 + "┴" + "─" * 15 + "┘")

        return "\n".join(lines)

    def _make_traceability_table(self, matrix: RequirementCoverageMatrix) -> str:
        """Create traceability matrix table."""
        lines = []

        lines.append("   ┌" + "─" * 15 + "┬" + "─" * 15 + "┬" + "─" * 15 + "┬" + "─" * 15 + "┐")
        lines.append(f"   │{'Requirement':<15}│{'Clause':<15}│{'Test Case':<15}│{'Status':<15}│")
        lines.append("   ├" + "─" * 15 + "┼" + "─" * 15 + "┼" + "─" * 15 + "┼" + "─" * 15 + "┤")

        for item in matrix.items[:15]:  # Limit rows
            req_id = item.requirement_id[:13]
            clause = item.source_clause[:13]
            tc = item.covered_by_tests[0][:13] if item.covered_by_tests else "None"
            status = "Covered" if item.coverage_status == "covered" else "NOT COVERED"
            lines.append(f"   │{req_id:<15}│{clause:<15}│{tc:<15}│{status:<15}│")

        lines.append("   └" + "─" * 15 + "┴" + "─" * 15 + "┴" + "─" * 15 + "┴" + "─" * 15 + "┘")

        return "\n".join(lines)

    # =========================================================
    # Markdown Export
    # =========================================================

    def export_to_markdown(self, test_plan: TestPlan) -> str:
        """Export test plan to Markdown format."""
        lines = []

        lines.append(f"# EMC TEST PLAN")
        lines.append("")
        lines.append(f"**Document No:** {test_plan.document_number}")
        lines.append(f"**Revision:** {test_plan.revision}")
        lines.append(f"**Date:** {test_plan.date}")
        lines.append("")

        lines.append("## 1. Scope")
        lines.append(test_plan.scope)
        lines.append("")

        lines.append("## 2. Applicable Standards")
        for std in test_plan.applicable_standards:
            lines.append(f"- {std}")
        lines.append("")

        lines.append("## 3. Test Cases")
        for tc in test_plan.test_cases:
            lines.append(f"### {tc.test_case_id}: {tc.title}")
            lines.append(f"**Requirement:** {tc.source_clause}")
            lines.append(f"**Objective:** {tc.objective}")
            lines.append("")
            lines.append("**Procedure:**")
            for step in tc.procedure_steps:
                lines.append(f"{step.step_number}. {step.action}")
            lines.append("")
            lines.append("**Pass Criteria:**")
            for crit in tc.pass_fail_criteria:
                lines.append(f"- {crit.description}")
            lines.append("")

        lines.append("## 4. Coverage Summary")
        lines.append(f"- Total Requirements: {test_plan.coverage_matrix.total_requirements}")
        lines.append(f"- Covered: {test_plan.coverage_matrix.covered_requirements}")
        lines.append(f"- Coverage: {test_plan.coverage_matrix.coverage_percentage}%")
        lines.append("")

        return "\n".join(lines)

    # =========================================================
    # JSON Export
    # =========================================================

    def export_to_json(self, test_plan: TestPlan) -> str:
        """Export test plan to JSON format."""
        import json
        return json.dumps(test_plan.dict(), indent=2, default=str)

    # =========================================================
    # PDF Export (Professional Format)
    # =========================================================

    def export_to_pdf(self, test_plan: TestPlan) -> bytes:
        """
        Export test plan to professional PDF format.
        Returns PDF as bytes.
        """
        if not PDF_AVAILABLE:
            raise ImportError("reportlab is required for PDF export. Install with: pip install reportlab")

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=25*mm,
            bottomMargin=25*mm
        )

        # Create styles
        styles = self._create_pdf_styles()
        story = []

        # Title Page
        story.extend(self._create_title_page(test_plan, styles))
        story.append(PageBreak())

        # Table of Contents placeholder
        story.append(Paragraph("TABLE OF CONTENTS", styles['SectionHeader']))
        story.append(Spacer(1, 10*mm))
        toc_items = [
            ("1. Scope", "3"),
            ("2. Applicable Standards", "3"),
            ("3. Equipment Under Test (EUT)", "3"),
            ("4. Test Equipment Required", "4"),
            ("5. Environmental Conditions", "4"),
            ("6. Test Cases", "5"),
            ("7. Requirement Traceability Matrix", "..."),
            ("8. Summary", "..."),
            ("9. Approval", "..."),
        ]
        for item, page in toc_items:
            story.append(Paragraph(f"{item} {'.' * (60 - len(item))} {page}", styles['Normal']))
        story.append(PageBreak())

        # Section 1: Scope
        story.append(Paragraph("1. SCOPE", styles['SectionHeader']))
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph(test_plan.scope, styles['TPBodyText']))
        story.append(Spacer(1, 8*mm))

        # Section 2: Applicable Standards
        story.append(Paragraph("2. APPLICABLE STANDARDS", styles['SectionHeader']))
        story.append(Spacer(1, 5*mm))
        for std in test_plan.applicable_standards:
            story.append(Paragraph(f"• {std}", styles['BulletItem']))
        story.append(Spacer(1, 8*mm))

        # Section 3: EUT
        story.append(Paragraph("3. EQUIPMENT UNDER TEST (EUT)", styles['SectionHeader']))
        story.append(Spacer(1, 5*mm))
        eut_desc = test_plan.eut_description or "As specified in test request"
        story.append(Paragraph(f"<b>Description:</b> {eut_desc}", styles['TPBodyText']))
        story.append(Paragraph(f"<b>Operating Conditions:</b> Normal operation", styles['TPBodyText']))
        story.append(Spacer(1, 8*mm))

        # Section 4: Test Equipment
        story.append(Paragraph("4. TEST EQUIPMENT REQUIRED", styles['SectionHeader']))
        story.append(Spacer(1, 5*mm))
        story.append(self._create_equipment_table(test_plan.all_equipment, styles))
        story.append(Spacer(1, 8*mm))

        # Section 5: Environmental Conditions
        story.append(Paragraph("5. ENVIRONMENTAL CONDITIONS", styles['SectionHeader']))
        story.append(Spacer(1, 5*mm))
        if test_plan.environmental_conditions:
            for cond in test_plan.environmental_conditions:
                story.append(Paragraph(f"• {cond.parameter}: {cond.value}", styles['BulletItem']))
        else:
            story.append(Paragraph("• Temperature: 23 ± 5°C", styles['BulletItem']))
            story.append(Paragraph("• Humidity: 45-75% RH", styles['BulletItem']))
            story.append(Paragraph("• Atmospheric Pressure: 86-106 kPa", styles['BulletItem']))
        story.append(Spacer(1, 8*mm))

        # Section 6: Test Cases
        story.append(Paragraph("6. TEST CASES", styles['SectionHeader']))
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph(f"<b>Total Test Cases:</b> {test_plan.total_test_cases}", styles['TPBodyText']))
        story.append(Spacer(1, 5*mm))

        for tc in test_plan.test_cases:
            story.extend(self._create_test_case_section(tc, styles))
            story.append(Spacer(1, 5*mm))

        # Section 7: Traceability Matrix
        story.append(PageBreak())
        story.append(Paragraph("7. REQUIREMENT TRACEABILITY MATRIX", styles['SectionHeader']))
        story.append(Spacer(1, 5*mm))
        story.append(self._create_traceability_table(test_plan.coverage_matrix, styles))
        story.append(Spacer(1, 8*mm))

        # Section 8: Summary
        story.append(Paragraph("8. SUMMARY", styles['SectionHeader']))
        story.append(Spacer(1, 5*mm))
        summary_data = [
            ["Total Requirements", str(test_plan.coverage_matrix.total_requirements)],
            ["Covered Requirements", str(test_plan.coverage_matrix.covered_requirements)],
            ["Coverage Percentage", f"{test_plan.coverage_matrix.coverage_percentage}%"],
            ["Validation Status", "PASSED" if test_plan.validation.is_valid else "WARNINGS"],
        ]
        summary_table = Table(summary_data, colWidths=[120*mm, 50*mm])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f3f4f6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1f2937')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 8*mm))

        if test_plan.validation.warnings:
            story.append(Paragraph("<b>Warnings:</b>", styles['TPBodyText']))
            for warning in test_plan.validation.warnings[:5]:
                story.append(Paragraph(f"• {warning}", styles['WarningItem']))
        story.append(Spacer(1, 8*mm))

        # Section 9: Approval
        story.append(Paragraph("9. APPROVAL", styles['SectionHeader']))
        story.append(Spacer(1, 10*mm))
        approval_data = [
            ["Role", "Name", "Signature", "Date"],
            ["Prepared by", "", "", ""],
            ["Reviewed by", "", "", ""],
            ["Approved by", "", "", ""],
        ]
        approval_table = Table(approval_data, colWidths=[35*mm, 50*mm, 50*mm, 35*mm])
        approval_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f2937')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
            ('PADDING', (0, 0), (-1, -1), 12),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWHEIGHT', (0, 1), (-1, -1), 25*mm),
        ]))
        story.append(approval_table)

        # Footer
        story.append(Spacer(1, 15*mm))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1f2937')))
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph("END OF DOCUMENT", styles['CenterText']))

        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes

    def _create_pdf_styles(self):
        """Create custom PDF styles."""
        styles = getSampleStyleSheet()

        # Title style
        styles.add(ParagraphStyle(
            name='DocTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1f2937'),
            spaceAfter=20,
            alignment=TA_CENTER
        ))

        # Section header
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading1'],
            fontSize=14,
            textColor=colors.HexColor('#1f2937'),
            spaceBefore=15,
            spaceAfter=10,
            borderWidth=0,
            borderPadding=0,
            borderColor=colors.HexColor('#3b82f6'),
            backColor=colors.HexColor('#eff6ff'),
            leftIndent=0,
            rightIndent=0,
            borderRadius=3
        ))

        # Subsection header
        styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=styles['Heading2'],
            fontSize=11,
            textColor=colors.HexColor('#374151'),
            spaceBefore=10,
            spaceAfter=5,
            fontName='Helvetica-Bold'
        ))

        # Custom body text (use TPBodyText to avoid conflict)
        styles.add(ParagraphStyle(
            name='TPBodyText',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#374151'),
            spaceAfter=5,
            alignment=TA_JUSTIFY,
            leading=14
        ))

        # Bullet item
        styles.add(ParagraphStyle(
            name='BulletItem',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#374151'),
            leftIndent=15,
            spaceAfter=3
        ))

        # Warning item
        styles.add(ParagraphStyle(
            name='WarningItem',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#dc2626'),
            leftIndent=15,
            spaceAfter=3
        ))

        # Center text
        styles.add(ParagraphStyle(
            name='CenterText',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#6b7280'),
            alignment=TA_CENTER
        ))

        # Table header
        styles.add(ParagraphStyle(
            name='TableHeader',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.white,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER
        ))

        # Table cell
        styles.add(ParagraphStyle(
            name='TableCell',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#374151'),
            alignment=TA_LEFT
        ))

        return styles

    def _create_title_page(self, test_plan: TestPlan, styles) -> list:
        """Create title page elements."""
        elements = []

        elements.append(Spacer(1, 30*mm))

        # Company/Organization placeholder
        elements.append(Paragraph("EMC TEST LABORATORY", styles['CenterText']))
        elements.append(Spacer(1, 20*mm))

        # Main title
        elements.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor('#3b82f6')))
        elements.append(Spacer(1, 10*mm))
        elements.append(Paragraph("EMC TEST PLAN", styles['DocTitle']))
        elements.append(Spacer(1, 5*mm))
        elements.append(Paragraph(test_plan.title, styles['SubsectionHeader']))
        elements.append(Spacer(1, 10*mm))
        elements.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor('#3b82f6')))

        elements.append(Spacer(1, 30*mm))

        # Document info table
        info_data = [
            ["Document Number:", test_plan.document_number],
            ["Revision:", test_plan.revision],
            ["Date:", test_plan.date],
            ["Test Plan ID:", test_plan.test_plan_id],
        ]
        info_table = Table(info_data, colWidths=[50*mm, 80*mm])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#374151')),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(info_table)

        elements.append(Spacer(1, 40*mm))

        # Footer note
        elements.append(Paragraph(
            "This document is generated automatically based on applicable EMC standards.",
            styles['CenterText']
        ))
        elements.append(Paragraph(
            f"Generated on: {test_plan.generated_at[:10]}",
            styles['CenterText']
        ))

        return elements

    def _create_equipment_table(self, equipment: List[EquipmentItem], styles) -> Table:
        """Create equipment table for PDF."""
        headers = ["Equipment", "Specification", "Calibration Required"]
        data = [headers]

        if equipment:
            for eq in equipment[:10]:
                data.append([
                    eq.name[:40],
                    (eq.specification or "As per standard")[:30],
                    "Yes" if eq.calibration_required else "No"
                ])
        else:
            # Default equipment
            data.append(["EMI Receiver", "Per applicable standard", "Yes"])
            data.append(["Test Antenna", "Per applicable standard", "Yes"])
            data.append(["LISN", "50µH/50Ω", "Yes"])
            data.append(["Signal Generator", "Per applicable standard", "Yes"])

        table = Table(data, colWidths=[70*mm, 70*mm, 30*mm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f2937')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
        ]))
        return table

    def _create_test_case_section(self, tc: TestCase, styles) -> list:
        """Create test case section for PDF."""
        elements = []

        # Test case header
        header_text = f"Test Case {tc.test_case_id}: {tc.title[:60]}"
        elements.append(Paragraph(header_text, styles['SubsectionHeader']))

        # Test case details table
        priority_color = {
            'critical': '#dc2626',
            'high': '#ea580c',
            'medium': '#ca8a04',
            'low': '#16a34a'
        }.get(tc.priority.value, '#6b7280')

        req_text = tc.requirement_text[:200] + "..." if len(tc.requirement_text) > 200 else tc.requirement_text

        details = [
            ["Source Clause:", tc.source_clause],
            ["Requirement Type:", tc.requirement_type.value.upper()],
            ["Priority:", tc.priority.value.upper()],
            ["Test Type:", tc.test_type.value.replace('_', ' ').upper()],
        ]
        details_table = Table(details, colWidths=[40*mm, 130*mm])
        details_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#374151')),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        elements.append(details_table)
        elements.append(Spacer(1, 3*mm))

        # Requirement text
        elements.append(Paragraph(f"<b>Requirement:</b> \"{req_text}\"", styles['TPBodyText']))
        elements.append(Spacer(1, 3*mm))

        # Objective
        elements.append(Paragraph(f"<b>Objective:</b> {tc.objective}", styles['TPBodyText']))
        elements.append(Spacer(1, 3*mm))

        # Pre-conditions
        if tc.pre_conditions:
            elements.append(Paragraph("<b>Pre-conditions:</b>", styles['TPBodyText']))
            for pre in tc.pre_conditions[:5]:
                elements.append(Paragraph(f"• {pre}", styles['BulletItem']))

        # Procedure
        if tc.procedure_steps:
            elements.append(Paragraph("<b>Procedure:</b>", styles['TPBodyText']))
            for step in tc.procedure_steps[:10]:
                step_text = f"Step {step.step_number}: {step.action[:100]}"
                elements.append(Paragraph(f"  {step_text}", styles['BulletItem']))

        # Pass/Fail Criteria
        if tc.pass_fail_criteria:
            elements.append(Paragraph("<b>Pass/Fail Criteria:</b>", styles['TPBodyText']))
            for crit in tc.pass_fail_criteria[:5]:
                elements.append(Paragraph(f"• {crit.description}", styles['BulletItem']))

        # Source reference
        if tc.source_chunks:
            elements.append(Spacer(1, 2*mm))
            elements.append(Paragraph(f"<i>Source: {tc.source_chunks[0]}</i>", styles['CenterText']))

        # Separator
        elements.append(Spacer(1, 3*mm))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e5e7eb')))

        return elements

    def _create_traceability_table(self, matrix: RequirementCoverageMatrix, styles) -> Table:
        """Create traceability matrix table for PDF."""
        headers = ["Requirement ID", "Source Clause", "Test Case", "Status"]
        data = [headers]

        for item in matrix.items[:20]:  # Limit rows
            status = "✓ Covered" if item.coverage_status == "covered" else "✗ Not Covered"
            test_cases = item.covered_by_tests[0] if item.covered_by_tests else "None"
            data.append([
                item.requirement_id[:15],
                item.source_clause[:15],
                test_cases[:15],
                status
            ])

        table = Table(data, colWidths=[40*mm, 40*mm, 40*mm, 50*mm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f2937')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#d1d5db')),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9fafb')]),
        ]))
        return table
