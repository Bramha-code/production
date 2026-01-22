import re
import os
from docx import Document
from docx.shared import Pt
from openai import OpenAI
# Assuming these are in your local pdf_generator.py
from pdf_generator import generate_docx_report, extract_component_name

client = OpenAI(
    base_url="http://192.168.1.30:1234/v1",
    api_key="lm-studio"  # dummy value
)

def architect_plan(query: str) -> list:
    """Agent 1: Generates a dense 10-chapter structure for a 50-page report."""
    print("[DEBUG] Architect is planning a 50-page technical structure...")
    prompt = (
        f"Create an exhaustive 10-chapter technical outline for a 50-page report on: {query}. "
        "Each chapter must have exactly 4 specific sub-sections. "
        "Return ONLY a markdown list of headers (e.g., # Chapter 1, ## Sub-section 1.1)."
    )
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}]
    )
    outline = response.choices[0].message.content
    return re.findall(r'^(#{1,2}\s+.*)', outline, re.MULTILINE)

def writer_agent(query: str, chapter_title: str, full_outline: str) -> str:
    """Agent 2: Writes highly technical content for each section."""
    print(f"[DEBUG] Writing section: {chapter_title}")
    # We use a system prompt here to enforce technical rigor
    prompt = (
        f"You are a senior technical writer. You are writing a 50-page deep-dive report on {query}.\n"
        f"Current Outline Roadmap:\n{full_outline}\n\n"
        f"TASK: Write the content for the section: '{chapter_title}'.\n"
        "REQUIREMENTS:\n"
        "1.Highly technical content.\n"
        "2. Include markdown tables for data specifications where relevant.\n"
        "3. Use professional terminology and detailed analysis.\n"
        "4. DO NOT write a conclusion unless the chapter title is 'Conclusion'."
    )
    response = client.chat.completions.create(
        model="qwen/qwen3-vl-4b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def generate_comprehensive_report(query: str, template_path: str):
    # 1. Identity Setup
    comp_name = extract_component_name(query)
    
    # 2. Planning
    outline_headers = architect_plan(query)
    full_outline_str = "\n".join(outline_headers)
    
    # 3. Execution
    full_content = []
    total_sections = len(outline_headers)
    print(f"[DEBUG] Starting generation of {total_sections} high-density sections...")
    
    for i, header in enumerate(outline_headers):
        print(f"[DEBUG] Progress: {i+1}/{total_sections}")
        section_content = writer_agent(query, header, full_outline_str)
        full_content.append(section_content)
        
    # Combine and Parse
    final_markdown = "\n\n".join(full_content)
    
    print("[DEBUG] Handing off to DOCX Generator...")
    generate_docx_report(final_markdown, template_path, comp_name)

if __name__ == '__main__':
    user_query = input("Enter your subject for the 50-page report: ")
    generate_comprehensive_report(user_query, 'template.docx')