import asyncio
import json
import re
import time
import requests
from openai import AsyncOpenAI
from pdf_utils import generate_docx_report
import os
from datetime import datetime

# --------------------
# CONFIG
# --------------------
LLM_BASE_URL = "http://192.168.0.254:1234/v1"
API_URL = "https://junction-expanded-biological-hypothetical.trycloudflare.com/api/query/grounded"

PAGE_CONFIG = {
    30: {"chapters": 6,  "words_per_chapter": 900},
    50: {"chapters": 8,  "words_per_chapter": 1200},
    70: {"chapters": 10, "words_per_chapter": 1600},
    85: {"chapters": 12, "words_per_chapter": 2000},
}



class FileLogger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"report_run_{ts}.log")

    def _write(self, text):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def write(self, msg):
        print(msg)
        self._write(msg)

    def json(self, obj):
        formatted = json.dumps(obj, indent=2)
        print(formatted)
        self._write(formatted)



# --------------------
# CONTEXT AGENT
# --------------------
class ContextAgent:
    """Optional Knowledge Graph grounding."""
    @staticmethod
    def get_grounding(query: str):
        try:
            payload = {"query": query, "top_k": 10}
            response = requests.post(API_URL, json=payload, timeout=8)
            results = response.json().get("results", [])
            if not results:
                return ""
            return "\n".join(
                f"Source [{r.get('document_id','N/A')}]: {r.get('content','')}"
                for r in results
            )
        except Exception:
            return ""  # Non-blocking by design

# --------------------
# REPORT AGENT
# --------------------
class ReportAgent:
    def __init__(self, log_container, progress_bar):
        self.client = AsyncOpenAI(
            base_url=LLM_BASE_URL,
            api_key="lm-studio"
        )
        self.log_container = log_container
        self.progress_bar = progress_bar
        self.metrics = {
            "llm_calls": 0,
            "llm_time": 0.0,
            "chapter_timings": []
        }

    def log(self, msg):
        self.log_container.write(f"🔄 {msg}")

    async def call_llm(self, sys, usr):
        start = time.perf_counter()
        res = await self.client.chat.completions.create(
            model="qwen/qwen3-vl-4b",
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": usr}
            ],
            temperature=0.3
        )
        elapsed = time.perf_counter() - start
        self.metrics["llm_calls"] += 1
        self.metrics["llm_time"] += elapsed
        return res.choices[0].message.content

    # --------------------
    # MAIN PIPELINE
    # --------------------
    async def run_pipeline(
        self,
        query: str,
        report_pages: int = 50,
        template_path: str = "template.docx"
    ):
        start_time = time.perf_counter()

        if report_pages not in PAGE_CONFIG:
            raise ValueError("Report pages must be one of 30, 50, 70, 85")

        cfg = PAGE_CONFIG[report_pages]

        self.log("Identifying subject...")
        comp_name = await self.call_llm(
            "Extract subject name only.",
            f"Return only the subject name from: {query}"
        )

        self.log(f"Fetching Knowledge Graph context for {comp_name} (optional)...")
        kg_context = ContextAgent.get_grounding(comp_name)

        self.log("Architecting report structure...")
        plan = await self.call_llm(
            "Technical Architect.",
            f"""
            Generate a JSON list of exactly {cfg['chapters']} 
            highly technical chapter titles for a {report_pages}-page report 
            on {comp_name}.
            """
        )

        chapters = json.loads(re.search(r'\[.*\]', plan, re.DOTALL).group())

        full_content = []
        total = len(chapters)

        for idx, title in enumerate(chapters):
            chap_start = time.perf_counter()

            self.log(f"Generating Chapter {idx+1}/{total}: {title}")

            sys_msg = f"""
            Senior Technical Writer.
            You MUST proceed even if reference context is empty.

            Reference Context (optional):
            {kg_context[:3000]}

            Write a deeply technical chapter titled "{title}".
            Start with '# {title}'.
            Target length: ~{cfg['words_per_chapter']} words.
            No meta commentary.
            """

            content = await self.call_llm(
                sys_msg,
                f"Write the full technical chapter: {title}"
            )

            elapsed = time.perf_counter() - chap_start
            self.metrics["chapter_timings"].append({
                "chapter": title,
                "seconds": round(elapsed, 2)
            })

            full_content.append(content)
            self.progress_bar.progress((idx + 1) / total)

        self.log("Converting to DOCX...")
        file_path = generate_docx_report(
            "\n\n".join(full_content),
            template_path,
            comp_name
        )

        total_time = time.perf_counter() - start_time

        # --------------------
        # EXECUTION REPORT
        # --------------------
        execution_report = {
            "component": comp_name,
            "report_pages": report_pages,
            "chapters": cfg["chapters"],
            "words_per_chapter": cfg["words_per_chapter"],
            "kg_context_used": bool(kg_context),
            "total_llm_calls": self.metrics["llm_calls"],
            "total_llm_time_sec": round(self.metrics["llm_time"], 2),
            "avg_llm_latency_sec": round(
                self.metrics["llm_time"] / max(1, self.metrics["llm_calls"]), 2
            ),
            "chapter_generation_times": self.metrics["chapter_timings"],
            "total_execution_time_sec": round(total_time, 2)
        }

        self.log("Execution Summary:")
        self.log_container.json(execution_report)

        return file_path, execution_report

if __name__ == "__main__":
    import asyncio

    logger = FileLogger()

    class DummyProgress:
        def progress(self, value):
            pass

    agent = ReportAgent(
        log_container=logger,
        progress_bar=DummyProgress()
    )

    asyncio.run(
        agent.run_pipeline(
            query=input("Enter your Query"),
            report_pages=input("Enter the number of Pages")
        )
    )

    print(f"\n📄 Logs saved to: {logger.log_path}")
