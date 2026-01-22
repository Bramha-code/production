"""
Audit & Observability Layer (Phase 7)

Production-grade monitoring and traceability with:
1. Full lineage tracking (query → retrieval → generation)
2. Observability metrics (latency, groundedness, recall)
3. Security audit logging
4. Evaluation loops (feedback + drift detection)

This ensures end-to-end visibility for quality control.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import json
import hashlib
from pathlib import Path

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


tracer = trace.get_tracer(__name__)


# =========================================================
# Audit Event Models
# =========================================================

class AuditEventType(str, Enum):
    """Types of audit events"""
    QUERY_RECEIVED = "query_received"
    QUERY_ORCHESTRATED = "query_orchestrated"
    RETRIEVAL_PERFORMED = "retrieval_performed"
    LLM_GENERATED = "llm_generated"
    RESPONSE_RETURNED = "response_returned"
    FEEDBACK_RECEIVED = "feedback_received"
    ERROR_OCCURRED = "error_occurred"


class AuditEvent(BaseModel):
    """A single audit event"""
    event_id: str = Field(default_factory=lambda: hashlib.sha256(
        str(datetime.utcnow()).encode()
    ).hexdigest()[:16])
    event_type: AuditEventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Context
    session_id: str
    user_id: Optional[str] = None
    
    # Event data
    data: Dict[str, Any] = Field(default_factory=dict)
    
    # Lineage tracking
    parent_event_id: Optional[str] = None
    trace_id: Optional[str] = None
    
    # Security
    sanitized: bool = Field(default=False, description="PII removed")


class QueryLineage(BaseModel):
    """
    Complete lineage of a query from start to finish.
    
    Tracks:
    - Original query and rewrites
    - Retrieved chunks with scores
    - Generated response
    - User feedback
    """
    trace_id: str
    session_id: str
    
    # Query evolution
    original_query: str
    rewritten_query: Optional[str] = None
    sub_queries: List[str] = Field(default_factory=list)
    
    # Orchestration
    query_intent: str
    retrieval_strategy: str
    
    # Retrieval
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    vector_scores: Dict[str, float] = Field(default_factory=dict)
    graph_expansion_count: int = 0
    
    # Generation
    prompt_sent: str = ""
    response_generated: str = ""
    citations: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0
    
    # Metrics
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    # Feedback
    user_feedback: Optional[str] = None  # thumbs_up, thumbs_down
    user_comment: Optional[str] = None
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


# =========================================================
# Observability Metrics
# =========================================================

class GroundednessScore(BaseModel):
    """
    Measures how well LLM output is supported by context.
    
    Methods:
    - Citation overlap: % of claims with citations
    - Factual consistency: LLM judges answer vs context
    - Hallucination detection: Identify unsupported claims
    """
    score: float = Field(ge=0.0, le=1.0)
    citation_overlap: float = Field(ge=0.0, le=1.0)
    factual_consistency: float = Field(ge=0.0, le=1.0)
    has_hallucinations: bool = False
    hallucination_examples: List[str] = Field(default_factory=list)


class RetrievalMetrics(BaseModel):
    """Retrieval quality metrics"""
    
    # Recall@K: Did we find the right documents?
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    
    # Precision: Are retrieved docs relevant?
    precision_at_5: Optional[float] = None
    
    # MRR: Mean Reciprocal Rank
    mrr: Optional[float] = None
    
    # Hit rate
    hit_rate: Optional[float] = None


class LatencyBreakdown(BaseModel):
    """Detailed latency breakdown"""
    query_orchestration_ms: float = 0.0
    vector_search_ms: float = 0.0
    graph_traversal_ms: float = 0.0
    context_assembly_ms: float = 0.0
    llm_generation_ms: float = 0.0
    total_ms: float = 0.0
    
    def get_slowest_stage(self) -> str:
        """Identify bottleneck"""
        stages = {
            "orchestration": self.query_orchestration_ms,
            "vector_search": self.vector_search_ms,
            "graph_traversal": self.graph_traversal_ms,
            "context_assembly": self.context_assembly_ms,
            "llm_generation": self.llm_generation_ms
        }
        return max(stages.items(), key=lambda x: x[1])[0]


class ObservabilityMetrics(BaseModel):
    """Complete metrics for a query"""
    trace_id: str
    
    # Quality metrics
    groundedness: GroundednessScore
    retrieval: RetrievalMetrics
    
    # Performance metrics
    latency: LatencyBreakdown
    
    # User satisfaction
    user_satisfaction: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Timestamp
    measured_at: datetime = Field(default_factory=datetime.utcnow)


# =========================================================
# Audit Logger
# =========================================================

class AuditLogger:
    """
    Production audit logger with:
    - Structured logging
    - PII sanitization
    - Log rotation
    - Query for compliance
    """
    
    def __init__(self, log_dir: Path, rotate_days: int = 30):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rotate_days = rotate_days
        
        # Current log file
        self.current_log = self.log_dir / f"audit_{datetime.utcnow().date()}.jsonl"
    
    @tracer.start_as_current_span("log_audit_event")
    def log_event(self, event: AuditEvent):
        """Log audit event to file"""
        span = trace.get_current_span()
        span.set_attribute("event_type", event.event_type)
        
        # Sanitize if needed
        if not event.sanitized:
            event = self._sanitize_event(event)
        
        # Append to log file
        with open(self.current_log, 'a') as f:
            f.write(event.json() + "\n")
    
    def log_query_lineage(self, lineage: QueryLineage):
        """Log complete query lineage"""
        # Store in separate lineage file
        lineage_file = self.log_dir / f"lineage_{datetime.utcnow().date()}.jsonl"
        
        with open(lineage_file, 'a') as f:
            f.write(lineage.json() + "\n")
    
    def _sanitize_event(self, event: AuditEvent) -> AuditEvent:
        """Remove PII from event data"""
        # Simple sanitization - would be more sophisticated in production
        
        # Redact email addresses
        import re
        for key, value in event.data.items():
            if isinstance(value, str):
                value = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                              '[REDACTED_EMAIL]', value)
                event.data[key] = value
        
        event.sanitized = True
        return event
    
    def query_events(
        self,
        event_type: Optional[AuditEventType] = None,
        session_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[AuditEvent]:
        """Query audit events"""
        events = []
        
        # Determine which log files to read
        log_files = list(self.log_dir.glob("audit_*.jsonl"))
        
        for log_file in log_files:
            with open(log_file, 'r') as f:
                for line in f:
                    event = AuditEvent.parse_raw(line)
                    
                    # Apply filters
                    if event_type and event.event_type != event_type:
                        continue
                    if session_id and event.session_id != session_id:
                        continue
                    if start_date and event.timestamp < start_date:
                        continue
                    if end_date and event.timestamp > end_date:
                        continue
                    
                    events.append(event)
        
        return events
    
    def rotate_logs(self):
        """Delete old log files"""
        cutoff = datetime.utcnow() - timedelta(days=self.rotate_days)
        
        for log_file in self.log_dir.glob("*.jsonl"):
            # Parse date from filename
            try:
                date_str = log_file.stem.split("_")[1]
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                if file_date < cutoff:
                    log_file.unlink()
                    print(f"Rotated old log: {log_file}")
            except:
                continue


# =========================================================
# Metrics Collector
# =========================================================

class MetricsCollector:
    """
    Collects and computes observability metrics.
    """
    
    def __init__(self):
        self.metrics_cache: Dict[str, ObservabilityMetrics] = {}
    
    @tracer.start_as_current_span("compute_groundedness")
    def compute_groundedness(
        self,
        response: str,
        context_chunks: List[Dict[str, Any]],
        citations: List[str]
    ) -> GroundednessScore:
        """
        Compute groundedness score.
        
        Args:
            response: Generated response
            context_chunks: Retrieved context
            citations: Provided citations
        
        Returns:
            GroundednessScore
        """
        span = trace.get_current_span()
        
        # Metric 1: Citation overlap
        # Count sentences in response
        sentences = response.split(". ")
        cited_sentences = sum(1 for s in sentences if any(c in s for c in citations))
        citation_overlap = cited_sentences / len(sentences) if sentences else 0.0
        
        # Metric 2: Factual consistency (simplified)
        # Check if key phrases from response appear in context
        context_text = " ".join([c.get("content_text", "") for c in context_chunks])
        
        # Extract key phrases (nouns, entities)
        import re
        key_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response)
        
        consistent_phrases = sum(1 for phrase in key_phrases if phrase in context_text)
        factual_consistency = consistent_phrases / len(key_phrases) if key_phrases else 0.0
        
        # Metric 3: Hallucination detection
        # Look for speculation keywords
        hallucination_indicators = [
            "probably", "likely", "might be", "could be",
            "I think", "perhaps", "possibly"
        ]
        
        has_hallucinations = any(ind in response.lower() for ind in hallucination_indicators)
        hallucination_examples = [ind for ind in hallucination_indicators if ind in response.lower()]
        
        # Combined score
        score = (citation_overlap * 0.5 + factual_consistency * 0.5) * (0.5 if has_hallucinations else 1.0)
        
        span.set_attribute("groundedness.score", score)
        
        return GroundednessScore(
            score=score,
            citation_overlap=citation_overlap,
            factual_consistency=factual_consistency,
            has_hallucinations=has_hallucinations,
            hallucination_examples=hallucination_examples
        )
    
    def compute_retrieval_metrics(
        self,
        retrieved_chunk_ids: List[str],
        ground_truth_chunk_ids: Optional[List[str]] = None
    ) -> RetrievalMetrics:
        """
        Compute retrieval quality metrics.
        
        Requires ground truth labels for accurate measurement.
        """
        metrics = RetrievalMetrics()
        
        if not ground_truth_chunk_ids:
            return metrics
        
        # Recall@K: What fraction of relevant docs were retrieved?
        relevant_in_top_5 = sum(1 for cid in retrieved_chunk_ids[:5] if cid in ground_truth_chunk_ids)
        relevant_in_top_10 = sum(1 for cid in retrieved_chunk_ids[:10] if cid in ground_truth_chunk_ids)
        
        metrics.recall_at_5 = relevant_in_top_5 / len(ground_truth_chunk_ids)
        metrics.recall_at_10 = relevant_in_top_10 / len(ground_truth_chunk_ids)
        
        # Precision@5: What fraction of top 5 are relevant?
        metrics.precision_at_5 = relevant_in_top_5 / 5
        
        # MRR: Mean Reciprocal Rank
        for i, cid in enumerate(retrieved_chunk_ids, 1):
            if cid in ground_truth_chunk_ids:
                metrics.mrr = 1.0 / i
                break
        
        # Hit rate: Did we find any relevant doc?
        metrics.hit_rate = 1.0 if any(cid in ground_truth_chunk_ids for cid in retrieved_chunk_ids) else 0.0
        
        return metrics
    
    def record_metrics(self, trace_id: str, metrics: ObservabilityMetrics):
        """Store metrics for later analysis"""
        self.metrics_cache[trace_id] = metrics
    
    def get_aggregate_metrics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Compute aggregate metrics over time period"""
        relevant_metrics = [
            m for m in self.metrics_cache.values()
            if (not start_date or m.measured_at >= start_date) and
               (not end_date or m.measured_at <= end_date)
        ]
        
        if not relevant_metrics:
            return {}
        
        return {
            "avg_groundedness": sum(m.groundedness.score for m in relevant_metrics) / len(relevant_metrics),
            "avg_latency_ms": sum(m.latency.total_ms for m in relevant_metrics) / len(relevant_metrics),
            "hallucination_rate": sum(1 for m in relevant_metrics if m.groundedness.has_hallucinations) / len(relevant_metrics),
            "user_satisfaction": sum(m.user_satisfaction for m in relevant_metrics if m.user_satisfaction) / 
                                sum(1 for m in relevant_metrics if m.user_satisfaction)
        }


# =========================================================
# Evaluation Loop
# =========================================================

class GoldStandardQA(BaseModel):
    """A gold standard Q&A pair for evaluation"""
    question: str
    expected_answer: str
    expected_sources: List[str]
    category: str  # factual_lookup, compliance_check, etc.


class EvaluationLoop:
    """
    Continuous evaluation using:
    1. User feedback (thumbs up/down)
    2. Gold standard Q&A pairs
    3. Drift detection
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        gold_standard_file: Optional[Path] = None
    ):
        self.metrics_collector = metrics_collector
        self.gold_standard: List[GoldStandardQA] = []
        
        if gold_standard_file and gold_standard_file.exists():
            self._load_gold_standard(gold_standard_file)
    
    def _load_gold_standard(self, file_path: Path):
        """Load gold standard Q&A pairs"""
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.gold_standard = [GoldStandardQA(**qa) for qa in data]
    
    def record_feedback(
        self,
        trace_id: str,
        feedback: str,
        comment: Optional[str] = None
    ):
        """Record user feedback"""
        if trace_id in self.metrics_collector.metrics_cache:
            metrics = self.metrics_collector.metrics_cache[trace_id]
            
            # Convert feedback to satisfaction score
            satisfaction_map = {
                "thumbs_up": 1.0,
                "thumbs_down": 0.0,
                "neutral": 0.5
            }
            metrics.user_satisfaction = satisfaction_map.get(feedback, 0.5)
    
    def run_evaluation(self) -> Dict[str, float]:
        """
        Run evaluation on gold standard set.
        
        Returns:
            Dict with accuracy, groundedness, etc.
        """
        results = {
            "total_questions": len(self.gold_standard),
            "correct_answers": 0,
            "avg_groundedness": 0.0,
            "citation_accuracy": 0.0
        }
        
        # Would run actual evaluation here
        # For production, integrate with ReasoningEngine
        
        return results
    
    def detect_drift(self, window_days: int = 7) -> Dict[str, Any]:
        """
        Detect model drift by comparing recent metrics to baseline.
        
        Returns:
            Drift report with alerts
        """
        cutoff = datetime.utcnow() - timedelta(days=window_days)
        
        recent_metrics = self.metrics_collector.get_aggregate_metrics(
            start_date=cutoff
        )
        
        # Compare to baseline (would be stored separately)
        baseline = {
            "avg_groundedness": 0.85,
            "hallucination_rate": 0.05,
            "user_satisfaction": 0.80
        }
        
        drift_report = {
            "window_days": window_days,
            "metrics": recent_metrics,
            "baseline": baseline,
            "alerts": []
        }
        
        # Check for significant drift
        if recent_metrics.get("avg_groundedness", 0) < baseline["avg_groundedness"] - 0.1:
            drift_report["alerts"].append("Groundedness dropped below threshold")
        
        if recent_metrics.get("hallucination_rate", 0) > baseline["hallucination_rate"] + 0.05:
            drift_report["alerts"].append("Hallucination rate increased")
        
        return drift_report


# =========================================================
# Main Observability Service
# =========================================================

class ObservabilityService:
    """
    Main service coordinating all observability functions.
    """
    
    def __init__(self, log_dir: Path):
        self.audit_logger = AuditLogger(log_dir)
        self.metrics_collector = MetricsCollector()
        self.evaluation_loop = EvaluationLoop(self.metrics_collector)
    
    def track_query(self, lineage: QueryLineage):
        """Track complete query lineage"""
        self.audit_logger.log_query_lineage(lineage)
    
    def record_metrics(self, metrics: ObservabilityMetrics):
        """Record observability metrics"""
        self.metrics_collector.record_metrics(metrics.trace_id, metrics)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for observability dashboard"""
        return {
            "aggregate_metrics": self.metrics_collector.get_aggregate_metrics(),
            "drift_report": self.evaluation_loop.detect_drift(),
            "recent_queries": 0  # Would query audit logs
        }


# =========================================================
# CLI / Testing
# =========================================================

def main():
    """Test observability service"""
    
    log_dir = Path("/tmp/audit_logs")
    service = ObservabilityService(log_dir)
    
    # Test audit event
    event = AuditEvent(
        event_type=AuditEventType.QUERY_RECEIVED,
        session_id="test_session",
        data={"query": "What are the requirements?"}
    )
    
    service.audit_logger.log_event(event)
    print("Logged audit event")
    
    # Test metrics
    groundedness = service.metrics_collector.compute_groundedness(
        response="The organization shall establish requirements as per ISO 9001:4.4.1",
        context_chunks=[{"content_text": "The organization shall establish requirements"}],
        citations=["ISO 9001:4.4.1"]
    )
    
    print(f"\nGroundedness score: {groundedness.score:.2f}")
    print(f"Citation overlap: {groundedness.citation_overlap:.2f}")
    print(f"Has hallucinations: {groundedness.has_hallucinations}")


if __name__ == "__main__":
    main()
