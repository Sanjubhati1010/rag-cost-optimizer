import re
from typing import List, Tuple, Dict, Any, Optional

from backend.app.config import settings
from backend.app.logging_config import logger
from backend.app.schemas import RoutingDecision
from backend.app.llm import ollama_client


class QueryRouter:
    """Deterministic cost-aware query complexity analyzer and model router."""

    # Keywords signaling comparison and tradeoff evaluation (+3)
    COMPARISON_PATTERNS = [
        r"\bcompare\b",
        r"\bcontrast\b",
        r"\bdifference(s)?\s+between\b",
        r"\bversus\b",
        r"\b\s+vs\.?\s+\b",
        r"\btrade-?offs?\b",
        r"\bpros\s+and\s+cons\b",
        r"\badadvantages\s+and\s+disadvantages\b",
        r"\bbetter\s+than\b",
        r"\bdistinguish\s+between\b",
    ]

    # Keywords signaling deep explanation and architectural breakdowns (+2)
    DEEP_EXPLANATION_PATTERNS = [
        r"\bexplain\s+how\b",
        r"\bexplain\s+why\b",
        r"\bin\s+detail\b",
        r"\bdetailed\s+explanation\b",
        r"\bdeep\s+dive\b",
        r"\bstep-?by-?step\b",
        r"\bcomprehensively?\b",
        r"\bbreak\s*down\b",
        r"\bdiscuss\s+when\b",
        r"\bdesign\s+a\b",
        r"\barchitect\s+a\b",
        r"\banalyze\s+how\b",
        r"\bhow\s+does\s+.+\s+affect\b",
    ]

    # Technical concepts and multi-entity indicators (+1 per match)
    TECHNICAL_CONCEPTS = [
        r"\b1nf\b",
        r"\b2nf\b",
        r"\b3nf\b",
        r"\bbcnf\b",
        r"\b4nf\b",
        r"\bfunctional\s+dependenc(y|ies)\b",
        r"\bdistributed\s+(database|system)s?\b",
        r"\breplication\b",
        r"\bsharding\b",
        r"\bacid\b",
        r"\bcap\s+theorem\b",
        r"\bconsistency\s+and\s+availability\b",
        r"\bconcurrency\s+control\b",
        r"\bvector\s+embeddings?\b",
        r"\bhnsw\b",
        r"\bcosine\s+similarity\b",
    ]

    # Simple question patterns for fast-tracking simple definitions
    SIMPLE_DEFINITION_PATTERNS = [
        r"^(what\s+is|what\s+are|define|meaning\s+of|who\s+is|which\s+port|what\s+port)\s+[\w\s]{1,35}\??$",
    ]

    def analyze_query_complexity(self, query: str) -> Tuple[int, str, str, List[str]]:
        """Calculates a deterministic complexity score for a user query.

        Scoring Rules:
        - Query word count: 0 (<=7 words), +1 (8-13), +2 (14-23), +3 (24+ words)
        - Comparison requests: +3
        - Deep explanation / architectural requests: +2
        - Compound clauses / multiple conjunctions: +2
        - Technical concepts / domain entities: +1 per match (max +4)
        - Simple short definitions: fast-tracked score <= 2

        Returns:
            Tuple of (score, level, reason, factors)
        """
        trimmed = query.strip()
        lower_query = trimmed.lower()
        words = trimmed.split()
        word_count = len(words)
        factors: List[str] = []
        score = 0

        # Fast-track check: short simple definition (<= 5 words)
        if word_count <= 5:
            for pattern in self.SIMPLE_DEFINITION_PATTERNS:
                if re.match(pattern, lower_query):
                    return 1, "simple", "Low complexity definition query", ["Short single-concept definition query"]

        # 1. Length and word count scoring
        if word_count >= 24:
            score += 3
            factors.append(f"Long query ({word_count} words)")
        elif word_count >= 14:
            score += 2
            factors.append(f"Medium-length query ({word_count} words)")
        elif word_count >= 8:
            score += 1
            factors.append(f"Standard query length ({word_count} words)")

        # 2. Comparison & contrast indicators
        comparison_matches = [p for p in self.COMPARISON_PATTERNS if re.search(p, lower_query)]
        if comparison_matches:
            score += 3
            factors.append("Comparison/tradeoff analysis requested")

        # 3. Deep explanation & architectural design indicators
        explanation_matches = [p for p in self.DEEP_EXPLANATION_PATTERNS if re.search(p, lower_query)]
        if explanation_matches:
            score += 2
            factors.append("In-depth explanation / architectural breakdown requested")

        # 4. Multi-clause / Compound structure indicators
        conjunctions = len(re.findall(r"\b(and|as\s+well\s+as|furthermore|along\s+with|whereas|while)\b", lower_query))
        commas_semicolons = len(re.findall(r"[,;]", trimmed))
        if conjunctions >= 2 or (conjunctions >= 1 and commas_semicolons >= 2):
            score += 2
            factors.append("Multi-clause compound query")

        # 5. Technical concepts / multi-entity matches
        tech_matches = [p for p in self.TECHNICAL_CONCEPTS if re.search(p, lower_query)]
        if tech_matches:
            tech_score = min(len(tech_matches), 4)
            score += tech_score
            factors.append(f"Technical domain concepts detected ({len(tech_matches)} match(es))")

        # Determine level based on thresholds
        if score <= settings.SIMPLE_THRESHOLD:
            level = "simple"
            reason = "Low complexity query"
        elif score < settings.COMPLEX_THRESHOLD:
            level = "medium"
            reason = "Moderate complexity query"
        else:
            level = "complex"
            reason = "Multi-concept technical query / high complexity"

        return score, level, reason, factors

    def route_query(self, query: str) -> RoutingDecision:
        """Determines the optimal model route based on query complexity.

        Returns:
            RoutingDecision model containing score, level, selected model, and reason.
        """
        if not settings.COST_ROUTING_ENABLED:
            logger.info("[ROUTER] Cost-aware routing disabled via configuration.")
            return RoutingDecision(
                complexity_score=0,
                complexity_level="disabled",
                selected_model=settings.SMALL_MODEL,
                routing_reason="Routing disabled; using default model",
                factors=["Routing disabled via settings"]
            )

        score, level, reason, factors = self.analyze_query_complexity(query)

        logger.info(f"[ROUTER] query complexity score={score} level={level}")

        if level == "complex":
            # Check if configured large model is available locally
            large_model = settings.LARGE_MODEL
            if ollama_client.is_model_available(large_model):
                selected_model = large_model
                routing_reason = reason
            else:
                logger.warning(
                    f"[ROUTER] Configured large model '{large_model}' not available locally; "
                    f"falling back to small model '{settings.SMALL_MODEL}'."
                )
                selected_model = settings.SMALL_MODEL
                routing_reason = f"{reason} (routed to {settings.SMALL_MODEL} via fallback)"
        else:
            # Simple and Medium queries route to small model
            selected_model = settings.SMALL_MODEL
            routing_reason = reason

        logger.info(f"[ROUTER] selected model={selected_model}")
        logger.info(f"[ROUTER] reason={routing_reason}")

        return RoutingDecision(
            complexity_score=score,
            complexity_level=level,
            selected_model=selected_model,
            routing_reason=routing_reason,
            factors=factors
        )


query_router = QueryRouter()
