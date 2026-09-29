import sqlite3
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

from backend.app.config import settings, DATA_DIR
from backend.app.logging_config import logger
from backend.app.monitoring.telemetry import TelemetryRecord


DB_PATH = DATA_DIR / "telemetry.db"


class TelemetryStorage:
    """SQLite-backed persistent storage for RAG telemetry records."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a configured SQLite connection."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        """Initializes SQLite database and tables."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._get_connection() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS request_telemetry (
                        request_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        query TEXT NOT NULL,
                        status TEXT NOT NULL,
                        error_stage TEXT,
                        error_message TEXT,
                        cache_hit INTEGER NOT NULL,
                        cache_similarity REAL NOT NULL,
                        cache_lookup_latency_ms REAL,
                        complexity_score INTEGER,
                        complexity_level TEXT,
                        selected_model TEXT,
                        routing_reason TEXT,
                        routing_latency_ms REAL,
                        retrieval_latency_ms REAL,
                        number_of_sources INTEGER NOT NULL,
                        retrieval_top_k INTEGER NOT NULL,
                        model TEXT,
                        generation_latency_ms REAL,
                        input_tokens INTEGER,
                        output_tokens INTEGER,
                        total_tokens INTEGER,
                        load_duration_ms REAL,
                        prompt_eval_duration_ms REAL,
                        eval_duration_ms REAL,
                        total_request_latency_ms REAL NOT NULL,
                        actual_api_cost REAL NOT NULL,
                        estimated_compute_cost REAL,
                        cost_calculation_method TEXT NOT NULL
                    );
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON request_telemetry(timestamp DESC);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_status ON request_telemetry(status);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_model ON request_telemetry(model);")
                conn.commit()
            logger.info(f"Telemetry database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize telemetry database: {e}", exc_info=True)

    def save_record(self, record: TelemetryRecord) -> None:
        """Saves a single telemetry record to SQLite."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO request_telemetry (
                            request_id, timestamp, query, status, error_stage, error_message,
                            cache_hit, cache_similarity, cache_lookup_latency_ms,
                            complexity_score, complexity_level, selected_model, routing_reason, routing_latency_ms,
                            retrieval_latency_ms, number_of_sources, retrieval_top_k,
                            model, generation_latency_ms, input_tokens, output_tokens, total_tokens,
                            load_duration_ms, prompt_eval_duration_ms, eval_duration_ms,
                            total_request_latency_ms, actual_api_cost, estimated_compute_cost, cost_calculation_method
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.request_id,
                            record.timestamp,
                            record.query,
                            record.status,
                            record.error_stage,
                            record.error_message,
                            1 if record.cache_hit else 0,
                            record.cache_similarity,
                            record.cache_lookup_latency_ms,
                            record.complexity_score,
                            record.complexity_level,
                            record.selected_model,
                            record.routing_reason,
                            record.routing_latency_ms,
                            record.retrieval_latency_ms,
                            record.number_of_sources,
                            record.retrieval_top_k,
                            record.model,
                            record.generation_latency_ms,
                            record.input_tokens,
                            record.output_tokens,
                            record.total_tokens,
                            record.load_duration_ms,
                            record.prompt_eval_duration_ms,
                            record.eval_duration_ms,
                            record.total_request_latency_ms,
                            record.actual_api_cost,
                            record.estimated_compute_cost,
                            record.cost_calculation_method,
                        )
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"Error saving telemetry record {record.request_id}: {e}", exc_info=True)

    def get_recent(self, limit: int = 20) -> List[TelemetryRecord]:
        """Retrieves the most recent telemetry records (limit between 1 and 100)."""
        clamped_limit = max(1, min(100, int(limit)))
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute(
                        "SELECT * FROM request_telemetry ORDER BY timestamp DESC LIMIT ?",
                        (clamped_limit,)
                    )
                    rows = cursor.fetchall()
                    records = []
                    for row in rows:
                        data = dict(row)
                        data["cache_hit"] = bool(data["cache_hit"])
                        records.append(TelemetryRecord(**data))
                    return records
            except Exception as e:
                logger.error(f"Error reading recent telemetry records: {e}", exc_info=True)
                return []

    def get_all_records(self) -> List[TelemetryRecord]:
        """Retrieves all telemetry records."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute("SELECT * FROM request_telemetry ORDER BY timestamp ASC")
                    rows = cursor.fetchall()
                    records = []
                    for row in rows:
                        data = dict(row)
                        data["cache_hit"] = bool(data["cache_hit"])
                        records.append(TelemetryRecord(**data))
                    return records
            except Exception as e:
                logger.error(f"Error reading all telemetry records: {e}", exc_info=True)
                return []

    def count(self) -> int:
        """Returns total records in telemetry storage."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM request_telemetry")
                    return cursor.fetchone()[0]
            except Exception as e:
                logger.error(f"Error counting telemetry records: {e}")
                return 0

    def clear(self) -> None:
        """Clears all records from telemetry storage."""
        with self._lock:
            try:
                with self._get_connection() as conn:
                    conn.execute("DELETE FROM request_telemetry")
                    conn.commit()
                logger.info("Telemetry database cleared.")
            except Exception as e:
                logger.error(f"Error clearing telemetry database: {e}", exc_info=True)


telemetry_storage = TelemetryStorage()
