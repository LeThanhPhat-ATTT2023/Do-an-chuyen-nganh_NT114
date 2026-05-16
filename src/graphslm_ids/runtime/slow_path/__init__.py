"""Slow-path evidence construction and XAI report generation."""

from graphslm_ids.runtime.slow_path.context_hydrator import ContextHydrator
from graphslm_ids.runtime.slow_path.evidence_builder import EvidenceBuilder, EvidenceBuilderConfig
from graphslm_ids.runtime.slow_path.evidence_bundle import EvidenceBundle
from graphslm_ids.runtime.slow_path.evidence_ranker import EvidenceRanker, RankerConfig
from graphslm_ids.runtime.slow_path.hot_buffer_adapter import HotBufferAdapter
from graphslm_ids.runtime.slow_path.report_generator import (
    ReportGenerationError,
    ReportGenerator,
    ReportGeneratorConfig,
)
from graphslm_ids.runtime.slow_path.report_validator import ReportValidator, ValidatorConfig
from graphslm_ids.runtime.slow_path.slm_client import OllamaClient
from graphslm_ids.runtime.slow_path.slow_path_worker import SlowPathConfig, SlowPathResult, SlowPathWorker
from graphslm_ids.runtime.slow_path.types import FlowContext, GraphContext, MitreMetadata, PacketContext, SlowPathJob

__all__ = [
    "ContextHydrator",
    "EvidenceBuilder",
    "EvidenceBuilderConfig",
    "EvidenceBundle",
    "EvidenceRanker",
    "RankerConfig",
    "HotBufferAdapter",
    "ReportGenerator",
    "ReportGenerationError",
    "ReportGeneratorConfig",
    "ReportValidator",
    "ValidatorConfig",
    "OllamaClient",
    "SlowPathConfig",
    "SlowPathResult",
    "SlowPathWorker",
    "SlowPathJob",
    "FlowContext",
    "GraphContext",
    "MitreMetadata",
    "PacketContext",
]
