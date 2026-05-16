from __future__ import annotations

import queue as queue_module
from dataclasses import dataclass
from typing import Any, Mapping

from graphslm_ids.runtime.slow_path.context_hydrator import ContextHydrator
from graphslm_ids.runtime.slow_path.evidence_builder import EvidenceBuilder, EvidenceBuilderConfig
from graphslm_ids.runtime.slow_path.evidence_ranker import EvidenceRanker, RankerConfig
from graphslm_ids.runtime.slow_path.fallback_template import render_minimal, render_template
from graphslm_ids.runtime.slow_path.report_generator import ReportGenerationError, ReportGenerator
from graphslm_ids.runtime.slow_path.report_validator import ReportValidator, ValidatorConfig
from graphslm_ids.runtime.slow_path.types import SlowPathJob


@dataclass
class SlowPathConfig:
    queue_timeout: float = 1.0
    max_cf_packets: int = 10
    top_packets: int = 5
    top_techniques: int = 5
    top_paths: int = 5
    alert_threshold: float = 0.70
    max_payload_preview_bytes: int = 64


@dataclass
class SlowPathResult:
    report: str
    bundle: Any
    validation: Any
    fallback_tier: int


class SlowPathWorker:
    def __init__(
        self,
        config: SlowPathConfig | None = None,
        context_hydrator: ContextHydrator | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        evidence_ranker: EvidenceRanker | None = None,
        report_generator: ReportGenerator | None = None,
        validator: ReportValidator | None = None,
        cold_store: Any | None = None,
        counterfactual_model: Any | None = None,
        label_to_index: Mapping[str, int] | None = None,
    ) -> None:
        self.config = config or SlowPathConfig()
        self.context_hydrator = context_hydrator or ContextHydrator()
        self.evidence_builder = evidence_builder or EvidenceBuilder(
            EvidenceBuilderConfig(
                alert_threshold=self.config.alert_threshold,
                max_payload_preview_bytes=self.config.max_payload_preview_bytes,
            ),
            counterfactual_model=counterfactual_model,
            label_to_index=label_to_index,
        )
        self.evidence_ranker = evidence_ranker or EvidenceRanker(RankerConfig())
        self.report_generator = report_generator or ReportGenerator()
        self.validator = validator or ReportValidator(ValidatorConfig())
        self.cold_store = cold_store

    def process_job(
        self,
        job: SlowPathJob,
        hot_buffer: Any | None = None,
        cold_store: Any | None = None,
    ) -> SlowPathResult:
        store = cold_store or self.cold_store
        try:
            context = self.context_hydrator.hydrate(job.flow_id, hot_buffer, store)
            bundle = self.evidence_builder.build(
                job=job,
                context=context,
                max_cf_packets=self.config.max_cf_packets,
            )
            bundle = self.evidence_ranker.rank_and_truncate(
                bundle,
                top_packets=self.config.top_packets,
                top_techniques=self.config.top_techniques,
                top_paths=self.config.top_paths,
            )

            tier1_result = None
            try:
                tier1_report = self.report_generator.generate(bundle, tier=1)
                tier1_result = self.validator.validate(tier1_report, bundle)
            except (ReportGenerationError, TimeoutError):
                tier1_report = None

            if tier1_result is not None and tier1_result.overall_pass:
                final_report = tier1_report
                final_validation = tier1_result
                final_tier = 1
            else:
                mini_bundle = self.evidence_ranker.truncate_to_mini(bundle)
                tier2_result = None
                try:
                    tier2_report = self.report_generator.generate(mini_bundle, tier=2)
                    tier2_result = self.validator.validate(tier2_report, bundle)
                except (ReportGenerationError, TimeoutError):
                    tier2_report = None

                if tier2_result is not None and tier2_result.overall_pass:
                    final_report = tier2_report
                    final_validation = tier2_result
                    final_tier = 2
                else:
                    final_report = render_template(bundle, template_tag="TEMPLATE FALLBACK")
                    final_validation = self.validator.validate(final_report, bundle)
                    final_tier = 3

            self._persist(job, bundle, final_report, final_validation, final_tier, store)
            return SlowPathResult(final_report, bundle, final_validation, final_tier)

        except TimeoutError:
            report = render_minimal(job)
            self._persist(job, None, report, None, 3, store)
            return SlowPathResult(report, None, None, 3)

    def run_queue(
        self,
        slow_path_queue: Any,
        hot_buffer: Any | None = None,
        cold_store: Any | None = None,
        stop_event: Any | None = None,
        max_jobs: int | None = None,
    ) -> list[SlowPathResult]:
        results: list[SlowPathResult] = []
        processed = 0

        while max_jobs is None or processed < max_jobs:
            if stop_event is not None and stop_event.is_set():
                break

            try:
                job = slow_path_queue.get(timeout=self.config.queue_timeout)
            except queue_module.Empty:
                if max_jobs is not None:
                    break
                continue

            try:
                if job is None:
                    break
                result = self.process_job(job, hot_buffer=hot_buffer, cold_store=cold_store)
                results.append(result)
                processed += 1
            finally:
                if hasattr(slow_path_queue, "task_done"):
                    slow_path_queue.task_done()

        return results

    def _persist(
        self,
        job: SlowPathJob,
        bundle: Any,
        report: str,
        validation: Any,
        fallback_tier: int,
        store: Any | None,
    ) -> None:
        if store is None:
            return
        if hasattr(store, "save_report"):
            store.save_report(
                alert_id=job.alert_id,
                bundle=bundle,
                report=report,
                validation=validation,
                fallback_tier=fallback_tier,
            )
