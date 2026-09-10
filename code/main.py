import argparse
import copy
import csv
import hashlib
import json
import math
import os
import importlib.metadata
import random
import re
import resource
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from data_process import get_anchers, validate_anchor_audit


STRICT_EVIDENCE_INSTRUCTION = (
    "Strictly base your answer ONLY on the provided social media comments and "
    "retrieved memory. Do not use external knowledge or unstated historical facts."
)

DEFAULT_EVALUATOR_PROMPT = (
    "You are an event analysis assistant. Your task is to infer the event name "
    "discussed in the provided social media comments and extract related keywords.\n"
    f"{STRICT_EVIDENCE_INSTRUCTION}\n"
    "First, read all COMMENTS and identify only their shared core content.\n"
    "Second, summarize one concise EVENT_NAME supported by the COMMENTS.\n"
    "Third, extract at most 10 KEYWORDS. Every keyword must be a single word that "
    "appears verbatim in the COMMENTS.\n"
    "Return only a JSON object with EVENT_NAME (string) and KEYWORDS (list)."
)

DEFAULT_DETECTOR_PROMPT = (
    "You are a social-media event classifier. The supplied knowledge contains "
    "candidate EVENT identifiers and their KEYWORDS.\n"
    f"{STRICT_EVIDENCE_INSTRUCTION}\n"
    "Choose exactly one supplied EVENT for the INPUT. If every candidate is "
    "irrelevant, or the supplied knowledge is empty, return 'Others'.\n"
    "Return only a JSON object with INPUT and EVENT. Do not use chat history.\n"
    "Here is the optional platform knowledge variable:\n{knowledge}"
)

CQC_DETECTOR_PROMPT = (
    "You are an evidence-grounded social-media event classifier. The supplied "
    "knowledge contains candidate EVENT identifiers, KEYWORDS, and a source "
    "EVIDENCE span selected by cross-query consistency.\n"
    f"{STRICT_EVIDENCE_INSTRUCTION}\n"
    "Choose a supplied EVENT only when its EVIDENCE supports the same specific "
    "occurrence as the INPUT. A shared person, organization, location, or broad "
    "topic alone is insufficient. If the supplied knowledge is empty or the "
    "evidence is not specific enough, return 'Others'.\n"
    "Return only a JSON object with INPUT and EVENT. Do not use chat history.\n"
    "Here is the optional platform knowledge variable:\n{knowledge}"
)

RAGSEDE_ORIGINAL_EVALUATOR_PROMPT = (
    "You are an event analysis assistant. Your task is to infer the event names "
    "discussed in the provided social media comments and extract keywords "
    "related to the event.\n"
    "First, carefully read all the COMMENTS and understand the core content "
    "they discuss.\n"
    "Second, summarize a concise and accurate EVENT NAME based on the COMMENTS.\n"
    "Third, extract no more than 10 KEYWORDS related to the event from the "
    "comments. Each KEYWORD must be a single word that appears in the COMMENTS.\n"
    "Answer in JSON format with EVENT_NAME and KEYWORDS. Do not include any "
    "other information."
)

RAGSEDE_ORIGINAL_DETECTOR_PROMPT = (
    "The knowledge base contains EVENTs and corresponding KEYWORDs.\n"
    "You are a social media comment classifier determining which one EVENT the "
    "INPUT belongs to in the knowledge base.\n"
    "Answer in JSON format with INPUT and EVENT. Do not include any other "
    "information.\n"
    "When all knowledge base content is irrelevant to the INPUT, or when the "
    "knowledge base is empty, EVENT must be 'Others'.\n"
    "Do not consider chat history.\n"
    "Here is the knowledge base:\n{knowledge}\n"
    "The above is the knowledge base."
)

ENTITY_PATTERN = re.compile(
    r"(?<!\w)@[A-Za-z0-9_]+|https?://\S+|www\.\S+|"
    r"(?<!\w)#[\w-]+|\b(?:[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'-]{2,}|[A-Z]{2,})\b"
)


def set_reproducible(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


def load_data_blocks(dataset, data_path=None):
    print(f"Loading data from '{dataset}'... ", end="")
    path_to_data = data_path or os.path.join(
        "datasets", "cache", f"{dataset}.json"
    )
    with open(path_to_data, "r", encoding="utf-8") as file:
        data_blocks = json.load(file)
    print("Done")
    return data_blocks


def apply_stress_test(blocks, args):
    if args.stress_test == "none":
        return blocks
    transformed = copy.deepcopy(blocks)
    rng = random.Random(args.seed)
    for block in transformed:
        for split_value in block.values():
            if not isinstance(split_value, list):
                continue
            for message in split_value:
                if not isinstance(message, dict) or "text" not in message:
                    continue
                text = message["text"]
                if args.stress_test == "entity_collision":
                    text = ENTITY_PATTERN.sub("[SHARED_ENTITY]", text)
                elif args.stress_test == "time_location_shift":
                    text = re.sub(r"\b(?:19|20)\d{2}\b", "[SHIFTED_YEAR]", text)
                    text = re.sub(
                        r"\b[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ'-]{2,}\b",
                        "[SHIFTED_PLACE]",
                        text,
                    )
                elif args.stress_test == "lexical_noise":
                    words = text.split()
                    for index in range(len(words)):
                        if (
                            len(words[index]) >= 5
                            and rng.random() < args.stress_noise_rate
                        ):
                            word = words[index]
                            position = rng.randrange(1, len(word) - 1)
                            words[index] = (
                                word[:position]
                                + word[position + 1 :]
                            )
                    text = " ".join(words)
                message["text"] = text
    print(
        f"Applied stress test '{args.stress_test}' "
        f"with seed {args.seed}."
    )
    return transformed


def estimate_tokens(text):
    # RAGFlow streaming responses do not always expose provider token usage.
    # This transparent approximation is replaced by provider usage when available.
    return max(1, math.ceil(len(text or "") / 4))


def _usage_value(usage, keys):
    if usage is None:
        return None
    for key in keys:
        if isinstance(usage, dict) and key in usage:
            return usage[key]
        if hasattr(usage, key):
            return getattr(usage, key)
    return None


def get_peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


class RunStats:
    def __init__(self, args, block_id):
        self.args = args
        self.block_id = block_id
        self.started_at = time.perf_counter()
        self.llm_calls = 0
        self.llm_successful_calls = 0
        self.llm_failed_attempts = 0
        self.llm_calls_by_role = defaultdict(int)
        self.llm_latencies_by_role = defaultdict(list)
        self.llm_error_types = defaultdict(int)
        self.filtered_keyword_hallucinations = 0
        self.content_policy_blocks = 0
        self.others_predictions = 0
        self.confusion_predictions = 0
        self.mvra_confident_reassignments = 0
        self.mvra_consensus_opportunities = 0
        self.mvra_consensus_reassignments = 0
        self.mvra_consensus_agreements = 0
        self.mvra_single_view_reassignments = 0
        self.mvra_consensus_overrides = 0
        self.mvra_consensus_candidates = 0
        self.mvra_fallback_attempts = 0
        self.mvra_fallback_accepts = 0
        self.mvra_fallback_rejections = 0
        self.conrag_queries = 0
        self.conrag_view_uses = defaultdict(int)
        self.conrag_multi_supported_candidates = 0
        self.conrag_hub_penalized_matches = 0
        self.conrag_evidence_units_built = 0
        self.conrag_entity_objects_built = 0
        self.conrag_relation_objects_built = 0
        self.conrag_object_hits_by_view = defaultdict(int)
        self.conrag_source_projections = 0
        self.conrag_evidence_candidates = 0
        self.conrag_structural_only_rejections = 0
        self.conrag_admissible_evidence_candidates = 0
        # CQC-inspired event-assignment verification. These counters are
        # separate from ConRAG's object-view consensus because CQC operates
        # across meaning-preserving query formulations over one fixed event
        # candidate pool.
        self.cqc_anchors_checked = 0
        self.cqc_query_views_generated = 0
        self.cqc_query_views_used = 0
        self.cqc_semantic_drift_rejections = 0
        self.cqc_shared_pool_candidates = 0
        self.cqc_candidates_evaluated = 0
        self.cqc_candidates_accepted = 0
        self.cqc_candidates_rejected = 0
        self.cqc_majority_agreements = 0
        self.cqc_abstentions = 0
        self.cqc_variance_penalty_sum = 0.0
        # Uncertainty-Gated Residual MVRA (UGR-MVRA). The text-only decision
        # remains the primary path; object-level multi-view retrieval is
        # invoked only after an Others/Confusion result and must pass a
        # fully label-free evidence gate before it can reuse an event.
        self.ugr_primary_predictions = 0
        self.ugr_primary_preserved = 0
        self.ugr_fallback_attempts = 0
        self.ugr_fallback_accepts = 0
        self.ugr_fallback_rejections = 0
        self.ugr_direct_gate_accepts = 0
        self.ugr_prediction_changes = 0
        self.ugr_gate_passes = 0
        self.ugr_gate_rejections = 0
        self.ugr_gate_rejections_by_reason = defaultdict(int)
        self.ugr_pre_llm_short_circuits = 0
        self.ugr_detector_disagreements = 0
        self.ugr_candidate_text_similarity_sum = 0.0
        self.ugr_candidate_fused_score_sum = 0.0
        self.ugr_candidate_margin_sum = 0.0
        # Stage-aware selective MVRA: the monolithic text query creates one
        # fixed candidate pool; decomposed entity/relation/temporal views may
        # only rerank that pool. Auxiliary views are selected by positive,
        # text-aligned marginal utility rather than fused unconditionally.
        self.stage_aware_queries = 0
        self.stage_aware_candidate_pool_total = 0
        self.stage_aware_reranks = 0
        self.stage_aware_text_only = 0
        self.stage_aware_all_view_avoided = 0
        self.stage_aware_views_considered = defaultdict(int)
        self.stage_aware_views_selected = defaultdict(int)
        self.stage_aware_view_rejections = defaultdict(int)
        self.stage_aware_selected_view_count_sum = 0
        self.stage_aware_temporal_candidates = 0
        self.stage_aware_temporal_supports = 0
        self.message_latencies = []
        self.llm_seconds = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.provider_usage_calls = 0
        self.estimated_usage_calls = 0
        self.embedding_seconds = 0.0
        self.encoded_texts = 0
        self.retrieved_candidates = 0
        self.retrieval_seconds = 0.0
        self.input_messages = 0
        self.covered_messages = 0
        self.stage1_anchors = 0
        self.raw_anchors = 0
        self.anchors = 0
        self.dkms_initial_largest_cluster = 0
        self.dkms_largest_cluster = 0
        self.dkms_clusters_over_legacy_limit = 0
        self.dkms_recursive_splits = 0
        self.dkms_low_cohesion_splits = 0
        self.dkms_forced_splits = 0
        # Auditable second-stage quantities. They remain zero for a
        # single-stage run and are populated by key_message_sampling() for the
        # label-free dynamic two-stage protocol.
        self.dkms_block_density = 0.0
        self.dkms_unfloored_dynamic_threshold = 0.0
        self.dkms_dynamic_threshold = 0.0
        self.dkms_threshold_floor_applied = False
        self.dkms_stage2_group_count = 0
        self.dkms_stage2_largest_group = 0
        self.dkms_stage2_min_within_group_similarity = 1.0
        self.dkms_seconds = 0.0
        self.hslg_seconds = 0.0
        self.hslg_entropy_merges = 0
        self.hslg_reciprocal_merges = 0
        self.hslg_mutual_knn_merges = 0
        self.hslg_adaptive_threshold = 0.0
        self.hslg_lexical_vetoes = 0
        self.hslg_component_cap_rejections = 0
        self.hslg_consensus_candidate_edges = 0
        self.hslg_consensus_mutual_edges = 0
        self.hslg_consensus_merges = 0
        self.hslg_consensus_rejections = 0
        self.hslg_consensus_rejections_by_reason = defaultdict(int)
        self.novelty_buffered_anchors = 0
        self.novelty_buffer_groups_created = 0
        self.novelty_buffer_multi_anchor_groups = 0
        self.novelty_buffer_anchors_consolidated = 0
        self.novelty_buffer_singletons_flushed = 0
        self.novelty_buffer_candidate_edges = 0
        self.novelty_buffer_mutual_edges = 0
        self.novelty_buffer_rejections_by_reason = defaultdict(int)
        self.novelty_buffer_peak_size = 0
        self.novelty_buffer_evaluator_calls = 0
        self.novelty_buffer_seconds = 0.0
        self.messages = 0
        self.anchor_seconds = 0.0
        self.initialization_seconds = 0.0
        self.shared_model_initialization_seconds = 0.0
        self.audit_records = []
        self.retrieval_audit_records = []
        self.conrag_embedding_cache = {}

    def budget_available(self):
        if (
            self.args.max_llm_calls is not None
            and self.llm_calls >= self.args.max_llm_calls
        ):
            return False
        if (
            self.args.max_total_tokens is not None
            and self.prompt_tokens + self.completion_tokens
            >= self.args.max_total_tokens
        ):
            return False
        return True

    def record_embedding(self, count, elapsed):
        self.encoded_texts += int(count)
        self.embedding_seconds += elapsed

    def record_llm(
        self, role, prompt, response, elapsed, answer=None, json_parsed=False
    ):
        self.llm_calls += 1
        self.llm_calls_by_role[role] += 1
        self.llm_latencies_by_role[role].append(float(elapsed))
        self.llm_seconds += elapsed

        system_prompt = getattr(self.args, f"{role}_prompt", "")
        usage = getattr(answer, "usage", None) if answer is not None else None
        prompt_tokens = _usage_value(usage, ("prompt_tokens", "input_tokens"))
        completion_tokens = _usage_value(
            usage, ("completion_tokens", "output_tokens")
        )
        if prompt_tokens is None or completion_tokens is None:
            prompt_tokens = estimate_tokens(system_prompt + "\n" + prompt)
            completion_tokens = estimate_tokens(response)
            self.estimated_usage_calls += 1
        else:
            self.provider_usage_calls += 1
        self.prompt_tokens += int(prompt_tokens)
        self.completion_tokens += int(completion_tokens)

        if self.args.save_audit_log:
            self.audit_records.append(
                {
                    "role": role,
                    "system_prompt": system_prompt,
                    "prompt": prompt,
                    "response": response,
                    "json_parsed": json_parsed,
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                }
            )

    def record_llm_failure(self, error=None):
        self.llm_failed_attempts += 1
        if error is not None:
            if isinstance(error, json.JSONDecodeError):
                error_type = "json_parse"
            elif isinstance(error, LLMBudgetExceeded):
                error_type = "budget_exhausted"
            elif isinstance(error, LLMContentPolicyError):
                error_type = "content_policy"
            elif isinstance(error, LLMTransientError):
                error_type = "transient_provider_error"
            else:
                message = str(error).lower()
                if "required" in message or "invalid type" in message:
                    error_type = "schema_validation"
                elif "candidate" in message or "supplied" in message:
                    error_type = "invalid_event_selection"
                elif "reserved" in message:
                    error_type = "reserved_event_name"
                else:
                    error_type = type(error).__name__
            self.llm_error_types[error_type] += 1

    def record_llm_success(self):
        self.llm_successful_calls += 1
        if self.args.save_audit_log and self.audit_records:
            self.audit_records[-1]["validated"] = True

    def finish(self):
        processing_elapsed = time.perf_counter() - self.started_at
        elapsed = processing_elapsed + self.shared_model_initialization_seconds
        estimated_cost = (
            self.prompt_tokens * self.args.input_cost_per_million
            + self.completion_tokens * self.args.output_cost_per_million
        ) / 1_000_000
        gpu_peak_mb = 0.0
        if torch.cuda.is_available():
            gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        latency_summary = {}
        for role, values in self.llm_latencies_by_role.items():
            latency_summary[role] = {
                "mean_seconds": float(np.mean(values)),
                "p50_seconds": float(np.percentile(values, 50)),
                "p95_seconds": float(np.percentile(values, 95)),
                "count": len(values),
            }
        message_latency = {}
        if self.message_latencies:
            message_latency = {
                "mean_seconds": float(np.mean(self.message_latencies)),
                "p50_seconds": float(np.percentile(self.message_latencies, 50)),
                "p95_seconds": float(np.percentile(self.message_latencies, 95)),
                "count": len(self.message_latencies),
            }
        return {
            "elapsed_seconds": elapsed,
            "processing_elapsed_seconds": processing_elapsed,
            "messages": self.messages,
            "input_messages": self.input_messages,
            "covered_messages": self.covered_messages,
            "evaluation_coverage": (
                self.covered_messages / self.input_messages
                if self.input_messages
                else 0.0
            ),
            "stage1_anchors": self.stage1_anchors,
            # Kept for compatibility with earlier result files. In the new
            # single-stage protocol this equals stage1_anchors.
            "raw_anchors": self.raw_anchors,
            "dkms_initial_largest_cluster": (
                self.dkms_initial_largest_cluster
            ),
            "anchors": self.anchors,
            "dkms_largest_cluster": self.dkms_largest_cluster,
            "dkms_clusters_over_legacy_limit": (
                self.dkms_clusters_over_legacy_limit
            ),
            "dkms_recursive_splits": self.dkms_recursive_splits,
            "dkms_low_cohesion_splits": self.dkms_low_cohesion_splits,
            "dkms_forced_splits": self.dkms_forced_splits,
            "dkms_block_density": self.dkms_block_density,
            "dkms_unfloored_dynamic_threshold": (
                self.dkms_unfloored_dynamic_threshold
            ),
            "dkms_dynamic_threshold": self.dkms_dynamic_threshold,
            "dkms_threshold_floor_applied": (
                self.dkms_threshold_floor_applied
            ),
            "dkms_stage2_group_count": self.dkms_stage2_group_count,
            "dkms_stage2_largest_group": self.dkms_stage2_largest_group,
            "dkms_stage2_min_within_group_similarity": (
                self.dkms_stage2_min_within_group_similarity
            ),
            "stage2_anchor_reduction_ratio": (
                1.0 - self.anchors / self.stage1_anchors
                if self.stage1_anchors
                and self.args.anchor_pipeline
                in {"dynamic_two_stage", "legacy_two_stage"}
                else 0.0
            ),
            "throughput_messages_per_second": (
                self.messages / elapsed if elapsed > 0 else 0.0
            ),
            "llm_calls": self.llm_calls,
            "llm_successful_calls": self.llm_successful_calls,
            "llm_failed_attempts": self.llm_failed_attempts,
            "llm_success_rate": (
                self.llm_successful_calls / self.llm_calls
                if self.llm_calls
                else 0.0
            ),
            "llm_calls_by_role": dict(self.llm_calls_by_role),
            "llm_latency_by_role": latency_summary,
            "llm_error_types": dict(self.llm_error_types),
            "content_policy_blocks": self.content_policy_blocks,
            "filtered_keyword_hallucinations": (
                self.filtered_keyword_hallucinations
            ),
            "others_predictions": self.others_predictions,
            "confusion_predictions": self.confusion_predictions,
            "mvra_confident_reassignments": (
                self.mvra_confident_reassignments
            ),
            "mvra_consensus_opportunities": (
                self.mvra_consensus_opportunities
            ),
            "mvra_consensus_reassignments": (
                self.mvra_consensus_reassignments
            ),
            "mvra_consensus_agreements": self.mvra_consensus_agreements,
            "mvra_single_view_reassignments": (
                self.mvra_single_view_reassignments
            ),
            "mvra_consensus_overrides": self.mvra_consensus_overrides,
            "mvra_consensus_candidates": self.mvra_consensus_candidates,
            "mvra_fallback_attempts": self.mvra_fallback_attempts,
            "mvra_fallback_accepts": self.mvra_fallback_accepts,
            "mvra_fallback_rejections": self.mvra_fallback_rejections,
            "conrag_queries": self.conrag_queries,
            "conrag_view_uses": dict(self.conrag_view_uses),
            "conrag_multi_supported_candidates": (
                self.conrag_multi_supported_candidates
            ),
            "conrag_hub_penalized_matches": (
                self.conrag_hub_penalized_matches
            ),
            "conrag_evidence_units_built": (
                self.conrag_evidence_units_built
            ),
            "conrag_entity_objects_built": (
                self.conrag_entity_objects_built
            ),
            "conrag_relation_objects_built": (
                self.conrag_relation_objects_built
            ),
            "conrag_object_hits_by_view": dict(
                self.conrag_object_hits_by_view
            ),
            "conrag_source_projections": self.conrag_source_projections,
            "conrag_evidence_candidates": self.conrag_evidence_candidates,
            "conrag_structural_only_rejections": (
                self.conrag_structural_only_rejections
            ),
            "conrag_admissible_evidence_candidates": (
                self.conrag_admissible_evidence_candidates
            ),
            "cqc_anchors_checked": self.cqc_anchors_checked,
            "cqc_query_views_generated": self.cqc_query_views_generated,
            "cqc_query_views_used": self.cqc_query_views_used,
            "cqc_semantic_drift_rejections": (
                self.cqc_semantic_drift_rejections
            ),
            "cqc_shared_pool_candidates": self.cqc_shared_pool_candidates,
            "cqc_candidates_evaluated": self.cqc_candidates_evaluated,
            "cqc_candidates_accepted": self.cqc_candidates_accepted,
            "cqc_candidates_rejected": self.cqc_candidates_rejected,
            "cqc_majority_agreements": self.cqc_majority_agreements,
            "cqc_abstentions": self.cqc_abstentions,
            "cqc_variance_penalty_sum": self.cqc_variance_penalty_sum,
            "ugr_primary_predictions": self.ugr_primary_predictions,
            "ugr_primary_preserved": self.ugr_primary_preserved,
            "ugr_fallback_attempts": self.ugr_fallback_attempts,
            "ugr_fallback_accepts": self.ugr_fallback_accepts,
            "ugr_fallback_rejections": self.ugr_fallback_rejections,
            "ugr_direct_gate_accepts": self.ugr_direct_gate_accepts,
            "ugr_prediction_changes": self.ugr_prediction_changes,
            "ugr_fallback_acceptance_rate": (
                self.ugr_fallback_accepts / self.ugr_fallback_attempts
                if self.ugr_fallback_attempts
                else 0.0
            ),
            "ugr_gate_passes": self.ugr_gate_passes,
            "ugr_gate_rejections": self.ugr_gate_rejections,
            "ugr_gate_rejections_by_reason": dict(
                self.ugr_gate_rejections_by_reason
            ),
            "ugr_pre_llm_short_circuits": (
                self.ugr_pre_llm_short_circuits
            ),
            "ugr_detector_disagreements": self.ugr_detector_disagreements,
            "ugr_mean_candidate_text_similarity": (
                self.ugr_candidate_text_similarity_sum
                / self.ugr_gate_passes
                if self.ugr_gate_passes
                else 0.0
            ),
            "ugr_mean_candidate_fused_score": (
                self.ugr_candidate_fused_score_sum
                / self.ugr_gate_passes
                if self.ugr_gate_passes
                else 0.0
            ),
            "ugr_mean_candidate_margin": (
                self.ugr_candidate_margin_sum / self.ugr_gate_passes
                if self.ugr_gate_passes
                else 0.0
            ),
            "stage_aware_queries": self.stage_aware_queries,
            "stage_aware_candidate_pool_total": (
                self.stage_aware_candidate_pool_total
            ),
            "stage_aware_mean_candidate_pool_size": (
                self.stage_aware_candidate_pool_total
                / self.stage_aware_queries
                if self.stage_aware_queries
                else 0.0
            ),
            "stage_aware_reranks": self.stage_aware_reranks,
            "stage_aware_text_only": self.stage_aware_text_only,
            "stage_aware_all_view_avoided": (
                self.stage_aware_all_view_avoided
            ),
            "stage_aware_views_considered": dict(
                self.stage_aware_views_considered
            ),
            "stage_aware_views_selected": dict(
                self.stage_aware_views_selected
            ),
            "stage_aware_view_rejections": dict(
                self.stage_aware_view_rejections
            ),
            "stage_aware_mean_selected_view_count": (
                self.stage_aware_selected_view_count_sum
                / self.stage_aware_queries
                if self.stage_aware_queries
                else 0.0
            ),
            "stage_aware_temporal_candidates": (
                self.stage_aware_temporal_candidates
            ),
            "stage_aware_temporal_supports": (
                self.stage_aware_temporal_supports
            ),
            "hslg_entropy_merges": self.hslg_entropy_merges,
            "hslg_reciprocal_merges": self.hslg_reciprocal_merges,
            "hslg_mutual_knn_merges": self.hslg_mutual_knn_merges,
            "hslg_adaptive_threshold": self.hslg_adaptive_threshold,
            "hslg_lexical_vetoes": self.hslg_lexical_vetoes,
            "hslg_component_cap_rejections": (
                self.hslg_component_cap_rejections
            ),
            "hslg_consensus_candidate_edges": (
                self.hslg_consensus_candidate_edges
            ),
            "hslg_consensus_mutual_edges": (
                self.hslg_consensus_mutual_edges
            ),
            "hslg_consensus_merges": self.hslg_consensus_merges,
            "hslg_consensus_rejections": (
                self.hslg_consensus_rejections
            ),
            "hslg_consensus_rejections_by_reason": dict(
                self.hslg_consensus_rejections_by_reason
            ),
            "novelty_buffered_anchors": self.novelty_buffered_anchors,
            "novelty_buffer_groups_created": (
                self.novelty_buffer_groups_created
            ),
            "novelty_buffer_multi_anchor_groups": (
                self.novelty_buffer_multi_anchor_groups
            ),
            "novelty_buffer_anchors_consolidated": (
                self.novelty_buffer_anchors_consolidated
            ),
            "novelty_buffer_singletons_flushed": (
                self.novelty_buffer_singletons_flushed
            ),
            "novelty_buffer_candidate_edges": (
                self.novelty_buffer_candidate_edges
            ),
            "novelty_buffer_mutual_edges": (
                self.novelty_buffer_mutual_edges
            ),
            "novelty_buffer_rejections_by_reason": dict(
                self.novelty_buffer_rejections_by_reason
            ),
            "novelty_buffer_peak_size": self.novelty_buffer_peak_size,
            "novelty_buffer_evaluator_calls": (
                self.novelty_buffer_evaluator_calls
            ),
            "novelty_buffer_seconds": self.novelty_buffer_seconds,
            "message_latency": message_latency,
            "llm_seconds": self.llm_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "token_usage_source": {
                "provider_reported_calls": self.provider_usage_calls,
                "estimated_calls": self.estimated_usage_calls,
                "estimation_rule": "ceil(characters/4) when provider usage is absent",
            },
            "estimated_cost": estimated_cost,
            "input_cost_per_million": self.args.input_cost_per_million,
            "output_cost_per_million": self.args.output_cost_per_million,
            "encoded_texts": self.encoded_texts,
            "embedding_seconds": self.embedding_seconds,
            "retrieved_candidates": self.retrieved_candidates,
            "retrieval_seconds": self.retrieval_seconds,
            "anchor_seconds": self.anchor_seconds,
            "dkms_seconds": self.dkms_seconds,
            "hslg_seconds": self.hslg_seconds,
            "initialization_seconds": self.initialization_seconds,
            "shared_model_initialization_seconds": (
                self.shared_model_initialization_seconds
            ),
            "peak_rss_mb": get_peak_rss_mb(),
            "peak_gpu_memory_mb": gpu_peak_mb,
            "anchor_audit": getattr(self, "anchor_audit", {}),
        }


def encode_texts(model, texts, stats, **kwargs):
    count = 1 if isinstance(texts, str) else len(texts)
    started = time.perf_counter()
    result = model.encode(texts, **kwargs)
    stats.record_embedding(count, time.perf_counter() - started)
    return result


class LLMProviderError(RuntimeError):
    """A provider/RAGFlow error that should stop the run without JSON retries."""


class LLMContentPolicyError(RuntimeError):
    """A single prompt/response rejected by the model provider's safety filter."""


class LLMTransientError(RuntimeError):
    """A temporary provider/RAGFlow failure that is safe to retry."""


def parse_json_object(content):
    content = (content or "").strip()
    # DeepSeek-R1 family models may return an explicit reasoning section before
    # their final answer. Provider-like phrases inside this reasoning are normal
    # language and must never be interpreted as API failures.
    content = re.sub(
        r"<think>.*?</think>", "", content, flags=re.I | re.S
    ).strip()
    if "<think>" in content.lower() and "</think>" not in content.lower():
        raise ValueError(
            "DeepSeek reasoning output was truncated before the final JSON. "
            "Increase --max_tokens (for example, to 2048)."
        )

    # Real RAGFlow/provider failures start with an explicit error prefix. Do not
    # scan arbitrary response text for phrases such as "model not found" because
    # reasoning models can naturally produce those words while solving a task.
    if re.match(
        r"^(?:\*\*ERROR\*\*|ERROR\b|ACCESS DENIED\b|"
        r"INSUFFICIENT BALANCE\b|ARREARAGE\b|OVERDUE-PAYMENT\b)",
        content,
        flags=re.I,
    ):
        lowered_content = content.lower()
        if "inappropriate content" in lowered_content:
            raise LLMContentPolicyError(
                f"LLM content policy rejected this call: {content[:500]}"
            )
        transient_markers = (
            "timed out",
            "timeout",
            "rate limit",
            "too many requests",
            "temporarily unavailable",
            "service unavailable",
            "connection reset",
            "connection aborted",
            "bad gateway",
            "gateway timeout",
            "internal server error",
            # Alibaba Model Studio/RAGFlow can expose backend HTTP-500
            # failures through provider-specific error names rather than the
            # literal phrase "internal server error".  These failures are
            # transient and retrying the same request is appropriate; they
            # must not be confused with authentication, balance, or model
            # configuration errors, which remain fatal below.
            "internalerror",
            "modelservingerror",
            "modelservingwithdetailerror",
            "receive batching backend response failed",
            "batching backend",
            "code: 500",
            "code 500",
            "http 500",
        )
        if any(marker in lowered_content for marker in transient_markers):
            raise LLMTransientError(
                f"Temporary LLM provider error: {content[:500]}"
            )
        raise LLMProviderError(
            f"LLM provider returned an error: {content[:500]}"
        )

    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", content, re.I | re.S)
    search_spaces = fenced + [content]
    decoder = json.JSONDecoder()
    for search_space in reversed(search_spaces):
        for start in reversed(
            [index for index, char in enumerate(search_space) if char == "{"]
        ):
            try:
                parsed, _ = decoder.raw_decode(search_space[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise ValueError(f"No valid JSON object found in LLM response: {content[:500]!r}")


def mask_entities(text):
    counters = defaultdict(int)

    def replace(match):
        token = match.group(0)
        if token.startswith(("http://", "https://", "www.")):
            kind = "URL"
        elif token.startswith("@"):
            kind = "USER"
        elif token.startswith("#"):
            kind = "HASHTAG"
        else:
            kind = "ENTITY"
        counters[kind] += 1
        return f"[{kind}_{counters[kind]}]"

    return ENTITY_PATTERN.sub(replace, text)


def prepare_llm_text(text, args):
    if args.knowledge_control in {"entity_mask", "both"}:
        return mask_entities(text)
    return text


RESERVED_EVENTS = {"", "Others", "Confusion"}
LEXICAL_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "with", "about", "news", "people", "today", "un", "une", "de", "des",
    "du", "et", "en", "la", "le", "les", "pour", "sur",
}

VIEW_TOKEN_PATTERN = re.compile(
    r"(?u)#[\w-]+|@[A-Za-z0-9_]+|"
    r"[A-Za-zÀ-ÖØ-öø-ÿ][\wÀ-ÖØ-öø-ÿ'-]*|"
    r"\b(?:19|20)\d{2}\b|\b\d{1,2}[:/-]\d{1,2}(?:[:/-]\d{2,4})?\b"
)
TEMPORAL_PATTERN = re.compile(
    r"^(?:(?:19|20)\d{2}|\d{1,2}[:/-]\d{1,2}(?:[:/-]\d{2,4})?)$"
)


def remember_event_keywords(event_keywords, event, keywords, args):
    """Update local event memory without discarding earlier evidence."""
    if event in RESERVED_EVENTS:
        return
    cleaned = [
        keyword.strip()
        for keyword in keywords
        if isinstance(keyword, str) and keyword.strip()
    ]
    if not args.accumulate_event_keywords:
        event_keywords[event] = cleaned
        return

    merged = []
    seen = set()
    for keyword in event_keywords.get(event, []) + cleaned:
        normalized = keyword.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        merged.append(keyword)
        if len(merged) >= args.event_keyword_memory_size:
            break
    event_keywords[event] = merged


def event_memory_text(event, keywords, args):
    """Implement Text(event, W_event) from the MvSED memory equation."""
    pieces = []
    if (
        args.include_event_name_in_memory
        and args.knowledge_control not in {"event_alias", "both"}
        and event not in RESERVED_EVENTS
    ):
        pieces.append(event.replace("_", " "))
    pieces.extend(
        keyword
        for keyword in keywords
        if isinstance(keyword, str) and keyword.strip()
    )
    return " ".join(pieces).strip() or event


def remember_event_evidence(event_evidence, event, text, args):
    """Retain a bounded, label-free source-text memory for aligned views."""
    if event in RESERVED_EVENTS or not isinstance(text, str):
        return
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return
    memory = event_evidence.setdefault(event, [])
    if cleaned not in memory:
        memory.append(cleaned)
    limit = max(1, int(args.conrag_event_evidence_size))
    if len(memory) > limit:
        del memory[:-limit]


def _timestamp_hours(value):
    """Parse one dataset timestamp to UTC-agnostic epoch hours."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp() / 3600.0
    except (TypeError, ValueError, OverflowError):
        return None


def build_anchor_metadata(anchor_ids, block, args):
    """Aggregate label-free time/platform metadata for fixed anchors.

    The event labels stored in the benchmark are deliberately never read.
    Metadata is derived only from raw records addressed by each fixed anchor.
    """
    records = block.get("test", [])
    all_times = [
        timestamp
        for timestamp in (
            _timestamp_hours(record.get("created_at"))
            for record in records
        )
        if timestamp is not None
    ]
    if all_times:
        block_start = min(all_times)
        block_span = max(all_times) - block_start
    else:
        block_start = 0.0
        block_span = 0.0

    bin_width = max(float(args.stage_pulse_bin_hours), 1e-6)
    bin_counts = Counter(
        int((timestamp - block_start) // bin_width)
        for timestamp in all_times
    )
    count_values = np.asarray(list(bin_counts.values()), dtype=float)
    count_median = (
        float(np.median(count_values)) if len(count_values) else 0.0
    )
    count_mad = (
        float(np.median(np.abs(count_values - count_median)))
        if len(count_values)
        else 0.0
    )
    robust_scale = max(1.0, 1.4826 * count_mad)

    output = []
    for cluster in anchor_ids:
        indices = [
            int(index)
            for index in cluster
            if 0 <= int(index) < len(records)
        ]
        sources = [records[index] for index in indices]
        times = [
            timestamp
            for timestamp in (
                _timestamp_hours(source.get("created_at"))
                for source in sources
            )
            if timestamp is not None
        ]
        median_time = float(np.median(times)) if times else None
        if median_time is not None:
            pulse_bin = int((median_time - block_start) // bin_width)
            pulse_count = int(bin_counts.get(pulse_bin, 0))
            pulse_strength = max(
                0.0, (pulse_count - count_median) / robust_scale
            )
        else:
            pulse_bin = None
            pulse_count = 0
            pulse_strength = 0.0

        hashtags = _unique_in_order(
            str(value).strip("#")
            for source in sources
            for value in source.get("hashtags", [])
            if str(value).strip("#")
        )
        mentions = _unique_in_order(
            str(value).strip("@")
            for source in sources
            for value in source.get("user_mentions", [])
            if str(value).strip("@")
        )
        locations = _unique_in_order(
            value
            for source in sources
            for value in (
                source.get("place_full_name", ""),
                source.get("user_loc", ""),
            )
            if isinstance(value, str) and value.strip()
        )
        output.append(
            {
                "timestamp_hours": median_time,
                "time_span_hours": (
                    float(max(times) - min(times)) if times else 0.0
                ),
                "block_span_hours": float(block_span),
                "pulse_bin": pulse_bin,
                "pulse_count": pulse_count,
                "pulse_strength": float(pulse_strength),
                "hashtags": hashtags,
                "mentions": mentions,
                "locations": locations,
            }
        )
    return output


def remember_event_metadata(event_metadata, event, metadata, args):
    """Maintain a bounded online metadata prototype without event labels."""
    if event in RESERVED_EVENTS or not isinstance(metadata, dict):
        return
    memory = event_metadata.setdefault(
        event,
        {
            "timestamp_hours": [],
            "pulse_strength": [],
            "hashtags": [],
            "mentions": [],
            "locations": [],
        },
    )
    timestamp = metadata.get("timestamp_hours")
    if timestamp is not None:
        timestamp = float(timestamp)
        if timestamp not in memory["timestamp_hours"]:
            memory["timestamp_hours"].append(timestamp)
            memory["pulse_strength"].append(
                float(metadata.get("pulse_strength", 0.0))
            )
    for key in ("hashtags", "mentions", "locations"):
        memory[key] = _unique_in_order(
            memory.get(key, []) + list(metadata.get(key, [])),
            limit=args.stage_event_metadata_size,
        )
    limit = max(1, int(args.stage_event_metadata_size))
    for key in ("timestamp_hours", "pulse_strength"):
        if len(memory[key]) > limit:
            del memory[key][:-limit]


def _unique_in_order(values, limit=None):
    output = []
    seen = set()
    for value in values:
        normalized = value.casefold().strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
        if limit is not None and len(output) >= limit:
            break
    return output


def _is_entity_token(raw):
    normalized = (raw or "").casefold().strip()
    return bool(
        normalized
        and normalized not in LEXICAL_STOPWORDS
        and (
            raw.startswith(("#", "@"))
            or TEMPORAL_PATTERN.match(raw) is not None
            or (raw[:1].isupper() and len(raw) >= 3)
            or (raw.isupper() and len(raw) >= 2)
        )
    )


def extract_entity_anchors(text, limit=32):
    """Extract auditable entity/time anchors without an extra LLM call."""
    anchors = []
    for match in VIEW_TOKEN_PATTERN.finditer(text or ""):
        raw = match.group(0).strip()
        if _is_entity_token(raw):
            anchors.append(raw)
    return _unique_in_order(anchors, limit=limit)


def _content_tokens(text):
    tokens = []
    for match in VIEW_TOKEN_PATTERN.finditer(text or ""):
        token = match.group(0).casefold().strip()
        if (
            token in LEXICAL_STOPWORDS
            or token.startswith("http")
            or len(token.strip("#@")) < 2
        ):
            continue
        tokens.append(token)
    return tokens


def extract_relation_units(text, view_separator=" | ", limit=32):
    """Create local action/context relation units grounded in source text.

    Social posts rarely provide clean OpenIE triples.  Consecutive content
    pairs and entity-to-nearby-token links provide a deterministic structural
    proxy while remaining fully traceable to the supplied messages.
    """
    units = []
    for segment in (text or "").split(view_separator):
        tokens = _content_tokens(segment)
        for left, right in zip(tokens, tokens[1:]):
            if left != right:
                units.append(f"{left} {right}")
        entity_positions = {
            index
            for index, token in enumerate(tokens)
            if token.startswith(("#", "@"))
            or TEMPORAL_PATTERN.match(token) is not None
        }
        for index in sorted(entity_positions):
            for neighbour in tokens[index + 1 : index + 4]:
                if neighbour != tokens[index]:
                    units.append(f"{tokens[index]} {neighbour}")
    return _unique_in_order(units, limit=limit)


def extract_entity_object_texts(text, limit=32, context_window=4):
    """Textualize individual entity objects with source-grounded context.

    This is the corpus-side ``Connection`` representation used by the object
    ConRAG variants.  Each returned key is one graph object; its value contains
    only words observed near that object in the source evidence unit.
    """
    matches = [match.group(0).strip() for match in VIEW_TOKEN_PATTERN.finditer(text or "")]
    records = [
        {
            "raw": raw,
            "normalized": raw.casefold(),
            "is_entity": _is_entity_token(raw),
        }
        for raw in matches
        if raw.casefold() not in LEXICAL_STOPWORDS
        and not raw.casefold().startswith("http")
    ]
    object_contexts = defaultdict(list)
    for index, record in enumerate(records):
        if not record["is_entity"]:
            continue
        if (
            record["normalized"] not in object_contexts
            and len(object_contexts) >= limit
        ):
            continue
        start = max(0, index - context_window)
        end = min(len(records), index + context_window + 1)
        context = [item["normalized"] for item in records[start:end]]
        representation = " ".join(
            ["entity", record["normalized"], "context"] + context
        )
        object_contexts[record["normalized"]].append(representation)
    return {
        entity: _unique_in_order(contexts)
        for entity, contexts in list(object_contexts.items())[:limit]
    }


def extract_relation_objects(
    text,
    view_separator=" | ",
    limit=32,
    max_entity_span=10,
):
    """Extract independent, traceable relation objects from short posts.

    Explicit entity pairs use the intervening content as a lightweight
    relation phrase.  Lower-case posts without detectable entities fall back
    to local content pairs.  The graph object always remains linked to the
    source text and no LLM-generated fact is introduced.
    """
    units = []
    for segment in (text or "").split(view_separator):
        raw_tokens = [
            match.group(0).strip()
            for match in VIEW_TOKEN_PATTERN.finditer(segment)
        ]
        records = [
            {
                "token": raw.casefold(),
                "is_entity": _is_entity_token(raw),
            }
            for raw in raw_tokens
            if raw.casefold() not in LEXICAL_STOPWORDS
            and not raw.casefold().startswith("http")
            and len(raw.strip("#@")) >= 2
        ]
        entity_positions = [
            index for index, record in enumerate(records) if record["is_entity"]
        ]

        for left_pos, right_pos in zip(entity_positions, entity_positions[1:]):
            if right_pos - left_pos > max_entity_span:
                continue
            middle = [
                record["token"]
                for record in records[left_pos + 1 : right_pos]
                if not record["is_entity"]
            ][:4]
            relation = [records[left_pos]["token"]] + middle + [
                records[right_pos]["token"]
            ]
            if len(relation) >= 2:
                units.append(" ".join(relation))

        for entity_pos in entity_positions:
            neighbours = []
            for record in records[entity_pos + 1 : entity_pos + 5]:
                if not record["is_entity"]:
                    neighbours.append(record["token"])
            for neighbour in neighbours[:2]:
                units.append(
                    f"{records[entity_pos]['token']} {neighbour}"
                )

        if not entity_positions:
            content = [record["token"] for record in records]
            for left, right in zip(content, content[1:]):
                if left != right:
                    units.append(f"{left} {right}")
    return _unique_in_order(units, limit=limit)


def _build_conrag_evidence_graph(
    valid_events,
    event_keywords_dict,
    event_evidence_dict,
    args,
):
    """Build evidence units plus entity/relation-to-source mappings.

    The graph is reconstructed from the current dynamic event memory because
    streaming SED continuously adds evidence.  Every structural object keeps a
    ``sources`` set pointing to auditable evidence-unit indices.
    """
    evidence_units = []
    for event_index, event in enumerate(valid_events):
        source_texts = event_evidence_dict.get(event, [])
        source_texts = source_texts[-args.conrag_event_evidence_size :]
        seen = set()
        for source_text in source_texts:
            cleaned = " ".join((source_text or "").split()).strip()
            if not cleaned:
                continue
            cleaned = cleaned[: args.conrag_event_text_chars]
            fingerprint = cleaned.casefold()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            evidence_units.append(
                {
                    "event_index": event_index,
                    "event": event,
                    "kind": "source",
                    "text": cleaned,
                    "graph_eligible": True,
                }
            )

        memory = event_memory_text(
            event, event_keywords_dict.get(event, []), args
        )
        memory = " ".join((memory or "").split()).strip()
        if memory:
            memory = memory[: args.conrag_event_text_chars]
            fingerprint = memory.casefold()
            if fingerprint not in seen:
                evidence_units.append(
                    {
                        "event_index": event_index,
                        "event": event,
                        "kind": "memory",
                        "text": memory,
                        # Generated event names are useful to the text control,
                        # but graph objects should be grounded in source posts.
                        "graph_eligible": not bool(source_texts),
                    }
                )

    if not args.conrag_use_entity and not args.conrag_use_relation:
        return (
            evidence_units,
            [],
            [],
            np.ones(len(evidence_units), dtype=float),
        )

    entity_map = {}
    relation_map = {}
    for evidence_index, evidence in enumerate(evidence_units):
        if not evidence["graph_eligible"]:
            continue
        entity_records = (
            extract_entity_object_texts(
                evidence["text"],
                limit=args.conrag_max_view_units,
                context_window=args.conrag_entity_context_window,
            )
            if args.conrag_use_entity
            else {}
        )
        relation_records = (
            extract_relation_objects(
                evidence["text"],
                view_separator=args.view_separator,
                limit=args.conrag_max_view_units,
            )
            if args.conrag_use_entity or args.conrag_use_relation
            else []
        )
        for entity, contexts in entity_records.items():
            record = entity_map.setdefault(
                entity,
                {"key": entity, "contexts": [], "sources": set(), "relations": set()},
            )
            record["sources"].add(evidence_index)
            for context in contexts:
                if context not in record["contexts"]:
                    record["contexts"].append(context)
        if args.conrag_use_relation:
            for relation in relation_records:
                record = relation_map.setdefault(
                    relation,
                    {"key": relation, "sources": set()},
                )
                record["sources"].add(evidence_index)
        for entity in entity_records:
            entity_map[entity]["relations"].update(
                relation
                for relation in relation_records
                if entity in relation.split()
            )

    entity_objects = []
    for key in sorted(entity_map):
        record = entity_map[key]
        contexts = record["contexts"][: args.conrag_object_contexts]
        relation_summaries = sorted(record["relations"])[
            : args.conrag_object_contexts
        ]
        text_parts = contexts[:]
        if relation_summaries:
            text_parts.append(
                "local relations " + " ; ".join(relation_summaries)
            )
        text = " ; ".join(text_parts) or f"entity {key}"
        entity_objects.append(
            {
                "key": key,
                "text": text[: args.conrag_event_text_chars],
                "sources": sorted(record["sources"]),
                "degree": max(
                    1,
                    len(record["sources"]),
                    len(record["relations"]),
                ),
            }
        )

    relation_objects = [
        {
            "key": key,
            "text": key,
            "sources": sorted(relation_map[key]["sources"]),
        }
        for key in sorted(relation_map)
    ]

    evidence_neighbours = [set([index]) for index in range(len(evidence_units))]
    for graph_object in entity_objects + relation_objects:
        sources = set(graph_object["sources"])
        for evidence_index in sources:
            evidence_neighbours[evidence_index].update(sources)
    evidence_degrees = np.asarray(
        [max(1, len(neighbours)) for neighbours in evidence_neighbours],
        dtype=float,
    )
    return evidence_units, entity_objects, relation_objects, evidence_degrees


def _cached_conrag_embeddings(texts, namespace, sbert_model, stats):
    if not texts:
        return np.empty((0, 0), dtype=float)
    embeddings = [None] * len(texts)
    missing_indices = []
    missing_texts = []
    missing_keys = []
    for index, text in enumerate(texts):
        fingerprint = hashlib.sha1(text.encode("utf-8")).hexdigest()
        cache_key = (namespace, fingerprint)
        if cache_key in stats.conrag_embedding_cache:
            embeddings[index] = stats.conrag_embedding_cache[cache_key]
        else:
            missing_indices.append(index)
            missing_texts.append(text)
            missing_keys.append(cache_key)
    if missing_texts:
        encoded = encode_texts(
            sbert_model,
            missing_texts,
            stats,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for index, cache_key, embedding in zip(
            missing_indices, missing_keys, encoded
        ):
            value = np.asarray(embedding, dtype=float)
            stats.conrag_embedding_cache[cache_key] = value
            embeddings[index] = value
    return np.stack(embeddings)


def _object_view_hits(
    query_embedding,
    object_texts,
    namespace,
    sbert_model,
    args,
    stats,
):
    """Retrieve top graph objects before projecting them to source evidence."""
    if not object_texts:
        return []
    object_embeddings = _cached_conrag_embeddings(
        object_texts, namespace, sbert_model, stats
    )
    similarities = object_embeddings @ query_embedding
    # ConRAG's consensus counts positive support only.  A negative top-k hit is
    # therefore never allowed to create a structural vote.
    threshold = max(0.0, float(args.conrag_view_score_threshold))
    output = []
    for index in np.argsort(-similarities).tolist():
        score = float(similarities[index])
        if score <= threshold:
            continue
        output.append((score, int(index)))
        if len(output) >= args.conrag_object_top_k:
            break
    return output


def object_conrag_rank(
    query_text,
    valid_events,
    event_keywords_dict,
    event_evidence_dict,
    sbert_model,
    args,
    stats,
):
    """ConRAG Connection+Consensus adapted to streaming event retrieval.

    Entity and relation objects are independently retrieved, projected through
    auditable source mappings to a unified evidence-unit space, normalized per
    view, and only then aggregated to event candidates.  The multi-hop QA-only
    slot-binding component is intentionally excluded.
    """
    query_text = (query_text or "").replace(args.view_separator, " ").strip()
    if not query_text or not valid_events:
        return [], np.zeros(len(valid_events), dtype=int), [], {}

    (
        evidence_units,
        entity_objects,
        relation_objects,
        evidence_degrees,
    ) = _build_conrag_evidence_graph(
        valid_events, event_keywords_dict, event_evidence_dict, args
    )
    if not evidence_units:
        return [], np.zeros(len(valid_events), dtype=int), [], {}

    stats.conrag_evidence_units_built += len(evidence_units)
    stats.conrag_entity_objects_built += len(entity_objects)
    stats.conrag_relation_objects_built += len(relation_objects)

    query_embedding = np.asarray(
        encode_texts(
            sbert_model,
            [query_text],
            stats,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0],
        dtype=float,
    )
    evidence_count = len(evidence_units)
    view_results = {}
    object_audit = {}

    text_hits = _object_view_hits(
        query_embedding,
        [unit["text"] for unit in evidence_units],
        "object_conrag_text_evidence",
        sbert_model,
        args,
        stats,
    )
    if text_hits:
        raw_scores = np.zeros(evidence_count, dtype=float)
        hit_mask = np.zeros(evidence_count, dtype=bool)
        for score, evidence_index in text_hits:
            raw_scores[evidence_index] = score
            hit_mask[evidence_index] = True
        view_results["text"] = (
            _normalize_hit_scores(raw_scores, hit_mask),
            hit_mask,
            float(args.conrag_text_weight),
            raw_scores,
        )
        object_audit["text"] = [
            {
                "evidence_index": evidence_index,
                "event": evidence_units[evidence_index]["event"],
                "score": score,
            }
            for score, evidence_index in text_hits
        ]
        stats.conrag_object_hits_by_view["text"] += len(text_hits)

    if args.conrag_use_entity and entity_objects:
        entity_hits = _object_view_hits(
            query_embedding,
            [item["text"] for item in entity_objects],
            "object_conrag_entity",
            sbert_model,
            args,
            stats,
        )
        if entity_hits:
            raw_scores = np.zeros(evidence_count, dtype=float)
            for score, object_index in entity_hits:
                graph_object = entity_objects[object_index]
                degree_penalty = (
                    1.0
                    if graph_object["degree"] <= 1
                    else 1.0 / (1.0 + math.log(graph_object["degree"]))
                )
                if degree_penalty < 1.0:
                    stats.conrag_hub_penalized_matches += 1
                for evidence_index in graph_object["sources"]:
                    raw_scores[evidence_index] += score * degree_penalty
                    stats.conrag_source_projections += 1
            evidence_penalty = 1.0 / (1.0 + np.log(evidence_degrees))
            raw_scores *= evidence_penalty
            hit_mask = raw_scores > 0.0
            if np.any(hit_mask):
                view_results["entity"] = (
                    _normalize_hit_scores(raw_scores, hit_mask),
                    hit_mask,
                    float(args.conrag_entity_weight),
                    raw_scores,
                )
                stats.conrag_object_hits_by_view["entity"] += len(entity_hits)
                object_audit["entity"] = [
                    {
                        "object": entity_objects[index]["key"],
                        "score": score,
                        "degree": entity_objects[index]["degree"],
                        "source_count": len(entity_objects[index]["sources"]),
                    }
                    for score, index in entity_hits
                ]

    if args.conrag_use_relation and relation_objects:
        relation_hits = _object_view_hits(
            query_embedding,
            [item["text"] for item in relation_objects],
            "object_conrag_relation",
            sbert_model,
            args,
            stats,
        )
        if relation_hits:
            evidence_support = defaultdict(list)
            for score, object_index in relation_hits:
                graph_object = relation_objects[object_index]
                for evidence_index in graph_object["sources"]:
                    evidence_support[evidence_index].append(score)
                    stats.conrag_source_projections += 1
            raw_scores = np.zeros(evidence_count, dtype=float)
            for evidence_index, scores in evidence_support.items():
                ordered = sorted(scores, reverse=True)
                raw_scores[evidence_index] = ordered[0] + float(
                    args.conrag_relation_beta
                ) * sum(ordered[1:])
            hit_mask = raw_scores > 0.0
            if np.any(hit_mask):
                view_results["relation"] = (
                    _normalize_hit_scores(raw_scores, hit_mask),
                    hit_mask,
                    float(args.conrag_relation_weight),
                    raw_scores,
                )
                stats.conrag_object_hits_by_view["relation"] += len(relation_hits)
                object_audit["relation"] = [
                    {
                        "object": relation_objects[index]["key"],
                        "score": score,
                        "source_count": len(relation_objects[index]["sources"]),
                    }
                    for score, index in relation_hits
                ]

    active_views = list(view_results)
    if not active_views:
        return [], np.zeros(len(valid_events), dtype=int), [], {}

    candidate_mask = np.zeros(evidence_count, dtype=bool)
    support_counts = np.zeros(evidence_count, dtype=int)
    fused_scores = np.zeros(evidence_count, dtype=float)
    active_weight = sum(view_results[name][2] for name in active_views)
    if active_weight <= 0.0:
        active_weight = float(len(active_views))
    for view_name in active_views:
        normalized, hit_mask, weight, _ = view_results[view_name]
        candidate_mask |= hit_mask
        support_counts += hit_mask.astype(int)
        fused_scores += normalized * (weight / active_weight)
        stats.conrag_view_uses[view_name] += 1

    proposed_candidate_mask = candidate_mask.copy()
    rejected_structural_only = 0
    if args.conrag_require_structural_consensus:
        text_mask = (
            view_results["text"][1]
            if "text" in view_results
            else np.zeros(evidence_count, dtype=bool)
        )
        structural_support = np.zeros(evidence_count, dtype=int)
        for view_name in ("entity", "relation"):
            if view_name in view_results:
                structural_support += view_results[view_name][1].astype(int)
        admissible_mask = text_mask | (
            structural_support >= args.conrag_structural_min_views
        )
        rejected_structural_only = int(
            np.sum(proposed_candidate_mask & ~admissible_mask)
        )
        candidate_mask &= admissible_mask
        stats.conrag_structural_only_rejections += rejected_structural_only

    consensus_bonus = 1.0 + float(args.conrag_consensus_lambda) * (
        np.maximum(0, support_counts - 1) / 2.0
    )
    fused_scores *= consensus_bonus
    fused_scores[~candidate_mask] = -np.inf
    stats.conrag_evidence_candidates += int(
        np.sum(proposed_candidate_mask)
    )
    stats.conrag_admissible_evidence_candidates += int(
        np.sum(candidate_mask)
    )

    event_scores = np.full(len(valid_events), -np.inf, dtype=float)
    event_support = np.zeros(len(valid_events), dtype=int)
    # Preserve an unnormalized text-evidence score for the residual safety
    # gate. Fused scores are normalized within each view and can therefore be
    # high even when the absolute semantic match is weak; this independent
    # score prevents a structural coincidence from reusing an unrelated event.
    event_text_scores = np.full(len(valid_events), -np.inf, dtype=float)
    event_structural_support = np.zeros(len(valid_events), dtype=int)
    best_evidence = np.full(len(valid_events), -1, dtype=int)
    for evidence_index, evidence in enumerate(evidence_units):
        event_index = int(evidence["event_index"])
        if "text" in view_results:
            text_raw_scores = view_results["text"][3]
            if text_raw_scores[evidence_index] > event_text_scores[event_index]:
                event_text_scores[event_index] = float(
                    text_raw_scores[evidence_index]
                )
        structural_count = sum(
            int(view_results[view_name][1][evidence_index])
            for view_name in ("entity", "relation")
            if view_name in view_results
        )
        event_structural_support[event_index] = max(
            event_structural_support[event_index], structural_count
        )
        if not np.isfinite(fused_scores[evidence_index]):
            continue
        if fused_scores[evidence_index] > event_scores[event_index]:
            event_scores[event_index] = fused_scores[evidence_index]
            event_support[event_index] = support_counts[evidence_index]
            best_evidence[event_index] = evidence_index

    top_indices = np.argsort(-event_scores)[
        : min(args.retrieval_top_k, len(valid_events))
    ]
    top_pairs = [
        (float(event_scores[index]), int(index))
        for index in top_indices
        if np.isfinite(event_scores[index])
    ]

    stats.conrag_queries += 1
    stats.conrag_multi_supported_candidates += int(
        np.sum(event_support >= 2)
    )
    if args.save_audit_log:
        stats.retrieval_audit_records.append(
            {
                "block": stats.block_id,
                "mode": "object_connection_consensus",
                "query": query_text[: args.max_input_chars],
                "active_views": active_views,
                "evidence_unit_count": len(evidence_units),
                "entity_object_count": len(entity_objects),
                "relation_object_count": len(relation_objects),
                "candidate_policy": (
                    "text_or_entity_relation_consensus"
                    if args.conrag_require_structural_consensus
                    else "union_of_views"
                ),
                "proposed_evidence_candidates": int(
                    np.sum(proposed_candidate_mask)
                ),
                "admissible_evidence_candidates": int(
                    np.sum(candidate_mask)
                ),
                "rejected_structural_only_candidates": (
                    rejected_structural_only
                ),
                "object_hits": object_audit,
                "candidates": [
                    {
                        "event": valid_events[index],
                        "fused_score": score,
                        "supporting_views": int(event_support[index]),
                        "raw_text_similarity": (
                            float(event_text_scores[index])
                            if np.isfinite(event_text_scores[index])
                            else None
                        ),
                        "structural_supporting_views": int(
                            event_structural_support[index]
                        ),
                        "best_evidence": (
                            evidence_units[best_evidence[index]]["text"][:300]
                            if best_evidence[index] >= 0
                            else ""
                        ),
                        "view_scores": {
                            view_name: float(
                                view_results[view_name][3][best_evidence[index]]
                            )
                            for view_name in active_views
                            if best_evidence[index] >= 0
                        },
                    }
                    for score, index in top_pairs
                ],
            }
        )
    diagnostics = {
        "event_text_scores": event_text_scores,
        "event_structural_support": event_structural_support,
        "best_evidence": best_evidence,
        "evidence_units": evidence_units,
    }
    return top_pairs, event_support, active_views, diagnostics


def build_cqc_query_views(query_text, args):
    """Build deterministic, meaning-preserving views without another LLM.

    The source comments are copied verbatim; only their order and the neutral
    retrieval instruction change. Named entities, dates, locations, hashtags,
    and event actions are therefore frozen by construction. This keeps CQC
    auditable and avoids the extra sequential generation calls that the local
    RAGFlow deployment cannot batch like the original QA implementation.
    """
    segments = [
        " ".join(segment.split()).strip()
        for segment in (query_text or "").split(args.view_separator)
        if segment.strip()
    ]
    if not segments:
        compact = " ".join((query_text or "").split()).strip()
        segments = [compact] if compact else []
    if not segments:
        return []

    original = f" {args.view_separator} ".join(segments)
    views = [original]
    instructions = (
        "Identify the specific social event described by this evidence:",
        "Match this evidence to the same underlying social event:",
        "Determine which existing event concerns this exact occurrence:",
        "Find the event supported by all of the following comments:",
    )
    for rewrite_index in range(args.cqc_paraphrase_count):
        if len(segments) > 1:
            shift = (rewrite_index + 1) % len(segments)
            reordered = segments[shift:] + segments[:shift]
            if rewrite_index % 2 == 0:
                reordered = list(reversed(reordered))
        else:
            reordered = segments
        evidence = f" {args.view_separator} ".join(reordered)
        views.append(
            f"{instructions[rewrite_index % len(instructions)]} {evidence}"
        )

    unique_views = []
    seen = set()
    for view in views:
        normalized = " ".join(view.split()).strip()
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique_views.append(normalized[: args.cqc_query_chars])
    return unique_views


def cqc_rerank_shared_candidates(
    query_text,
    base_top_pairs,
    valid_events,
    event_keywords_dict,
    event_evidence_dict,
    sbert_model,
    args,
    stats,
):
    """Verify one fixed ConRAG candidate pool across equivalent queries.

    This is an event-clustering adaptation of cross-query consistency, not a
    claim to reproduce CQC-RAG's token-logit evaluator. Each candidate is
    scored against its source-grounded event evidence under every accepted
    query view. The final score is mean confidence minus the adaptive variance
    penalty lambda_0 * mean * variance. An unstable top event causes an
    explicit abstention, allowing the streaming detector to create a new event
    rather than forcing a potentially destructive merge.
    """
    stats.cqc_anchors_checked += 1
    shared_indices = []
    seen_indices = set()
    for _, event_index in base_top_pairs:
        event_index = int(event_index)
        if event_index in seen_indices:
            continue
        seen_indices.add(event_index)
        shared_indices.append(event_index)
        if len(shared_indices) >= args.cqc_shared_pool_k:
            break
    stats.cqc_shared_pool_candidates += len(shared_indices)
    if not shared_indices:
        stats.cqc_abstentions += 1
        return [], {}

    query_views = build_cqc_query_views(query_text, args)
    stats.cqc_query_views_generated += len(query_views)
    if not query_views:
        stats.cqc_abstentions += 1
        return [], {}
    query_embeddings = np.asarray(
        encode_texts(
            sbert_model,
            query_views,
            stats,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=float,
    )
    semantic_similarities = query_embeddings @ query_embeddings[0]
    keep_mask = semantic_similarities >= float(
        args.cqc_semantic_similarity_threshold
    )
    keep_mask[0] = True
    rejected_views = int(np.sum(~keep_mask))
    stats.cqc_semantic_drift_rejections += rejected_views
    query_views = [
        view for view, keep in zip(query_views, keep_mask.tolist()) if keep
    ]
    query_embeddings = query_embeddings[keep_mask]
    semantic_similarities = semantic_similarities[keep_mask]
    stats.cqc_query_views_used += len(query_views)

    # At least two independently ordered/formulated views are necessary for a
    # variance or majority signal. Falling back to one view would silently
    # turn the CQC condition into the original single-query method.
    if len(query_views) < 2:
        stats.cqc_candidates_evaluated += len(shared_indices)
        stats.cqc_candidates_rejected += len(shared_indices)
        stats.cqc_abstentions += 1
        return [], {}

    evidence_texts = []
    evidence_records = []
    candidate_evidence_indices = []
    for event_index in shared_indices:
        event = valid_events[event_index]
        sources = []
        recent_sources = event_evidence_dict.get(event, [])
        for source in recent_sources[-args.cqc_event_evidence_size :]:
            cleaned = " ".join((source or "").split()).strip()
            if cleaned and cleaned not in sources:
                sources.append(cleaned[: args.cqc_evidence_chars])
        grounded_source_count = len(sources)
        memory = event_memory_text(
            event, event_keywords_dict.get(event, []), args
        )
        memory = " ".join(memory.split()).strip()
        if memory and memory not in sources:
            sources.append(memory[: args.cqc_evidence_chars])
        if not sources:
            sources = [event]

        local_indices = []
        for source_index, source in enumerate(sources):
            local_indices.append(len(evidence_texts))
            evidence_texts.append(source)
            evidence_records.append(
                {
                    "event": event,
                    "event_index": event_index,
                    "text": source,
                    "source_grounded": source_index < grounded_source_count,
                }
            )
        candidate_evidence_indices.append(local_indices)

    evidence_embeddings = _cached_conrag_embeddings(
        evidence_texts, "cqc_shared_event_evidence", sbert_model, stats
    )
    evidence_scores = query_embeddings @ evidence_embeddings.T
    score_matrix = np.full(
        (len(query_views), len(shared_indices)), -1.0, dtype=float
    )
    best_evidence_by_candidate = []
    for candidate_position, evidence_indices in enumerate(
        candidate_evidence_indices
    ):
        local_scores = evidence_scores[:, evidence_indices]
        score_matrix[:, candidate_position] = np.max(local_scores, axis=1)
        mean_evidence_scores = np.mean(local_scores, axis=0)
        grounded_locals = [
            local_position
            for local_position, evidence_index in enumerate(evidence_indices)
            if evidence_records[evidence_index]["source_grounded"]
        ]
        if grounded_locals:
            best_local = max(
                grounded_locals,
                key=lambda local_position: float(
                    mean_evidence_scores[local_position]
                ),
            )
        else:
            best_local = int(np.argmax(mean_evidence_scores))
        best_evidence_by_candidate.append(evidence_indices[best_local])

    mean_scores = np.mean(score_matrix, axis=0)
    variance_scores = np.var(score_matrix, axis=0)
    penalties = (
        float(args.cqc_variance_lambda)
        * np.maximum(0.0, mean_scores)
        * variance_scores
    )
    consistency_scores = mean_scores - penalties
    vote_counts = np.zeros(len(shared_indices), dtype=int)
    for winner in np.argmax(score_matrix, axis=1).tolist():
        vote_counts[int(winner)] += 1

    ranked_positions = np.argsort(-consistency_scores)
    top_position = int(ranked_positions[0])
    runner_score = (
        float(consistency_scores[ranked_positions[1]])
        if len(ranked_positions) > 1
        else float("-inf")
    )
    margin = (
        float(consistency_scores[top_position]) - runner_score
        if np.isfinite(runner_score)
        else float("inf")
    )
    minimum_votes = max(2, int(args.cqc_min_votes))
    accepted = bool(
        vote_counts[top_position] >= minimum_votes
        and mean_scores[top_position] >= args.cqc_min_mean_score
        and margin >= args.cqc_margin
    )

    stats.cqc_candidates_evaluated += len(shared_indices)
    stats.cqc_variance_penalty_sum += float(np.sum(penalties))
    if vote_counts[top_position] >= minimum_votes:
        stats.cqc_majority_agreements += 1
    if accepted:
        stats.cqc_candidates_accepted += 1
        stats.cqc_candidates_rejected += max(0, len(shared_indices) - 1)
    else:
        stats.cqc_candidates_rejected += len(shared_indices)
        stats.cqc_abstentions += 1

    top_event_index = shared_indices[top_position]
    top_event = valid_events[top_event_index]
    best_evidence_record = evidence_records[
        best_evidence_by_candidate[top_position]
    ]
    evidence_by_event = (
        {top_event: best_evidence_record["text"]} if accepted else {}
    )

    if args.save_audit_log:
        base_scores = {
            int(event_index): float(score)
            for score, event_index in base_top_pairs
        }
        stats.retrieval_audit_records.append(
            {
                "block": stats.block_id,
                "mode": "cross_query_consistency_event_gate",
                "original_query": query_text[: args.max_input_chars],
                "query_views": query_views,
                "query_semantic_similarities": [
                    float(value) for value in semantic_similarities.tolist()
                ],
                "shared_pool_size": len(shared_indices),
                "accepted": accepted,
                "minimum_votes": minimum_votes,
                "minimum_mean_score": float(args.cqc_min_mean_score),
                "minimum_margin": float(args.cqc_margin),
                "variance_lambda": float(args.cqc_variance_lambda),
                "top_event": top_event,
                "top_margin": margin,
                "candidates": [
                    {
                        "event": valid_events[shared_indices[position]],
                        "base_conrag_score": base_scores.get(
                            shared_indices[position]
                        ),
                        "per_query_scores": [
                            float(value)
                            for value in score_matrix[:, position].tolist()
                        ],
                        "mean_score": float(mean_scores[position]),
                        "variance": float(variance_scores[position]),
                        "variance_penalty": float(penalties[position]),
                        "consistency_score": float(
                            consistency_scores[position]
                        ),
                        "top1_votes": int(vote_counts[position]),
                        "best_evidence": evidence_records[
                            best_evidence_by_candidate[position]
                        ]["text"][:300],
                    }
                    for position in ranked_positions.tolist()
                ],
            }
        )

    if not accepted:
        print(
            "(CQC gate) Abstained: "
            f"top={top_event!r}, votes={vote_counts[top_position]}/"
            f"{len(query_views)}, mean={mean_scores[top_position]:.3f}, "
            f"variance={variance_scores[top_position]:.4f}, "
            f"margin={margin:.3f}."
        )
        return [], {}

    print(
        "(CQC gate) Accepted stable candidate "
        f"{top_event!r} (votes={vote_counts[top_position]}/"
        f"{len(query_views)}, mean={mean_scores[top_position]:.3f}, "
        f"variance={variance_scores[top_position]:.4f}, "
        f"margin={margin:.3f})."
    )
    return [
        (float(consistency_scores[top_position]), int(top_event_index))
    ], evidence_by_event


def _normalize_hit_scores(raw_scores, hit_mask):
    normalized = np.zeros_like(raw_scores, dtype=float)
    hit_indices = np.flatnonzero(hit_mask)
    if not len(hit_indices):
        return normalized
    values = raw_scores[hit_indices]
    score_range = float(np.max(values) - np.min(values))
    if score_range <= 1e-12:
        normalized[hit_indices] = 1.0
    else:
        normalized[hit_indices] = (
            values - float(np.min(values))
        ) / score_range
    return normalized


def _aligned_view_scores(
    query_text,
    event_texts,
    view_name,
    sbert_model,
    args,
    stats,
    eligible_mask=None,
):
    """Return raw scores and top-k hit mask for one aligned event view."""
    candidate_count = len(event_texts)
    raw_scores = np.full(candidate_count, -1.0, dtype=float)
    hit_mask = np.zeros(candidate_count, dtype=bool)
    query_text = (query_text or "").strip()
    active_indices = [
        index
        for index, value in enumerate(event_texts)
        if isinstance(value, str)
        and value.strip()
        and (eligible_mask is None or bool(eligible_mask[index]))
    ]
    if not query_text or not active_indices:
        return raw_scores, hit_mask

    query_embedding = np.asarray(
        encode_texts(
            sbert_model,
            [query_text],
            stats,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0],
        dtype=float,
    )
    missing_indices = []
    missing_texts = []
    cache_keys = {}
    for index in active_indices:
        fingerprint = hashlib.sha1(
            event_texts[index].encode("utf-8")
        ).hexdigest()
        cache_key = (view_name, fingerprint)
        cache_keys[index] = cache_key
        if cache_key not in stats.conrag_embedding_cache:
            missing_indices.append(index)
            missing_texts.append(event_texts[index])
    if missing_texts:
        missing_embeddings = encode_texts(
            sbert_model,
            missing_texts,
            stats,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for index, embedding in zip(missing_indices, missing_embeddings):
            stats.conrag_embedding_cache[cache_keys[index]] = np.asarray(
                embedding, dtype=float
            )
    event_embeddings = np.stack(
        [stats.conrag_embedding_cache[cache_keys[index]] for index in active_indices]
    )
    similarities = event_embeddings @ query_embedding
    raw_scores[np.asarray(active_indices, dtype=int)] = similarities
    ranked = np.argsort(-raw_scores)
    kept = 0
    for index in ranked.tolist():
        if index not in active_indices:
            continue
        if raw_scores[index] < args.conrag_view_score_threshold:
            continue
        hit_mask[index] = True
        kept += 1
        if kept >= args.conrag_per_view_top_k:
            break
    return raw_scores, hit_mask


def aligned_conrag_rank(
    query_text,
    valid_events,
    event_keywords_dict,
    event_evidence_dict,
    sbert_model,
    args,
    stats,
):
    """Rank events with ConRAG-style aligned text/entity/relation views.

    The three heterogeneous signals are normalized independently and mapped
    back to the same event IDs.  Structural views remain auxiliary and the
    consensus term is only a small ranking bonus; it never assigns a cluster.
    """
    source_query = (query_text or "").strip()
    query_text = source_query.replace(args.view_separator, " ").strip()
    event_sources = []
    for event in valid_events:
        evidence = " ".join(
            event_evidence_dict.get(event, [])[
                -args.conrag_event_evidence_size :
            ]
        )
        memory = event_memory_text(
            event, event_keywords_dict.get(event, []), args
        )
        combined = " ".join(part for part in (memory, evidence) if part)
        event_sources.append(combined[-args.conrag_event_text_chars :])

    view_payloads = {
        "text": {
            "enabled": True,
            "query": query_text,
            "events": event_sources,
            "weight": float(args.conrag_text_weight),
            "eligible": None,
        }
    }

    query_entities = extract_entity_anchors(
        query_text, limit=args.conrag_max_view_units
    )
    event_entities = [
        extract_entity_anchors(
            source, limit=args.conrag_max_view_units
        )
        for source in event_sources
    ]
    if args.conrag_use_entity:
        document_frequency = Counter(
            entity
            for entities in event_entities
            for entity in set(entities)
        )
        query_entity_set = set(query_entities)
        entity_eligible = np.asarray(
            [bool(query_entity_set.intersection(entities)) for entities in event_entities],
            dtype=bool,
        )
        view_payloads["entity"] = {
            "enabled": bool(query_entities),
            "query": " ".join(query_entities),
            "events": [" ".join(values) for values in event_entities],
            "weight": float(args.conrag_entity_weight),
            "eligible": entity_eligible,
            "document_frequency": document_frequency,
        }

    query_relations = extract_relation_units(
        source_query,
        view_separator=args.view_separator,
        limit=args.conrag_max_view_units,
    )
    event_relations = [
        extract_relation_units(
            source,
            view_separator=args.view_separator,
            limit=args.conrag_max_view_units,
        )
        for source in event_sources
    ]
    if args.conrag_use_relation:
        view_payloads["relation"] = {
            "enabled": bool(query_relations),
            "query": " ; ".join(query_relations),
            "events": [" ; ".join(values) for values in event_relations],
            "weight": float(args.conrag_relation_weight),
            "eligible": None,
        }

    candidate_count = len(valid_events)
    fused_scores = np.zeros(candidate_count, dtype=float)
    support_counts = np.zeros(candidate_count, dtype=int)
    candidate_mask = np.zeros(candidate_count, dtype=bool)
    active_views = []
    per_view_scores = {}
    view_results = {}

    for view_name, payload in view_payloads.items():
        if not payload.get("enabled"):
            continue
        raw_scores, hit_mask = _aligned_view_scores(
            payload["query"],
            payload["events"],
            view_name,
            sbert_model,
            args,
            stats,
            eligible_mask=payload.get("eligible"),
        )
        if not np.any(hit_mask):
            continue

        if view_name == "entity":
            document_frequency = payload["document_frequency"]
            query_entity_set = set(query_entities)
            for index, entities in enumerate(event_entities):
                overlap = query_entity_set.intersection(entities)
                if not overlap:
                    continue
                hub_weight = float(
                    np.mean(
                        [
                            1.0
                            / (1.0 + math.log(max(1, document_frequency[item])))
                            for item in overlap
                        ]
                    )
                )
                if hub_weight < 1.0:
                    stats.conrag_hub_penalized_matches += 1
                raw_scores[index] *= hub_weight
            # Rebuild the view's top-k after hub suppression so a ubiquitous
            # entity cannot retain support merely because it ranked highly
            # before the degree-aware penalty was applied.
            hit_mask[:] = False
            eligible = payload.get("eligible")
            kept = 0
            for index in np.argsort(-raw_scores).tolist():
                if eligible is not None and not bool(eligible[index]):
                    continue
                if raw_scores[index] < args.conrag_view_score_threshold:
                    continue
                hit_mask[index] = True
                kept += 1
                if kept >= args.conrag_per_view_top_k:
                    break

        normalized = _normalize_hit_scores(raw_scores, hit_mask)
        active_views.append(view_name)
        stats.conrag_view_uses[view_name] += 1
        candidate_mask |= hit_mask
        support_counts += hit_mask.astype(int)
        view_results[view_name] = (
            normalized,
            hit_mask,
            float(payload["weight"]),
        )
        per_view_scores[view_name] = raw_scores

    if not active_views:
        return [], np.zeros(candidate_count, dtype=int), []

    active_weight = sum(view_results[name][2] for name in active_views)
    if active_weight <= 0.0:
        active_weight = float(len(active_views))
    for view_name in active_views:
        normalized, _, weight = view_results[view_name]
        fused_scores += normalized * (weight / active_weight)

    consensus_bonus = 1.0 + float(args.conrag_consensus_lambda) * (
        np.maximum(0, support_counts - 1) / 2.0
    )
    fused_scores *= consensus_bonus
    fused_scores[~candidate_mask] = -np.inf
    top_indices = np.argsort(-fused_scores)[
        : min(args.retrieval_top_k, candidate_count)
    ]
    top_pairs = [
        (float(fused_scores[index]), int(index))
        for index in top_indices
        if np.isfinite(fused_scores[index])
    ]

    stats.conrag_queries += 1
    stats.conrag_multi_supported_candidates += int(
        np.sum(support_counts >= 2)
    )
    if args.save_audit_log:
        stats.retrieval_audit_records.append(
            {
                "block": stats.block_id,
                "query": query_text[: args.max_input_chars],
                "active_views": active_views,
                "query_entities": query_entities,
                "query_relations": query_relations,
                "candidates": [
                    {
                        "event": valid_events[index],
                        "fused_score": score,
                        "supporting_views": int(support_counts[index]),
                        "view_scores": {
                            view_name: float(per_view_scores[view_name][index])
                            for view_name in active_views
                        },
                    }
                    for score, index in top_pairs
                ],
            }
        )
    return top_pairs, support_counts, active_views


def _ranking_profile_alignment(left, right, indices):
    """Cosine alignment of centered candidate-score profiles."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 2:
        return 0.0
    left_values = np.asarray(left, dtype=float)[indices]
    right_values = np.asarray(right, dtype=float)[indices]
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    if int(np.sum(finite)) < 2:
        return 0.0
    left_values = left_values[finite]
    right_values = right_values[finite]
    left_values = left_values - float(np.mean(left_values))
    right_values = right_values - float(np.mean(right_values))
    denominator = float(
        np.linalg.norm(left_values) * np.linalg.norm(right_values)
    )
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left_values, right_values) / denominator)


def stage_aware_selective_rank(
    query_text,
    valid_events,
    event_keywords_dict,
    event_evidence_dict,
    event_metadata_dict,
    event_assignment_counts,
    anchor_metadata,
    sbert_model,
    args,
    stats,
):
    """Monolithic retrieval followed by selective multi-view reranking.

    This is a label-free adaptation of two source principles: decomposed views
    never expand the initial text candidate pool, and a view is fused only if
    its text-aligned marginal utility remains positive after redundancy cost.
    The timestamp/pulse signal is auxiliary and can never retrieve an event by
    itself. The text view is always retained with a fixed majority weight.
    """
    query_text = (query_text or "").replace(args.view_separator, " ").strip()
    candidate_count = len(valid_events)
    if not query_text or not candidate_count:
        return [], np.zeros(candidate_count, dtype=int), [], {}

    event_sources = []
    for event in valid_events:
        recent_evidence = event_evidence_dict.get(event, [])[
            -args.conrag_event_evidence_size :
        ]
        evidence = " ".join(recent_evidence)
        memory = event_memory_text(
            event, event_keywords_dict.get(event, []), args
        )
        combined = " ".join(part for part in (memory, evidence) if part)
        event_sources.append(combined[-args.conrag_event_text_chars :])

    query_embedding = np.asarray(
        encode_texts(
            sbert_model,
            [query_text],
            stats,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0],
        dtype=float,
    )
    event_embeddings = _cached_conrag_embeddings(
        event_sources,
        "stage_aware_monolithic_event",
        sbert_model,
        stats,
    )
    text_scores = event_embeddings @ query_embedding
    pool_size = min(int(args.stage_candidate_pool_k), candidate_count)
    pool_indices = np.argsort(-text_scores)[:pool_size]
    pool_mask = np.zeros(candidate_count, dtype=bool)
    pool_mask[pool_indices] = True
    stats.stage_aware_queries += 1
    stats.stage_aware_candidate_pool_total += len(pool_indices)

    text_hit_mask = pool_mask.copy()
    text_normalized = _normalize_hit_scores(text_scores, text_hit_mask)
    view_profiles = {"text": text_normalized}
    raw_view_scores = {"text": np.asarray(text_scores, dtype=float)}
    considered = {}

    if args.enable_mvra and len(pool_indices) >= 2:
        query_entities = extract_entity_anchors(
            query_text, limit=args.conrag_max_view_units
        )
        event_entities = [
            extract_entity_anchors(
                source, limit=args.conrag_max_view_units
            )
            for source in event_sources
        ]
        if query_entities:
            entity_scores, entity_hits = _aligned_view_scores(
                " ".join(query_entities),
                [" ".join(values) for values in event_entities],
                "stage_aware_entity",
                sbert_model,
                args,
                stats,
                eligible_mask=pool_mask,
            )
            if np.any(entity_hits & pool_mask):
                considered["entity"] = (
                    entity_scores,
                    _normalize_hit_scores(entity_scores, entity_hits),
                )

        query_relations = extract_relation_units(
            query_text,
            view_separator=args.view_separator,
            limit=args.conrag_max_view_units,
        )
        if query_relations:
            event_relations = [
                extract_relation_units(
                    source,
                    view_separator=args.view_separator,
                    limit=args.conrag_max_view_units,
                )
                for source in event_sources
            ]
            relation_scores, relation_hits = _aligned_view_scores(
                " ; ".join(query_relations),
                [" ; ".join(values) for values in event_relations],
                "stage_aware_relation",
                sbert_model,
                args,
                stats,
                eligible_mask=pool_mask,
            )
            if np.any(relation_hits & pool_mask):
                considered["relation"] = (
                    relation_scores,
                    _normalize_hit_scores(relation_scores, relation_hits),
                )

        query_time = (
            anchor_metadata.get("timestamp_hours")
            if isinstance(anchor_metadata, dict)
            else None
        )
        if args.stage_use_temporal and query_time is not None:
            temporal_scores = np.full(candidate_count, -1.0, dtype=float)
            temporal_hits = np.zeros(candidate_count, dtype=bool)
            block_span = float(anchor_metadata.get("block_span_hours", 0.0))
            temporal_scale = max(
                float(args.stage_temporal_scale_min_hours),
                block_span * float(args.stage_temporal_scale_fraction),
            )
            query_pulse = float(anchor_metadata.get("pulse_strength", 0.0))
            for event_index in pool_indices.tolist():
                event = valid_events[event_index]
                metadata = event_metadata_dict.get(event, {})
                event_times = metadata.get("timestamp_hours", [])
                if not event_times:
                    continue
                event_time = float(np.median(event_times))
                time_kernel = math.exp(
                    -abs(float(query_time) - event_time) / temporal_scale
                )
                event_pulses = metadata.get("pulse_strength", [])
                event_pulse = (
                    float(np.median(event_pulses))
                    if event_pulses
                    else 0.0
                )
                pulse_kernel = math.exp(-abs(query_pulse - event_pulse))
                temporal_scores[event_index] = time_kernel * (
                    0.75 + 0.25 * pulse_kernel
                )
                temporal_hits[event_index] = True
            if int(np.sum(temporal_hits)) >= 2:
                considered["temporal"] = (
                    temporal_scores,
                    _normalize_hit_scores(temporal_scores, temporal_hits),
                )
                stats.stage_aware_temporal_candidates += int(
                    np.sum(temporal_hits)
                )

    # A label-free relevance-minus-redundancy selector. The complete text
    # ranking is the reference signal, not a task-label kernel; this is an
    # adaptation of the cited selection principle rather than KAGES itself.
    proposals = []
    text_top = int(pool_indices[0])
    support_k = min(int(args.stage_support_top_k), len(pool_indices))
    for view_name, (raw_scores, normalized) in considered.items():
        stats.stage_aware_views_considered[view_name] += 1
        alignment = _ranking_profile_alignment(
            normalized, text_normalized, pool_indices
        )
        spread = float(np.std(normalized[pool_indices]))
        view_top = np.argsort(-raw_scores[pool_indices])[:support_k]
        view_top_events = {
            int(pool_indices[position]) for position in view_top.tolist()
        }
        top_consistent = text_top in view_top_events
        if alignment < args.stage_view_min_alignment:
            stats.stage_aware_view_rejections["text_misalignment"] += 1
            continue
        base_gain = max(0.0, alignment) * spread * (
            1.0 if top_consistent else 0.5
        )
        proposals.append(
            {
                "name": view_name,
                "raw": raw_scores,
                "normalized": normalized,
                "alignment": alignment,
                "spread": spread,
                "base_gain": base_gain,
                "top_consistent": top_consistent,
            }
        )

    proposals.sort(key=lambda item: (-item["base_gain"], item["name"]))
    selected = []
    selected_profiles = []
    auxiliary_budget = max(0, int(args.stage_max_views) - 1)
    for proposal in proposals:
        if len(selected) >= auxiliary_budget:
            stats.stage_aware_view_rejections["view_budget"] += 1
            continue
        redundancy = max(
            (
                abs(
                    _ranking_profile_alignment(
                        proposal["normalized"], profile, pool_indices
                    )
                )
                for profile in selected_profiles
            ),
            default=0.0,
        )
        marginal_gain = proposal["base_gain"] - (
            float(args.stage_view_redundancy_lambda)
            * redundancy
            * proposal["spread"]
        )
        if marginal_gain < args.stage_view_min_gain:
            reason = "redundant" if redundancy > 0.0 else "low_marginal_gain"
            stats.stage_aware_view_rejections[reason] += 1
            continue
        proposal["marginal_gain"] = float(marginal_gain)
        proposal["redundancy"] = float(redundancy)
        selected.append(proposal)
        selected_profiles.append(proposal["normalized"])
        stats.stage_aware_views_selected[proposal["name"]] += 1

    active_views = ["text"] + [item["name"] for item in selected]
    stats.stage_aware_selected_view_count_sum += len(active_views)
    if selected:
        stats.stage_aware_reranks += 1
    else:
        stats.stage_aware_text_only += 1
    if len(considered) > len(selected):
        stats.stage_aware_all_view_avoided += 1

    fused_scores = np.asarray(text_normalized, dtype=float) * float(
        args.stage_text_weight
    )
    if selected:
        gains = np.asarray(
            [item["marginal_gain"] for item in selected], dtype=float
        )
        gains /= max(float(np.sum(gains)), 1e-12)
        auxiliary = np.zeros(candidate_count, dtype=float)
        for weight, item in zip(gains.tolist(), selected):
            auxiliary += float(weight) * item["normalized"]
            view_profiles[item["name"]] = item["normalized"]
            raw_view_scores[item["name"]] = item["raw"]
        fused_scores += (1.0 - float(args.stage_text_weight)) * auxiliary
    fused_scores[~pool_mask] = -np.inf

    # Keep ranking support separate from gate-eligible support.  In the
    # auxiliary-only temporal condition, time may change the fused ordering
    # but cannot satisfy either the multi-view or structural gate by itself.
    total_support_counts = np.zeros(candidate_count, dtype=int)
    gate_support_counts = np.zeros(candidate_count, dtype=int)
    total_structural_support = np.zeros(candidate_count, dtype=int)
    event_structural_support = np.zeros(candidate_count, dtype=int)
    supporting_view_names = [set() for _ in range(candidate_count)]
    gate_supporting_view_names = [set() for _ in range(candidate_count)]
    for view_name in active_views:
        profile = view_profiles[view_name]
        ranked_positions = np.argsort(-profile[pool_indices])[:support_k]
        supported_events = [
            int(pool_indices[position]) for position in ranked_positions.tolist()
        ]
        total_support_counts[supported_events] += 1
        for event_index in supported_events:
            supporting_view_names[event_index].add(view_name)
        gate_eligible_view = not (
            args.stage_temporal_aux_only and view_name == "temporal"
        )
        if gate_eligible_view:
            gate_support_counts[supported_events] += 1
            for event_index in supported_events:
                gate_supporting_view_names[event_index].add(view_name)
        if view_name != "text":
            total_structural_support[supported_events] += 1
            if gate_eligible_view:
                event_structural_support[supported_events] += 1
            if view_name == "temporal":
                stats.stage_aware_temporal_supports += len(supported_events)

    top_indices = np.argsort(-fused_scores)[
        : min(args.retrieval_top_k, candidate_count)
    ]
    top_pairs = [
        (float(fused_scores[index]), int(index))
        for index in top_indices
        if np.isfinite(fused_scores[index])
    ]
    stats.conrag_queries += 1
    for view_name in active_views:
        stats.conrag_view_uses[view_name] += 1
    stats.conrag_multi_supported_candidates += int(
        np.sum(gate_support_counts >= 2)
    )

    if args.save_audit_log:
        stats.retrieval_audit_records.append(
            {
                "block": stats.block_id,
                "mode": "stage_aware_selective_mvra",
                "query": query_text[: args.max_input_chars],
                "candidate_pool": [
                    valid_events[index] for index in pool_indices.tolist()
                ],
                "selected_views": active_views,
                "view_selection": [
                    {
                        "view": item["name"],
                        "alignment_to_text": item["alignment"],
                        "spread": item["spread"],
                        "redundancy": item["redundancy"],
                        "marginal_gain": item["marginal_gain"],
                        "top_consistent": item["top_consistent"],
                    }
                    for item in selected
                ],
                "candidates": [
                    {
                        "event": valid_events[index],
                        "fused_score": score,
                        "raw_text_similarity": float(text_scores[index]),
                        "supporting_views": int(
                            total_support_counts[index]
                        ),
                        "supporting_view_names": sorted(
                            supporting_view_names[index]
                        ),
                        "gate_supporting_views": int(
                            gate_support_counts[index]
                        ),
                        "gate_supporting_view_names": sorted(
                            gate_supporting_view_names[index]
                        ),
                        "total_structural_supporting_views": int(
                            total_structural_support[index]
                        ),
                        "structural_supporting_views": int(
                            event_structural_support[index]
                        ),
                        "target_event_assignment_anchors": int(
                            event_assignment_counts.get(
                                valid_events[index], {}
                            ).get("anchors", 0)
                        ),
                        "target_event_assignment_messages": int(
                            event_assignment_counts.get(
                                valid_events[index], {}
                            ).get("messages", 0)
                        ),
                    }
                    for score, index in top_pairs
                ],
            }
        )

    return top_pairs, gate_support_counts, active_views, {
        "event_text_scores": np.asarray(text_scores, dtype=float),
        "event_structural_support": event_structural_support,
        "event_total_structural_support": total_structural_support,
        "event_total_support": total_support_counts,
        "event_gate_support": gate_support_counts,
        "event_supporting_view_names": supporting_view_names,
        "event_gate_supporting_view_names": gate_supporting_view_names,
        "event_assignment_counts": event_assignment_counts,
        "selected_views": active_views,
        "candidate_pool": pool_indices,
    }


def next_anonymous_event_id(events_list):
    used = set(events_list)
    index = 1
    while f"EventMem-{index:04d}" in used:
        index += 1
    return f"EventMem-{index:04d}"


def _ask_json(chat, session_name, prompt, role, stats):
    if not stats.budget_available():
        raise LLMBudgetExceeded(
            "Matched LLM call/token budget has been exhausted."
        )
    for session in chat.list_sessions(name=session_name):
        chat.delete_sessions(ids=[session.id])
    session = chat.create_session(name=session_name)
    content = ""
    final_answer = None
    json_parsed = False
    parsed = None
    started = time.perf_counter()
    try:
        for answer in session.ask(prompt, stream=True):
            final_answer = answer
            content = answer.content
        parsed = parse_json_object(content)
        json_parsed = True
    finally:
        stats.record_llm(
            role,
            prompt,
            content,
            time.perf_counter() - started,
            answer=final_answer,
            json_parsed=json_parsed,
        )
    return parsed


class LLMBudgetExceeded(RuntimeError):
    pass


def kb_update(evaluator, doc, text, events_list, args, stats, event_name=None):
    print("KB updating...")
    event_json = {"EVENT_NAME": "Confusion", "KEYWORDS": []}
    text_for_llm = prepare_llm_text(text[-args.kb_context_chars :], args)
    prompt = "COMMENTS: " + text_for_llm

    for attempt in range(1, args.max_llm_attempts + 1):
        try:
            event_json = _ask_json(evaluator, "kw", prompt, "evaluator", stats)
            if "EVENT_NAME" not in event_json or "KEYWORDS" not in event_json:
                raise ValueError("Required EVENT_NAME or KEYWORDS is missing.")
            if not isinstance(event_json["EVENT_NAME"], str) or not isinstance(
                event_json["KEYWORDS"], list
            ):
                raise ValueError("EVENT_NAME or KEYWORDS has an invalid type.")
            event_json["EVENT_NAME"] = event_json["EVENT_NAME"].strip().rstrip(".")
            if event_json["EVENT_NAME"] in {"Others", "Confusion", ""}:
                raise ValueError("EVENT_NAME is empty or reserved.")
            raw_keywords = [
                keyword.strip()
                for keyword in event_json["KEYWORDS"]
                if isinstance(keyword, str) and keyword.strip()
            ]
            supported_keywords = [
                keyword.strip()
                for keyword in raw_keywords
                if keyword.strip().lower() in text_for_llm.lower()
            ]
            stats.filtered_keyword_hallucinations += max(
                0, len(raw_keywords) - len(supported_keywords)
            )
            event_json["KEYWORDS"] = supported_keywords[: args.max_keywords]
            if (
                event_name is None
                and args.knowledge_control in {"event_alias", "both"}
            ):
                # The semantic name proposed by the evaluator is intentionally
                # discarded so pretrained historical-event names cannot enter
                # retrieval memory or detector candidates.
                event_json["EVENT_NAME"] = next_anonymous_event_id(events_list)
            stats.record_llm_success()
            break
        except LLMBudgetExceeded as exc:
            stats.record_llm_failure(exc)
            print(f"(KB update) {exc}")
            return events_list, "Confusion", []
        except LLMContentPolicyError as exc:
            stats.record_llm_failure(exc)
            stats.content_policy_blocks += 1
            print(f"(KB update) Content-policy block; using Confusion: {exc}")
            return events_list, "Confusion", []
        except LLMTransientError as exc:
            stats.record_llm_failure(exc)
            print(
                f"(KB update) {exc}; retry "
                f"{attempt}/{args.max_llm_attempts}"
            )
            if attempt == args.max_llm_attempts:
                print("(KB update) Retry limit reached; using Confusion.")
                return events_list, "Confusion", []
            time.sleep(args.llm_retry_delay)
        except LLMProviderError:
            # Authentication, balance and model-configuration failures will not
            # recover by retrying the same request; fail fast to protect results.
            raise
        except Exception as exc:
            stats.record_llm_failure(exc)
            print(f"(KB update) {exc}; retry {attempt}/{args.max_llm_attempts}")
            if attempt == args.max_llm_attempts:
                return events_list, "Confusion", []

    keywords = event_json["KEYWORDS"]
    if event_name is None:
        chunk_text = (
            f"EVENT: {event_json['EVENT_NAME']}. KEYWORDS: "
            + ", ".join(keywords)
            + "."
        )
        doc.add_chunk(
            content=chunk_text,
            important_keywords=[event_json["EVENT_NAME"]],
        )
        if args.kb_write_delay:
            time.sleep(args.kb_write_delay)
        if event_json["EVENT_NAME"] not in events_list:
            events_list.append(event_json["EVENT_NAME"])
        return events_list, event_json["EVENT_NAME"], keywords

    if event_name == "Confusion":
        return events_list, event_name, keywords
    print(f"Update chunk: {event_name}...")
    chunks = doc.list_chunks(keywords=event_name, page=1, page_size=999)
    if chunks:
        chunk_text = f"EVENT: {event_name}. KEYWORDS: " + ", ".join(keywords) + "."
        chunks[0].update(
            {
                "content": chunk_text,
                "important_keywords": [event_name],
                "available": True,
            }
        )
        if args.kb_write_delay:
            time.sleep(args.kb_write_delay)
    return events_list, event_name, keywords


def materialize_novelty_group(
    group_items,
    evaluator,
    document,
    events_list,
    event_keywords,
    event_evidence,
    event_messages,
    event_to_id,
    event_ids,
    args,
    stats,
):
    """Create one event for a confirmed unresolved-anchor micro-cluster."""
    ordered_items = sorted(group_items, key=lambda item: item["arrival_index"])
    context_items = ordered_items[-args.novelty_buffer_context_anchors :]
    context = " ".join(item["text"] for item in context_items)
    stats.novelty_buffer_evaluator_calls += 1
    events_list, event_name, keywords = kb_update(
        evaluator,
        document,
        context,
        events_list,
        args,
        stats,
    )
    stats.novelty_buffer_groups_created += 1
    if len(ordered_items) >= args.novelty_buffer_min_cluster_anchors:
        stats.novelty_buffer_multi_anchor_groups += 1
        stats.novelty_buffer_anchors_consolidated += len(ordered_items)
    else:
        stats.novelty_buffer_singletons_flushed += 1

    if event_name != "Confusion":
        remember_event_keywords(event_keywords, event_name, keywords, args)
    else:
        stats.confusion_predictions += len(ordered_items)
    if event_name not in event_to_id:
        event_to_id[event_name] = len(event_to_id)
    assigned_id = event_to_id[event_name]

    for item in ordered_items:
        for position in item["positions"]:
            event_ids[position] = assigned_id
        event_messages.setdefault(event_name, []).append(item["text"])
        remember_event_evidence(
            event_evidence,
            event_name,
            item["text"],
            args,
        )
    return events_list, event_name


def conrag_consensus_fusion(score_array, args):
    """Fuse representative-message views in one aligned event space.

    Each view is normalized independently before fusion. Candidate support is
    counted over a per-view top-k neighbourhood rather than only the top-1
    event. A central representative view receives the primary weight, while a
    small consensus bonus promotes candidates supported by multiple views.
    All operations are label-free and deterministic.
    """
    scores = np.asarray(score_array, dtype=float)
    if scores.ndim != 2:
        raise ValueError("MVRA score array must be two-dimensional.")
    view_count, candidate_count = scores.shape
    if view_count == 0 or candidate_count == 0:
        return (
            np.asarray([], dtype=float),
            np.zeros(candidate_count, dtype=int),
            np.zeros(view_count, dtype=float),
            np.zeros(candidate_count, dtype=bool),
        )

    row_min = np.min(scores, axis=1, keepdims=True)
    row_range = np.max(scores, axis=1, keepdims=True) - row_min
    normalized = np.divide(
        scores - row_min,
        row_range,
        out=np.zeros_like(scores),
        where=row_range > 1e-12,
    )
    # With a single event memory there is no within-view ranking range. Keep a
    # valid raw match usable, while the raw support threshold still rejects an
    # unrelated candidate.
    flat_rows = np.squeeze(row_range <= 1e-12, axis=1)
    normalized[flat_rows] = 1.0

    if view_count == 1:
        primary_view = 0
        view_weights = np.ones(1, dtype=float)
    else:
        profile_norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        unit_profiles = np.divide(
            normalized,
            profile_norms,
            out=np.zeros_like(normalized),
            where=profile_norms > 1e-12,
        )
        profile_similarity = unit_profiles @ unit_profiles.T
        centrality = (
            np.sum(profile_similarity, axis=1)
            - np.diag(profile_similarity)
        ) / (view_count - 1)
        primary_view = int(np.argmax(centrality))
        primary_weight = float(args.mvra_conrag_primary_weight)
        auxiliary_weight = (1.0 - primary_weight) / (view_count - 1)
        view_weights = np.full(view_count, auxiliary_weight, dtype=float)
        view_weights[primary_view] = primary_weight

    support_counts = np.zeros(candidate_count, dtype=int)
    candidate_mask = np.zeros(candidate_count, dtype=bool)
    per_view_top_k = min(
        int(args.mvra_candidate_top_k_per_view), candidate_count
    )
    for view_index in range(view_count):
        ranked = np.argsort(-scores[view_index])[:per_view_top_k]
        if not len(ranked):
            continue
        top_score = float(scores[view_index, ranked[0]])
        support_floor = max(
            float(args.mvra_view_accept_threshold),
            top_score - float(args.mvra_support_score_gap),
        )
        supported = [
            int(index)
            for index in ranked.tolist()
            if float(scores[view_index, index]) >= support_floor
        ]
        if supported:
            support_counts[supported] += 1
            candidate_mask[supported] = True

    base_scores = view_weights @ normalized
    denominator = max(1, view_count - 1)
    consensus_bonus = 1.0 + float(args.mvra_consensus_bonus) * (
        np.maximum(0, support_counts - 1) / denominator
    )
    fused_scores = base_scores * consensus_bonus
    fused_scores[~candidate_mask] = -np.inf
    return fused_scores, support_counts, view_weights, candidate_mask


def evaluate_ugr_gate(
    supporting_views,
    structural_views,
    raw_text_similarity,
    fused_score,
    margin,
    args,
):
    """Return a label-free UGR decision and auditable rejection reasons."""
    reasons = []
    if supporting_views < args.ugr_min_support_views:
        reasons.append("insufficient_view_support")
    if structural_views < args.ugr_min_structural_views:
        reasons.append("insufficient_structural_support")
    if raw_text_similarity < args.ugr_min_text_similarity:
        reasons.append("low_text_similarity")
    if fused_score < args.ugr_min_fused_score:
        reasons.append("low_fused_score")
    if margin < args.ugr_min_margin:
        reasons.append("low_margin")
    return not reasons, reasons


def social_event_detect(
    detector,
    text,
    events_list,
    event_keywords_dict,
    event_evidence_dict,
    sbert_model,
    block_id,
    args,
    stats,
    anchor_metadata=None,
    event_metadata_dict=None,
    event_assignment_counts=None,
):
    anchor_metadata = anchor_metadata or {}
    event_metadata_dict = event_metadata_dict or {}
    if event_assignment_counts is None:
        event_assignment_counts = {}
    # UGR-MVRA is asymmetric by construction: multi-view evidence may repair
    # an uncertain text-only result, but it may never overwrite a non-Others
    # primary event. This residual protocol limits error propagation in the
    # streaming memory while preserving an auditable matched text control.
    if args.ugr_mode and not getattr(args, "ugr_internal_pass", False):
        primary_args = copy.copy(args)
        primary_args.ugr_internal_pass = True
        primary_args.ugr_enforce_gate = False
        primary_args.enable_mvra = False
        primary_args.conrag_use_entity = False
        primary_args.conrag_use_relation = False
        primary_args.conrag_require_structural_consensus = False
        primary_args.mvra_consensus_gate = False
        primary_args.mvra_confidence_gate = False
        primary_args.mvra_single_view_fallback = False
        primary_args.mvra_override_non_others = False
        primary_args.mvra_require_consensus_for_non_others = False
        primary_result = social_event_detect(
            detector,
            text,
            events_list,
            event_keywords_dict,
            event_evidence_dict,
            sbert_model,
            block_id,
            primary_args,
            stats,
            anchor_metadata=anchor_metadata,
            event_metadata_dict=event_metadata_dict,
            event_assignment_counts=event_assignment_counts,
        )
        stats.ugr_primary_predictions += 1
        primary_event = primary_result.get("EVENT", "Confusion")

        def annotate_ugr_result(
            final_result,
            *,
            attempted=False,
            gate_passed=False,
            gate_candidate=None,
            accepted=False,
            repair_mode="none",
        ):
            """Attach non-prompt audit metadata for paired MVRA evaluation."""
            annotated = dict(final_result)
            annotated["_ugr_primary_event"] = primary_event
            annotated["_ugr_final_event"] = annotated.get(
                "EVENT", "Confusion"
            )
            annotated["_ugr_fallback_attempted"] = bool(attempted)
            annotated["_ugr_gate_passed"] = bool(gate_passed)
            annotated["_ugr_gate_candidate"] = gate_candidate
            annotated["_ugr_repair_accepted"] = bool(accepted)
            annotated["_ugr_repair_mode"] = repair_mode
            return annotated

        fallback_events = {"Others"}
        if args.mvra_fallback_on_confusion:
            fallback_events.add("Confusion")
        if primary_event not in fallback_events:
            stats.ugr_primary_preserved += 1
            return annotate_ugr_result(primary_result)

        has_event_memory = any(
            event not in RESERVED_EVENTS and event in event_keywords_dict
            for event in events_list
        )
        if not has_event_memory:
            stats.ugr_primary_preserved += 1
            stats.ugr_gate_rejections += 1
            stats.ugr_gate_rejections_by_reason["empty_event_memory"] += 1
            return annotate_ugr_result(primary_result)

        stats.ugr_fallback_attempts += 1
        gate_passes_before = stats.ugr_gate_passes
        fallback_args = copy.copy(args)
        fallback_args.ugr_internal_pass = True
        fallback_args.ugr_enforce_gate = True
        fallback_args.enable_mvra = True
        fallback_args.conrag_use_entity = True
        fallback_args.conrag_use_relation = True
        fallback_args.conrag_require_structural_consensus = True
        fallback_args.mvra_consensus_gate = False
        fallback_args.mvra_confidence_gate = False
        fallback_args.mvra_single_view_fallback = False
        fallback_args.mvra_override_non_others = False
        fallback_args.mvra_require_consensus_for_non_others = True
        fallback_result = social_event_detect(
            detector,
            text,
            events_list,
            event_keywords_dict,
            event_evidence_dict,
            sbert_model,
            block_id,
            fallback_args,
            stats,
            anchor_metadata=anchor_metadata,
            event_metadata_dict=event_metadata_dict,
            event_assignment_counts=event_assignment_counts,
        )
        fallback_event = fallback_result.get("EVENT", "Confusion")
        gate_passed = bool(
            fallback_result.get(
                "_ugr_gate_passed",
                stats.ugr_gate_passes > gate_passes_before,
            )
        )
        gate_candidate = fallback_result.get("_ugr_gate_candidate")
        direct_gate = bool(fallback_result.get("_ugr_direct_gate", False))
        if fallback_event not in {"Others", "Confusion"}:
            stats.ugr_fallback_accepts += 1
            if direct_gate:
                stats.ugr_direct_gate_accepts += 1
            if fallback_event != primary_event:
                stats.ugr_prediction_changes += 1
            print(
                "(UGR-MVRA) Accepted a gated residual repair: "
                f"{primary_event!r} -> {fallback_event!r}."
            )
            return annotate_ugr_result(
                fallback_result,
                attempted=True,
                gate_passed=gate_passed,
                gate_candidate=gate_candidate or fallback_event,
                accepted=True,
                repair_mode=(
                    "direct_gate" if direct_gate else "llm_verified"
                ),
            )

        if stats.ugr_gate_passes > gate_passes_before:
            stats.ugr_detector_disagreements += 1
        stats.ugr_fallback_rejections += 1
        print(
            "(UGR-MVRA) No detector-confirmed admissible repair; keeping "
            f"the primary result {primary_event!r}."
        )
        return annotate_ugr_result(
            primary_result,
            attempted=True,
            gate_passed=gate_passed,
            gate_candidate=gate_candidate,
            accepted=False,
            repair_mode=(
                "direct_gate" if direct_gate else "llm_verified"
            ),
        )

    # Conservative MVRA is a fallback, not a replacement for the original
    # single-query detector. It leaves every confident primary decision
    # untouched and spends a second detector call only for Others/Confusion.
    # The fallback result is accepted only when at least two views provide a
    # sufficiently strong and separated consensus. No task label is used.
    if args.mvra_fallback_only:
        primary_args = copy.copy(args)
        primary_args.mvra_fallback_only = False
        primary_args.enable_mvra = False
        primary_args.mvra_consensus_gate = False
        primary_args.mvra_confidence_gate = False
        primary_args.mvra_single_view_fallback = False
        primary_args.mvra_override_non_others = False
        primary_args.mvra_require_consensus_for_non_others = False
        primary_result = social_event_detect(
            detector,
            text,
            events_list,
            event_keywords_dict,
            event_evidence_dict,
            sbert_model,
            block_id,
            primary_args,
            stats,
            anchor_metadata=anchor_metadata,
            event_metadata_dict=event_metadata_dict,
            event_assignment_counts=event_assignment_counts,
        )
        primary_event = primary_result.get("EVENT", "Confusion")
        fallback_events = {"Others"}
        if args.mvra_fallback_on_confusion:
            fallback_events.add("Confusion")
        if primary_event not in fallback_events:
            return primary_result

        stats.mvra_fallback_attempts += 1
        fallback_args = copy.copy(args)
        fallback_args.mvra_fallback_only = False
        fallback_args.enable_mvra = True
        fallback_args.mvra_consensus_gate = True
        fallback_args.mvra_confidence_gate = False
        fallback_args.mvra_single_view_fallback = False
        fallback_args.mvra_override_non_others = False
        fallback_args.mvra_require_consensus_for_non_others = True
        fallback_result = social_event_detect(
            detector,
            text,
            events_list,
            event_keywords_dict,
            event_evidence_dict,
            sbert_model,
            block_id,
            fallback_args,
            stats,
            anchor_metadata=anchor_metadata,
            event_metadata_dict=event_metadata_dict,
            event_assignment_counts=event_assignment_counts,
        )
        fallback_event = fallback_result.get("EVENT", "Confusion")
        if fallback_event not in {"Others", "Confusion"}:
            stats.mvra_fallback_accepts += 1
            print(
                "(Conservative MVRA) Accepted consensus fallback: "
                f"{primary_event!r} -> {fallback_event!r}."
            )
            return fallback_result

        stats.mvra_fallback_rejections += 1
        print(
            "(Conservative MVRA) No admissible cross-view consensus; "
            f"keeping the primary result {primary_event!r}."
        )
        return primary_result

    text_clean = text.replace('"', "")
    if args.variant == "ragsede_original":
        input_for_llm = text_clean[: args.max_input_chars]
        prompt = "INPUT: " + input_for_llm
        for attempt in range(1, args.max_llm_attempts + 1):
            try:
                result = _ask_json(
                    detector, "sed", prompt, "detector", stats
                )
                if "INPUT" not in result or "EVENT" not in result:
                    raise ValueError("Required INPUT or EVENT is missing.")
                if not isinstance(result["EVENT"], str):
                    raise ValueError("EVENT has an invalid type.")
                predicted_event = result["EVENT"].strip().rstrip(".")
                if predicted_event not in set(events_list):
                    raise ValueError(
                        "EVENT is not present in the RagSEDE event memory."
                    )
                result["EVENT"] = predicted_event
                stats.record_llm_success()
                return result
            except LLMBudgetExceeded as exc:
                stats.record_llm_failure(exc)
                print(f"(RagSEDE detector) {exc}")
                return {"INPUT": input_for_llm, "EVENT": "Confusion"}
            except LLMContentPolicyError as exc:
                stats.record_llm_failure(exc)
                stats.content_policy_blocks += 1
                print(
                    "(RagSEDE detector) Content-policy block; "
                    f"using Confusion: {exc}"
                )
                return {"INPUT": input_for_llm, "EVENT": "Confusion"}
            except LLMTransientError as exc:
                stats.record_llm_failure(exc)
                print(
                    f"(RagSEDE detector) {exc}; retry "
                    f"{attempt}/{args.max_llm_attempts}"
                )
                if attempt == args.max_llm_attempts:
                    print(
                        "(RagSEDE detector) Retry limit reached; "
                        "using Confusion."
                    )
                    return {"INPUT": input_for_llm, "EVENT": "Confusion"}
                time.sleep(args.llm_retry_delay)
            except LLMProviderError:
                raise
            except Exception as exc:
                stats.record_llm_failure(exc)
                print(
                    f"(RagSEDE detector) {exc}; retry "
                    f"{attempt}/{args.max_llm_attempts}"
                )
        return {"INPUT": input_for_llm, "EVENT": "Confusion"}

    retrieval_started = time.perf_counter()
    retrieval_text = prepare_llm_text(text_clean, args)
    dynamic_kb_lines = []
    alias_to_event = {}
    ranked_candidates = []
    consensus_support = {}
    consensus_decision = None
    consensus_top_event = None
    candidate_evidence = {}
    object_diagnostics = {}
    gate_result_metadata = {}

    valid_events = [
        event
        for event in events_list
        if event in event_keywords_dict and event != "Confusion"
    ]
    if valid_events:
        keyword_texts = [
            " ".join(
                keyword
                for keyword in event_keywords_dict[event]
                if isinstance(keyword, str) and keyword.strip()
            )
            for event in valid_events
        ]
        if args.object_conrag_mode:
            if args.stage_aware_mode:
                (
                    top_pairs,
                    support_counts,
                    active_views,
                    object_diagnostics,
                ) = stage_aware_selective_rank(
                    retrieval_text,
                    valid_events,
                    event_keywords_dict,
                    event_evidence_dict,
                    event_metadata_dict,
                    event_assignment_counts,
                    anchor_metadata,
                    sbert_model,
                    args,
                    stats,
                )
            else:
                (
                    top_pairs,
                    support_counts,
                    active_views,
                    object_diagnostics,
                ) = object_conrag_rank(
                    retrieval_text,
                    valid_events,
                    event_keywords_dict,
                    event_evidence_dict,
                    sbert_model,
                    args,
                    stats,
                )
            if args.cqc_mode:
                top_pairs, candidate_evidence = (
                    cqc_rerank_shared_candidates(
                        retrieval_text,
                        top_pairs,
                        valid_events,
                        event_keywords_dict,
                        event_evidence_dict,
                        sbert_model,
                        args,
                        stats,
                    )
                )
            consensus_support = {
                valid_events[index]: int(support_counts[index])
                for _, index in top_pairs
            }
            # Object ConRAG changes candidate ranking only. The detector still
            # makes the event decision from the same bounded candidate prompt.
            use_mvra = len(active_views) > 1
            # Ordinary object variants use consensus only as a ranking rule.
            # The UGR residual pass additionally requires an explicit gate so
            # that a fallback event cannot be selected on weak evidence.
            use_consensus = bool(getattr(args, "ugr_enforce_gate", False))
            queries = [retrieval_text[: args.max_input_chars]]
        elif args.aligned_conrag_mode:
            top_pairs, support_counts, active_views = aligned_conrag_rank(
                retrieval_text,
                valid_events,
                event_keywords_dict,
                event_evidence_dict,
                sbert_model,
                args,
                stats,
            )
            consensus_support = {
                valid_events[index]: int(support_counts[index])
                for _, index in top_pairs
            }
            # ConRAG consensus is a ranking bonus only.  It must never enter
            # the legacy hard reassignment gates below.
            use_mvra = len(active_views) > 1
            use_consensus = False
            queries = [retrieval_text[: args.max_input_chars]]
        else:
            kb_texts = [
                event_memory_text(event, event_keywords_dict[event], args)
                for event in valid_events
            ]
            kb_embs = encode_texts(
                sbert_model, kb_texts, stats, convert_to_tensor=True
            )

            use_mvra = args.enable_mvra and block_id >= args.mvra_start_block
            if use_mvra:
                queries = [
                    query.strip()
                    for query in retrieval_text.split(args.view_separator)
                    if query.strip()
                ][: args.view_count]
            else:
                queries = [retrieval_text[: args.max_input_chars]]
            if not queries:
                queries = [retrieval_text[: args.max_input_chars]]
            if args.mvra_fusion == "concat" and len(queries) > 1:
                queries = [args.view_separator.join(queries)]

            query_embs = encode_texts(
                sbert_model, queries, stats, convert_to_tensor=True
            )
            score_matrix = util.cos_sim(query_embs, kb_embs)
            use_consensus = (
                args.mvra_consensus_gate
                and use_mvra
                and len(queries) >= args.mvra_consensus_min_views
            )
            if use_consensus:
                score_array = score_matrix.detach().cpu().numpy()
                if args.mvra_fusion == "conrag":
                    (
                        fused_array,
                        support_counts,
                        _,
                        candidate_mask,
                    ) = conrag_consensus_fusion(score_array, args)
                    stats.mvra_consensus_candidates += int(
                        np.sum(candidate_mask)
                    )
                else:
                    view_top_indices = np.argmax(score_array, axis=1)
                    view_top_scores = np.max(score_array, axis=1)
                    support_counts = np.zeros(len(valid_events), dtype=int)
                    for event_index, view_score in zip(
                        view_top_indices.tolist(), view_top_scores.tolist()
                    ):
                        if float(view_score) >= args.mvra_view_accept_threshold:
                            support_counts[int(event_index)] += 1
                    mean_scores = np.mean(score_array, axis=0)
                    max_scores = np.max(score_array, axis=0)
                    support_ratio = support_counts / max(1, len(queries))
                    # Legacy balanced fusion is retained for controlled ablation.
                    fused_array = (
                        0.45 * mean_scores
                        + 0.45 * max_scores
                        + 0.10 * support_ratio
                    )
                top_indices = np.argsort(-fused_array)[
                    : min(args.retrieval_top_k, len(valid_events))
                ]
                top_pairs = [
                    (float(fused_array[index]), int(index))
                    for index in top_indices
                ]
                consensus_support = {
                    valid_events[index]: int(support_counts[index])
                    for index in top_indices
                }
            elif args.mvra_fusion == "mean":
                fused_scores = torch.mean(score_matrix, dim=0)
                top_results = torch.topk(
                    fused_scores, k=min(args.retrieval_top_k, len(valid_events))
                )
                top_pairs = [
                    (float(score), int(index))
                    for score, index in zip(
                        top_results.values, top_results.indices
                    )
                ]
            else:
                fused_scores = torch.max(score_matrix, dim=0).values
                top_results = torch.topk(
                    fused_scores, k=min(args.retrieval_top_k, len(valid_events))
                )
                top_pairs = [
                    (float(score), int(index))
                    for score, index in zip(
                        top_results.values, top_results.indices
                    )
                ]

        aliases = {
            event: f"Event-{index + 1:04d}"
            for index, event in enumerate(valid_events)
        }
        for score, index in top_pairs:
            if score < args.retrieval_score_threshold:
                continue
            event = valid_events[index]
            visible_event = (
                aliases[event]
                if args.knowledge_control in {"event_alias", "both"}
                else event
            )
            alias_to_event[visible_event] = event
            ranked_candidates.append((visible_event, event, score))
            candidate_line = (
                f"- EVENT: {visible_event}, "
                f"KEYWORDS: {keyword_texts[index]}"
            )
            if event in candidate_evidence:
                evidence = " ".join(
                    candidate_evidence[event].split()
                )[: args.cqc_evidence_chars]
                candidate_line += f", EVIDENCE: {evidence}"
            dynamic_kb_lines.append(candidate_line)
        if use_consensus and ranked_candidates:
            best_visible, best_event, best_score = ranked_candidates[0]
            if len(ranked_candidates) >= 2:
                margin = best_score - ranked_candidates[1][2]
            elif args.mvra_require_consensus_for_non_others:
                # In fallback mode, all runner-up events may legitimately be
                # below the retrieval floor. Treat the floor as the strongest
                # admissible runner-up; the separate multi-view support test
                # still requires evidence from at least two views.
                margin = best_score - args.retrieval_score_threshold
            else:
                margin = float("-inf")
            supporting_views = consensus_support.get(best_event, 0)
            required_views = (
                args.ugr_min_support_views
                if getattr(args, "ugr_enforce_gate", False)
                else args.mvra_consensus_min_views
            )
            if supporting_views >= required_views:
                consensus_top_event = best_event

            gate_reasons = []
            raw_text_similarity = float("-inf")
            structural_views = 0
            total_supporting_views = int(supporting_views)
            total_structural_views = 0
            supporting_view_names = []
            gate_supporting_view_names = []
            selected_view_names = list(
                object_diagnostics.get("selected_views", [])
            )
            if getattr(args, "ugr_enforce_gate", False):
                candidate_index = valid_events.index(best_event)
                text_scores = object_diagnostics.get("event_text_scores")
                structural_scores = object_diagnostics.get(
                    "event_structural_support"
                )
                if text_scores is not None:
                    raw_text_similarity = float(text_scores[candidate_index])
                if structural_scores is not None:
                    structural_views = int(
                        structural_scores[candidate_index]
                    )
                total_support_scores = object_diagnostics.get(
                    "event_total_support"
                )
                if total_support_scores is not None:
                    total_supporting_views = int(
                        total_support_scores[candidate_index]
                    )
                total_structural_scores = object_diagnostics.get(
                    "event_total_structural_support"
                )
                if total_structural_scores is not None:
                    total_structural_views = int(
                        total_structural_scores[candidate_index]
                    )
                support_name_sets = object_diagnostics.get(
                    "event_supporting_view_names"
                )
                if support_name_sets is not None:
                    supporting_view_names = sorted(
                        support_name_sets[candidate_index]
                    )
                gate_support_name_sets = object_diagnostics.get(
                    "event_gate_supporting_view_names"
                )
                if gate_support_name_sets is not None:
                    gate_supporting_view_names = sorted(
                        gate_support_name_sets[candidate_index]
                    )
                gate_passed, gate_reasons = evaluate_ugr_gate(
                    supporting_views,
                    structural_views,
                    raw_text_similarity,
                    best_score,
                    margin,
                    args,
                )
            else:
                gate_passed = (
                    supporting_views >= args.mvra_consensus_min_views
                    and best_score >= args.mvra_consensus_accept_threshold
                    and margin >= args.mvra_consensus_margin
                )

            target_assignment = event_assignment_counts.get(
                best_event, {}
            )
            gate_result_metadata = {
                "_ugr_supporting_views": int(supporting_views),
                "_ugr_total_supporting_views": int(
                    total_supporting_views
                ),
                "_ugr_structural_views": int(structural_views),
                "_ugr_total_structural_views": int(
                    total_structural_views
                ),
                "_ugr_supporting_view_names": supporting_view_names,
                "_ugr_gate_supporting_view_names": (
                    gate_supporting_view_names
                ),
                "_ugr_selected_view_names": selected_view_names,
                "_ugr_raw_text_similarity": (
                    float(raw_text_similarity)
                    if np.isfinite(raw_text_similarity)
                    else None
                ),
                "_ugr_fused_score": float(best_score),
                "_ugr_margin": float(margin),
                "_ugr_target_event_assignment_anchors": int(
                    target_assignment.get("anchors", 0)
                ),
                "_ugr_target_event_assignment_messages": int(
                    target_assignment.get("messages", 0)
                ),
                "_ugr_temporal_selected": "temporal"
                in selected_view_names,
                "_ugr_temporal_supported_candidate": "temporal"
                in supporting_view_names,
                "_ugr_temporal_aux_only": bool(
                    args.stage_temporal_aux_only
                ),
            }

            if gate_passed:
                consensus_decision = (
                    best_visible,
                    best_event,
                    best_score,
                    margin,
                    supporting_views,
                )
                stats.mvra_consensus_opportunities += 1
                if getattr(args, "ugr_enforce_gate", False):
                    stats.ugr_gate_passes += 1
                    stats.ugr_candidate_text_similarity_sum += (
                        raw_text_similarity
                    )
                    stats.ugr_candidate_fused_score_sum += best_score
                    stats.ugr_candidate_margin_sum += margin
            elif getattr(args, "ugr_enforce_gate", False):
                stats.ugr_gate_rejections += 1
                for reason in gate_reasons:
                    stats.ugr_gate_rejections_by_reason[reason] += 1

            if (
                getattr(args, "ugr_enforce_gate", False)
                and args.save_audit_log
            ):
                stats.retrieval_audit_records.append(
                    {
                        "block": stats.block_id,
                        "mode": "ugr_residual_gate",
                        "primary_uncertain": True,
                        "candidate_event": best_event,
                        "supporting_views": int(supporting_views),
                        "total_supporting_views": int(
                            total_supporting_views
                        ),
                        "supporting_view_names": supporting_view_names,
                        "gate_supporting_view_names": (
                            gate_supporting_view_names
                        ),
                        "selected_view_names": selected_view_names,
                        "structural_supporting_views": structural_views,
                        "total_structural_supporting_views": int(
                            total_structural_views
                        ),
                        "raw_text_similarity": (
                            raw_text_similarity
                            if np.isfinite(raw_text_similarity)
                            else None
                        ),
                        "fused_score": float(best_score),
                        "margin": float(margin),
                        "target_event_assignment_anchors": int(
                            target_assignment.get("anchors", 0)
                        ),
                        "target_event_assignment_messages": int(
                            target_assignment.get("messages", 0)
                        ),
                        "temporal_selected": "temporal"
                        in selected_view_names,
                        "temporal_supported_candidate": "temporal"
                        in supporting_view_names,
                        "temporal_aux_only": bool(
                            args.stage_temporal_aux_only
                        ),
                        "accepted_by_gate": bool(gate_passed),
                        "rejection_reasons": gate_reasons,
                    }
                )
        elif use_consensus and getattr(args, "ugr_enforce_gate", False):
            stats.ugr_gate_rejections += 1
            stats.ugr_gate_rejections_by_reason["no_candidate"] += 1
        stats.retrieved_candidates += len(dynamic_kb_lines)
    stats.retrieval_seconds += time.perf_counter() - retrieval_started

    dynamic_kb = "\n".join(dynamic_kb_lines) if dynamic_kb_lines else "Empty."
    allowed_aliases = set(alias_to_event)
    allowed_events = (
        allowed_aliases
        if args.knowledge_control in {"event_alias", "both"}
        else set(alias_to_event.values())
    )
    input_for_llm = prepare_llm_text(
        text_clean[: args.max_input_chars], args
    )
    # Gate-first execution is decision-equivalent to the former implementation:
    # a failed UGR gate could never produce an admissible non-Others fallback,
    # because the post-LLM validator rejected it unconditionally. Returning
    # Others here avoids a detector call while the outer residual wrapper still
    # restores the original text-only result.
    if (
        getattr(args, "ugr_enforce_gate", False)
        and consensus_decision is None
    ):
        stats.ugr_pre_llm_short_circuits += 1
        print(
            "(UGR-MVRA) Evidence gate failed before detector invocation; "
            "skipping the residual LLM call."
        )
        return {
            "INPUT": input_for_llm,
            "EVENT": "Others",
            "_ugr_gate_passed": False,
            "_ugr_gate_candidate": None,
            "_ugr_direct_gate": False,
            **gate_result_metadata,
        }

    # Deterministic gate-only condition.  Candidate generation, view
    # selection and all thresholds are identical to the LLM-verified UGR
    # condition; the sole experimental difference is that a candidate which
    # already passed every frozen label-free gate is accepted directly.  This
    # avoids both provider randomness and the extra residual detector call.
    if (
        getattr(args, "ugr_enforce_gate", False)
        and getattr(args, "ugr_direct_gate", False)
        and consensus_decision is not None
    ):
        _, best_event, best_score, margin, supporting_views = (
            consensus_decision
        )
        print(
            "(UGR direct gate) Accepted the gated candidate "
            f"{best_event!r} (views={supporting_views}, "
            f"score={best_score:.3f}, margin={margin:.3f})."
        )
        return {
            "INPUT": input_for_llm,
            "EVENT": best_event,
            "_ugr_gate_passed": True,
            "_ugr_gate_candidate": best_event,
            "_ugr_direct_gate": True,
            **gate_result_metadata,
        }

    prompt = (
        f"INPUT: {input_for_llm}\n\n"
        f"CURRENT RELEVANT KNOWLEDGE BASE:\n{dynamic_kb}"
    )

    for attempt in range(1, args.max_llm_attempts + 1):
        try:
            result = _ask_json(detector, "sed", prompt, "detector", stats)
            if "INPUT" not in result or "EVENT" not in result:
                raise ValueError("Required INPUT or EVENT is missing.")
            if not isinstance(result["EVENT"], str):
                raise ValueError("EVENT has an invalid type.")
            predicted_event = result["EVENT"].strip().rstrip(".")
            if allowed_events:
                if (
                    predicted_event not in allowed_events
                    and predicted_event != "Others"
                ):
                    raise ValueError("EVENT is not among the retrieved candidates.")
                if (
                    args.knowledge_control in {"event_alias", "both"}
                    and predicted_event in alias_to_event
                ):
                    predicted_event = alias_to_event[predicted_event]
            elif predicted_event != "Others":
                raise ValueError(
                    "No candidate was supplied, so EVENT must be Others."
                )

            # A fallback-only MVRA call may reuse an existing event only when
            # that same event passes the explicit cross-view consensus gate.
            # This prevents a second LLM call from changing a prediction on
            # single-view evidence alone.
            if (
                args.mvra_require_consensus_for_non_others
                and predicted_event != "Others"
                and (
                    consensus_decision is None
                    or predicted_event != consensus_decision[1]
                )
            ):
                print(
                    (
                        "(UGR-MVRA)"
                        if getattr(args, "ugr_enforce_gate", False)
                        else "(Conservative MVRA)"
                    )
                    + " Rejected a non-consensus fallback "
                    f"selection {predicted_event!r}."
                )
                predicted_event = "Others"

            if (
                consensus_top_event is not None
                and predicted_event == consensus_top_event
            ):
                stats.mvra_consensus_agreements += 1

            if (
                predicted_event != "Others"
                and args.mvra_override_non_others
                and consensus_decision is not None
            ):
                (
                    best_visible,
                    best_event,
                    best_score,
                    margin,
                    supporting_views,
                ) = consensus_decision
                selected_support = consensus_support.get(
                    predicted_event, 0
                )
                if (
                    predicted_event != best_event
                    and selected_support < args.mvra_consensus_min_views
                    and best_score >= args.mvra_override_accept_threshold
                    and margin >= args.mvra_override_margin
                ):
                    previous_event = predicted_event
                    predicted_event = best_event
                    stats.mvra_consensus_overrides += 1
                    print(
                        "(MVRA consensus) Replaced a weak single-view "
                        f"selection {previous_event!r} with {best_visible} "
                        f"(views={supporting_views}/{len(queries)}, "
                        f"score={best_score:.3f}, margin={margin:.3f})."
                    )

            # DeepSeek can be conservative and return Others despite one
            # clearly dominant multi-view match.  The refined path accepts the
            # top event only when both an absolute score and a top-1/top-2
            # margin agree.  This decision uses no labels and is fully logged.
            if (
                predicted_event == "Others"
                and args.mvra_consensus_gate
                and consensus_decision is not None
            ):
                (
                    best_visible,
                    best_event,
                    best_score,
                    margin,
                    supporting_views,
                ) = consensus_decision
                predicted_event = best_event
                stats.mvra_consensus_reassignments += 1
                print(
                    "(MVRA consensus) Reassigned conservative Others to "
                    f"{best_visible} (views={supporting_views}/{len(queries)}, "
                    f"score={best_score:.3f}, margin={margin:.3f})."
                )
            elif (
                predicted_event == "Others"
                and args.mvra_confidence_gate
                and len(ranked_candidates) >= args.mvra_gate_min_candidates
            ):
                best_visible, best_event, best_score = ranked_candidates[0]
                runner_up_score = ranked_candidates[1][2]
                margin = best_score - runner_up_score
                if (
                    best_score >= args.mvra_accept_threshold
                    and margin >= args.mvra_accept_margin
                ):
                    predicted_event = best_event
                    stats.mvra_confident_reassignments += 1
                    print(
                        "(MVRA) Reassigned conservative Others to "
                        f"{best_visible} (score={best_score:.3f}, "
                        f"margin={margin:.3f})."
                    )
            elif (
                predicted_event == "Others"
                and args.mvra_single_view_fallback
                and len(queries) < args.mvra_consensus_min_views
                and len(ranked_candidates) >= 2
            ):
                best_visible, best_event, best_score = ranked_candidates[0]
                margin = best_score - ranked_candidates[1][2]
                if (
                    best_score >= args.mvra_single_view_accept_threshold
                    and margin >= args.mvra_single_view_margin
                ):
                    predicted_event = best_event
                    stats.mvra_single_view_reassignments += 1
                    print(
                        "(MVRA single-view) Reassigned conservative Others to "
                        f"{best_visible} (score={best_score:.3f}, "
                        f"margin={margin:.3f})."
                    )
            result["EVENT"] = predicted_event
            if getattr(args, "ugr_enforce_gate", False):
                result["_ugr_gate_passed"] = bool(
                    consensus_decision is not None
                )
                result["_ugr_gate_candidate"] = (
                    consensus_decision[1]
                    if consensus_decision is not None
                    else None
                )
                result["_ugr_direct_gate"] = False
                result.update(gate_result_metadata)
            stats.record_llm_success()
            return result
        except LLMBudgetExceeded as exc:
            stats.record_llm_failure(exc)
            print(f"(Detector) {exc}")
            return {"INPUT": input_for_llm, "EVENT": "Confusion"}
        except LLMContentPolicyError as exc:
            stats.record_llm_failure(exc)
            stats.content_policy_blocks += 1
            print(f"(Detector) Content-policy block; using Confusion: {exc}")
            return {"INPUT": input_for_llm, "EVENT": "Confusion"}
        except LLMTransientError as exc:
            stats.record_llm_failure(exc)
            print(
                f"(Detector) {exc}; retry "
                f"{attempt}/{args.max_llm_attempts}"
            )
            if attempt == args.max_llm_attempts:
                print("(Detector) Retry limit reached; using Confusion.")
                return {"INPUT": input_for_llm, "EVENT": "Confusion"}
            time.sleep(args.llm_retry_delay)
        except LLMProviderError:
            raise
        except Exception as exc:
            stats.record_llm_failure(exc)
            print(f"(Detector) {exc}; retry {attempt}/{args.max_llm_attempts}")

    return {"INPUT": input_for_llm, "EVENT": "Confusion"}


def key_message_sampling(
    anchors, anchors_label, base_threshold, sbert_model, args, stats
):
    started = time.perf_counter()
    if not args.enable_dkms:
        print("\nDKMS disabled for this variant.")
        stats.dkms_seconds += time.perf_counter() - started
        return anchors, anchors_label
    print("\nExecuting Density-aware Key Message Sampling (DKMS)...")
    if not anchors:
        stats.dkms_seconds += time.perf_counter() - started
        return [], []

    embeddings = encode_texts(
        sbert_model, anchors, stats, convert_to_tensor=True
    )
    similarity_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    pair_indices = np.triu_indices_from(similarity_matrix, k=1)
    pairwise_similarities = similarity_matrix[pair_indices]

    if len(pairwise_similarities):
        block_density = float(np.mean(pairwise_similarities))
        dynamic_threshold = (
            base_threshold
            + (block_density - args.dkms_density_reference)
            * args.dkms_density_scale
        )
        dynamic_threshold = float(
            np.clip(
                dynamic_threshold,
                args.dkms_threshold_min,
                args.dkms_threshold_max,
            )
        )
    else:
        block_density = 0.0
        dynamic_threshold = base_threshold

    unfloored_dynamic_threshold = dynamic_threshold
    threshold_floor_applied = bool(
        args.dkms_never_below_base
        and dynamic_threshold < base_threshold
    )
    if args.dkms_never_below_base:
        dynamic_threshold = max(base_threshold, dynamic_threshold)

    print(
        "Block semantic density: "
        f"{block_density:.3f}; DKMS threshold: {dynamic_threshold:.3f}"
        + (
            f" (density-only value {unfloored_dynamic_threshold:.3f}; "
            "safe floor applied)"
            if threshold_floor_applied
            else ""
        )
    )
    stats.dkms_block_density = block_density
    stats.dkms_unfloored_dynamic_threshold = (
        unfloored_dynamic_threshold
    )
    stats.dkms_dynamic_threshold = dynamic_threshold
    stats.dkms_threshold_floor_applied = threshold_floor_applied

    if args.dkms_stage2_linkage == "complete":
        if len(anchors) == 1:
            stage2_groups = [[0]]
        else:
            distance_matrix = np.clip(
                1.0 - similarity_matrix, 0.0, 2.0
            )
            np.fill_diagonal(distance_matrix, 0.0)
            stage2_labels = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="complete",
                distance_threshold=1.0 - dynamic_threshold,
            ).fit_predict(distance_matrix)
            groups_by_label = defaultdict(list)
            for item, label in enumerate(stage2_labels):
                groups_by_label[int(label)].append(item)
            stage2_groups = sorted(
                groups_by_label.values(), key=lambda group: min(group)
            )
        print(
            "Stage-2 grouping: complete-link pairwise-consistent "
            "clustering."
        )
    else:
        # Backward-compatible star grouping used by the historical dynamic
        # variants. It is retained only for controlled ablation.
        visited = set()
        stage2_groups = []
        for index in range(len(anchors)):
            if index in visited:
                continue
            similar_indices = np.where(
                similarity_matrix[index] >= dynamic_threshold
            )[0]
            cluster_indices = [
                item for item in similar_indices if item not in visited
            ] or [index]
            visited.update(cluster_indices)
            stage2_groups.append(cluster_indices)

    stats.dkms_stage2_group_count = len(stage2_groups)
    stats.dkms_stage2_largest_group = max(
        (len(group) for group in stage2_groups), default=0
    )
    within_group_minima = []
    for group in stage2_groups:
        if len(group) < 2:
            continue
        local_similarity = similarity_matrix[np.ix_(group, group)]
        upper = local_similarity[np.triu_indices(len(group), k=1)]
        if len(upper):
            within_group_minima.append(float(np.min(upper)))
    stats.dkms_stage2_min_within_group_similarity = min(
        within_group_minima, default=1.0
    )
    if (
        args.dkms_stage2_linkage == "complete"
        and stats.dkms_stage2_min_within_group_similarity
        < dynamic_threshold - 1e-6
    ):
        raise RuntimeError(
            "Complete-link stage-2 invariant failed: minimum within-group "
            f"similarity {stats.dkms_stage2_min_within_group_similarity:.6f} "
            f"is below threshold {dynamic_threshold:.6f}."
        )

    sampled_anchors = []
    sampled_labels = []

    for cluster_indices in stage2_groups:
        # A sampled anchor represents every original anchor in the DKMS group.
        # Keeping only the first label list (the previous behavior) silently
        # discarded gold instances and biased all reported metrics.
        group_labels = []
        for item in cluster_indices:
            labels = anchors_label[item]
            group_labels.extend(labels if isinstance(labels, list) else [labels])

        if len(cluster_indices) <= args.view_count:
            selected_indices = cluster_indices
        elif args.view_selection == "random":
            selected_indices = random.sample(
                cluster_indices, k=args.view_count
            )
        elif args.view_selection == "first":
            selected_indices = cluster_indices[: args.view_count]
        else:
            cluster_embeddings = embeddings[cluster_indices]
            centroid = torch.mean(cluster_embeddings, dim=0, keepdim=True)
            representativeness = (
                util.cos_sim(cluster_embeddings, centroid)
                .cpu()
                .numpy()
                .flatten()
            )
            cluster_similarity = similarity_matrix[
                np.ix_(cluster_indices, cluster_indices)
            ]
            diversity = []
            for local_index in range(len(cluster_indices)):
                mask = np.ones(len(cluster_indices), dtype=bool)
                mask[local_index] = False
                diversity.append(
                    1.0 - float(np.mean(cluster_similarity[local_index][mask]))
                )
            final_scores = (
                args.dkms_lambda * representativeness
                + (1.0 - args.dkms_lambda) * np.asarray(diversity)
            )
            selected_local = np.argsort(final_scores)[
                -args.view_count :
            ][::-1]
            selected_indices = [
                cluster_indices[item] for item in selected_local
            ]

        sampled_anchors.append(
            args.view_separator.join(anchors[item] for item in selected_indices)
        )
        sampled_labels.append(group_labels)

    print(
        f"DKMS reduced {len(anchors)} anchors to {len(sampled_anchors)} "
        f"with at most L={args.view_count} representative-message views.\n"
    )
    stats.dkms_seconds += time.perf_counter() - started
    return sampled_anchors, sampled_labels


def minimize_structural_entropy(
    similarity_matrix,
    base_threshold,
    entropy_epsilon=1e-5,
    max_community_size=None,
):
    node_count = similarity_matrix.shape[0]
    weights = np.copy(similarity_matrix)
    weights[weights < base_threshold] = 0
    np.fill_diagonal(weights, 0)
    total_volume = np.sum(weights)
    if total_volume == 0:
        return {index: index for index in range(node_count)}

    communities = {index: [index] for index in range(node_count)}
    degrees = np.sum(weights, axis=1)

    def calculate_entropy(partition):
        entropy = 0.0
        for nodes in partition.values():
            volume = np.sum(degrees[nodes])
            if volume == 0:
                continue
            internal_edges = np.sum(weights[np.ix_(nodes, nodes)])
            boundary = volume - internal_edges
            if boundary > 0:
                entropy -= (boundary / total_volume) * np.log2(
                    volume / total_volume
                )
            for node in nodes:
                if degrees[node] > 0:
                    entropy -= (degrees[node] / total_volume) * np.log2(
                        degrees[node] / volume
                    )
        return entropy

    while True:
        best_delta = 0.0
        best_pair = None
        community_ids = list(communities)
        current_entropy = calculate_entropy(communities)
        for left_index in range(len(community_ids)):
            for right_index in range(left_index + 1, len(community_ids)):
                left = community_ids[left_index]
                right = community_ids[right_index]
                left_nodes, right_nodes = communities[left], communities[right]
                if (
                    max_community_size
                    and len(left_nodes) + len(right_nodes)
                    > max_community_size
                ):
                    continue
                if np.sum(weights[np.ix_(left_nodes, right_nodes)]) == 0:
                    continue
                merged = {
                    key: value
                    for key, value in communities.items()
                    if key not in (left, right)
                }
                merged[left] = left_nodes + right_nodes
                delta = calculate_entropy(merged) - current_entropy
                if delta < best_delta:
                    best_delta = delta
                    best_pair = (left, right)
        if best_pair is None or best_delta >= -entropy_epsilon:
            break
        left, right = best_pair
        communities[left].extend(communities[right])
        del communities[right]

    mapping = {}
    for nodes in communities.values():
        canonical_id = min(nodes)
        for node in nodes:
            mapping[node] = canonical_id
    return mapping


def _set_jaccard_matrix(value_sets):
    """Return a deterministic Jaccard matrix for auditable graph objects."""
    count = len(value_sets)
    output = np.zeros((count, count), dtype=float)
    for left in range(count):
        output[left, left] = 1.0 if value_sets[left] else 0.0
        for right in range(left + 1, count):
            union = value_sets[left] | value_sets[right]
            score = (
                len(value_sets[left] & value_sets[right]) / len(union)
                if union
                else 0.0
            )
            output[left, right] = score
            output[right, left] = score
    return output


def consensus_hslg_partition(
    semantic_similarity,
    entity_similarity,
    relation_similarity,
    args,
):
    """Build a conservative multi-view residual event partition.

    A candidate edge must have strong text similarity plus entity or relation
    evidence, must be mutual top-k, and must preserve complete-link evidence
    when components grow. No task labels or clustering metrics enter the rule.
    """
    node_count = semantic_similarity.shape[0]
    identity = {index: index for index in range(node_count)}
    if node_count <= 1:
        return identity, {
            "candidate_edges": 0,
            "mutual_edges": 0,
            "merges": 0,
            "rejections": {},
        }

    text_support = (
        semantic_similarity >= args.hslg_consensus_semantic_threshold
    )
    entity_support = (
        entity_similarity >= args.hslg_consensus_entity_threshold
    )
    relation_support = (
        relation_similarity >= args.hslg_consensus_relation_threshold
    )
    view_support = (
        text_support.astype(int)
        + entity_support.astype(int)
        + relation_support.astype(int)
    )
    structural_support = entity_support | relation_support

    weights = np.asarray(
        [
            args.conrag_text_weight,
            args.conrag_entity_weight,
            args.conrag_relation_weight,
        ],
        dtype=float,
    )
    weights /= max(float(np.sum(weights)), 1e-12)
    fused_similarity = (
        weights[0] * semantic_similarity
        + weights[1] * entity_similarity
        + weights[2] * relation_similarity
    )
    candidate_mask = (
        text_support
        & structural_support
        & (view_support >= args.hslg_consensus_min_views)
        & (fused_similarity >= args.hslg_consensus_score_threshold)
    )
    np.fill_diagonal(candidate_mask, False)
    candidate_edges = int(
        np.sum(candidate_mask[np.triu_indices(node_count, k=1)])
    )

    neighbour_scores = np.where(candidate_mask, fused_similarity, -np.inf)
    neighbour_count = min(
        args.hslg_consensus_top_k, node_count - 1
    )
    top_neighbours = np.argsort(-neighbour_scores, axis=1)[
        :, :neighbour_count
    ]
    neighbour_sets = []
    for node, row in enumerate(top_neighbours):
        neighbour_sets.append(
            {
                int(candidate)
                for candidate in row.tolist()
                if candidate_mask[node, int(candidate)]
            }
        )

    rejection_counts = defaultdict(int)
    mutual_edges = []
    for left, right in zip(*np.triu_indices(node_count, k=1)):
        if not candidate_mask[left, right]:
            continue
        if right not in neighbour_sets[left] or left not in neighbour_sets[right]:
            rejection_counts["not_mutual_top_k"] += 1
            continue
        mutual_edges.append(
            (float(fused_similarity[left, right]), int(left), int(right))
        )
    mutual_edges.sort(reverse=True)

    parents = list(range(node_count))
    members = [{index} for index in range(node_count)]

    def find(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    merges = 0
    for _, left, right in mutual_edges:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        combined_size = len(members[left_root]) + len(members[right_root])
        if combined_size > args.hslg_consensus_max_component_size:
            rejection_counts["component_cap"] += 1
            continue
        # Complete-link validation prevents a chain of individually plausible
        # similarities from joining two events with no direct evidence.
        if any(
            not candidate_mask[left_node, right_node]
            for left_node in members[left_root]
            for right_node in members[right_root]
        ):
            rejection_counts["complete_link"] += 1
            continue
        canonical = min(left_root, right_root)
        other = max(left_root, right_root)
        parents[other] = canonical
        members[canonical] |= members[other]
        members[other] = set()
        merges += 1

    mapping = {node: find(node) for node in range(node_count)}
    return mapping, {
        "candidate_edges": candidate_edges,
        "mutual_edges": len(mutual_edges),
        "merges": merges,
        "rejections": dict(rejection_counts),
    }


def _safe_complete_link_partition(
    score_matrix,
    candidate_mask,
    top_k,
    max_component_size,
):
    """Contract mutual graph edges without transitive-chain over-merging."""
    node_count = score_matrix.shape[0]
    identity = {index: index for index in range(node_count)}
    if node_count <= 1:
        return identity, {
            "mutual_edges": 0,
            "merges": 0,
            "rejections": {},
        }

    candidates = np.asarray(candidate_mask, dtype=bool).copy()
    np.fill_diagonal(candidates, False)
    neighbour_scores = np.where(candidates, score_matrix, -np.inf)
    neighbour_count = min(int(top_k), node_count - 1)
    top_neighbours = np.argsort(-neighbour_scores, axis=1)[
        :, :neighbour_count
    ]
    neighbour_sets = [
        {
            int(candidate)
            for candidate in row.tolist()
            if candidates[node, int(candidate)]
        }
        for node, row in enumerate(top_neighbours)
    ]

    rejection_counts = defaultdict(int)
    mutual_edges = []
    for left, right in zip(*np.triu_indices(node_count, k=1)):
        if not candidates[left, right]:
            continue
        if right not in neighbour_sets[left] or left not in neighbour_sets[right]:
            rejection_counts["not_mutual_top_k"] += 1
            continue
        mutual_edges.append(
            (float(score_matrix[left, right]), int(left), int(right))
        )
    mutual_edges.sort(reverse=True)

    parents = list(range(node_count))
    members = [{index} for index in range(node_count)]

    def find(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    merges = 0
    for _, left, right in mutual_edges:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        combined_size = len(members[left_root]) + len(members[right_root])
        if combined_size > int(max_component_size):
            rejection_counts["component_cap"] += 1
            continue
        if any(
            not candidates[left_node, right_node]
            for left_node in members[left_root]
            for right_node in members[right_root]
        ):
            rejection_counts["complete_link"] += 1
            continue
        canonical = min(left_root, right_root)
        other = max(left_root, right_root)
        parents[other] = canonical
        members[canonical] |= members[other]
        members[other] = set()
        merges += 1

    return {node: find(node) for node in range(node_count)}, {
        "mutual_edges": len(mutual_edges),
        "merges": merges,
        "rejections": dict(rejection_counts),
    }


def buffered_novelty_groups(
    pending_items,
    sbert_model,
    args,
    stats,
):
    """Partition unresolved anchors using only local, label-free evidence."""
    if len(pending_items) <= 1:
        return [[index] for index in range(len(pending_items))], {
            "candidate_edges": 0,
            "mutual_edges": 0,
            "merges": 0,
            "rejections": {},
        }

    texts = [item["text"] for item in pending_items]
    embeddings = encode_texts(
        sbert_model,
        texts,
        stats,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    semantic_similarity = util.cos_sim(embeddings, embeddings).cpu().numpy()

    if args.novelty_buffer_local_hslg:
        entity_sets = [
            set(
                extract_entity_anchors(
                    text,
                    limit=args.conrag_max_view_units,
                )
            )
            for text in texts
        ]
        relation_sets = [
            set(
                extract_relation_units(
                    text,
                    view_separator=args.view_separator,
                    limit=args.conrag_max_view_units,
                )
            )
            for text in texts
        ]
        profile = copy.copy(args)
        profile.hslg_consensus_semantic_threshold = (
            args.novelty_buffer_semantic_threshold
        )
        profile.hslg_consensus_entity_threshold = (
            args.novelty_buffer_entity_threshold
        )
        profile.hslg_consensus_relation_threshold = (
            args.novelty_buffer_relation_threshold
        )
        profile.hslg_consensus_min_views = (
            args.novelty_buffer_min_views
        )
        profile.hslg_consensus_score_threshold = (
            args.novelty_buffer_score_threshold
        )
        profile.hslg_consensus_top_k = args.novelty_buffer_top_k
        profile.hslg_consensus_max_component_size = (
            args.novelty_buffer_max_component_size
        )
        mapping, audit = consensus_hslg_partition(
            semantic_similarity,
            _set_jaccard_matrix(entity_sets),
            _set_jaccard_matrix(relation_sets),
            profile,
        )
    else:
        candidate_mask = (
            semantic_similarity >= args.novelty_buffer_semantic_threshold
        )
        np.fill_diagonal(candidate_mask, False)
        mapping, audit = _safe_complete_link_partition(
            semantic_similarity,
            candidate_mask,
            args.novelty_buffer_top_k,
            args.novelty_buffer_max_component_size,
        )
        audit["candidate_edges"] = int(
            np.sum(
                candidate_mask[
                    np.triu_indices(len(pending_items), k=1)
                ]
            )
        )

    grouped = defaultdict(list)
    for item_index, group_id in mapping.items():
        grouped[int(group_id)].append(int(item_index))
    groups = sorted(
        grouped.values(),
        key=lambda group: (min(group), len(group)),
    )
    return groups, audit


def apply_hslg(
    event_to_id,
    event_keywords_dict,
    event_evidence_dict,
    event_id_list,
    sbert_model,
    args,
    stats,
):
    started = time.perf_counter()
    unique_events = list(event_to_id)
    predictions = np.asarray(event_id_list)
    active_events = [
        event
        for event in unique_events
        if event not in RESERVED_EVENTS and event in event_keywords_dict
    ]
    if args.hslg_mode == "none" or len(active_events) <= 1:
        print("HSLG disabled or unnecessary.")
        stats.hslg_seconds += time.perf_counter() - started
        return predictions, 0

    knowledge_texts = [
        event_memory_text(event, event_keywords_dict.get(event, []), args)
        for event in active_events
    ]
    embeddings = encode_texts(
        sbert_model, knowledge_texts, stats, convert_to_tensor=True
    )
    semantic_similarity = util.cos_sim(embeddings, embeddings).cpu().numpy()
    lexical_similarity = np.zeros_like(semantic_similarity)
    keyword_sets = [
        {
            token
            for keyword in event_keywords_dict.get(event, [])
            if isinstance(keyword, str)
            for token in re.findall(r"[\w'-]+", keyword.casefold())
            if token not in LEXICAL_STOPWORDS and len(token) > 1
        }
        for event in active_events
    ]
    for left, left_set in enumerate(keyword_sets):
        for right, right_set in enumerate(keyword_sets):
            union = left_set | right_set
            if union:
                lexical_similarity[left, right] = len(
                    left_set & right_set
                ) / len(union)

    if args.hslg_mode == "consensus":
        entity_sets = []
        relation_sets = []
        for event in active_events:
            sources = event_evidence_dict.get(event, [])[
                -args.conrag_event_evidence_size :
            ]
            grounded_text = args.view_separator.join(
                source
                for source in sources
                if isinstance(source, str) and source.strip()
            )
            if not grounded_text:
                grounded_text = " ".join(
                    keyword
                    for keyword in event_keywords_dict.get(event, [])
                    if isinstance(keyword, str) and keyword.strip()
                )
            entity_sets.append(
                set(
                    extract_entity_anchors(
                        grounded_text,
                        limit=args.conrag_max_view_units,
                    )
                )
            )
            relation_sets.append(
                set(
                    extract_relation_units(
                        grounded_text,
                        view_separator=args.view_separator,
                        limit=args.conrag_max_view_units,
                    )
                )
            )

        entity_similarity = _set_jaccard_matrix(entity_sets)
        relation_similarity = _set_jaccard_matrix(relation_sets)
        local_mapping, audit = consensus_hslg_partition(
            semantic_similarity,
            entity_similarity,
            relation_similarity,
            args,
        )
        active_ids = [event_to_id[event] for event in active_events]
        global_mapping = {
            event_id: event_id for event_id in event_to_id.values()
        }
        for local_id, global_id in enumerate(active_ids):
            global_mapping[global_id] = active_ids[local_mapping[local_id]]
        predictions = np.asarray(
            [global_mapping[event_id] for event_id in event_id_list]
        )
        merged_count = len(unique_events) - len(
            set(global_mapping.values())
        )
        stats.hslg_entropy_merges = 0
        stats.hslg_reciprocal_merges = 0
        stats.hslg_mutual_knn_merges = audit["merges"]
        stats.hslg_consensus_candidate_edges = audit["candidate_edges"]
        stats.hslg_consensus_mutual_edges = audit["mutual_edges"]
        stats.hslg_consensus_merges = audit["merges"]
        stats.hslg_consensus_rejections = sum(
            audit["rejections"].values()
        )
        stats.hslg_consensus_rejections_by_reason.update(
            audit["rejections"]
        )
        stats.hslg_adaptive_threshold = (
            args.hslg_consensus_score_threshold
        )
        print(
            "HSLG mode=consensus merged "
            f"{merged_count} detected sub-events "
            f"({audit['candidate_edges']} candidate edges, "
            f"{audit['mutual_edges']} mutual edges, "
            f"{audit['merges']} complete-link merges); "
            f"threshold={args.hslg_consensus_score_threshold:.3f}."
        )
        stats.hslg_seconds += time.perf_counter() - started
        return predictions, merged_count

    if args.hslg_mode == "semantic":
        fused_similarity = semantic_similarity
    elif args.hslg_mode == "lexical":
        fused_similarity = lexical_similarity
    else:
        fused_similarity = (
            args.hslg_semantic_weight * semantic_similarity
            + (1.0 - args.hslg_semantic_weight) * lexical_similarity
        )
    graph_similarity = np.array(fused_similarity, copy=True)
    graph_threshold = args.hslg_threshold
    lexical_vetoes = 0
    if args.hslg_adaptive_graph and len(active_events) > 1:
        # Ordinary semantic proximity is insufficient when event memories have
        # unrelated lexical evidence. Balanced mode also estimates its graph
        # quantile only from edges that pass this evidence rule.
        allowed_edges = (
            lexical_similarity >= args.hslg_min_lexical_overlap
        ) | (semantic_similarity >= args.hslg_semantic_override)
        np.fill_diagonal(allowed_edges, True)
        pair_indices = np.triu_indices(len(active_events), k=1)
        pairwise_scores = graph_similarity[pair_indices]
        if args.hslg_allowed_edge_quantile:
            allowed_pair_mask = allowed_edges[pair_indices]
            pairwise_scores = pairwise_scores[allowed_pair_mask]
        if len(pairwise_scores):
            quantile_threshold = float(
                np.quantile(pairwise_scores, args.hslg_adaptive_quantile)
            )
            graph_threshold = float(
                np.clip(
                    max(args.hslg_threshold, quantile_threshold),
                    args.hslg_adaptive_threshold_min,
                    args.hslg_adaptive_threshold_max,
                )
            )

        above_threshold = graph_similarity >= graph_threshold
        veto_mask = above_threshold & ~allowed_edges
        lexical_vetoes = int(
            np.sum(veto_mask[np.triu_indices(len(active_events), k=1)])
        )
        graph_similarity[~allowed_edges] = 0.0
        np.fill_diagonal(graph_similarity, 1.0)

    entropy_mapping = minimize_structural_entropy(
        graph_similarity,
        base_threshold=graph_threshold,
        entropy_epsilon=args.entropy_epsilon,
        max_community_size=(
            args.hslg_max_community_size
            if args.hslg_max_community_size > 0
            else None
        ),
    )
    entropy_merges = len(active_events) - len(set(entropy_mapping.values()))

    # Preserve the structural-entropy partition and add only very high-
    # confidence reciprocal-nearest-neighbour contractions.  This repairs
    # duplicate sub-event names without allowing a hub event to absorb every
    # semantically related event in the block.
    parents = list(range(len(active_events)))
    component_sizes = [1] * len(active_events)

    def find(node):
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(left, right, max_component_size=None):
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return False
        if (
            max_component_size
            and component_sizes[left_root] + component_sizes[right_root]
            > max_component_size
        ):
            return None
        canonical = min(left_root, right_root)
        other = max(left_root, right_root)
        parents[other] = canonical
        component_sizes[canonical] += component_sizes[other]
        return True

    for node, canonical in entropy_mapping.items():
        union(node, canonical)

    reciprocal_merges = 0
    if args.hslg_reciprocal_merge and len(active_events) > 1:
        neighbour_scores = np.array(graph_similarity, copy=True)
        np.fill_diagonal(neighbour_scores, -np.inf)
        nearest = np.argmax(neighbour_scores, axis=1)
        for left, right in enumerate(nearest.tolist()):
            if left >= right or int(nearest[right]) != left:
                continue
            if (
                float(graph_similarity[left, right])
                < args.hslg_reciprocal_threshold
            ):
                continue
            if union(left, right):
                reciprocal_merges += 1

    mutual_knn_merges = 0
    if args.hslg_mutual_knn_merge and len(active_events) > 1:
        neighbour_scores = np.array(graph_similarity, copy=True)
        np.fill_diagonal(neighbour_scores, -np.inf)
        neighbour_count = min(
            args.hslg_mutual_top_k, len(active_events) - 1
        )
        top_neighbours = np.argsort(-neighbour_scores, axis=1)[
            :, :neighbour_count
        ]
        neighbour_sets = [set(row.tolist()) for row in top_neighbours]
        merge_threshold = max(
            args.hslg_mutual_threshold, graph_threshold
        )
        for left in range(len(active_events)):
            for right in sorted(neighbour_sets[left]):
                if left >= right or left not in neighbour_sets[right]:
                    continue
                if float(graph_similarity[left, right]) < merge_threshold:
                    continue
                merged = union(
                    left,
                    right,
                    max_component_size=(
                        args.hslg_max_community_size
                        if args.hslg_max_community_size > 0
                        else None
                    ),
                )
                if merged is None:
                    stats.hslg_component_cap_rejections += 1
                elif merged:
                    mutual_knn_merges += 1

    active_ids = [event_to_id[event] for event in active_events]
    root_to_global = {}
    for local_id, global_id in enumerate(active_ids):
        root = find(local_id)
        root_to_global[root] = min(
            global_id, root_to_global.get(root, global_id)
        )
    global_mapping = {event_id: event_id for event_id in event_to_id.values()}
    for local_id, global_id in enumerate(active_ids):
        global_mapping[global_id] = root_to_global[find(local_id)]

    predictions = np.asarray(
        [global_mapping[event_id] for event_id in event_id_list]
    )
    merged_count = len(unique_events) - len(set(global_mapping.values()))
    stats.hslg_entropy_merges = entropy_merges
    stats.hslg_reciprocal_merges = reciprocal_merges
    stats.hslg_mutual_knn_merges = mutual_knn_merges
    stats.hslg_adaptive_threshold = graph_threshold
    stats.hslg_lexical_vetoes = lexical_vetoes
    print(
        f"HSLG mode={args.hslg_mode} merged {merged_count} detected "
        f"sub-events ({entropy_merges} entropy, "
        f"{reciprocal_merges} reciprocal, "
        f"{mutual_knn_merges} mutual-kNN); "
        f"threshold={graph_threshold:.3f}, lexical vetoes={lexical_vetoes}."
    )
    stats.hslg_seconds += time.perf_counter() - started
    return predictions, merged_count


def get_scores(predictions, labels):
    if len(predictions) != len(labels):
        raise ValueError(
            f"Prediction/label length mismatch: {len(predictions)} vs {len(labels)}"
        )
    score_functions = [
        ("NMI", metrics.normalized_mutual_info_score),
        ("AMI", metrics.adjusted_mutual_info_score),
        ("ARI", metrics.adjusted_rand_score),
    ]
    return {
        name: float(function(labels, predictions))
        for name, function in score_functions
    }


def config_for_log(args):
    excluded = {"API_KEY"}
    return {
        key: value
        for key, value in vars(args).items()
        if key not in excluded and not callable(value)
    }


def safe_identifier(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_for_log():
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit_learn": package_version("scikit-learn"),
        "sentence_transformers": package_version("sentence-transformers"),
        "ragflow_sdk": package_version("ragflow-sdk"),
        "gpu": gpu,
    }


def append_score_to_txt(block_id, scores, filename, args):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a", encoding="utf-8") as output:
        output.write(
            f"Block {block_id} | variant={args.variant} | "
            f"anchor_pipeline={args.anchor_pipeline} | seed={args.seed}\n"
        )
        for metric_name, value in scores.items():
            output.write(f"  {metric_name}: {value:.6f}\n")
        output.write("\n")


def clustering_error_analysis(labels, predictions):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    predicted_to_gold = defaultdict(set)
    gold_to_predicted = defaultdict(set)
    predicted_sizes = defaultdict(int)
    gold_sizes = defaultdict(int)
    for gold, predicted in zip(labels.tolist(), predictions.tolist()):
        predicted_to_gold[str(predicted)].add(str(gold))
        gold_to_predicted[str(gold)].add(str(predicted))
        predicted_sizes[str(predicted)] += 1
        gold_sizes[str(gold)] += 1
    mixed_clusters = {
        cluster: sorted(gold_events)
        for cluster, gold_events in predicted_to_gold.items()
        if len(gold_events) > 1
    }
    fragmented_events = {
        event: sorted(clusters)
        for event, clusters in gold_to_predicted.items()
        if len(clusters) > 1
    }
    return {
        "mixed_predicted_clusters": len(mixed_clusters),
        "messages_in_mixed_clusters": sum(
            predicted_sizes[cluster] for cluster in mixed_clusters
        ),
        "fragmented_gold_events": len(fragmented_events),
        "messages_in_fragmented_events": sum(
            gold_sizes[event] for event in fragmented_events
        ),
        "mixed_cluster_details": mixed_clusters,
        "fragmented_event_details": fragmented_events,
    }


def paired_mvra_analysis(
    labels,
    primary_predictions,
    post_mvra_predictions,
    repair_records,
):
    """Evaluate MVRA before/after on one shared streaming trajectory.

    The arrays are populated inside the same run, so evaluator memory, primary
    LLM decisions and provider failures are shared.  Gold labels are consumed
    only here, after inference, to score rather than to select a repair.
    Per-repair helpful/harmful labels are leave-one-repair-out ARI effects
    conditional on all other accepted repairs.
    """
    labels = np.asarray(labels)
    primary_values = [str(value) for value in primary_predictions]
    post_values = [str(value) for value in post_mvra_predictions]
    max_label_chars = max(
        [1]
        + [len(value) for value in primary_values]
        + [len(value) for value in post_values]
    )
    shared_dtype = f"<U{max_label_chars}"
    primary = np.asarray(primary_values, dtype=shared_dtype)
    post_mvra = np.asarray(post_values, dtype=shared_dtype)
    if len(primary) != len(labels) or len(post_mvra) != len(labels):
        raise ValueError(
            "Paired MVRA prediction/label length mismatch: "
            f"primary={len(primary)}, post={len(post_mvra)}, "
            f"labels={len(labels)}"
        )

    pre_scores = get_scores(primary, labels)
    post_scores = get_scores(post_mvra, labels)
    scored_records = []
    helpful = 0
    harmful = 0
    neutral = 0
    for raw_record in repair_records:
        record = dict(raw_record)
        if record.get("accepted"):
            start = int(record["message_start"])
            end = int(record["message_end"])
            counterfactual = np.array(post_mvra, copy=True)
            counterfactual[start:end] = primary[start:end]
            counterfactual_scores = get_scores(counterfactual, labels)
            metric_delta = {
                metric: post_scores[metric] - counterfactual_scores[metric]
                for metric in ("NMI", "AMI", "ARI")
            }
            record["metric_delta_if_applied"] = metric_delta
            ari_delta = metric_delta["ARI"]
            if ari_delta > 1e-12:
                outcome = "helpful"
                helpful += 1
            elif ari_delta < -1e-12:
                outcome = "harmful"
                harmful += 1
            else:
                outcome = "neutral"
                neutral += 1
            record["posthoc_outcome"] = outcome
        else:
            record["metric_delta_if_applied"] = None
            record["posthoc_outcome"] = "not_accepted"
        scored_records.append(record)

    changed_mask = primary != post_mvra
    accepted_repairs = sum(
        1 for record in repair_records if record.get("accepted")
    )
    return {
        "pre_mvra_macro_scores": pre_scores,
        "post_mvra_macro_scores": post_scores,
        "macro_score_delta": {
            metric: post_scores[metric] - pre_scores[metric]
            for metric in ("NMI", "AMI", "ARI")
        },
        "pre_mvra_predicted_cluster_count": len(set(primary.tolist())),
        "post_mvra_predicted_cluster_count": len(
            set(post_mvra.tolist())
        ),
        "changed_predictions": int(np.sum(changed_mask)),
        "changed_anchors": sum(
            1 for record in repair_records if record.get("accepted")
        ),
        "accepted_repairs": accepted_repairs,
        "helpful_repairs": helpful,
        "harmful_repairs": harmful,
        "neutral_repairs": neutral,
        "repair_records": scored_records,
    }


def save_block_outputs(
    args,
    block_id,
    scores,
    stats,
    labels,
    predictions,
    merged_count,
    pre_hslg_predictions=None,
    hslg_state=None,
    primary_mvra_predictions=None,
    post_mvra_predictions=None,
    mvra_paired_result=None,
):
    block_dir = Path("ckpts") / args.dataset / f"M{block_id}"
    block_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"M{block_id}_{safe_identifier(args.variant)}_"
        f"{safe_identifier(args.anchor_pipeline)}_s{args.seed}"
    )
    result = {
        "dataset": args.dataset,
        "block": block_id,
        "variant": args.variant,
        "anchor_pipeline": args.anchor_pipeline,
        "seed": args.seed,
        "macro_scores": scores,
        "efficiency": stats.finish(),
        "merged_sub_events": merged_count,
        "configuration": config_for_log(args),
        "environment": environment_for_log(),
        "gold_cluster_count": len(set(np.asarray(labels).tolist())),
        "predicted_cluster_count": len(
            set(np.asarray(predictions).tolist())
        ),
        "pre_hslg_predicted_cluster_count": (
            len(set(np.asarray(pre_hslg_predictions).tolist()))
            if pre_hslg_predictions is not None
            else None
        ),
        "pre_hslg_macro_scores": (
            get_scores(np.asarray(pre_hslg_predictions), np.asarray(labels))
            if pre_hslg_predictions is not None
            else None
        ),
        "pre_mvra_macro_scores": (
            mvra_paired_result.get("pre_mvra_macro_scores")
            if mvra_paired_result is not None
            else None
        ),
        "post_mvra_macro_scores": (
            mvra_paired_result.get("post_mvra_macro_scores")
            if mvra_paired_result is not None
            else None
        ),
        "mvra_macro_score_delta": (
            mvra_paired_result.get("macro_score_delta")
            if mvra_paired_result is not None
            else None
        ),
        "pre_mvra_predicted_cluster_count": (
            mvra_paired_result.get("pre_mvra_predicted_cluster_count")
            if mvra_paired_result is not None
            else None
        ),
        "post_mvra_predicted_cluster_count": (
            mvra_paired_result.get("post_mvra_predicted_cluster_count")
            if mvra_paired_result is not None
            else None
        ),
        "changed_predictions": (
            mvra_paired_result.get("changed_predictions", 0)
            if mvra_paired_result is not None
            else 0
        ),
        "changed_anchors": (
            mvra_paired_result.get("changed_anchors", 0)
            if mvra_paired_result is not None
            else 0
        ),
        "accepted_repairs": (
            mvra_paired_result.get("accepted_repairs", 0)
            if mvra_paired_result is not None
            else 0
        ),
        "helpful_repairs": (
            mvra_paired_result.get("helpful_repairs", 0)
            if mvra_paired_result is not None
            else 0
        ),
        "harmful_repairs": (
            mvra_paired_result.get("harmful_repairs", 0)
            if mvra_paired_result is not None
            else 0
        ),
        "neutral_repairs": (
            mvra_paired_result.get("neutral_repairs", 0)
            if mvra_paired_result is not None
            else 0
        ),
        "repair_records": (
            mvra_paired_result.get("repair_records", [])
            if mvra_paired_result is not None
            else []
        ),
        "pre_mvra_clustering_error_analysis": (
            clustering_error_analysis(labels, primary_mvra_predictions)
            if primary_mvra_predictions is not None
            else None
        ),
        "post_mvra_clustering_error_analysis": (
            clustering_error_analysis(labels, post_mvra_predictions)
            if post_mvra_predictions is not None
            else None
        ),
        "clustering_error_analysis": clustering_error_analysis(
            labels, predictions
        ),
    }
    with open(block_dir / f"{stem}_metrics.json", "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    with open(
        block_dir / f"{stem}_predictions.json", "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "gold_labels": np.asarray(labels).tolist(),
                "pre_hslg_predicted_clusters": (
                    np.asarray(pre_hslg_predictions).tolist()
                    if pre_hslg_predictions is not None
                    else None
                ),
                "pre_mvra_predicted_clusters": (
                    np.asarray(primary_mvra_predictions).tolist()
                    if primary_mvra_predictions is not None
                    else None
                ),
                "post_mvra_predicted_clusters": (
                    np.asarray(post_mvra_predictions).tolist()
                    if post_mvra_predictions is not None
                    else None
                ),
                "predicted_clusters": np.asarray(predictions).tolist(),
            },
            file,
            ensure_ascii=False,
        )
    if args.save_intermediate_state and hslg_state is not None:
        with open(
            block_dir / f"{stem}_hslg_state.json", "w", encoding="utf-8"
        ) as file:
            json.dump(hslg_state, file, ensure_ascii=False, indent=2)
    if args.save_audit_log:
        with open(
            block_dir / f"{stem}_llm_audit.json", "w", encoding="utf-8"
        ) as file:
            json.dump(stats.audit_records, file, ensure_ascii=False, indent=2)
        if stats.retrieval_audit_records:
            with open(
                block_dir / f"{stem}_retrieval_audit.json",
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    stats.retrieval_audit_records,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )
    append_score_to_txt(
        block_id,
        scores,
        filename=str(block_dir / f"M{block_id}_scores.txt"),
        args=args,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_llm_free_block(args, block_id, anchors, anchors_label, model, stats):
    if not anchors:
        return np.asarray([]), np.asarray([]), 0
    embeddings = encode_texts(
        model,
        anchors,
        stats,
        convert_to_tensor=True,
        normalize_embeddings=True,
    ).cpu()
    centroids = []
    counts = []
    predictions = []
    labels = []
    for index, embedding in enumerate(embeddings):
        current_labels = (
            anchors_label[index]
            if isinstance(anchors_label[index], list)
            else [anchors_label[index]]
        )
        labels.extend(current_labels)
        if not centroids:
            cluster_id = 0
            centroids.append(embedding.clone())
            counts.append(1)
        else:
            centroid_matrix = torch.stack(centroids)
            similarities = util.cos_sim(
                embedding.unsqueeze(0), centroid_matrix
            )[0]
            best_score, best_index = torch.max(similarities, dim=0)
            if float(best_score) >= args.llm_free_threshold:
                cluster_id = int(best_index)
                counts[cluster_id] += 1
                updated = (
                    centroids[cluster_id] * (counts[cluster_id] - 1) + embedding
                ) / counts[cluster_id]
                centroids[cluster_id] = torch.nn.functional.normalize(
                    updated, dim=0
                )
            else:
                cluster_id = len(centroids)
                centroids.append(embedding.clone())
                counts.append(1)
        predictions.extend([cluster_id] * len(current_labels))
    stats.messages = len(labels)
    return np.asarray(predictions), np.asarray(labels), 0


def choose_device(requested):
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def selected_blocks(blocks, args):
    # The original pipeline treats blocks[0] as initialization and evaluates
    # blocks[1:] as M1, M2, ... . Keep that mapping explicit and configurable.
    evaluation_blocks = blocks[1:] if args.skip_initial_block else blocks
    for block_id, block in enumerate(evaluation_blocks, start=1):
        if block_id < args.start_block:
            continue
        if args.end_block is not None and block_id > args.end_block:
            continue
        yield block_id, block


def start_run(args, blocks):
    device = choose_device(args.embedding_device)
    print(f"Loading local encoder '{args.local_embedding_model}' on {device}...")
    model_started = time.perf_counter()
    local_model = SentenceTransformer(args.local_embedding_model, device=device)
    model_initialization_seconds = time.perf_counter() - model_started
    ragflow = None
    if args.variant != "llm_free" and not args.anchor_only:
        from RAGFlow import RAGFlowInstance

        ragflow = RAGFlowInstance(
            args.HOST_ADDRESS,
            args.API_KEY,
            llm_model=args.llm_model,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            ragflow_top_n=args.ragflow_top_n,
            ragflow_top_k=args.ragflow_top_k,
            dataset_embedding_model=args.ragflow_embedding_model,
            dataset_ready_attempts=args.ragflow_ready_attempts,
            dataset_ready_delay=args.ragflow_retry_delay,
        )

    blocks_to_run = list(selected_blocks(blocks, args))
    for run_index, (block_id, block) in enumerate(blocks_to_run):
        set_reproducible(args.seed)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        stats = RunStats(args, block_id)
        stats.shared_model_initialization_seconds = (
            model_initialization_seconds if run_index == 0 else 0.0
        )

        anchor_started = time.perf_counter()
        stage1_anchors, stage1_anchor_ids, stage1_anchor_labels = get_anchers(
            args,
            block_id,
            block,
            embedding_model=local_model,
            stats=stats,
        )
        stage1_anchor_metadata = (
            build_anchor_metadata(
                stage1_anchor_ids,
                block,
                args,
            )
            if args.stage_aware_mode
            else [{} for _ in stage1_anchors]
        )
        stats.anchor_seconds = time.perf_counter() - anchor_started
        stats.input_messages = len(block.get("test", []))
        stats.stage1_anchors = len(stage1_anchors)
        stats.raw_anchors = len(stage1_anchors)
        base_threshold = (
            args.twitter12_threshold
            if args.dataset == "twitter12"
            else (
                args.twitter18_threshold
                if args.dataset == "twitter18"
                else args.dataset_threshold
            )
        )
        if args.anchor_pipeline in {"dynamic_two_stage", "legacy_two_stage"}:
            anchors, anchor_labels = key_message_sampling(
                stage1_anchors,
                stage1_anchor_labels,
                args.dkms_base_threshold
                if args.dkms_base_threshold is not None
                else base_threshold,
                local_model,
                args,
                stats,
            )
            anchor_metadata = [{} for _ in anchors]
        else:
            # DKMS has already been applied directly to raw messages inside
            # get_anchers(). Do not re-cluster the generated super-anchors.
            anchors, anchor_labels = stage1_anchors, stage1_anchor_labels
            anchor_metadata = stage1_anchor_metadata
            print(
                "Single-stage anchor pipeline: second-stage anchor "
                "re-clustering is disabled.\n"
            )
        stats.anchors = len(anchors)
        stats.covered_messages = sum(
            len(labels) if isinstance(labels, list) else 1
            for labels in anchor_labels
        )
        if stats.covered_messages != stats.input_messages:
            raise RuntimeError(
                "Anchor label coverage mismatch: "
                f"covered {stats.covered_messages} of "
                f"{stats.input_messages} input messages."
            )
        print(
            "Evaluation coverage check: "
            f"{stats.covered_messages}/{stats.input_messages} messages "
            "covered exactly once.\n"
        )

        if args.expected_anchor_audit:
            with open(args.expected_anchor_audit, encoding="utf-8") as file:
                expected = json.load(file)
            if (expected.get("dataset"), expected.get("block"), expected.get("seed")) != (args.dataset, block_id, args.seed):
                raise ValueError("Expected anchor-audit identity does not match this run")
            validate_anchor_audit(stats.anchor_audit, expected["anchor_audit"])
        if args.anchor_only:
            block_dir = Path("ckpts") / args.dataset / f"M{block_id}"
            block_dir.mkdir(parents=True, exist_ok=True)
            path = block_dir / (
                f"M{block_id}_{safe_identifier(args.variant)}_"
                f"{safe_identifier(args.anchor_pipeline)}_s{args.seed}_anchor_audit.json"
            )
            with path.open("w", encoding="utf-8") as file:
                json.dump({"dataset": args.dataset, "block": block_id, "seed": args.seed,
                           "variant": args.variant, "anchor_audit": stats.anchor_audit,
                           "configuration": config_for_log(args), "environment": environment_for_log()},
                          file, ensure_ascii=False, indent=2)
            print(f"Anchor-only audit saved: {path}; no LLM calls made.")
            continue

        if args.variant == "llm_free":
            predictions, labels, merged_count = run_llm_free_block(
                args, block_id, anchors, anchor_labels, local_model, stats
            )
            scores = get_scores(predictions, labels)
            save_block_outputs(
                args,
                block_id,
                scores,
                stats,
                labels,
                predictions,
                merged_count,
            )
            continue

        unique_suffix = (
            f"{time.time_ns()}_{safe_identifier(args.variant)}_s{args.seed}"
        )
        initialization_started = time.perf_counter()
        evaluator = ragflow.creat_evaluator(
            name=f"{args.dataset}_Evaluator_M{block_id}_{unique_suffix}",
            evaluator_prompt=(
                RAGSEDE_ORIGINAL_EVALUATOR_PROMPT
                if args.variant == "ragsede_original"
                else args.evaluator_prompt
            ),
            evaluator_presence_penalty=args.presence_penalty,
            evaluator_frequency_penalty=args.frequency_penalty,
            evaluator_temperature=args.temperature,
        )
        if args.chat_ready_delay:
            time.sleep(args.chat_ready_delay)
        document, events_list = ragflow.creat_dataset(
            name=f"{args.dataset}_SED_M{block_id}_{unique_suffix}"
        )
        event_keywords = {}
        event_messages = {}
        event_evidence = {}
        event_metadata = {}
        event_assignment_counts = {}

        if anchors:
            events_list, event_name, keywords = kb_update(
                evaluator,
                document,
                anchors[0],
                events_list,
                args,
                stats,
            )
            if event_name != "Confusion":
                remember_event_keywords(
                    event_keywords, event_name, keywords, args
                )
                remember_event_evidence(
                    event_evidence, event_name, anchors[0], args
                )
                remember_event_metadata(
                    event_metadata,
                    event_name,
                    anchor_metadata[0],
                    args,
                )

        detector = None
        for attempt in range(1, args.ragflow_ready_attempts + 1):
            try:
                detector = ragflow.creat_detector(
                    name=f"{args.dataset}_Detector_M{block_id}_{unique_suffix}",
                    detector_prompt=(
                        RAGSEDE_ORIGINAL_DETECTOR_PROMPT
                        if args.variant == "ragsede_original"
                        else args.detector_prompt
                    ),
                    detector_similarity_threshold=args.detector_similarity_threshold,
                    detector_keywords_similarity_weight=(
                        args.detector_keywords_similarity_weight
                    ),
                    detector_presence_penalty=args.presence_penalty,
                    detector_frequency_penalty=args.frequency_penalty,
                    detector_temperature=args.temperature,
                    attach_dataset=(args.variant == "ragsede_original"),
                )
                break
            except Exception as exc:
                if attempt == args.ragflow_ready_attempts:
                    raise
                print(
                    f"RAGFlow not ready ({exc}); retry "
                    f"{attempt}/{args.ragflow_ready_attempts}"
                )
                time.sleep(args.ragflow_retry_delay)
        if detector is None:
            raise TimeoutError("RAGFlow detector creation timed out.")
        if args.detector_ready_delay:
            time.sleep(args.detector_ready_delay)
        stats.initialization_seconds = (
            time.perf_counter() - initialization_started
        )

        labels = []
        event_ids = []
        event_to_id = {}
        pending_novelty = []
        primary_mvra_event_names = []
        post_mvra_event_names = []
        mvra_repair_records = []

        def process_novelty_buffer(force=False, current_index=None):
            nonlocal events_list, pending_novelty
            if not args.novelty_buffer_mode or not pending_novelty:
                return
            buffer_started = time.perf_counter()
            groups, audit = buffered_novelty_groups(
                pending_novelty,
                local_model,
                args,
                stats,
            )
            stats.novelty_buffer_candidate_edges += int(
                audit.get("candidate_edges", 0)
            )
            stats.novelty_buffer_mutual_edges += int(
                audit.get("mutual_edges", 0)
            )
            for reason, count in audit.get("rejections", {}).items():
                stats.novelty_buffer_rejections_by_reason[reason] += int(
                    count
                )

            confirmed_groups = [
                group
                for group in groups
                if force
                or len(group) >= args.novelty_buffer_min_cluster_anchors
            ]
            consumed = set()
            for group in confirmed_groups:
                group_items = [pending_novelty[index] for index in group]
                events_list, _ = materialize_novelty_group(
                    group_items,
                    evaluator,
                    document,
                    events_list,
                    event_keywords,
                    event_evidence,
                    event_messages,
                    event_to_id,
                    event_ids,
                    args,
                    stats,
                )
                consumed.update(group)
            if consumed:
                pending_novelty = [
                    item
                    for index, item in enumerate(pending_novelty)
                    if index not in consumed
                ]

            # A bounded buffer preserves streaming behaviour. An unresolved
            # oldest anchor is materialized alone only after capacity/TTL is
            # reached; it is never forced into a weak graph component.
            while pending_novelty and not force:
                oldest_age = (
                    int(current_index)
                    - int(pending_novelty[0]["arrival_index"])
                    if current_index is not None
                    else 0
                )
                must_flush = (
                    len(pending_novelty) > args.novelty_buffer_max_anchors
                    or oldest_age >= args.novelty_buffer_ttl_anchors
                )
                if not must_flush:
                    break
                oldest = pending_novelty.pop(0)
                events_list, _ = materialize_novelty_group(
                    [oldest],
                    evaluator,
                    document,
                    events_list,
                    event_keywords,
                    event_evidence,
                    event_messages,
                    event_to_id,
                    event_ids,
                    args,
                    stats,
                )
            stats.novelty_buffer_seconds += (
                time.perf_counter() - buffer_started
            )

        print(f"Testing data block M{block_id}...")
        for index, message in enumerate(
            tqdm(anchors, desc="Processing", unit="anchor")
        ):
            message_started = time.perf_counter()
            current_metadata = anchor_metadata[index]
            current_labels = (
                anchor_labels[index]
                if isinstance(anchor_labels[index], list)
                else [anchor_labels[index]]
            )
            labels.extend(current_labels)
            result = social_event_detect(
                detector,
                message,
                events_list,
                event_keywords,
                event_evidence,
                local_model,
                block_id,
                args,
                stats,
                anchor_metadata=current_metadata,
                event_metadata_dict=event_metadata,
                event_assignment_counts=event_assignment_counts,
            )
            detected_event = result.get("EVENT", "Confusion")
            primary_event = result.get(
                "_ugr_primary_event", detected_event
            )
            repair_attempted = bool(
                result.get("_ugr_fallback_attempted", False)
            )
            repair_accepted = bool(
                result.get("_ugr_repair_accepted", False)
            )
            if result["EVENT"] == "Others":
                stats.others_predictions += 1
                if args.novelty_buffer_mode:
                    start_position = len(event_ids)
                    positions = list(
                        range(
                            start_position,
                            start_position + len(current_labels),
                        )
                    )
                    # Negative placeholders are internal only and are fully
                    # backfilled before scoring or output serialization.
                    event_ids.extend([-1] * len(current_labels))
                    pending_novelty.append(
                        {
                            "text": message,
                            "positions": positions,
                            "arrival_index": index,
                        }
                    )
                    stats.novelty_buffered_anchors += 1
                    stats.novelty_buffer_peak_size = max(
                        stats.novelty_buffer_peak_size,
                        len(pending_novelty),
                    )
                    process_novelty_buffer(
                        force=False,
                        current_index=index,
                    )
                    stats.message_latencies.append(
                        time.perf_counter() - message_started
                    )
                    continue
                update_context = (
                    message
                    if args.variant == "ragsede_original"
                    else " ".join(
                        event_messages.get("Others", [])[
                            -args.update_context_size :
                        ]
                        + [message]
                    )
                )
                events_list, event_name, keywords = kb_update(
                    evaluator,
                    document,
                    update_context,
                    events_list,
                    args,
                    stats,
                )
                result["EVENT"] = event_name
                if event_name != "Confusion":
                    remember_event_keywords(
                        event_keywords, event_name, keywords, args
                    )

            event_name = result["EVENT"]
            if args.stage_aware_mode:
                message_start = len(post_mvra_event_names)
                message_end = message_start + len(current_labels)
                # A primary Others result would enter the novelty/materialize
                # path in the matched text condition.  When MVRA repairs it to
                # an existing event, represent that counterfactual as one new
                # anchor-level event without making another evaluator call or
                # mutating the shared streaming memory.
                if primary_event == "Others" and repair_accepted:
                    primary_name = (
                        f"__PRIMARY_NOVEL_M{block_id}_A{index:06d}"
                    )
                elif primary_event == "Others":
                    primary_name = event_name
                else:
                    primary_name = primary_event
                primary_mvra_event_names.extend(
                    [primary_name] * len(current_labels)
                )
                post_mvra_event_names.extend(
                    [event_name] * len(current_labels)
                )
                if repair_attempted:
                    mvra_repair_records.append(
                        {
                            "anchor_index": index,
                            "message_start": message_start,
                            "message_end": message_end,
                            "message_count": len(current_labels),
                            "primary_event": primary_event,
                            "gate_candidate": result.get(
                                "_ugr_gate_candidate"
                            ),
                            "final_event": event_name,
                            "gate_passed": bool(
                                result.get("_ugr_gate_passed", False)
                            ),
                            "accepted": repair_accepted,
                            "repair_mode": result.get(
                                "_ugr_repair_mode", "none"
                            ),
                            "supporting_views": result.get(
                                "_ugr_supporting_views"
                            ),
                            "total_supporting_views": result.get(
                                "_ugr_total_supporting_views"
                            ),
                            "supporting_view_names": result.get(
                                "_ugr_supporting_view_names", []
                            ),
                            "gate_supporting_view_names": result.get(
                                "_ugr_gate_supporting_view_names", []
                            ),
                            "selected_view_names": result.get(
                                "_ugr_selected_view_names", []
                            ),
                            "structural_supporting_views": result.get(
                                "_ugr_structural_views"
                            ),
                            "total_structural_supporting_views": result.get(
                                "_ugr_total_structural_views"
                            ),
                            "raw_text_similarity": result.get(
                                "_ugr_raw_text_similarity"
                            ),
                            "fused_score": result.get(
                                "_ugr_fused_score"
                            ),
                            "margin": result.get("_ugr_margin"),
                            "target_event_assignment_anchors": result.get(
                                "_ugr_target_event_assignment_anchors"
                            ),
                            "target_event_assignment_messages": result.get(
                                "_ugr_target_event_assignment_messages"
                            ),
                            "temporal_selected": bool(
                                result.get("_ugr_temporal_selected", False)
                            ),
                            "temporal_supported_candidate": bool(
                                result.get(
                                    "_ugr_temporal_supported_candidate",
                                    False,
                                )
                            ),
                            "temporal_aux_only": bool(
                                result.get(
                                    "_ugr_temporal_aux_only", False
                                )
                            ),
                        }
                    )
            if event_name == "Confusion":
                stats.confusion_predictions += 1
            if event_name not in event_to_id:
                event_to_id[event_name] = len(event_to_id)
            event_ids.extend([event_to_id[event_name]] * len(current_labels))
            assignment_count = event_assignment_counts.setdefault(
                event_name, {"anchors": 0, "messages": 0}
            )
            assignment_count["anchors"] += 1
            assignment_count["messages"] += len(current_labels)
            event_messages.setdefault(event_name, []).append(message)
            remember_event_evidence(
                event_evidence, event_name, message, args
            )
            remember_event_metadata(
                event_metadata,
                event_name,
                current_metadata,
                args,
            )

            if (
                len(event_messages[event_name])
                >= args.max_messages_per_event
            ):
                update_text = " ".join(event_messages[event_name])
                _, updated_name, keywords = kb_update(
                    evaluator,
                    document,
                    update_text,
                    events_list,
                    args,
                    stats,
                    event_name=event_name,
                )
                if updated_name != "Confusion":
                    remember_event_keywords(
                        event_keywords, updated_name, keywords, args
                    )
                event_messages[event_name] = []
            process_novelty_buffer(
                force=False,
                current_index=index,
            )
            stats.message_latencies.append(
                time.perf_counter() - message_started
            )

        process_novelty_buffer(force=True, current_index=len(anchors))
        if any(event_id < 0 for event_id in event_ids):
            raise RuntimeError(
                "Novelty-buffer invariant failed: unresolved prediction IDs."
            )
        stats.messages = len(labels)
        pre_hslg_predictions = np.asarray(event_ids)
        mvra_paired_result = None
        primary_mvra_predictions = None
        post_mvra_predictions = None
        if args.stage_aware_mode:
            primary_mvra_predictions = np.asarray(
                primary_mvra_event_names, dtype=str
            )
            post_mvra_predictions = np.asarray(
                post_mvra_event_names, dtype=str
            )
            mvra_paired_result = paired_mvra_analysis(
                labels,
                primary_mvra_predictions,
                post_mvra_predictions,
                mvra_repair_records,
            )
            delta = mvra_paired_result["macro_score_delta"]
            print(
                "Paired MVRA evaluation on shared primary decisions: "
                f"accepted={mvra_paired_result['accepted_repairs']}, "
                f"helpful={mvra_paired_result['helpful_repairs']}, "
                f"harmful={mvra_paired_result['harmful_repairs']}, "
                f"delta_NMI={delta['NMI']:+.6f}, "
                f"delta_AMI={delta['AMI']:+.6f}, "
                f"delta_ARI={delta['ARI']:+.6f}."
            )
        predictions, merged_count = apply_hslg(
            event_to_id,
            event_keywords,
            event_evidence,
            event_ids,
            local_model,
            args,
            stats,
        )
        labels = np.asarray(labels)
        scores = get_scores(predictions, labels)
        save_block_outputs(
            args,
            block_id,
            scores,
            stats,
            labels,
            predictions,
            merged_count,
            pre_hslg_predictions=pre_hslg_predictions,
            primary_mvra_predictions=primary_mvra_predictions,
            post_mvra_predictions=post_mvra_predictions,
            mvra_paired_result=mvra_paired_result,
            hslg_state={
                "event_to_id": event_to_id,
                "event_keywords": event_keywords,
                "event_evidence": event_evidence,
                "event_metadata": event_metadata,
                "event_assignment_counts": event_assignment_counts,
                "event_id_list": list(event_ids),
            },
        )


def build_pareto_report(args):
    input_dir = Path(args.build_pareto)
    metric_files = list(input_dir.rglob("*_metrics.json"))
    if not metric_files:
        raise FileNotFoundError(f"No *_metrics.json files under {input_dir}")

    per_seed = defaultdict(list)
    for metric_file in metric_files:
        with open(metric_file, "r", encoding="utf-8") as file:
            result = json.load(file)
        pipeline = result.get("anchor_pipeline", "legacy_two_stage")
        condition = result.get("variant")
        if pipeline != "single_stage":
            condition = f"{condition}__{pipeline}"
        key = (
            result.get("dataset"),
            condition,
            result.get("seed"),
        )
        per_seed[key].append(result)

    seed_rows = []
    for (dataset, variant, seed), results in per_seed.items():
        elapsed = sum(item["efficiency"]["elapsed_seconds"] for item in results)
        cost = sum(item["efficiency"]["estimated_cost"] for item in results)
        score = float(
            np.mean([item["macro_scores"][args.pareto_metric] for item in results])
        )
        seed_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "blocks": len(results),
                "elapsed_seconds": elapsed,
                "estimated_cost": cost,
                args.pareto_metric: score,
            }
        )

    across_seeds = defaultdict(list)
    for row in seed_rows:
        across_seeds[(row["dataset"], row["variant"])].append(row)
    rows = []
    for (dataset, variant), variant_rows in across_seeds.items():
        x_values = np.asarray(
            [row[args.pareto_x] for row in variant_rows], dtype=float
        )
        metric_values = np.asarray(
            [row[args.pareto_metric] for row in variant_rows], dtype=float
        )
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seeds": ",".join(
                    str(row["seed"])
                    for row in sorted(variant_rows, key=lambda item: item["seed"])
                ),
                "runs": len(variant_rows),
                "blocks_per_run": min(row["blocks"] for row in variant_rows),
                f"{args.pareto_x}_mean": float(np.mean(x_values)),
                f"{args.pareto_x}_std": float(np.std(x_values)),
                f"{args.pareto_metric}_mean": float(np.mean(metric_values)),
                f"{args.pareto_metric}_std": float(np.std(metric_values)),
            }
        )
    x_mean = f"{args.pareto_x}_mean"
    x_std = f"{args.pareto_x}_std"
    metric_mean = f"{args.pareto_metric}_mean"
    metric_std = f"{args.pareto_metric}_std"
    rows.sort(key=lambda row: row[x_mean])

    for row in rows:
        row["pareto_frontier"] = not any(
            (
                other[x_mean] <= row[x_mean]
                and other[metric_mean] >= row[metric_mean]
                and (
                    other[x_mean] < row[x_mean]
                    or other[metric_mean] > row[metric_mean]
                )
            )
            for other in rows
            if other is not row
        )

    output_dir = Path(args.pareto_output or input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cost_performance_pareto.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required only for Pareto plotting. "
            f"The numeric Pareto CSV was still written to {csv_path}."
        ) from exc

    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    for row in rows:
        marker = "*" if row["pareto_frontier"] else "o"
        axis.errorbar(
            row[x_mean],
            row[metric_mean],
            xerr=row[x_std],
            yerr=row[metric_std],
            marker=marker,
            markersize=12 if marker == "*" else 8,
            capsize=3,
            linestyle="none",
        )
        axis.annotate(
            row["variant"],
            (row[x_mean], row[metric_mean]),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel(args.pareto_x.replace("_", " "))
    axis.set_ylabel(args.pareto_metric)
    axis.set_title("Cost vs. Performance Pareto Frontier")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure_path = output_dir / "cost_performance_pareto.png"
    figure.savefig(figure_path, dpi=220)
    print(f"Wrote {csv_path} and {figure_path}")


def _paired_effect_size(differences):
    differences = np.asarray(differences, dtype=float)
    if len(differences) < 2:
        return None
    deviation = float(np.std(differences, ddof=1))
    if deviation == 0:
        return 0.0 if float(np.mean(differences)) == 0 else None
    return float(np.mean(differences) / deviation)


def build_statistical_report(args):
    input_dir = Path(args.summarize_results)
    metric_files = list(input_dir.rglob("*_metrics.json"))
    if not metric_files:
        raise FileNotFoundError(f"No *_metrics.json files under {input_dir}")

    records = []
    for metric_file in metric_files:
        with open(metric_file, "r", encoding="utf-8") as file:
            result = json.load(file)
        pipeline = result.get("anchor_pipeline", "legacy_two_stage")
        condition = result["variant"]
        if pipeline != "single_stage":
            condition = f"{condition}__{pipeline}"
        records.append(
            {
                "dataset": result["dataset"],
                "block": int(result["block"]),
                "variant": condition,
                "seed": int(result["seed"]),
                **result["macro_scores"],
            }
        )

    output_dir = Path(args.summary_output or input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_means = []
    grouped_runs = defaultdict(list)
    for record in records:
        grouped_runs[
            (record["dataset"], record["variant"], record["seed"])
        ].append(record)
    for (dataset, variant, seed), run_records in grouped_runs.items():
        run_means.append(
            {
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "blocks": len(run_records),
                **{
                    metric_name: float(
                        np.mean(
                            [record[metric_name] for record in run_records]
                        )
                    )
                    for metric_name in ("NMI", "AMI", "ARI")
                },
            }
        )

    summary_rows = []
    for variant in sorted({record["variant"] for record in records}):
        variant_runs = [
            run for run in run_means if run["variant"] == variant
        ]
        for metric_name in ("NMI", "AMI", "ARI"):
            values = np.asarray(
                [run[metric_name] for run in variant_runs], dtype=float
            )
            ci95 = (
                1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
            summary_rows.append(
                {
                    "variant": variant,
                    "metric": metric_name,
                    "independent_runs": len(values),
                    "summary_unit": "dataset mean across blocks per seed",
                    "mean": float(np.mean(values)),
                    "std": (
                        float(np.std(values, ddof=1))
                        if len(values) > 1
                        else 0.0
                    ),
                    "ci95_half_width": ci95,
                }
            )
    summary_path = output_dir / "result_summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    comparison = args.compare_to
    paired_rows = []
    if comparison:
        try:
            from scipy.stats import wilcoxon
        except ImportError:
            wilcoxon = None
        lookup = {
            (
                record["dataset"],
                record["block"],
                record["seed"],
                record["variant"],
            ): record
            for record in records
        }
        variants = sorted(
            {record["variant"] for record in records if record["variant"] != comparison}
        )
        for variant in variants:
            for metric_name in ("NMI", "AMI", "ARI"):
                pairs = []
                for record in records:
                    if record["variant"] != variant:
                        continue
                    reference = lookup.get(
                        (
                            record["dataset"],
                            record["block"],
                            record["seed"],
                            comparison,
                        )
                    )
                    if reference is not None:
                        pairs.append(
                            (record[metric_name], reference[metric_name])
                        )
                if not pairs:
                    continue
                differences = np.asarray(
                    [left - right for left, right in pairs], dtype=float
                )
                p_value = None
                test_name = None
                if (
                    wilcoxon is not None
                    and len(differences) >= 2
                    and np.any(differences != 0)
                ):
                    p_value = float(wilcoxon(differences).pvalue)
                    test_name = "wilcoxon"
                elif len(differences) >= 2 and np.any(differences != 0):
                    positive = int(np.sum(differences > 0))
                    negative = int(np.sum(differences < 0))
                    nonzero = positive + negative
                    tail = min(positive, negative)
                    p_value = min(
                        1.0,
                        2.0
                        * sum(
                            math.comb(nonzero, value)
                            for value in range(tail + 1)
                        )
                        / (2**nonzero),
                    )
                    test_name = "exact_sign_test"
                tolerance = args.tie_tolerance
                paired_rows.append(
                    {
                        "variant": variant,
                        "reference": comparison,
                        "metric": metric_name,
                        "pair_unit": "dataset-block-seed",
                        "paired_observations": len(pairs),
                        "mean_difference": float(np.mean(differences)),
                        "paired_cohens_d": _paired_effect_size(differences),
                        "test": test_name,
                        "paired_test_p_value": p_value,
                        "wins": int(np.sum(differences > tolerance)),
                        "ties": int(np.sum(np.abs(differences) <= tolerance)),
                        "losses": int(np.sum(differences < -tolerance)),
                    }
                )
        if paired_rows:
            paired_path = output_dir / "paired_comparisons.csv"
            with open(paired_path, "w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(paired_rows[0]))
                writer.writeheader()
                writer.writerows(paired_rows)
            print(f"Wrote {paired_path}")
        elif comparison:
            print(
                f"No block/seed pairs were available for reference '{comparison}'."
            )
    print(f"Wrote {summary_path}")


def resolve_args(args):
    stage_aware_variants = {
        "stage_aware_text_control",
        "stage_aware_all_view_mvra",
        "stage_aware_selective_mvra_no_time",
        "stage_aware_selective_mvra",
        "stage_aware_selective_gate",
        "stage_aware_selective_gate_no_time",
        "stage_aware_selective_gate_time_aux",
    }
    stage_aware_ugr_variants = {
        "stage_aware_all_view_mvra",
        "stage_aware_selective_mvra_no_time",
        "stage_aware_selective_mvra",
        "stage_aware_selective_gate",
        "stage_aware_selective_gate_no_time",
        "stage_aware_selective_gate_time_aux",
    }
    buffered_novelty_variants = {
        "ugr_mvra_buffered",
        "ugr_mvra_buffered_local_hslg",
    }
    ugr_variants = {
        "ugr_mvra",
        "ugr_mvra_hslg",
        "ugr_mvra_consensus_hslg",
        *buffered_novelty_variants,
        *stage_aware_ugr_variants,
    }
    object_conrag_variants = {
        "conrag_object_text",
        "conrag_object_text_hslg",
        "conrag_object_three",
        "conrag_object_three_hslg",
        "conrag_object_consensus",
        "conrag_object_consensus_hslg",
        "conrag_cqc_consensus",
        "conrag_cqc_consensus_hslg",
        *ugr_variants,
        *stage_aware_variants,
    }
    cqc_variants = {
        "conrag_cqc_consensus",
        "conrag_cqc_consensus_hslg",
    }
    aligned_conrag_variants = {
        "conrag_aligned_text",
        "conrag_aligned_text_entity",
        "conrag_aligned_text_relation",
        "conrag_aligned_three_no_bonus",
        "conrag_aligned_three",
        "conrag_aligned_three_hslg",
    }
    fixed_anchor_factorial_variants = {
        "fixed_anchor_matched_control",
        "fixed_anchor_mvra",
        "fixed_anchor_hslg",
        "fixed_anchor_mvra_hslg",
    }
    consistent_dynamic_two_stage_variants = {
        "two_stage_consistent_dynamic",
        "two_stage_consistent_dynamic_mvra",
        "two_stage_consistent_dynamic_full",
    }
    safe_dynamic_two_stage_variants = {
        "two_stage_safe_dynamic",
        "two_stage_safe_dynamic_mvra",
        "two_stage_safe_dynamic_full",
    }
    dynamic_two_stage_variants = {
        "two_stage_dynamic",
        "two_stage_dynamic_mvra",
        "two_stage_dynamic_full",
        *safe_dynamic_two_stage_variants,
        *consistent_dynamic_two_stage_variants,
    }
    two_stage_mvra_variants = {
        "two_stage_dynamic_mvra",
        "two_stage_dynamic_full",
        "two_stage_safe_dynamic_mvra",
        "two_stage_safe_dynamic_full",
        "two_stage_consistent_dynamic_mvra",
        "two_stage_consistent_dynamic_full",
    }
    two_stage_experiment_variants = {
        "fixed_anchor_control",
        *fixed_anchor_factorial_variants,
        *dynamic_two_stage_variants,
    }
    conrag_mvra_variants = {
        "mvra_conrag",
        "mvra_conrag_no_hslg",
        "mvra_conrag_no_bonus",
    }
    mvra_experiment_variants = {
        "mvra_legacy_control",
        "mvra_no_mvra_control",
        *conrag_mvra_variants,
    }
    refined_v1_variants = {
        "full_refined",
        "refined_w/o_dkms",
        "refined_w/o_mvra",
        "refined_w/o_hslg",
    }
    refined_v2_variants = {
        "full_refined_v2",
        "refined_v2_w/o_dkms",
        "refined_v2_w/o_mvra",
        "refined_v2_w/o_hslg",
    }
    balanced_variants = {
        "full_balanced",
        "balanced_w/o_dkms",
        "balanced_w/o_mvra",
        "balanced_w/o_hslg",
        *mvra_experiment_variants,
        *two_stage_experiment_variants,
        *aligned_conrag_variants,
        *object_conrag_variants,
    }
    args.object_conrag_mode = args.variant in object_conrag_variants
    args.cqc_mode = args.variant in cqc_variants
    args.ugr_mode = args.variant in ugr_variants
    args.stage_aware_mode = args.variant in stage_aware_variants
    args.stage_use_temporal = args.variant in {
        "stage_aware_all_view_mvra",
        "stage_aware_selective_mvra",
        "stage_aware_selective_gate",
        "stage_aware_selective_gate_time_aux",
    }
    args.stage_force_all_views = (
        args.variant == "stage_aware_all_view_mvra"
    )
    args.novelty_buffer_mode = (
        args.variant in buffered_novelty_variants
    )
    args.novelty_buffer_local_hslg = (
        args.variant == "ugr_mvra_buffered_local_hslg"
    )
    args.ugr_internal_pass = False
    args.ugr_enforce_gate = False
    args.ugr_direct_gate = args.variant in {
        "stage_aware_selective_gate",
        "stage_aware_selective_gate_no_time",
        "stage_aware_selective_gate_time_aux",
    }
    args.stage_temporal_aux_only = (
        args.variant == "stage_aware_selective_gate_time_aux"
    )
    args.aligned_conrag_mode = args.variant in aligned_conrag_variants
    args.conrag_mvra_mode = args.variant in conrag_mvra_variants
    args.refined_v2_mode = args.variant in refined_v2_variants
    args.balanced_mode = args.variant in balanced_variants
    args.refined_mode = (
        args.variant in refined_v1_variants
        or args.refined_v2_mode
        or args.balanced_mode
    )
    if args.cqc_mode and args.detector_prompt == DEFAULT_DETECTOR_PROMPT:
        args.detector_prompt = CQC_DETECTOR_PROMPT
    if args.stage_force_all_views:
        # Diagnostic Fuse-All path from the view-selection paper: all
        # available auxiliary views are retained inside the same fixed pool.
        args.stage_max_views = 4
        args.stage_view_min_alignment = -1.0
        args.stage_view_min_gain = 0.0
        args.stage_view_redundancy_lambda = 0.0

    # Frozen label-free profile derived from the observed mechanism failures:
    # v2 over-split 24 low-cohesion clusters, while its mutual graph never
    # fired. These settings reduce recursive splitting and relax only graph
    # edges that carry lexical or exceptionally strong semantic evidence.
    if args.balanced_mode:
        balanced_profile = {
            "dkms_refine_trigger_size": 200,
            "dkms_refine_hard_max_size": 600,
            "dkms_min_cohesion": 0.40,
            "dkms_split_threshold_step": 0.04,
            "dkms_split_threshold_max": 0.68,
            "dkms_split_max_depth": 4,
            "mvra_view_accept_threshold": 0.35,
            "mvra_consensus_accept_threshold": 0.40,
            "mvra_consensus_margin": 0.01,
            "hslg_threshold": 0.25,
            "hslg_adaptive_quantile": 0.70,
            "hslg_adaptive_threshold_min": 0.25,
            "hslg_adaptive_threshold_max": 0.65,
            "hslg_min_lexical_overlap": 0.04,
            "hslg_semantic_override": 0.72,
            "hslg_mutual_top_k": 3,
            "hslg_mutual_threshold": 0.45,
            "hslg_max_community_size": 8,
        }
        for parameter, value in balanced_profile.items():
            setattr(args, parameter, value)
    if args.object_conrag_mode:
        # ConRAG Connection+Consensus adapted to online event memory. Matched
        # text/object multi-view x no-HSLG/HSLG conditions are retained, with
        # an additional conservative structural-consensus pair. Multi-hop QA
        # slot binding is deliberately excluded.
        three_view_variants = {
            "conrag_object_three",
            "conrag_object_three_hslg",
            "conrag_object_consensus",
            "conrag_object_consensus_hslg",
            "conrag_cqc_consensus",
            "conrag_cqc_consensus_hslg",
            *ugr_variants,
        }
        hslg_variants = {
            "conrag_object_text_hslg",
            "conrag_object_three_hslg",
            "conrag_object_consensus_hslg",
            "conrag_cqc_consensus_hslg",
            "ugr_mvra_hslg",
        }
        args.anchor_pipeline = "single_stage"
        args.enable_dkms = False
        args.preserve_anchor_views = True
        args.enable_mvra = args.variant in three_view_variants
        args.conrag_use_entity = args.enable_mvra
        args.conrag_use_relation = args.enable_mvra
        args.conrag_require_structural_consensus = args.variant in {
            "conrag_object_consensus",
            "conrag_object_consensus_hslg",
            "conrag_cqc_consensus",
            "conrag_cqc_consensus_hslg",
            *ugr_variants,
        }
        args.mvra_fallback_only = False
        args.mvra_override_non_others = False
        if args.variant == "ugr_mvra_consensus_hslg":
            args.hslg_mode = "consensus"
        else:
            args.hslg_mode = (
                "hybrid" if args.variant in hslg_variants else "none"
            )
    elif args.aligned_conrag_mode:
        # Faithful adaptation of ConRAG's evidence-aligned retrieval idea:
        # heterogeneous views are projected to one event space, text remains
        # dominant, and consensus is only a small ranking bonus.
        entity_variants = {
            "conrag_aligned_text_entity",
            "conrag_aligned_three_no_bonus",
            "conrag_aligned_three",
            "conrag_aligned_three_hslg",
        }
        relation_variants = {
            "conrag_aligned_text_relation",
            "conrag_aligned_three_no_bonus",
            "conrag_aligned_three",
            "conrag_aligned_three_hslg",
        }
        args.anchor_pipeline = "single_stage"
        args.enable_dkms = False
        args.preserve_anchor_views = True
        args.enable_mvra = (
            args.variant in entity_variants
            or args.variant in relation_variants
        )
        args.conrag_use_entity = args.variant in entity_variants
        args.conrag_use_relation = args.variant in relation_variants
        args.mvra_fallback_only = False
        args.mvra_override_non_others = False
        args.conrag_consensus_lambda = (
            0.0
            if args.variant == "conrag_aligned_three_no_bonus"
            else args.conrag_consensus_lambda
        )
        args.hslg_mode = (
            "hybrid"
            if args.variant == "conrag_aligned_three_hslg"
            else "none"
        )
    elif args.variant in fixed_anchor_factorial_variants:
        # A 2x2 matched experiment on RagSEDE's fixed-threshold anchors. All
        # four conditions retain identical representative-message separators;
        # only conservative MVRA and/or HSLG is toggled.
        mvra_variants = {"fixed_anchor_mvra", "fixed_anchor_mvra_hslg"}
        hslg_variants = {"fixed_anchor_hslg", "fixed_anchor_mvra_hslg"}
        args.anchor_pipeline = "single_stage"
        args.enable_dkms = False
        args.preserve_anchor_views = True
        args.enable_mvra = args.variant in mvra_variants
        args.mvra_fusion = "conrag" if args.enable_mvra else "max"
        args.mvra_fallback_only = args.enable_mvra
        args.mvra_override_non_others = False
        args.mvra_consensus_accept_threshold = 0.45
        args.mvra_consensus_margin = 0.03
        args.hslg_mode = (
            "hybrid" if args.variant in hslg_variants else "none"
        )
    elif args.variant == "fixed_anchor_control":
        # Matched first-stage control for the two-stage contribution chain.
        # It shares the same prompts and event-memory profile as the variants
        # below, while changing only the requested mechanisms.
        args.anchor_pipeline = "single_stage"
        args.enable_dkms = False
        args.enable_mvra = False
        args.hslg_mode = "none"
        args.mvra_override_non_others = False
    elif args.variant in dynamic_two_stage_variants:
        # Stage 1 is exactly RagSEDE's fixed-threshold anchor construction.
        # Stage 2 derives one block-level threshold only from the cosine
        # density of stage-1 anchors; no event labels or clustering metrics are
        # available to this computation.
        args.anchor_pipeline = "dynamic_two_stage"
        args.enable_dkms = True
        args.dkms_never_below_base = (
            args.variant in safe_dynamic_two_stage_variants
            or args.variant in consistent_dynamic_two_stage_variants
        )
        args.dkms_stage2_linkage = (
            "complete"
            if args.variant in consistent_dynamic_two_stage_variants
            else "star"
        )
        args.enable_mvra = args.variant in two_stage_mvra_variants
        args.mvra_fusion = "conrag" if args.enable_mvra else "max"
        args.mvra_override_non_others = args.enable_mvra
        args.hslg_mode = (
            "hybrid"
            if args.variant
            in {
                "two_stage_dynamic_full",
                "two_stage_safe_dynamic_full",
                "two_stage_consistent_dynamic_full",
            }
            else "none"
        )
    elif args.conrag_mvra_mode:
        # Isolate the retrieval contribution on RagSEDE's fixed-threshold,
        # max-size-controlled anchors.  This avoids the impure giant anchors
        # observed in the earlier balanced DKMS profile.
        args.enable_dkms = False
        args.enable_mvra = True
        args.mvra_fusion = "conrag"
        args.mvra_consensus_accept_threshold = 0.45
        args.mvra_consensus_margin = 0.03
        args.mvra_override_non_others = True
        if args.variant == "mvra_conrag_no_bonus":
            args.mvra_consensus_bonus = 0.0
        if args.variant == "mvra_conrag_no_hslg":
            args.hslg_mode = "none"
    elif args.variant == "mvra_legacy_control":
        args.enable_dkms = False
        args.enable_mvra = True
        args.mvra_fusion = "max"
        args.mvra_override_non_others = False
    elif args.variant == "mvra_no_mvra_control":
        args.enable_dkms = False
        args.enable_mvra = False
    if args.language is None:
        args.language = "French" if args.dataset == "twitter18" else "English"
    if args.local_embedding_model is None:
        args.local_embedding_model = (
            "all-MiniLM-L6-v2"
            if args.language == "English"
            else "distiluse-base-multilingual-cased-v1"
        )
    if (
        args.variant in mvra_experiment_variants
        or args.variant in two_stage_experiment_variants
    ):
        pass
    elif args.variant in {
        "w/o_dkms", "refined_w/o_dkms", "refined_v2_w/o_dkms",
        "balanced_w/o_dkms",
    }:
        args.enable_dkms = False
    elif args.variant in {
        "w/o_mvra", "refined_w/o_mvra", "refined_v2_w/o_mvra",
        "balanced_w/o_mvra",
        "single_query",
    }:
        args.enable_mvra = False
    elif args.variant in {
        "w/o_hslg", "refined_w/o_hslg", "refined_v2_w/o_hslg",
        "balanced_w/o_hslg",
    }:
        args.hslg_mode = "none"
    elif args.variant == "ragsede_control":
        # Matched control: same LLM/encoder/budget, without the three proposed
        # mechanisms; semantic consolidation is retained as the closest existing
        # RagSEDE-style structural component in this repository.
        args.enable_dkms = False
        args.enable_mvra = False
        args.hslg_mode = "semantic"
    elif args.variant == "ragsede_original":
        # Source-faithful SED path: one fixed-threshold anchor constructor,
        # RAGFlow's attached-dataset retrieval, and no MvSED consolidation.
        args.enable_dkms = False
        args.enable_mvra = False
        args.hslg_mode = "none"
        args.evaluator_prompt = RAGSEDE_ORIGINAL_EVALUATOR_PROMPT
        args.detector_prompt = RAGSEDE_ORIGINAL_DETECTOR_PROMPT
    elif args.variant == "ragsede_plus_dkms":
        args.enable_dkms = True
        args.enable_mvra = False
        args.hslg_mode = "semantic"
    elif args.variant == "ragsede_plus_dkms_mvra":
        args.enable_dkms = True
        args.enable_mvra = True
        args.hslg_mode = "semantic"
    args.dkms_preserve_large_clusters = (
        args.refined_mode and args.enable_dkms
    )
    args.dkms_cohesion_refinement = (
        (args.refined_v2_mode or args.balanced_mode) and args.enable_dkms
    )
    args.include_event_name_in_memory = args.refined_mode
    args.accumulate_event_keywords = args.refined_mode
    args.mvra_confidence_gate = (
        args.refined_mode
        and not args.refined_v2_mode
        and not args.balanced_mode
        and args.enable_mvra
    )
    args.mvra_consensus_gate = (
        (args.refined_v2_mode or args.balanced_mode) and args.enable_mvra
    )
    args.mvra_single_view_fallback = (
        args.balanced_mode and args.enable_mvra
    )
    if args.aligned_conrag_mode or args.object_conrag_mode:
        # Aligned ConRAG uses consensus only inside candidate ranking.  All
        # legacy deterministic reassignment paths must remain disabled.
        args.mvra_confidence_gate = False
        args.mvra_consensus_gate = False
        args.mvra_single_view_fallback = False
        args.mvra_override_non_others = False
        args.mvra_fallback_only = False
    if args.mvra_fallback_only:
        # Fallback acceptance requires real cross-view agreement. A one-view
        # shortcut or non-Others override would violate that protocol.
        args.mvra_single_view_fallback = False
        args.mvra_override_non_others = False
    args.hslg_reciprocal_merge = (
        args.refined_mode
        and not args.refined_v2_mode
        and not args.balanced_mode
        and args.hslg_mode != "none"
    )
    args.hslg_adaptive_graph = (
        (args.refined_v2_mode or args.balanced_mode)
        and args.hslg_mode != "none"
    )
    args.hslg_mutual_knn_merge = (
        (args.refined_v2_mode or args.balanced_mode)
        and args.hslg_mode != "none"
    )
    args.hslg_allowed_edge_quantile = (
        args.balanced_mode and args.hslg_mode != "none"
    )
    if args.object_conrag_mode:
        enabled_views = ["text-evidence"]
        if args.conrag_use_entity:
            enabled_views.append("entity-object")
        if args.conrag_use_relation:
            enabled_views.append("relation-object")
        print(
            "Object-level ConRAG condition enabled: fixed anchors, views="
            + ",".join(enabled_views)
            + f", object_top_k={args.conrag_object_top_k}, "
            + "candidate_policy="
            + (
                "text-or-entity+relation-consensus"
                if args.conrag_require_structural_consensus
                else "union-of-views"
            )
            + ", "
            + f"consensus_lambda={args.conrag_consensus_lambda:.3f}, "
            + (
                "CQC=shared-pool-mean-variance-gate, "
                if args.cqc_mode
                else (
                    "UGR=uncertainty-gated-residual-repair, "
                    if args.ugr_mode
                    else "CQC/UGR=disabled, "
                )
            )
            + f"HSLG={args.hslg_mode}."
        )
        if args.ugr_mode:
            print(
                "UGR gate (label-free, fixed globally): "
                f"views>={args.ugr_min_support_views}, "
                f"structural_views>={args.ugr_min_structural_views}, "
                f"text_similarity>={args.ugr_min_text_similarity:.3f}, "
                f"fused_score>={args.ugr_min_fused_score:.3f}, "
                f"margin>={args.ugr_min_margin:.3f}."
            )
        if args.novelty_buffer_mode:
            buffer_graph = (
                "text+entity+relation consensus"
                if args.novelty_buffer_local_hslg
                else "text-semantic"
            )
            print(
                "Buffered novelty resolution enabled (label-free, fixed "
                f"globally): graph={buffer_graph}, "
                f"capacity={args.novelty_buffer_max_anchors}, "
                f"TTL={args.novelty_buffer_ttl_anchors} anchors, "
                f"minimum_group={args.novelty_buffer_min_cluster_anchors}, "
                f"complete-link cap={args.novelty_buffer_max_component_size}."
            )
        if args.stage_aware_mode:
            stage_profile = (
                "text-only matched control"
                if args.variant == "stage_aware_text_control"
                else "forced all-view reranking"
                if args.stage_force_all_views
                else "selective deterministic gate without time"
                if args.ugr_direct_gate and not args.stage_use_temporal
                else "selective deterministic gate with auxiliary-only time"
                if args.stage_temporal_aux_only
                else "selective reranking without time"
                if not args.stage_use_temporal
                else "selective reranking with deterministic gate"
                if args.ugr_direct_gate
                else "selective reranking with temporal pulse"
            )
            print(
                "Stage-aware selective MVRA enabled (label-free): "
                "monolithic text candidate generation, fixed-pool "
                "entity/relation/temporal reranking, relevance-minus-"
                "redundancy view selection, "
                f"condition={stage_profile}, "
                f"max_views={args.stage_max_views}, "
                f"text_weight={args.stage_text_weight:.2f}."
            )
    elif args.aligned_conrag_mode:
        enabled_views = ["text"]
        if args.conrag_use_entity:
            enabled_views.append("entity-anchor")
        if args.conrag_use_relation:
            enabled_views.append("relation/action")
        print(
            "Evidence-aligned ConRAG condition enabled: views="
            + ",".join(enabled_views)
            + f", consensus_lambda={args.conrag_consensus_lambda:.3f}, "
            + f"HSLG={args.hslg_mode}."
        )
    elif args.variant in fixed_anchor_factorial_variants:
        enabled = []
        if args.enable_mvra:
            enabled.append("fallback-only consensus MVRA")
        if args.hslg_mode != "none":
            enabled.append("evidence-gated HSLG")
        print(
            "Fixed-anchor factorial condition enabled: RagSEDE "
            "fixed-threshold anchors"
            + (" + " + " + ".join(enabled) if enabled else " only")
            + "."
        )
    elif args.variant == "fixed_anchor_control":
        print(
            "Fixed-anchor matched control enabled: RagSEDE fixed-threshold "
            "anchors, without dynamic re-clustering, MVRA, or HSLG."
        )
    elif args.variant in dynamic_two_stage_variants:
        dynamic_name = (
            "pairwise-consistent density-aware anchor re-clustering"
            if args.variant in consistent_dynamic_two_stage_variants
            else
            "safe label-free dynamic anchor re-clustering"
            if args.variant in safe_dynamic_two_stage_variants
            else "label-free dynamic anchor re-clustering"
        )
        enabled = [dynamic_name]
        if args.enable_mvra:
            enabled.append("consensus multi-view retrieval")
        if args.hslg_mode != "none":
            enabled.append("HSLG")
        print(
            "Dynamic two-stage protocol enabled: fixed-threshold stage 1 + "
            + " + ".join(enabled)
            + "."
        )
    elif args.conrag_mvra_mode:
        print(
            "ConRAG-style MVRA enabled: fixed-threshold controlled anchors, "
            "per-view normalization, aligned candidate fusion, and "
            "multi-view consensus scoring."
        )
    elif args.variant == "mvra_legacy_control":
        print(
            "Legacy MVRA control enabled on fixed-threshold controlled "
            "anchors with the same balanced event memory and HSLG."
        )
    elif args.variant == "mvra_no_mvra_control":
        print(
            "No-MVRA control enabled on fixed-threshold controlled anchors "
            "with the same balanced event memory and HSLG."
        )
    elif args.balanced_mode:
        print(
            "Balanced MvSED enabled: conservative cohesion-aware DKMS, "
            "multi/single-view MVRA fallback, and evidence-gated HSLG."
        )
    elif args.refined_v2_mode:
        print(
            "Refined MvSED v2 enabled: cohesion-aware DKMS, cross-view "
            "consensus MVRA, and adaptive lexical-veto HSLG."
        )
    elif args.refined_mode:
        print(
            "Refined MvSED enabled: faithful large-cluster DKMS, "
            "event-aware MVRA memory, and reciprocal HSLG consolidation."
        )
    if args.knowledge_control == "both":
        print("Knowledge control: entity masking + event identifier anonymization.")
    if not 0.0 <= args.dkms_lambda <= 1.0:
        raise ValueError("--dkms_lambda must be in [0, 1].")
    if not 0.0 <= args.hslg_semantic_weight <= 1.0:
        raise ValueError("--hslg_semantic_weight must be in [0, 1].")
    if args.view_count < 1:
        raise ValueError("--view_count must be positive.")
    if args.event_keyword_memory_size < args.max_keywords:
        raise ValueError(
            "--event_keyword_memory_size must be at least --max_keywords."
        )
    if not -1.0 <= args.mvra_accept_threshold <= 1.0:
        raise ValueError("--mvra_accept_threshold must be in [-1, 1].")
    if args.mvra_accept_margin < 0.0:
        raise ValueError("--mvra_accept_margin must be non-negative.")
    if args.mvra_gate_min_candidates < 2:
        raise ValueError("--mvra_gate_min_candidates must be at least 2.")
    if args.dkms_refine_trigger_size < 2:
        raise ValueError("--dkms_refine_trigger_size must be at least 2.")
    if args.dkms_refine_hard_max_size < args.dkms_refine_trigger_size:
        raise ValueError(
            "--dkms_refine_hard_max_size must be at least the trigger size."
        )
    if not -1.0 <= args.dkms_min_cohesion <= 1.0:
        raise ValueError("--dkms_min_cohesion must be in [-1, 1].")
    if args.dkms_split_threshold_step <= 0.0:
        raise ValueError("--dkms_split_threshold_step must be positive.")
    if not 0.0 < args.dkms_split_threshold_max <= 1.0:
        raise ValueError("--dkms_split_threshold_max must be in (0, 1].")
    if args.dkms_split_max_depth < 1:
        raise ValueError("--dkms_split_max_depth must be positive.")
    if args.mvra_consensus_min_views < 2:
        raise ValueError("--mvra_consensus_min_views must be at least 2.")
    if not -1.0 <= args.mvra_view_accept_threshold <= 1.0:
        raise ValueError("--mvra_view_accept_threshold must be in [-1, 1].")
    if not -1.0 <= args.mvra_consensus_accept_threshold <= 1.0:
        raise ValueError(
            "--mvra_consensus_accept_threshold must be in [-1, 1]."
        )
    if args.mvra_consensus_margin < 0.0:
        raise ValueError("--mvra_consensus_margin must be non-negative.")
    if not -1.0 <= args.mvra_single_view_accept_threshold <= 1.0:
        raise ValueError(
            "--mvra_single_view_accept_threshold must be in [-1, 1]."
        )
    if args.mvra_single_view_margin < 0.0:
        raise ValueError("--mvra_single_view_margin must be non-negative.")
    if args.mvra_candidate_top_k_per_view < 1:
        raise ValueError(
            "--mvra_candidate_top_k_per_view must be positive."
        )
    if args.mvra_support_score_gap < 0.0:
        raise ValueError("--mvra_support_score_gap must be non-negative.")
    if args.mvra_consensus_bonus < 0.0:
        raise ValueError("--mvra_consensus_bonus must be non-negative.")
    if not 0.0 <= args.mvra_conrag_primary_weight <= 1.0:
        raise ValueError(
            "--mvra_conrag_primary_weight must be in [0, 1]."
        )
    if not 0.0 <= args.mvra_override_accept_threshold <= 2.0:
        raise ValueError(
            "--mvra_override_accept_threshold must be in [0, 2]."
        )
    if args.mvra_override_margin < 0.0:
        raise ValueError("--mvra_override_margin must be non-negative.")
    conrag_weights = (
        args.conrag_text_weight,
        args.conrag_entity_weight,
        args.conrag_relation_weight,
    )
    if any(weight < 0.0 for weight in conrag_weights):
        raise ValueError("Aligned ConRAG view weights must be non-negative.")
    if sum(conrag_weights) <= 0.0:
        raise ValueError("At least one aligned ConRAG view weight must be positive.")
    if args.conrag_consensus_lambda < 0.0:
        raise ValueError("--conrag_consensus_lambda must be non-negative.")
    if args.conrag_per_view_top_k < 1:
        raise ValueError("--conrag_per_view_top_k must be positive.")
    if not -1.0 <= args.conrag_view_score_threshold <= 1.0:
        raise ValueError(
            "--conrag_view_score_threshold must be in [-1, 1]."
        )
    if args.conrag_event_evidence_size < 1:
        raise ValueError("--conrag_event_evidence_size must be positive.")
    if args.conrag_event_text_chars < 1:
        raise ValueError("--conrag_event_text_chars must be positive.")
    if args.conrag_max_view_units < 1:
        raise ValueError("--conrag_max_view_units must be positive.")
    if args.conrag_object_top_k < 1:
        raise ValueError("--conrag_object_top_k must be positive.")
    if args.conrag_relation_beta < 0.0:
        raise ValueError("--conrag_relation_beta must be non-negative.")
    if args.conrag_entity_context_window < 1:
        raise ValueError(
            "--conrag_entity_context_window must be positive."
        )
    if args.conrag_object_contexts < 1:
        raise ValueError("--conrag_object_contexts must be positive.")
    if not 1 <= args.conrag_structural_min_views <= 2:
        raise ValueError(
            "--conrag_structural_min_views must be either 1 or 2."
        )
    if not 2 <= args.ugr_min_support_views <= 3:
        raise ValueError("--ugr_min_support_views must be 2 or 3.")
    if not 1 <= args.ugr_min_structural_views <= 2:
        raise ValueError(
            "--ugr_min_structural_views must be either 1 or 2."
        )
    if not -1.0 <= args.ugr_min_text_similarity <= 1.0:
        raise ValueError(
            "--ugr_min_text_similarity must be in [-1, 1]."
        )
    if not -1.0 <= args.ugr_min_fused_score <= 2.0:
        raise ValueError("--ugr_min_fused_score must be in [-1, 2].")
    if args.ugr_min_margin < 0.0:
        raise ValueError("--ugr_min_margin must be non-negative.")
    if args.cqc_paraphrase_count < 0:
        raise ValueError("--cqc_paraphrase_count must be non-negative.")
    if args.cqc_query_chars < 1:
        raise ValueError("--cqc_query_chars must be positive.")
    if not -1.0 <= args.cqc_semantic_similarity_threshold <= 1.0:
        raise ValueError(
            "--cqc_semantic_similarity_threshold must be in [-1, 1]."
        )
    if args.cqc_shared_pool_k < 1:
        raise ValueError("--cqc_shared_pool_k must be positive.")
    if args.cqc_min_votes < 2:
        raise ValueError("--cqc_min_votes must be at least 2.")
    if args.cqc_mode and args.cqc_min_votes > args.cqc_paraphrase_count + 1:
        raise ValueError(
            "--cqc_min_votes cannot exceed the original query plus the "
            "configured paraphrases."
        )
    if not -1.0 <= args.cqc_min_mean_score <= 1.0:
        raise ValueError("--cqc_min_mean_score must be in [-1, 1].")
    if args.cqc_variance_lambda < 0.0:
        raise ValueError("--cqc_variance_lambda must be non-negative.")
    if args.cqc_margin < 0.0:
        raise ValueError("--cqc_margin must be non-negative.")
    if args.cqc_event_evidence_size < 1:
        raise ValueError("--cqc_event_evidence_size must be positive.")
    if args.cqc_evidence_chars < 1:
        raise ValueError("--cqc_evidence_chars must be positive.")
    if not -1.0 <= args.hslg_reciprocal_threshold <= 1.0:
        raise ValueError("--hslg_reciprocal_threshold must be in [-1, 1].")
    if not 0.0 <= args.hslg_adaptive_quantile <= 1.0:
        raise ValueError("--hslg_adaptive_quantile must be in [0, 1].")
    if not (
        0.0 <= args.hslg_adaptive_threshold_min
        <= args.hslg_adaptive_threshold_max <= 1.0
    ):
        raise ValueError("Invalid HSLG adaptive threshold bounds.")
    if not 0.0 <= args.hslg_min_lexical_overlap <= 1.0:
        raise ValueError("--hslg_min_lexical_overlap must be in [0, 1].")
    if not -1.0 <= args.hslg_semantic_override <= 1.0:
        raise ValueError("--hslg_semantic_override must be in [-1, 1].")
    if args.hslg_mutual_top_k < 1:
        raise ValueError("--hslg_mutual_top_k must be positive.")
    if not -1.0 <= args.hslg_mutual_threshold <= 1.0:
        raise ValueError("--hslg_mutual_threshold must be in [-1, 1].")
    if args.hslg_max_community_size < 0:
        raise ValueError("--hslg_max_community_size must be non-negative.")
    for parameter in (
        "hslg_consensus_semantic_threshold",
        "hslg_consensus_entity_threshold",
        "hslg_consensus_relation_threshold",
        "hslg_consensus_score_threshold",
    ):
        if not 0.0 <= getattr(args, parameter) <= 1.0:
            raise ValueError(f"--{parameter} must be in [0, 1].")
    if not 2 <= args.hslg_consensus_min_views <= 3:
        raise ValueError("--hslg_consensus_min_views must be 2 or 3.")
    if args.hslg_consensus_top_k < 1:
        raise ValueError("--hslg_consensus_top_k must be positive.")
    if args.hslg_consensus_max_component_size < 2:
        raise ValueError(
            "--hslg_consensus_max_component_size must be at least 2."
        )
    if args.novelty_buffer_max_anchors < 2:
        raise ValueError(
            "--novelty_buffer_max_anchors must be at least 2."
        )
    if args.novelty_buffer_ttl_anchors < 1:
        raise ValueError(
            "--novelty_buffer_ttl_anchors must be positive."
        )
    if args.novelty_buffer_min_cluster_anchors < 2:
        raise ValueError(
            "--novelty_buffer_min_cluster_anchors must be at least 2."
        )
    if args.novelty_buffer_context_anchors < 1:
        raise ValueError(
            "--novelty_buffer_context_anchors must be positive."
        )
    for parameter in (
        "novelty_buffer_semantic_threshold",
        "novelty_buffer_entity_threshold",
        "novelty_buffer_relation_threshold",
        "novelty_buffer_score_threshold",
    ):
        if not 0.0 <= getattr(args, parameter) <= 1.0:
            raise ValueError(f"--{parameter} must be in [0, 1].")
    if not 2 <= args.novelty_buffer_min_views <= 3:
        raise ValueError("--novelty_buffer_min_views must be 2 or 3.")
    if args.novelty_buffer_top_k < 1:
        raise ValueError("--novelty_buffer_top_k must be positive.")
    if (
        args.novelty_buffer_max_component_size
        < args.novelty_buffer_min_cluster_anchors
    ):
        raise ValueError(
            "--novelty_buffer_max_component_size must be at least "
            "--novelty_buffer_min_cluster_anchors."
        )
    if args.stage_candidate_pool_k < 2:
        raise ValueError("--stage_candidate_pool_k must be at least 2.")
    if args.stage_max_views < 1:
        raise ValueError("--stage_max_views must be positive.")
    if not 0.5 <= args.stage_text_weight <= 1.0:
        raise ValueError("--stage_text_weight must be in [0.5, 1].")
    if not -1.0 <= args.stage_view_min_alignment <= 1.0:
        raise ValueError(
            "--stage_view_min_alignment must be in [-1, 1]."
        )
    if args.stage_view_min_gain < 0.0:
        raise ValueError("--stage_view_min_gain must be non-negative.")
    if args.stage_view_redundancy_lambda < 0.0:
        raise ValueError(
            "--stage_view_redundancy_lambda must be non-negative."
        )
    if args.stage_support_top_k < 1:
        raise ValueError("--stage_support_top_k must be positive.")
    if not 0.0 < args.stage_temporal_scale_fraction <= 1.0:
        raise ValueError(
            "--stage_temporal_scale_fraction must be in (0, 1]."
        )
    if args.stage_temporal_scale_min_hours <= 0.0:
        raise ValueError(
            "--stage_temporal_scale_min_hours must be positive."
        )
    if args.stage_event_metadata_size < 1:
        raise ValueError("--stage_event_metadata_size must be positive.")
    if args.stage_pulse_bin_hours <= 0.0:
        raise ValueError("--stage_pulse_bin_hours must be positive.")
    if args.dataset not in {"twitter12", "twitter18"} and args.dataset_threshold is None:
        raise ValueError(
            "Custom datasets require --dataset_threshold."
        )
    if not 0.0 <= args.stress_noise_rate <= 1.0:
        raise ValueError("--stress_noise_rate must be in [0, 1].")
    return args


def build_parser():
    parser = argparse.ArgumentParser(
        description="Reviewer-controlled MvSED experiments based on RagSEDE."
    )
    parser.add_argument("--dataset", default="twitter12")
    parser.add_argument("--anchor_preprocessing", choices=("legacy", "ragsede", "language"), default="legacy",
                        help="Explicit anchor-only preprocessing. Legacy preserves historical variant-dependent behavior.")
    parser.add_argument("--anchor_only", action="store_true", help="Construct and audit anchors only; never call RAGFlow/LLMs.")
    parser.add_argument("--expected_anchor_audit", help="Fail before LLM work if anchor evidence differs from this audit JSON.")
    parser.add_argument(
        "--data_path",
        help="Optional JSON block file for a custom dataset.",
    )
    parser.add_argument(
        "--language",
        choices=("English", "French"),
        help="Preprocessing language; inferred for Twitter2012/2018.",
    )
    parser.add_argument(
        "--dataset_threshold",
        type=float,
        help="Anchor threshold for a custom dataset.",
    )
    parser.add_argument(
        "--skip_initial_block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the original RagSEDE convention that blocks[0] is initialization.",
    )
    parser.add_argument("--HOST_ADDRESS", default="http://127.0.0.1")
    parser.add_argument("--API_KEY", default=os.getenv("RAGFLOW_API_KEY"))
    parser.add_argument(
        "--variant",
        choices=(
            "full",
            "full_refined",
            "full_refined_v2",
            "full_balanced",
            "w/o_dkms",
            "w/o_mvra",
            "w/o_hslg",
            "refined_w/o_dkms",
            "refined_w/o_mvra",
            "refined_w/o_hslg",
            "refined_v2_w/o_dkms",
            "refined_v2_w/o_mvra",
            "refined_v2_w/o_hslg",
            "balanced_w/o_dkms",
            "balanced_w/o_mvra",
            "balanced_w/o_hslg",
            "mvra_conrag",
            "mvra_conrag_no_hslg",
            "mvra_conrag_no_bonus",
            "mvra_legacy_control",
            "mvra_no_mvra_control",
            "fixed_anchor_control",
            "fixed_anchor_matched_control",
            "fixed_anchor_mvra",
            "fixed_anchor_hslg",
            "fixed_anchor_mvra_hslg",
            "conrag_aligned_text",
            "conrag_aligned_text_entity",
            "conrag_aligned_text_relation",
            "conrag_aligned_three_no_bonus",
            "conrag_aligned_three",
            "conrag_aligned_three_hslg",
            "conrag_object_text",
            "conrag_object_text_hslg",
            "conrag_object_three",
            "conrag_object_three_hslg",
            "conrag_object_consensus",
            "conrag_object_consensus_hslg",
            "conrag_cqc_consensus",
            "conrag_cqc_consensus_hslg",
            "ugr_mvra",
            "ugr_mvra_hslg",
            "ugr_mvra_consensus_hslg",
            "ugr_mvra_buffered",
            "ugr_mvra_buffered_local_hslg",
            "stage_aware_text_control",
            "stage_aware_all_view_mvra",
            "stage_aware_selective_mvra_no_time",
            "stage_aware_selective_mvra",
            "stage_aware_selective_gate",
            "stage_aware_selective_gate_no_time",
            "stage_aware_selective_gate_time_aux",
            "two_stage_dynamic",
            "two_stage_dynamic_mvra",
            "two_stage_dynamic_full",
            "two_stage_safe_dynamic",
            "two_stage_safe_dynamic_mvra",
            "two_stage_safe_dynamic_full",
            "two_stage_consistent_dynamic",
            "two_stage_consistent_dynamic_mvra",
            "two_stage_consistent_dynamic_full",
            "single_query",
            "ragsede_original",
            "ragsede_control",
            "ragsede_plus_dkms",
            "ragsede_plus_dkms_mvra",
            "llm_free",
        ),
        default="full",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_block", type=int, default=1)
    parser.add_argument("--end_block", type=int)

    parser.add_argument("--evaluator_prompt", default=DEFAULT_EVALUATOR_PROMPT)
    parser.add_argument("--detector_prompt", default=DEFAULT_DETECTOR_PROMPT)
    parser.add_argument(
        "--llm_model", default="deepseek-r1-distill-qwen-32b"
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--max_llm_attempts", type=int, default=10)
    parser.add_argument(
        "--llm_retry_delay",
        type=float,
        default=5.0,
        help="Seconds to wait before retrying a temporary LLM/RAGFlow error.",
    )
    parser.add_argument(
        "--max_llm_calls",
        type=int,
        help="Optional matched call budget shared by all compared LLM variants.",
    )
    parser.add_argument(
        "--max_total_tokens",
        type=int,
        help="Optional matched prompt+completion token budget.",
    )
    parser.add_argument("--max_keywords", type=int, default=10)
    parser.add_argument(
        "--event_keyword_memory_size",
        type=int,
        default=30,
        help="Maximum accumulated evidence keywords per refined event memory.",
    )
    parser.add_argument("--kb_context_chars", type=int, default=400)
    parser.add_argument("--max_input_chars", type=int, default=200)

    parser.add_argument("--local_embedding_model")
    parser.add_argument(
        "--embedding_device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--ragflow_embedding_model", default="BAAI/bge-small-zh-v1.5"
    )
    parser.add_argument("--reuse_embedding_cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_cluster_size", type=int, default=100)
    parser.add_argument("--anchor_top_k", type=int, default=3)
    parser.add_argument("--anchor_lambda", type=float, default=0.7)
    parser.add_argument(
        "--anchor_pipeline",
        choices=(
            "single_stage",
            "dynamic_two_stage",
            "legacy_two_stage",
        ),
        default="single_stage",
        help=(
            "single_stage constructs anchors once; dynamic_two_stage first "
            "uses the fixed RagSEDE threshold and then applies a label-free "
            "density-derived threshold to the resulting super-anchors; "
            "legacy_two_stage is a backward-compatible alias"
        ),
    )
    parser.add_argument("--twitter12_threshold", type=float, default=0.40)
    parser.add_argument("--twitter18_threshold", type=float, default=0.35)
    parser.add_argument(
        "--ragsede_twitter18_threshold", type=float, default=0.30
    )

    parser.add_argument("--enable_dkms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dkms_base_threshold", type=float)
    parser.add_argument("--dkms_density_reference", type=float, default=0.15)
    parser.add_argument("--dkms_density_scale", type=float, default=0.8)
    parser.add_argument("--dkms_threshold_min", type=float, default=0.25)
    parser.add_argument("--dkms_threshold_max", type=float, default=0.55)
    parser.add_argument(
        "--dkms_never_below_base",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Prevent the second-stage density-aware threshold from falling "
            "below the first-stage fixed threshold. Safe dynamic variants "
            "enable this automatically."
        ),
    )
    parser.add_argument(
        "--dkms_stage2_linkage",
        choices=("star", "complete"),
        default="star",
        help=(
            "Second-stage super-anchor grouping rule. complete requires "
            "pairwise-consistent groups and is enabled automatically by "
            "the consistent dynamic variants."
        ),
    )
    parser.add_argument("--dkms_lambda", type=float, default=0.7)
    parser.add_argument(
        "--dkms_refine_trigger_size",
        type=int,
        default=100,
        help="Minimum cluster size considered for cohesion-aware refinement.",
    )
    parser.add_argument(
        "--dkms_refine_hard_max_size",
        type=int,
        default=400,
        help="Safety limit for a single refined semantic cluster.",
    )
    parser.add_argument(
        "--dkms_min_cohesion",
        type=float,
        default=0.50,
        help="Lower-quartile cosine cohesion required to preserve a large cluster.",
    )
    parser.add_argument(
        "--dkms_split_threshold_step", type=float, default=0.05
    )
    parser.add_argument(
        "--dkms_split_threshold_max", type=float, default=0.75
    )
    parser.add_argument("--dkms_split_max_depth", type=int, default=6)

    parser.add_argument("--enable_mvra", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--view_count", type=int, default=3)
    parser.add_argument("--view_separator", default=" | ")
    parser.add_argument(
        "--preserve_anchor_views",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Preserve representative-message separators even when MVRA is "
            "disabled, enabling exactly matched fixed-anchor controls."
        ),
    )
    parser.add_argument(
        "--mvra_fallback_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the original single-query detector first and invoke MVRA "
            "only for an Others/Confusion result."
        ),
    )
    parser.add_argument(
        "--mvra_fallback_on_confusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also attempt conservative MVRA after a Confusion result.",
    )
    parser.add_argument(
        "--mvra_require_consensus_for_non_others",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Require the selected fallback event to pass the explicit "
            "cross-view consensus gate."
        ),
    )
    parser.add_argument("--mvra_start_block", type=int, default=1)
    parser.add_argument("--retrieval_top_k", type=int, default=8)
    parser.add_argument("--retrieval_score_threshold", type=float, default=0.10)
    parser.add_argument(
        "--mvra_accept_threshold",
        type=float,
        default=0.60,
        help="Minimum top-1 MVRA score for the refined confidence gate.",
    )
    parser.add_argument(
        "--mvra_accept_margin",
        type=float,
        default=0.05,
        help="Minimum top-1/top-2 score margin for the refined confidence gate.",
    )
    parser.add_argument(
        "--mvra_gate_min_candidates",
        type=int,
        default=2,
        help="Minimum retrieved candidates before the refined gate may fire.",
    )
    parser.add_argument(
        "--mvra_consensus_min_views",
        type=int,
        default=2,
        help="Minimum agreeing views for the v2 MVRA consensus gate.",
    )
    parser.add_argument(
        "--mvra_view_accept_threshold",
        type=float,
        default=0.40,
        help="Minimum per-view cosine score counted as consensus support.",
    )
    parser.add_argument(
        "--mvra_consensus_accept_threshold",
        type=float,
        default=0.45,
        help="Minimum fused score for label-free Others reassignment.",
    )
    parser.add_argument(
        "--mvra_consensus_margin",
        type=float,
        default=0.02,
        help="Minimum consensus top-1/top-2 fused-score margin.",
    )
    parser.add_argument(
        "--mvra_single_view_accept_threshold",
        type=float,
        default=0.58,
        help="Strong-match threshold for balanced single-view fallback.",
    )
    parser.add_argument(
        "--mvra_single_view_margin",
        type=float,
        default=0.04,
        help="Top-1/top-2 margin for balanced single-view fallback.",
    )
    parser.add_argument(
        "--mvra_candidate_top_k_per_view",
        type=int,
        default=3,
        help=(
            "Per-view candidate depth used by ConRAG-style consensus."
        ),
    )
    parser.add_argument(
        "--mvra_support_score_gap",
        type=float,
        default=0.08,
        help=(
            "Maximum raw-score gap from a view's top event for that view "
            "to support another candidate."
        ),
    )
    parser.add_argument(
        "--mvra_consensus_bonus",
        type=float,
        default=0.05,
        help="Multiplicative bonus for support from multiple views.",
    )
    parser.add_argument(
        "--mvra_conrag_primary_weight",
        type=float,
        default=0.60,
        help=(
            "Weight assigned to the most central representative-message "
            "view; the remaining weight is shared by auxiliary views."
        ),
    )
    parser.add_argument(
        "--mvra_override_non_others",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow a strong multi-view consensus to replace an LLM event "
            "selection supported by fewer than the required views."
        ),
    )
    parser.add_argument(
        "--mvra_override_accept_threshold",
        type=float,
        default=0.55,
        help="Minimum fused score for replacing a non-Others selection.",
    )
    parser.add_argument(
        "--mvra_override_margin",
        type=float,
        default=0.05,
        help="Minimum fused top-1/top-2 margin for a consensus override.",
    )
    parser.add_argument(
        "--conrag_text_weight",
        type=float,
        default=0.65,
        help="Primary text-evidence weight in aligned ConRAG retrieval.",
    )
    parser.add_argument(
        "--conrag_entity_weight",
        type=float,
        default=0.15,
        help="Auxiliary entity-anchor weight in aligned ConRAG retrieval.",
    )
    parser.add_argument(
        "--conrag_relation_weight",
        type=float,
        default=0.20,
        help="Auxiliary relation/action weight in aligned ConRAG retrieval.",
    )
    parser.add_argument(
        "--conrag_consensus_lambda",
        type=float,
        default=0.05,
        help="Small multiplicative bonus for support from multiple views.",
    )
    parser.add_argument(
        "--conrag_per_view_top_k",
        type=int,
        default=8,
        help="Candidate depth retained independently by each aligned view.",
    )
    parser.add_argument(
        "--conrag_view_score_threshold",
        type=float,
        default=-1.0,
        help=(
            "Minimum raw cosine score for a view hit; -1 keeps the standard "
            "top-k retrieval protocol used by ConRAG."
        ),
    )
    parser.add_argument(
        "--conrag_event_evidence_size",
        type=int,
        default=12,
        help="Maximum recent source anchors retained for each event view.",
    )
    parser.add_argument(
        "--conrag_event_text_chars",
        type=int,
        default=1200,
        help="Maximum source-memory characters encoded per event.",
    )
    parser.add_argument(
        "--conrag_max_view_units",
        type=int,
        default=32,
        help="Maximum auditable entity or relation units per view.",
    )
    parser.add_argument(
        "--conrag_object_top_k",
        type=int,
        default=3,
        help=(
            "Top graph objects/evidence units retrieved independently in each "
            "object-level ConRAG view."
        ),
    )
    parser.add_argument(
        "--conrag_relation_beta",
        type=float,
        default=0.02,
        help=(
            "Residual weight for relation hits after the strongest relation "
            "supporting one evidence unit."
        ),
    )
    parser.add_argument(
        "--conrag_entity_context_window",
        type=int,
        default=4,
        help="Source-token radius used to textualize one entity object.",
    )
    parser.add_argument(
        "--conrag_object_contexts",
        type=int,
        default=4,
        help="Maximum grounded source contexts retained per entity object.",
    )
    parser.add_argument(
        "--conrag_require_structural_consensus",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep every text hit, but admit a structural-only evidence unit "
            "only when enough structural views support that same unit."
        ),
    )
    parser.add_argument(
        "--conrag_structural_min_views",
        type=int,
        default=2,
        help=(
            "Required structural views for a non-text evidence candidate in "
            "the conservative object-level ConRAG variants."
        ),
    )
    parser.add_argument(
        "--ugr_min_support_views",
        type=int,
        default=2,
        help=(
            "Minimum text/entity/relation views supporting the same event "
            "before UGR-MVRA may repair an uncertain primary result."
        ),
    )
    parser.add_argument(
        "--ugr_min_structural_views",
        type=int,
        default=1,
        help=(
            "Minimum entity/relation views supporting the UGR candidate; "
            "the independent text-evidence threshold is always required."
        ),
    )
    parser.add_argument(
        "--ugr_min_text_similarity",
        type=float,
        default=0.45,
        help=(
            "Minimum raw cosine similarity to grounded event text evidence "
            "for a residual repair."
        ),
    )
    parser.add_argument(
        "--ugr_min_fused_score",
        type=float,
        default=0.40,
        help="Minimum object-view fused score for a residual repair.",
    )
    parser.add_argument(
        "--ugr_min_margin",
        type=float,
        default=0.03,
        help="Minimum top-1/top-2 fused-score margin for a residual repair.",
    )
    parser.add_argument(
        "--cqc_paraphrase_count",
        type=int,
        default=2,
        help=(
            "Number of deterministic meaning-preserving query reformulations; "
            "the original query is always included as an additional view."
        ),
    )
    parser.add_argument(
        "--cqc_query_chars",
        type=int,
        default=1200,
        help="Maximum characters retained in each CQC query view.",
    )
    parser.add_argument(
        "--cqc_semantic_similarity_threshold",
        type=float,
        default=0.85,
        help=(
            "Minimum embedding similarity to the original query required for "
            "a reformulated query to enter consistency scoring."
        ),
    )
    parser.add_argument(
        "--cqc_shared_pool_k",
        type=int,
        default=8,
        help=(
            "Maximum candidates copied from the single original-query ConRAG "
            "pool; reformulations cannot add candidates."
        ),
    )
    parser.add_argument(
        "--cqc_min_votes",
        type=int,
        default=2,
        help="Minimum top-1 query-view votes required for event acceptance.",
    )
    parser.add_argument(
        "--cqc_min_mean_score",
        type=float,
        default=0.35,
        help="Minimum mean evidence similarity across accepted query views.",
    )
    parser.add_argument(
        "--cqc_variance_lambda",
        type=float,
        default=0.50,
        help=(
            "Base coefficient in the adaptive mean-minus-variance consistency "
            "score."
        ),
    )
    parser.add_argument(
        "--cqc_margin",
        type=float,
        default=0.03,
        help="Minimum consistency-score margin over the runner-up event.",
    )
    parser.add_argument(
        "--cqc_event_evidence_size",
        type=int,
        default=4,
        help="Recent grounded evidence messages scored for each shared candidate.",
    )
    parser.add_argument(
        "--cqc_evidence_chars",
        type=int,
        default=300,
        help="Maximum evidence characters exposed to scoring and the detector.",
    )

    parser.add_argument(
        "--stage_candidate_pool_k",
        type=int,
        default=8,
        help=(
            "Monolithic-text candidate pool size; decomposed views may only "
            "rerank these events."
        ),
    )
    parser.add_argument(
        "--stage_max_views",
        type=int,
        default=3,
        help="Maximum selected views including the mandatory text view.",
    )
    parser.add_argument(
        "--stage_text_weight",
        type=float,
        default=0.65,
        help="Mandatory monolithic-text weight after selective reranking.",
    )
    parser.add_argument(
        "--stage_view_min_alignment",
        type=float,
        default=0.05,
        help="Minimum centered score-profile alignment to the text ranking.",
    )
    parser.add_argument(
        "--stage_view_min_gain",
        type=float,
        default=0.015,
        help="Minimum label-free marginal utility for selecting an extra view.",
    )
    parser.add_argument(
        "--stage_view_redundancy_lambda",
        type=float,
        default=0.35,
        help="Penalty for score-profile redundancy among auxiliary views.",
    )
    parser.add_argument(
        "--stage_support_top_k",
        type=int,
        default=3,
        help="Per-selected-view support depth inside the fixed pool.",
    )
    parser.add_argument(
        "--stage_temporal_scale_fraction",
        type=float,
        default=0.25,
        help="Temporal kernel scale as a fraction of the block time span.",
    )
    parser.add_argument(
        "--stage_temporal_scale_min_hours",
        type=float,
        default=1.0,
        help="Minimum temporal-kernel scale in hours.",
    )
    parser.add_argument(
        "--stage_event_metadata_size",
        type=int,
        default=12,
        help="Maximum online metadata observations retained per event.",
    )
    parser.add_argument(
        "--stage_pulse_bin_hours",
        type=float,
        default=1.0,
        help="Unsupervised message-volume bin width for pulse strength.",
    )

    parser.add_argument(
        "--hslg_mode",
        choices=("hybrid", "semantic", "lexical", "consensus", "none"),
        default="hybrid",
    )
    parser.add_argument("--hslg_semantic_weight", type=float, default=0.7)
    parser.add_argument("--hslg_threshold", type=float, default=0.30)
    parser.add_argument(
        "--hslg_reciprocal_threshold",
        type=float,
        default=0.60,
        help=(
            "Minimum hybrid similarity for refined reciprocal-neighbour "
            "contraction."
        ),
    )
    parser.add_argument(
        "--hslg_adaptive_quantile", type=float, default=0.85
    )
    parser.add_argument(
        "--hslg_adaptive_threshold_min", type=float, default=0.30
    )
    parser.add_argument(
        "--hslg_adaptive_threshold_max", type=float, default=0.70
    )
    parser.add_argument(
        "--hslg_min_lexical_overlap", type=float, default=0.05
    )
    parser.add_argument(
        "--hslg_semantic_override", type=float, default=0.80
    )
    parser.add_argument("--hslg_mutual_top_k", type=int, default=2)
    parser.add_argument(
        "--hslg_mutual_threshold", type=float, default=0.55
    )
    parser.add_argument(
        "--hslg_max_community_size",
        type=int,
        default=0,
        help="Maximum event-memory nodes per HSLG community; 0 is unlimited.",
    )
    parser.add_argument(
        "--hslg_consensus_semantic_threshold",
        type=float,
        default=0.72,
        help="Minimum text-view similarity for a consensus HSLG edge.",
    )
    parser.add_argument(
        "--hslg_consensus_entity_threshold",
        type=float,
        default=0.10,
        help="Minimum entity-set Jaccard support for a consensus HSLG edge.",
    )
    parser.add_argument(
        "--hslg_consensus_relation_threshold",
        type=float,
        default=0.05,
        help="Minimum relation-set Jaccard support for a consensus HSLG edge.",
    )
    parser.add_argument(
        "--hslg_consensus_min_views",
        type=int,
        default=2,
        help="Minimum agreeing text/entity/relation views for HSLG.",
    )
    parser.add_argument(
        "--hslg_consensus_score_threshold",
        type=float,
        default=0.52,
        help="Minimum fused multi-view score for a consensus HSLG edge.",
    )
    parser.add_argument(
        "--hslg_consensus_top_k",
        type=int,
        default=3,
        help="Mutual-neighbour depth used by consensus HSLG.",
    )
    parser.add_argument(
        "--hslg_consensus_max_component_size",
        type=int,
        default=4,
        help="Complete-link event-node cap for consensus HSLG.",
    )
    parser.add_argument(
        "--novelty_buffer_max_anchors",
        type=int,
        default=32,
        help=(
            "Maximum unresolved Others anchors retained before the oldest "
            "anchor is safely materialized as a singleton event."
        ),
    )
    parser.add_argument(
        "--novelty_buffer_ttl_anchors",
        type=int,
        default=16,
        help=(
            "Maximum number of subsequent anchors an unresolved Others "
            "anchor may wait in the online buffer."
        ),
    )
    parser.add_argument(
        "--novelty_buffer_min_cluster_anchors",
        type=int,
        default=2,
        help="Minimum buffered anchors required to create a grouped event.",
    )
    parser.add_argument(
        "--novelty_buffer_context_anchors",
        type=int,
        default=4,
        help="Maximum buffered representative anchors sent to the evaluator.",
    )
    parser.add_argument(
        "--novelty_buffer_semantic_threshold",
        type=float,
        default=0.68,
        help="Minimum text similarity for a buffered novelty edge.",
    )
    parser.add_argument(
        "--novelty_buffer_entity_threshold",
        type=float,
        default=0.05,
        help="Minimum entity Jaccard support for a local consensus edge.",
    )
    parser.add_argument(
        "--novelty_buffer_relation_threshold",
        type=float,
        default=0.02,
        help="Minimum relation Jaccard support for a local consensus edge.",
    )
    parser.add_argument(
        "--novelty_buffer_min_views",
        type=int,
        default=2,
        help="Minimum agreeing text/entity/relation views for a local edge.",
    )
    parser.add_argument(
        "--novelty_buffer_score_threshold",
        type=float,
        default=0.46,
        help="Minimum fused score for a local multi-view novelty edge.",
    )
    parser.add_argument(
        "--novelty_buffer_top_k",
        type=int,
        default=3,
        help="Mutual-neighbour depth in the bounded novelty graph.",
    )
    parser.add_argument(
        "--novelty_buffer_max_component_size",
        type=int,
        default=8,
        help="Complete-link cap for a buffered novelty group.",
    )
    parser.add_argument("--entropy_epsilon", type=float, default=1e-5)

    parser.add_argument(
        "--knowledge_control",
        choices=("none", "event_alias", "entity_mask", "both"),
        default="none",
    )
    parser.add_argument("--llm_free_threshold", type=float, default=0.55)
    parser.add_argument("--save_audit_log", action="store_true")
    parser.add_argument(
        "--save_intermediate_state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save pre-HSLG predictions and event memory for offline analysis.",
    )

    parser.add_argument("--detector_similarity_threshold", type=float, default=0.0)
    parser.add_argument("--detector_keywords_similarity_weight", type=float, default=0.5)
    parser.add_argument("--ragflow_top_n", type=int, default=8)
    parser.add_argument("--ragflow_top_k", type=int, default=1024)
    parser.add_argument("--ragflow_ready_attempts", type=int, default=60)
    parser.add_argument("--ragflow_retry_delay", type=float, default=10.0)
    parser.add_argument("--chat_ready_delay", type=float, default=5.0)
    parser.add_argument("--detector_ready_delay", type=float, default=2.0)
    parser.add_argument("--kb_write_delay", type=float, default=1.0)
    parser.add_argument("--max_messages_per_event", type=int, default=10)
    parser.add_argument("--update_context_size", type=int, default=4)

    parser.add_argument("--input_cost_per_million", type=float, default=0.0)
    parser.add_argument("--output_cost_per_million", type=float, default=0.0)
    parser.add_argument("--build_pareto", help="Directory containing *_metrics.json files.")
    parser.add_argument(
        "--pareto_x",
        choices=("estimated_cost", "elapsed_seconds"),
        default="elapsed_seconds",
    )
    parser.add_argument(
        "--pareto_metric", choices=("NMI", "AMI", "ARI"), default="ARI"
    )
    parser.add_argument("--pareto_output")
    parser.add_argument(
        "--summarize_results",
        help="Directory containing repeated-run *_metrics.json files.",
    )
    parser.add_argument(
        "--compare_to",
        default="full",
        help="Reference variant used for paired tests and win/tie/loss.",
    )
    parser.add_argument("--summary_output")
    parser.add_argument("--tie_tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--view_selection",
        choices=("scored", "random", "first"),
        default="scored",
    )
    parser.add_argument(
        "--mvra_fusion",
        choices=("max", "mean", "concat", "conrag"),
        default="max",
    )
    parser.add_argument(
        "--stress_test",
        choices=(
            "none",
            "entity_collision",
            "time_location_shift",
            "lexical_noise",
        ),
        default="none",
    )
    parser.add_argument("--stress_noise_rate", type=float, default=0.15)
    return parser


def main():
    parser = build_parser()
    args = resolve_args(parser.parse_args())
    if args.build_pareto:
        build_pareto_report(args)
        return
    if args.summarize_results:
        build_statistical_report(args)
        return
    if args.variant != "llm_free" and not args.anchor_only and not args.API_KEY:
        parser.error("--API_KEY or RAGFLOW_API_KEY is required for LLM variants.")
    set_reproducible(args.seed)
    blocks = load_data_blocks(args.dataset, data_path=args.data_path)
    blocks = apply_stress_test(blocks, args)
    start_run(args, blocks)


if __name__ == "__main__":
    main()
