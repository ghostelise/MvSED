import hashlib
import json
import os
import pickle
import time

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from get_key_messages import select_representative_messages
from utils import preprocess_french_sentence, preprocess_sentence


ANCHOR_AUDIT_FIELDS = (
    "raw_text_sha256", "preprocessed_text_sha256", "embedding_model",
    "effective_threshold", "preprocessing", "message_count", "anchor_count",
    "ordered_membership_sha256", "ordered_representatives_sha256",
    "representative_text_sha256",
)


def anchor_preprocessor(args):
    """Explicit opt-in alignment; legacy remains unchanged for existing runs."""
    mode = getattr(args, "anchor_preprocessing", "legacy")
    if mode == "ragsede":
        return preprocess_sentence, "ragsede"
    if mode == "language":
        return (preprocess_sentence, "ragsede") if args.language == "English" else (preprocess_french_sentence, "french")
    if mode != "legacy":
        raise ValueError(f"Unknown anchor preprocessing mode: {mode}")
    return (preprocess_sentence, "ragsede") if (
        args.language == "English" or getattr(args, "variant", None) == "ragsede_original"
    ) else (preprocess_french_sentence, "french")


def _audit_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_anchor_audit(actual, expected):
    differences = [key for key in ANCHOR_AUDIT_FIELDS if key not in actual or key not in expected or actual[key] != expected[key]]
    if differences:
        raise ValueError("Anchor protocol mismatch before LLM calls: " + ", ".join(differences))


def _cache_paths(args, block_id):
    model_key = hashlib.sha1(
        args.local_embedding_model.encode("utf-8")
    ).hexdigest()[:10]
    output_dir = os.path.join("ckpts", args.dataset, f"M{block_id}")
    embedding_path = os.path.join(
        output_dir, f"M{block_id}_SBERT_embeddings_{model_key}.pkl"
    )
    metadata_path = embedding_path + ".json"
    return output_dir, embedding_path, metadata_path


def get_chunk_embeddings(args, block_id, block, embedding_model, stats=None):
    output_dir, embedding_path, metadata_path = _cache_paths(args, block_id)
    os.makedirs(output_dir, exist_ok=True)
    texts = [message["text"] for message in block["test"]]
    preprocess, preprocessing_name = anchor_preprocessor(args)
    processed_texts = [preprocess(text) for text in texts]
    fingerprint = hashlib.sha256(
        "\n".join(processed_texts).encode("utf-8")
    ).hexdigest()
    expected_metadata = {
        "dataset": args.dataset,
        "block": block_id,
        "model": args.local_embedding_model,
        "message_count": len(processed_texts),
        "text_sha256": fingerprint,
    }
    if stats is not None:
        stats.anchor_audit = {
            "raw_text_sha256": _audit_digest(texts),
            "preprocessed_text_sha256": _audit_digest(processed_texts),
            "embedding_model": args.local_embedding_model,
            "preprocessing": preprocessing_name,
            "message_count": len(texts),
        }

    cache_valid = False
    if args.reuse_embedding_cache and os.path.exists(embedding_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as file:
                cache_valid = json.load(file) == expected_metadata
        except (OSError, json.JSONDecodeError):
            cache_valid = False
    if cache_valid:
        return embedding_path

    print("Encoding preprocessed text anchors.")
    started = time.perf_counter()
    embeddings = embedding_model.encode(
        processed_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
    ).cpu()
    if stats is not None:
        stats.record_embedding(
            len(processed_texts), time.perf_counter() - started
        )
    with open(embedding_path, "wb") as file:
        pickle.dump(embeddings, file)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(expected_metadata, file, indent=2)
    print(f"Stored embedding cache: {embedding_path}")
    return embedding_path


def _cluster_cohesion(similarity_matrix, cluster):
    """Return a robust, label-free cohesion score for one semantic cluster."""
    if len(cluster) <= 1:
        return 1.0
    local_similarity = similarity_matrix[np.ix_(cluster, cluster)]
    pair_indices = np.triu_indices(len(cluster), k=1)
    pairwise = local_similarity[pair_indices]
    if not len(pairwise):
        return 1.0
    # The lower quartile is more sensitive than the mean to a minority topic
    # hidden inside a large cluster, while remaining deterministic and fully
    # independent of the task labels.
    return float(np.quantile(pairwise, 0.25))


def _agglomerative_partition(similarity_matrix, cluster, threshold, n_clusters=None):
    """Partition cluster indices using only their pairwise cosine distances."""
    if len(cluster) <= 1:
        return [cluster]
    local_similarity = similarity_matrix[np.ix_(cluster, cluster)]
    local_distance = np.clip(1.0 - local_similarity, 0.0, 2.0)
    np.fill_diagonal(local_distance, 0.0)
    if n_clusters is None:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=1.0 - threshold,
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=int(n_clusters),
            metric="precomputed",
            linkage="average",
        )
    labels = clustering.fit_predict(local_distance)
    partitions = [[] for _ in range(int(max(labels)) + 1)]
    for local_index, label in enumerate(labels):
        partitions[int(label)].append(cluster[local_index])
    return [partition for partition in partitions if partition]


def refine_semantic_clusters(
    similarity_matrix,
    clusters,
    initial_threshold,
    trigger_size,
    hard_max_size,
    min_cohesion,
    threshold_step,
    threshold_max,
    max_depth,
):
    """Recursively split only heterogeneous or pathologically large clusters.

    This replaces the legacy consecutive slicing rule with a semantic rule. A
    large but cohesive cluster is preserved; a low-cohesion cluster is split at
    a stricter cosine threshold. The hard limit is only a safety guard against
    a single generic hub swallowing a substantial fraction of a block.
    """
    refined = []
    diagnostics = {
        "initial_largest_cluster": max((len(cluster) for cluster in clusters), default=0),
        "recursive_splits": 0,
        "low_cohesion_splits": 0,
        "forced_splits": 0,
    }
    queue = [(cluster, float(initial_threshold), 0) for cluster in clusters]

    while queue:
        cluster, parent_threshold, depth = queue.pop(0)
        size = len(cluster)
        cohesion = _cluster_cohesion(similarity_matrix, cluster)
        is_low_cohesion = size > trigger_size and cohesion < min_cohesion
        exceeds_hard_limit = size > hard_max_size
        if not is_low_cohesion and not exceeds_hard_limit:
            refined.append(cluster)
            continue

        split_threshold = min(
            threshold_max, parent_threshold + threshold_step
        )
        partitions = _agglomerative_partition(
            similarity_matrix, cluster, split_threshold
        )

        # If the stricter cut still returns one giant component, continue
        # increasing the semantic threshold before considering a forced split.
        while len(partitions) == 1 and split_threshold < threshold_max:
            split_threshold = min(
                threshold_max, split_threshold + threshold_step
            )
            partitions = _agglomerative_partition(
                similarity_matrix, cluster, split_threshold
            )

        if len(partitions) == 1 and exceeds_hard_limit:
            target_count = int(np.ceil(size / hard_max_size))
            partitions = _agglomerative_partition(
                similarity_matrix,
                cluster,
                split_threshold,
                n_clusters=max(2, target_count),
            )
            diagnostics["forced_splits"] += 1

        if len(partitions) == 1:
            # The cluster remains semantically inseparable at the configured
            # maximum threshold, so preserving it is safer than order slicing.
            refined.append(cluster)
            continue

        diagnostics["recursive_splits"] += 1
        if is_low_cohesion:
            diagnostics["low_cohesion_splits"] += 1
        if depth + 1 >= max_depth:
            for partition in partitions:
                if len(partition) > hard_max_size:
                    target_count = int(
                        np.ceil(len(partition) / hard_max_size)
                    )
                    refined.extend(
                        _agglomerative_partition(
                            similarity_matrix,
                            partition,
                            split_threshold,
                            n_clusters=max(2, target_count),
                        )
                    )
                    diagnostics["forced_splits"] += 1
                else:
                    refined.append(partition)
            continue
        queue.extend(
            (partition, split_threshold, depth + 1)
            for partition in partitions
        )

    diagnostics["refined_cluster_count"] = len(refined)
    diagnostics["final_largest_cluster"] = max(
        (len(cluster) for cluster in refined), default=0
    )
    return refined, diagnostics


def cluster_embeddings(
    embeddings,
    similarity_threshold=0.8,
    max_cluster_size=5,
    cohesion_refinement=False,
    refinement_options=None,
    diagnostics=None,
):
    if len(embeddings) == 0:
        return []
    if len(embeddings) == 1:
        return [[0]]
    similarity_matrix = cosine_similarity(embeddings)
    
    distance_matrix = 1 - similarity_matrix
    
    clustering = AgglomerativeClustering(
        n_clusters=None,  
        metric='precomputed', 
        linkage='average', 
        distance_threshold=1 - similarity_threshold 
    )
    labels = clustering.fit_predict(distance_matrix)
    
    clusters = [[] for _ in range(max(labels) + 1)]
    for idx, label in enumerate(labels):
        clusters[label].append(idx)
    
    if cohesion_refinement:
        options = refinement_options or {}
        clusters, refinement_diagnostics = refine_semantic_clusters(
            similarity_matrix,
            clusters,
            initial_threshold=similarity_threshold,
            trigger_size=options.get("trigger_size", 100),
            hard_max_size=options.get("hard_max_size", 400),
            min_cohesion=options.get("min_cohesion", 0.50),
            threshold_step=options.get("threshold_step", 0.05),
            threshold_max=options.get("threshold_max", 0.75),
            max_depth=options.get("max_depth", 6),
        )
        if diagnostics is not None:
            diagnostics.update(refinement_diagnostics)
        return clusters

    # MvSED's DKMS definition selects representative messages directly from a
    # semantic cluster, regardless of how large that cluster is.  The previous
    # implementation always cut clusters into consecutive chunks of 100.  That
    # order-dependent split created multiple provisional events for one hot
    # topic and was a major source of fragmentation.  A non-positive/None limit
    # therefore means "preserve the semantic cluster"; the legacy controls keep
    # their historical finite limit.
    if max_cluster_size is None or max_cluster_size <= 0:
        return clusters

    final_clusters = []
    for cluster in clusters:
        if len(cluster) <= max_cluster_size:
            final_clusters.append(cluster)
        else:
            for i in range(0, len(cluster), max_cluster_size):
                final_clusters.append(cluster[i:i + max_cluster_size])
    
    return final_clusters

def create_anchors_blocks_and_ids(
    key_id, clusters, blk, view_separator=" | "
):
    text_list = []
    for message in blk['test']:
        text_list.append(message['text'])

    anchors = []
    anchors_id = []
    
    for i, cluster in enumerate(clusters):
        # Keep representative messages as explicit views. MVRA can split this
        # separator directly; RagSEDE-style controls simply treat the complete
        # string as one anchor.
        anchor = view_separator.join([text_list[idx] for idx in key_id[i]])
        anchors.append(anchor)
        anchors_id.append(cluster)
    
    return anchors, anchors_id

def get_anchers(args, b, blk, embedding_model, stats=None):
    embeddings_path = get_chunk_embeddings(
        args, b, blk, embedding_model, stats=stats
    )
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    sampling_started = time.perf_counter()
    
    if args.dataset == 'twitter12':
        base_threshold = args.twitter12_threshold
    elif args.dataset == 'twitter18':
        base_threshold = (
            args.ragsede_twitter18_threshold
            if getattr(args, "variant", None) == "ragsede_original"
            else args.twitter18_threshold
        )
    else:
        base_threshold = args.dataset_threshold

    single_stage_dkms = (
        getattr(args, "anchor_pipeline", "single_stage") == "single_stage"
        and getattr(args, "enable_dkms", False)
    )
    if single_stage_dkms and len(embeddings) > 1:
        similarity_matrix = cosine_similarity(embeddings)
        pair_indices = np.triu_indices_from(similarity_matrix, k=1)
        block_density = float(np.mean(similarity_matrix[pair_indices]))
        threshold = (
            args.dkms_base_threshold
            if args.dkms_base_threshold is not None
            else base_threshold
        )
        threshold += (
            block_density - args.dkms_density_reference
        ) * args.dkms_density_scale
        threshold = float(
            np.clip(
                threshold,
                args.dkms_threshold_min,
                args.dkms_threshold_max,
            )
        )
        print(
            "Single-stage DKMS on raw messages: "
            f"density={block_density:.3f}, threshold={threshold:.3f}."
        )
    else:
        block_density = 0.0
        threshold = base_threshold
        pipeline = getattr(args, "anchor_pipeline", "single_stage")
        protocol = (
            "first-stage RagSEDE fixed-threshold anchors"
            if pipeline in {"dynamic_two_stage", "legacy_two_stage"}
            else "RagSEDE fixed-threshold control"
        )
        print(f"Anchor protocol: {protocol}, threshold={threshold:.3f}.")

    preserve_large_clusters = (
        single_stage_dkms
        and getattr(args, "dkms_preserve_large_clusters", False)
    )
    cohesion_refinement = (
        single_stage_dkms
        and getattr(args, "dkms_cohesion_refinement", False)
    )
    refinement_diagnostics = {}
    clusters = cluster_embeddings(
        embeddings,
        similarity_threshold=threshold,
        max_cluster_size=(
            None if preserve_large_clusters else args.max_cluster_size
        ),
        cohesion_refinement=cohesion_refinement,
        refinement_options={
            "trigger_size": args.dkms_refine_trigger_size,
            "hard_max_size": args.dkms_refine_hard_max_size,
            "min_cohesion": args.dkms_min_cohesion,
            "threshold_step": args.dkms_split_threshold_step,
            "threshold_max": args.dkms_split_threshold_max,
            "max_depth": args.dkms_split_max_depth,
        },
        diagnostics=refinement_diagnostics,
    )

    if stats is not None and clusters:
        stats.dkms_initial_largest_cluster = refinement_diagnostics.get(
            "initial_largest_cluster", max(len(cluster) for cluster in clusters)
        )
        stats.dkms_largest_cluster = max(len(cluster) for cluster in clusters)
        stats.dkms_clusters_over_legacy_limit = sum(
            len(cluster) > args.max_cluster_size for cluster in clusters
        )
        stats.dkms_recursive_splits = refinement_diagnostics.get(
            "recursive_splits", 0
        )
        stats.dkms_low_cohesion_splits = refinement_diagnostics.get(
            "low_cohesion_splits", 0
        )
        stats.dkms_forced_splits = refinement_diagnostics.get(
            "forced_splits", 0
        )
    if cohesion_refinement:
        print(
            "Cohesion-aware DKMS refined "
            f"{refinement_diagnostics.get('initial_largest_cluster', 0)} -> "
            f"{refinement_diagnostics.get('final_largest_cluster', 0)} "
            "messages for the largest cluster; "
            f"recursive splits={refinement_diagnostics.get('recursive_splits', 0)}, "
            f"forced splits={refinement_diagnostics.get('forced_splits', 0)}."
        )
    elif preserve_large_clusters:
        print(
            "Refined DKMS preserves complete semantic clusters; "
            "order-based max-size splitting is disabled."
        )
    
    text_labels = []
    for message in blk['test']:
        text_labels.append(message['event_id'])


    anchors_label = []
    for i, cluster in enumerate(clusters):
        cluster_labels = [text_labels[idx] for idx in cluster]
        anchors_label.append(cluster_labels)

    key_id = []
    for i, cluster in enumerate(clusters):
        cluster_vecs = embeddings[cluster]
        top_local_indices = select_representative_messages(
            cluster_vecs.cpu().numpy(),
            top_k=(args.view_count if single_stage_dkms else args.anchor_top_k),
            lambda_val=(
                args.dkms_lambda if single_stage_dkms else args.anchor_lambda
            ),
        )
        top_global_indices = [cluster[idx] for idx in top_local_indices]
        key_id.append(top_global_indices)

    anchors, anchors_id = create_anchors_blocks_and_ids(
        key_id,
        clusters,
        blk,
        view_separator=(
            args.view_separator
            if (
                args.enable_mvra
                or getattr(args, "preserve_anchor_views", False)
            )
            and getattr(args, "anchor_pipeline", "single_stage") == "single_stage"
            else " "
        ),
    )

    # Compare memberships and representative evidence, not merely cluster counts.
    # Joining separators deliberately remain method-specific and are not claimed identical.
    if sorted(int(i) for group in anchors_id for i in group) != list(range(len(blk["test"]))):
        raise ValueError("Anchor membership must cover each input message exactly once")
    if stats is not None:
        stats.anchor_audit.update({
            "effective_threshold": float(threshold),
            "anchor_count": len(anchors),
            "ordered_membership_sha256": _audit_digest([[int(i) for i in group] for group in anchors_id]),
            "ordered_representatives_sha256": _audit_digest([[int(i) for i in group] for group in key_id]),
            "representative_text_sha256": _audit_digest([[blk["test"][int(i)]["text"] for i in group] for group in key_id]),
        })

    print(
        f"Single anchor construction produced {len(anchors)} anchors from "
        f"{len(blk['test'])} raw messages."
    )
    if stats is not None and single_stage_dkms:
        stats.dkms_seconds += time.perf_counter() - sampling_started

    return anchors, anchors_id, anchors_label
