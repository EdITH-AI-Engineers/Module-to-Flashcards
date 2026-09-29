import json
from dataclasses import replace

import pytest

from flashcard_validator import (
    ValidationError,
    are_near_duplicates,
    parse_review_issues,
    validate_module,
)
from tests.factories import valid_clusters, with_question


def test_near_duplicate_threshold_requires_both_metrics():
    assert are_near_duplicates(
        "Which number system uses base 2 to represent digital values?",
        "Which number system uses base 2 to represent digital values correctly?",
    )
    assert not are_near_duplicates(
        "Which number system uses base two?",
        "How does parity detect transmission errors?",
    )


def test_near_duplicate_does_not_confuse_a_substantive_term_substitution():
    assert not are_near_duplicates(
        "Which method uses a deterministic algorithm to process input data for binary encoding?",
        "Which method uses a deterministic algorithm to process input data for octal encoding?",
    )


def test_near_duplicate_ignores_question_word_variation_for_the_same_content():
    assert are_near_duplicates(
        "Which number system uses base 2 to represent digital values?",
        "What number system uses base 2 to represent digital values?",
    )


def test_module_accepts_twenty_valid_clusters():
    assert validate_module(valid_clusters()) == ()


def test_module_requires_twenty_clusters():
    errors = validate_module(valid_clusters()[:19])

    assert any("exactly 20 clusters" in error for error in errors)


def test_module_rejects_duplicate_cluster_uuid():
    clusters = list(valid_clusters())
    clusters[1] = replace(clusters[1], cluster=clusters[0].cluster)

    errors = validate_module(tuple(clusters))

    assert any("cluster UUIDs must be unique" in error for error in errors)


def test_module_rejects_invalid_cluster_uuid():
    clusters = list(valid_clusters())
    clusters[0] = replace(clusters[0], cluster="not-a-uuid")

    errors = validate_module(tuple(clusters))

    assert any("valid UUID" in error for error in errors)


def test_module_detects_cross_cluster_near_duplicate():
    original = valid_clusters()[0].cards[0].question
    clusters = with_question(valid_clusters(), 1, 0, original + " correctly")

    errors = validate_module(clusters)

    assert any("near-duplicate questions" in error for error in errors)


def test_module_allows_fuzzy_subject_overlap_across_different_card_types():
    clusters = list(valid_clusters())
    first_cards = list(clusters[0].cards)
    second_cards = list(clusters[1].cards)
    first_cards[0] = replace(
        first_cards[0],
        question="Which classification applies to topic1 alpha1 beta1 revision0?",
    )
    second_cards[2] = replace(
        second_cards[2],
        question="The classification applies to topic1 alpha1 beta1 revision0.",
    )
    clusters[0] = replace(clusters[0], cards=tuple(first_cards))
    clusters[1] = replace(clusters[1], cards=tuple(second_cards))

    errors = validate_module(tuple(clusters))

    assert not any(
        "cluster 1 card 1 and cluster 2 card 3" in error
        for error in errors
    )


def test_module_rejects_exact_normalized_question_across_card_types():
    clusters = list(valid_clusters())
    exact_question = clusters[0].cards[0].question
    second_cards = list(clusters[1].cards)
    second_cards[2] = replace(second_cards[2], question=exact_question)
    clusters[1] = replace(clusters[1], cards=tuple(second_cards))

    errors = validate_module(tuple(clusters))

    assert any(
        "cluster 1 card 1 and cluster 2 card 3" in error
        for error in errors
    )


def test_module_can_defer_cross_cluster_duplicates_until_global_review():
    original = valid_clusters()[0].cards[0].question
    clusters = with_question(valid_clusters(), 1, 0, original + " correctly")

    errors = validate_module(clusters, check_question_duplicates=False)

    assert not any("near-duplicate questions" in error for error in errors)


def test_review_parser_rejects_unknown_cluster():
    raw = json.dumps(
        {"issues": [{"cluster": "not-known", "reasons": ["unsupported claim"]}]}
    )
    with pytest.raises(ValidationError, match="unknown cluster"):
        parse_review_issues(raw, {"known-id"})


def test_review_parser_merges_reasons_for_the_same_cluster():
    cluster = valid_clusters()[0].cluster
    raw = json.dumps(
        {
            "issues": [
                {"cluster": cluster, "reasons": ["unsupported claim"]},
                {"cluster": cluster, "reasons": ["answer leakage"]},
            ]
        }
    )

    issues = parse_review_issues(raw, {cluster})

    assert len(issues) == 1
    assert issues[0].cluster == cluster
    assert issues[0].reasons == ("unsupported claim", "answer leakage")
