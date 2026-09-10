from __future__ import annotations

import argparse
import csv
<<<<<<< HEAD
import difflib
=======
>>>>>>> 57756b4a7cbb850c9cc4535c92977efa22d5b65b
import gc
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from structured_module import (
    extract_lesson_facts,
    graph_ready_text,
    parse_module_metadata,
)


DEFAULT_MODEL = "Babelscape/rebel-large"

<<<<<<< HEAD
CONTROLLED_RELATIONS = frozenset(
    {
        "defines",
        "created_by",
        "enacted_on",
        "part_of",
        "contrasts_with",
        "example_of",
        "mandates",
        "is_a",
    }
)
RELATION_ALIASES = {
    "define": "defines",
    "defines": "defines",
    "meaning": "defines",
    "created by": "created_by",
    "creates": "created_by",
    "created": "created_by",
    "enacted on": "enacted_on",
    "enacted": "enacted_on",
    "date of enactment": "enacted_on",
    "part of": "part_of",
    "belongs to": "part_of",
    "contrasts with": "contrasts_with",
    "opposite of": "contrasts_with",
    "example of": "example_of",
    "is an example of": "example_of",
    "mandates": "mandates",
    "requires": "mandates",
    "is a": "is_a",
    "type of": "is_a",
    "category of": "is_a",
}
LEGAL_IDENTIFIER = re.compile(
    r"\b(?P<kind>ra|republic act|act|law|eo|executive order|"
    r"proclamation|ordinance|resolution)\s*[-.]?\s*(?P<number>\d{2,})\b",
    re.I,
)
LEGAL_TITLE = re.compile(
    r"\b(?:act|law|order|proclamation|ordinance|resolution)\b$", re.I
)
ALIAS_CUE = re.compile(
    r"\b(?:also known as|known as|aka|otherwise called|officially called)\b|[()]",
    re.I,
)
LEGAL_KIND_ALIASES = {
    "ra": "act",
    "republic act": "act",
    "act": "act",
    "law": "act",
    "eo": "order",
    "executive order": "order",
    "proclamation": "proclamation",
    "ordinance": "ordinance",
    "resolution": "resolution",
}

=======
>>>>>>> 57756b4a7cbb850c9cc4535c92977efa22d5b65b

@dataclass
class RebelRuntime:
    tokenizer: object
    model: object
    device: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a knowledge graph from text using the REBEL model."
    )
    parser.add_argument("input", type=Path, help="UTF-8 .txt file to process")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("knowledge_graph_output")
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-tokens", type=int, default=384)
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto uses CUDA when available, otherwise CPU",
    )
    return parser.parse_args(argv)


def clean_text(text: str) -> str:
    """Remove structural noise while preserving the module's meaningful text."""
    text = text.replace("\x00", " ").replace("\r\n", "\n")
    text = re.sub(r"(?m)^[-=]{3,}\s*$", "\n", text)
    text = re.sub(r"(?m)^\s*[{}]\s*$", "\n", text)
    text = re.sub(r"(?m)^\s*(Content|Brief Explanation):\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def prepare_input_text(text: str) -> tuple[str, dict[str, str]]:
    """Project structured modules into graph-ready prose and retain identity."""
    metadata = parse_module_metadata(text)
    projected = graph_ready_text(text) if metadata else text
    return clean_text(projected), metadata


def make_chunks(
    text: str, tokenizer, chunk_tokens: int, overlap_tokens: int
) -> list[dict]:
    if overlap_tokens >= chunk_tokens:
        raise ValueError("--overlap-tokens must be smaller than --chunk-tokens")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    step = chunk_tokens - overlap_tokens
    chunks = []
    for index, start in enumerate(range(0, len(token_ids), step), start=1):
        ids = token_ids[start : start + chunk_tokens]
        if not ids:
            break
        chunk_text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        if chunk_text:
            slide_matches = re.findall(r"Slide\s+(\d+)", chunk_text, re.I)
            chunks.append(
                {
                    "id": index,
                    "text": chunk_text,
                    "slides": sorted({int(x) for x in slide_matches}),
                }
            )
        if start + chunk_tokens >= len(token_ids):
            break
    return chunks


def parse_rebel_output(decoded: str) -> list[tuple[str, str, str]]:
    """Parse REBEL's: <triplet> subject <subj> object <obj> relation."""
    decoded = decoded.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
    triples: list[tuple[str, str, str]] = []
    subject = object_ = relation = ""
    state = None

    for token in decoded.strip().split():
        if token == "<triplet>":
            if subject and object_ and relation:
                triples.append((subject.strip(), relation.strip(), object_.strip()))
            subject, object_, relation = "", "", ""
            state = "subject"
        elif token == "<subj>":
            state = "object"
        elif token == "<obj>":
            state = "relation"
        elif state == "subject":
            subject += (" " if subject else "") + token
        elif state == "object":
            object_ += (" " if object_ else "") + token
        elif state == "relation":
            relation += (" " if relation else "") + token

    if subject and object_ and relation:
        triples.append((subject.strip(), relation.strip(), object_.strip()))
    return triples


<<<<<<< HEAD
def normalize_relation(value: str) -> str | None:
    """Map REBEL wording to the small relation vocabulary used downstream."""
    normalized = re.sub(r"\s+", " ", value.replace("_", " ").strip().casefold())
    if normalized in CONTROLLED_RELATIONS:
        return normalized
    if normalized in RELATION_ALIASES:
        return RELATION_ALIASES[normalized]
    for phrase, relation in sorted(RELATION_ALIASES.items(), key=lambda item: -len(item[0])):
        if phrase in normalized:
            return relation
    return None


def _surface_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


def _surface_span(value: str, text: str) -> tuple[int, int] | None:
    value_tokens = _surface_tokens(value)
    text_tokens = _surface_tokens(text)
    if not value_tokens or not text_tokens:
        return None
    value_width = len(value_tokens)
    for start in range(len(text_tokens) - value_width + 1):
        if text_tokens[start : start + value_width] == value_tokens:
            return start, start + value_width
    value_text = " ".join(value_tokens)
    for start in range(len(text_tokens)):
        for window_width in range(
            max(1, value_width - 1),
            min(len(text_tokens), value_width + 2) + 1,
        ):
            window = " ".join(text_tokens[start : start + window_width])
            if difflib.SequenceMatcher(None, value_text, window).ratio() >= 0.9:
                return start, start + window_width
    return None


def _evidence_record(chunk: dict, text: str, slides: Sequence[int]) -> dict:
    word_count = len(_surface_tokens(text))
    return {
        "chunk_id": chunk["id"],
        "slides": list(slides),
        "text": text,
        "word_count": word_count,
        "confidence": 0.5 if word_count < 8 else 1.0,
    }


def _source_supports_order(subject: str, relation: str, object_: str, text: str) -> bool:
    subject_span = _surface_span(subject, text)
    object_span = _surface_span(object_, text)
    if subject_span is None or object_span is None:
        return False
    if subject_span[0] > object_span[0]:
        return False
    if relation == "enacted_on":
        return bool(re.search(r"\b(?:19|20)\d{2}\b|\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b", object_))
    if relation == "created_by":
        law_subject = bool(re.search(r"\b(?:ra|republic act|act|law)\b", subject, re.I))
        law_object = bool(re.search(r"\b(?:ra|republic act|act|law)\b", object_, re.I))
        if law_object and not law_subject:
            return False
    return True


def _chunk_evidence(chunk: dict, subject: str, object_: str) -> list[dict]:
    text = str(chunk["text"])
    slide_parts = list(re.finditer(r"(?im)(?:^|\n)\s*Slide\s+(\d+)\s*", text))
    if not slide_parts:
        return [_evidence_record(chunk, text, chunk.get("slides", []))]
    evidence: list[dict] = []
    for index, match in enumerate(slide_parts):
        end = slide_parts[index + 1].start() if index + 1 < len(slide_parts) else len(text)
        slide_text = text[match.end() : end].strip()
        if _source_supports_order(subject, "is_a", object_, slide_text):
            evidence.append(
                _evidence_record(chunk, slide_text, [int(match.group(1))])
            )
    return evidence or [_evidence_record(chunk, text, chunk.get("slides", []))]


=======
>>>>>>> 57756b4a7cbb850c9cc4535c92977efa22d5b65b
def batches(items: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def canonical(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


<<<<<<< HEAD
def canonical_entity_key(value: str) -> str:
    normalized = canonical(value).replace("–", "-")
    identifier = LEGAL_IDENTIFIER.search(normalized)
    if identifier:
        kind = LEGAL_KIND_ALIASES[identifier.group("kind")]
        return f"legal:{kind}:{identifier.group('number')}"
    if normalized.isdigit() and len(normalized) >= 2:
        return f"legal:unknown:{normalized}"
    return normalized


def _is_legal_identifier(value: str) -> bool:
    return LEGAL_IDENTIFIER.search(value) is not None or (
        value.strip().isdigit() and len(value.strip()) >= 2
    )


def _is_legal_title(value: str) -> bool:
    return bool(LEGAL_TITLE.search(canonical(value)))


def _union(parent: dict[str, str], left: str, right: str) -> None:
    def root(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    left_root = root(left)
    right_root = root(right)
    if left_root != right_root:
        parent[right_root] = left_root


def resolve_entity_aliases(triples: Sequence[dict]) -> tuple[dict, ...]:
    """Merge source-supported aliases before assigning graph node IDs."""
    labels: dict[str, list[str]] = defaultdict(list)
    values: set[str] = set()
    for triple in triples:
        for field in ("subject", "object"):
            value = str(triple[field]).strip()
            values.add(value)
            key = canonical_entity_key(value)
            if value not in labels[key]:
                labels[key].append(value)

    parent = {value: value for value in values}
    same_key: dict[str, list[str]] = defaultdict(list)
    for value in values:
        same_key[canonical_entity_key(value)].append(value)
    for aliases in same_key.values():
        for alias in aliases[1:]:
            _union(parent, aliases[0], alias)

    for triple in triples:
        subject = str(triple["subject"]).strip()
        object_ = str(triple["object"]).strip()
        subject_identifier = _is_legal_identifier(subject)
        object_identifier = _is_legal_identifier(object_)
        subject_title = _is_legal_title(subject)
        object_title = _is_legal_title(object_)
        evidence_texts = [
            str(item.get("text", ""))
            for item in triple.get("evidence", ())
            if isinstance(item, dict)
        ]
        alias_context = any(ALIAS_CUE.search(text) for text in evidence_texts)
        definitional_edge = canonical(str(triple.get("relation", ""))) == "defines"
        if (
            (subject_identifier and object_title)
            or (object_identifier and subject_title)
        ) and (alias_context or definitional_edge):
            _union(parent, subject, object_)

    groups: dict[str, list[str]] = defaultdict(list)
    for value in values:
        current = value
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = parent[current]
        if value not in groups[current]:
            groups[current].append(value)
    preferred: dict[str, str] = {}
    for group in groups.values():
        label = max(group, key=lambda value: (len(value), value.casefold()))
        for value in group:
            preferred[value] = label

    resolved: list[dict] = []
    for triple in triples:
        item = dict(triple)
        item["subject"] = preferred[str(triple["subject"]).strip()]
        item["object"] = preferred[str(triple["object"]).strip()]
        resolved.append(item)
    return tuple(resolved)


def sanitize_triples(triples: Sequence[dict]) -> tuple[dict, ...]:
    """Normalize relation names and discard triples that lack source support."""
    sanitized: list[dict] = []
    for triple in triples:
        subject = str(triple.get("subject", "")).strip()
        object_ = str(triple.get("object", "")).strip()
        relation = normalize_relation(str(triple.get("relation", "")))
        if not subject or not object_ or relation is None:
            continue
        item = dict(triple)
        item.update({"subject": subject, "relation": relation, "object": object_})
        evidence = item.get("evidence")
        if isinstance(evidence, list) and evidence:
            supported = any(
                isinstance(entry, dict)
                and _source_supports_order(
                    subject,
                    relation,
                    object_,
                    str(entry.get("text", "")),
                )
                for entry in evidence
            )
            if not supported:
                continue
        sanitized.append(item)
    return tuple(sanitized)


def extract_relations(
    chunks: list[dict],
    tokenizer,
    model,
    device: str,
    args,
    lesson_facts: Sequence[dict[str, object]] = (),
):
    import torch

    collected: dict[tuple[str, str, str], dict] = {}
    if lesson_facts:
        extraction_items = [
            {
                "id": fact.get("id", f"fact-{index}"),
                "text": str(fact.get("statement", "")),
                "slides": list(fact.get("slides", [])),
            }
            for index, fact in enumerate(lesson_facts, start=1)
            if str(fact.get("statement", "")).strip()
        ]
    else:
        extraction_items = chunks
    total_batches = (len(extraction_items) + args.batch_size - 1) // args.batch_size

    for batch_number, batch in enumerate(batches(extraction_items, args.batch_size), start=1):
=======
def extract_relations(chunks: list[dict], tokenizer, model, device: str, args):
    import torch

    collected: dict[tuple[str, str, str], dict] = {}
    total_batches = (len(chunks) + args.batch_size - 1) // args.batch_size

    for batch_number, batch in enumerate(batches(chunks, args.batch_size), start=1):
>>>>>>> 57756b4a7cbb850c9cc4535c92977efa22d5b65b
        encoded = tokenizer(
            [item["text"] for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.chunk_tokens + 2,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                length_penalty=0.8,
            )

        decoded_batch = tokenizer.batch_decode(generated, skip_special_tokens=False)
        for chunk, decoded in zip(batch, decoded_batch):
            print(
                f"[knowledge-graph] chunk {chunk['id']} generated output:\n{decoded}",
                flush=True,
            )
            for subject, relation, object_ in parse_rebel_output(decoded):
                if not subject or not relation or not object_:
                    continue
<<<<<<< HEAD
                normalized_relation = normalize_relation(relation)
                if normalized_relation is None:
                    print(
                        f"[knowledge-graph] discarded unsupported relation: {relation!r}",
                        flush=True,
                    )
                    continue
                if not _source_supports_order(
                    subject,
                    normalized_relation,
                    object_,
                    chunk["text"],
                ):
                    print(
                        f"[knowledge-graph] discarded ungrounded or reversed triple: "
                        f"{subject} | {relation} | {object_}",
                        flush=True,
                    )
                    continue
                key = (
                    canonical(subject),
                    normalized_relation,
                    canonical(object_),
                )
                if key not in collected:
                    collected[key] = {
                        "subject": subject,
                        "relation": normalized_relation,
                        "object": object_,
                        "evidence": [],
                    }
                for evidence in _chunk_evidence(chunk, subject, object_):
                    if evidence not in collected[key]["evidence"]:
                        collected[key]["evidence"].append(evidence)
=======
                key = (canonical(subject), canonical(relation), canonical(object_))
                if key not in collected:
                    collected[key] = {
                        "subject": subject,
                        "relation": relation,
                        "object": object_,
                        "evidence": [],
                    }
                evidence = {
                    "chunk_id": chunk["id"],
                    "slides": chunk["slides"],
                    "text": chunk["text"],
                }
                if evidence not in collected[key]["evidence"]:
                    collected[key]["evidence"].append(evidence)
>>>>>>> 57756b4a7cbb850c9cc4535c92977efa22d5b65b

        print(f"Processed batch {batch_number}/{total_batches}", flush=True)

    return list(collected.values())


def build_graph(
    triples: list[dict],
    source_file: Path,
    model_name: str | Path,
    module_metadata: dict[str, str] | None = None,
    lesson_facts: Sequence[dict[str, object]] = (),
) -> dict:
<<<<<<< HEAD
    triples = list(resolve_entity_aliases(sanitize_triples(triples)))
=======
>>>>>>> 57756b4a7cbb850c9cc4535c92977efa22d5b65b
    node_names: dict[str, str] = {}
    degrees: defaultdict[str, int] = defaultdict(int)
    for triple in triples:
        for field in ("subject", "object"):
            name = triple[field]
<<<<<<< HEAD
            key = canonical_entity_key(name)
=======
            key = canonical(name)
>>>>>>> 57756b4a7cbb850c9cc4535c92977efa22d5b65b
            node_names.setdefault(key, name)
            degrees[key] += 1

    ordered_keys = sorted(node_names, key=lambda k: (-degrees[k], node_names[k].casefold()))
    node_ids = {key: f"n{index}" for index, key in enumerate(ordered_keys, start=1)}
    nodes = [
        {"id": node_ids[key], "label": node_names[key], "degree": degrees[key]}
        for key in ordered_keys
    ]
    edges = []
    for index, triple in enumerate(triples, start=1):
<<<<<<< HEAD
        edge = {
                "id": f"e{index}",
                "source": node_ids[canonical_entity_key(triple["subject"])],
                "target": node_ids[canonical_entity_key(triple["object"])],
                **triple,
            }
        evidence = triple.get("evidence")
        if isinstance(evidence, list) and evidence:
            edge["confidence"] = min(
                float(item.get("confidence", 1.0))
                for item in evidence
                if isinstance(item, dict)
            )
        edges.append(edge)
=======
        edges.append(
            {
                "id": f"e{index}",
                "source": node_ids[canonical(triple["subject"])],
                "target": node_ids[canonical(triple["object"])],
                **triple,
            }
        )
>>>>>>> 57756b4a7cbb850c9cc4535c92977efa22d5b65b

    metadata = {
        "source_file": source_file.name,
        "model": str(model_name),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "fact_count": len(lesson_facts),
    }
    approved_metadata = {
        key: value
        for key, value in (module_metadata or {}).items()
        if key
        in {
            "format_version",
            "course_code",
            "module_number",
            "module_title",
            "source_file",
        }
        and str(value).strip()
    }
    metadata.update(approved_metadata)
    return {
        "metadata": metadata,
        "nodes": nodes,
        "edges": edges,
        "facts": [dict(fact) for fact in lesson_facts],
    }


def save_outputs(graph: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "knowledge_graph.json"
    csv_path = output_dir / "triples.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(graph, handle, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "subject",
                "relation",
                "object",
                "source_chunks",
                "source_slides",
                "evidence_count",
            ),
        )
        writer.writeheader()
        for edge in graph["edges"]:
            slide_numbers = sorted(
                {slide for item in edge["evidence"] for slide in item["slides"]}
            )
            writer.writerow(
                {
                    "subject": edge["subject"],
                    "relation": edge["relation"],
                    "object": edge["object"],
                    "source_chunks": ";".join(
                        str(item["chunk_id"]) for item in edge["evidence"]
                    ),
                    "source_slides": ";".join(map(str, slide_numbers)),
                    "evidence_count": len(edge["evidence"]),
                }
            )

    print(f"Saved {graph['metadata']['node_count']} nodes and "
          f"{graph['metadata']['edge_count']} edges")
    print(f"JSON: {json_path.resolve()}")
    print(f"CSV:  {csv_path.resolve()}")


def load_runtime(
    model_name: str | Path,
    device: str,
    *,
    local_files_only: bool = False,
) -> RebelRuntime:
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies. Run: python -m pip install -r requirements.txt"
        ) from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but PyTorch cannot access a CUDA GPU")

    print(f"Loading {model_name} on {device} ...", flush=True)
    load_options = {"local_files_only": True} if local_files_only else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **load_options)
    model = (
        AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_options)
        .to(device)
        .eval()
    )
    return RebelRuntime(tokenizer, model, device)


def release_runtime(runtime: RebelRuntime) -> None:
    del runtime.model
    gc.collect()
    if runtime.device == "cuda":
        import torch

        torch.cuda.empty_cache()


def _validate_run_args(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise ValueError(f"Input file not found: {args.input}")
    if args.chunk_tokens < 1:
        raise ValueError("--chunk-tokens must be at least 1")
    if args.overlap_tokens < 0:
        raise ValueError("--overlap-tokens must be non-negative")
    if args.overlap_tokens >= args.chunk_tokens:
        raise ValueError("--overlap-tokens must be smaller than --chunk-tokens")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.num_beams < 1:
        raise ValueError("--num-beams must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1")


def run(
    args: argparse.Namespace,
    *,
    runtime: RebelRuntime | None = None,
) -> tuple[Path, Path]:
    _validate_run_args(args)
    owns_runtime = runtime is None
    if runtime is None:
        runtime = load_runtime(args.model, args.device)

    try:
        source_text = args.input.read_text(encoding="utf-8-sig", errors="replace")
        text, module_metadata = prepare_input_text(source_text)
        lesson_facts = extract_lesson_facts(source_text)
        chunks = make_chunks(text, runtime.tokenizer, args.chunk_tokens, args.overlap_tokens)
        print(f"Created {len(chunks)} overlapping chunks", flush=True)

        triples = extract_relations(
<<<<<<< HEAD
            chunks,
            runtime.tokenizer,
            runtime.model,
            runtime.device,
            args,
            lesson_facts,
=======
            chunks, runtime.tokenizer, runtime.model, runtime.device, args
>>>>>>> 57756b4a7cbb850c9cc4535c92977efa22d5b65b
        )
        graph = build_graph(
            triples,
            args.input,
            args.model,
            module_metadata,
            lesson_facts,
        )
        save_outputs(graph, args.output_dir)
        return args.output_dir / "knowledge_graph.json", args.output_dir / "triples.csv"
    finally:
        if owns_runtime:
            release_runtime(runtime)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0
