from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from structured_module import graph_ready_text, parse_module_metadata


DEFAULT_MODEL = "Babelscape/rebel-large"


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


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


def batches(items: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def canonical(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def extract_relations(chunks: list[dict], tokenizer, model, device: str, args):
    import torch

    collected: dict[tuple[str, str, str], dict] = {}
    total_batches = (len(chunks) + args.batch_size - 1) // args.batch_size

    for batch_number, batch in enumerate(batches(chunks, args.batch_size), start=1):
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
            for subject, relation, object_ in parse_rebel_output(decoded):
                if not subject or not relation or not object_:
                    continue
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

        print(f"Processed batch {batch_number}/{total_batches}", flush=True)

    return list(collected.values())


def build_graph(
    triples: list[dict],
    source_file: Path,
    model_name: str,
    module_metadata: dict[str, str] | None = None,
) -> dict:
    node_names: dict[str, str] = {}
    degrees: defaultdict[str, int] = defaultdict(int)
    for triple in triples:
        for field in ("subject", "object"):
            name = triple[field]
            key = canonical(name)
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
        edges.append(
            {
                "id": f"e{index}",
                "source": node_ids[canonical(triple["subject"])],
                "target": node_ids[canonical(triple["object"])],
                **triple,
            }
        )

    metadata = {
        "source_file": source_file.name,
        "model": model_name,
        "node_count": len(nodes),
        "edge_count": len(edges),
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


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Run: pip install torch transformers sentencepiece"
        ) from exc

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but PyTorch cannot access a CUDA GPU")

    print(f"Loading {args.model} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device).eval()

    source_text = args.input.read_text(encoding="utf-8-sig", errors="replace")
    text, module_metadata = prepare_input_text(source_text)
    chunks = make_chunks(text, tokenizer, args.chunk_tokens, args.overlap_tokens)
    print(f"Created {len(chunks)} overlapping chunks", flush=True)

    triples = extract_relations(chunks, tokenizer, model, device, args)
    graph = build_graph(triples, args.input, args.model, module_metadata)
    save_outputs(graph, args.output_dir)


if __name__ == "__main__":
    main()
