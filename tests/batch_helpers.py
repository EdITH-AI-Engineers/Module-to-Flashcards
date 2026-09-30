from collections import Counter
from contextlib import contextmanager
import csv
import io
import json
import time

from flashcard_csv import CSV_COLUMNS
import pipeline
from batch_pipeline import BatchDependencies, BatchItem
from structured_module import StructuredModule, StructuredSlide, render_structured_module


def structured_content(module_number="01", source_file="Module One.txt"):
    return render_structured_module(
        StructuredModule(
            course_code="CPE0021",
            module_number=str(module_number),
            module_title="Architecture",
            source_file=source_file,
            slides=(
                StructuredSlide(
                    number=1,
                    extraction_method="text",
                    title="Processor",
                    content=("A processor executes instructions.",),
                    visual_text=("Not Specified",),
                    definitions=(),
                    knowledge_statements=("A processor contains an ALU.",),
                    brief_explanation="The page describes a processor.",
                ),
            ),
        )
    )


def make_items(tmp_path, *names):
    items = []
    for number, name in enumerate(names, start=1):
        source = tmp_path / name
        source.write_text(
            structured_content(number, name),
            encoding="utf-8",
        )
        args = pipeline.parse_args(
            [
                str(source),
                "--course-code",
                "CPE0021",
                "--module-number",
                str(number),
                "--output-root",
                str(tmp_path / "output"),
                "--kg-batch-size",
                "1",
                "--kg-num-beams",
                "1",
                "--skip-final-review",
                "--timeout",
                "0",
            ]
        )
        items.append(
            BatchItem(
                name,
                args,
                pipeline.pipeline_paths(
                    source,
                    args.output_root,
                    args.course_code,
                    args.module_number,
                ),
            )
        )
    return tuple(items)


def materialize(item, stage):
    if stage == "ingest":
        item.paths.structured_text.parent.mkdir(parents=True, exist_ok=True)
        item.paths.structured_text.write_text(
            item.args.input.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    elif stage == "graph":
        item.paths.unchecked_graph_dir.mkdir(parents=True, exist_ok=True)
        item.paths.unchecked_graph_json.write_text(
            json.dumps(
                {
                    "metadata": {"module_number": item.args.module_number},
                    "nodes": [],
                    "edges": [
                        {
                            "id": "e1",
                            "subject": "processor",
                            "relation": "contains",
                            "object": "ALU",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
    else:
        item.paths.graph_dir.mkdir(parents=True, exist_ok=True)
        graph = json.loads(item.paths.unchecked_graph_json.read_text(encoding="utf-8"))
        graph.setdefault("metadata", {})["graph_checker"] = {
            "status": "checked",
            "model": "test",
        }
        item.paths.graph_json.write_text(json.dumps(graph), encoding="utf-8")
        item.paths.flashcards.parent.mkdir(parents=True, exist_ok=True)
        item.paths.flashcards.write_text(
            flashcard_content(item.args.course_code, item.args.module_number),
            encoding="utf-8",
        )


def flashcard_content(course_code="CPE0021", module_number="01"):
    """Return a structurally complete two-block 100-card reuse fixture."""
    blocks = []
    for block_number, first_cluster in enumerate((1, 11), start=1):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for cluster_number in range(first_cluster, first_cluster + 10):
            cluster = f"00000000-0000-4000-8000-{cluster_number:012d}"
            for card_number in range(1, 6):
                writer.writerow(
                    (
                        "multiple-choice",
                        f"Question {cluster_number}-{card_number}",
                        "Correct",
                        "Wrong one",
                        "Wrong two",
                        "Wrong three",
                        "",
                        "Explanation",
                        "Hint",
                        "1",
                        cluster,
                        course_code,
                        str(module_number),
                    )
                )
        blocks.append(f"Module {module_number}.{block_number}\n{stream.getvalue()}")
    return "\n".join(blocks)


def fake_dependencies(
    events,
    qwen_loader,
    rebel_loader,
    *,
    fail_ingest=None,
    monotonic=None,
):
    def ingest(item, runtime):
        events.append(("ingest", item.filename, runtime))
        if item.filename == fail_ingest:
            raise RuntimeError("ingest failed")
        materialize(item, "ingest")

    def graph(item, runtime):
        events.append(("graph", item.filename, runtime))
        materialize(item, "graph")

    def flashcards(item, backend):
        events.append(("flashcards", item.filename, backend))
        materialize(item, "flashcards")

    return BatchDependencies(
        qwen_loader=qwen_loader,
        rebel_loader=rebel_loader,
        ingest_stage=ingest,
        graph_stage=graph,
        flashcard_stage=flashcards,
        monotonic=monotonic or time.monotonic,
    )


def counting_dependencies(counters: Counter):
    @contextmanager
    def qwen_loader(args):
        counters["qwen_load"] += 1
        yield object()

    @contextmanager
    def rebel_loader(args):
        counters["rebel_load"] += 1
        yield object()

    def ingest(item, runtime):
        counters["ingest"] += 1
        materialize(item, "ingest")

    def graph(item, runtime):
        counters["graph"] += 1
        materialize(item, "graph")

    def flashcards(item, backend):
        counters["flashcards"] += 1
        materialize(item, "flashcards")

    return BatchDependencies(
        qwen_loader=qwen_loader,
        rebel_loader=rebel_loader,
        ingest_stage=ingest,
        graph_stage=graph,
        flashcard_stage=flashcards,
        monotonic=time.monotonic,
    )
