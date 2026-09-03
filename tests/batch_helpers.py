from collections import Counter
from contextlib import contextmanager
import json
import time

import pipeline
from batch_pipeline import BatchDependencies, BatchItem
from structured_module import StructuredModule, StructuredSlide, render_structured_module


def structured_content():
    return render_structured_module(
        StructuredModule(
            course_code="CPE0021",
            module_number="01",
            module_title="Architecture",
            source_file="Module One.pdf",
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
        pdf = tmp_path / name
        pdf.write_bytes(b"pdf")
        args = pipeline.parse_args(
            [
                str(pdf),
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
                pipeline.pipeline_paths(pdf, args.output_root),
            )
        )
    return tuple(items)


def materialize(item, stage):
    if stage == "normalize":
        item.paths.structured_text.parent.mkdir(parents=True, exist_ok=True)
        item.paths.structured_text.write_text(structured_content(), encoding="utf-8")
    elif stage == "graph":
        item.paths.graph_dir.mkdir(parents=True, exist_ok=True)
        item.paths.graph_json.write_text(
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
        item.paths.flashcards.write_text(
            f"Module {item.args.module_number}.1\n"
            f"Module {item.args.module_number}.2\n",
            encoding="utf-8",
        )


def fake_dependencies(
    events,
    qwen_loader,
    rebel_loader,
    *,
    fail_normalize=None,
    monotonic=None,
):
    def normalize(item, backend):
        events.append(("normalize", item.filename, backend))
        if item.filename == fail_normalize:
            raise RuntimeError("normalization failed")
        materialize(item, "normalize")

    def graph(item, runtime):
        events.append(("graph", item.filename, runtime))
        materialize(item, "graph")

    def flashcards(item, backend):
        events.append(("flashcards", item.filename, backend))
        materialize(item, "flashcards")

    return BatchDependencies(
        qwen_loader=qwen_loader,
        rebel_loader=rebel_loader,
        normalize_stage=normalize,
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

    def normalize(item, backend):
        counters["normalize"] += 1
        materialize(item, "normalize")

    def graph(item, runtime):
        counters["graph"] += 1
        materialize(item, "graph")

    def flashcards(item, backend):
        counters["flashcards"] += 1
        materialize(item, "flashcards")

    return BatchDependencies(
        qwen_loader=qwen_loader,
        rebel_loader=rebel_loader,
        normalize_stage=normalize,
        graph_stage=graph,
        flashcard_stage=flashcards,
        monotonic=time.monotonic,
    )
