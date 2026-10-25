"""Discover and copy saved checkpoints without touching an interrupted run."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil

from artifact_paths import flashcard_output_path, module_file_label, safe_path_component
from flashcard_csv import valid_written_receipt
from graph_input import GraphInputError, extract_graph_facts, load_graph, resolve_identity
from knowledge_graph_checker import is_checked_graph
from pipeline import PipelinePaths, _valid_structured_text, pipeline_paths, stage_structured_module
from structured_module import parse_module_metadata, parse_slide_report_metadata, validate_slide_report


@dataclass(frozen=True)
class SavedGraphModule:
    graph: Path
    paths: PipelinePaths
    course_code: str
    module_number: str
    checked: bool
    completed: bool

    @property
    def source(self) -> Path:
        return self.graph


@dataclass(frozen=True)
class SavedStructuredModule:
    source: Path
    paths: PipelinePaths
    course_code: str
    module_number: str
    identity_from_filename: bool = False
    completed: bool = False


SavedCheckpoint = SavedGraphModule | SavedStructuredModule


def saved_run_root(runs_root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("Invalid saved run ID.")
    root = Path(runs_root).resolve()
    run = (root / run_id).resolve()
    if not run.is_relative_to(root) or not run.is_dir():
        raise ValueError("Saved run not found.")
    return run


def _contained(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def _checkpoint_paths(workspace: Path, run_root: Path, course: str, number: str) -> PipelinePaths:
    return PipelinePaths(
        workspace=workspace,
        structured_text=workspace / "structured_module.txt",
        unchecked_graph_dir=workspace / "unchecked_knowledge_graph",
        unchecked_graph_json=workspace / "unchecked_knowledge_graph" / "knowledge_graph.json",
        unchecked_triples_csv=workspace / "unchecked_knowledge_graph" / "triples.csv",
        graph_dir=workspace / "knowledge_graph",
        graph_json=workspace / "knowledge_graph" / "knowledge_graph.json",
        triples_csv=workspace / "knowledge_graph" / "triples.csv",
        flashcards=flashcard_output_path(run_root / "flashcards", course, number),
    )


def _validate_checkpoint_facts(graph: dict) -> None:
    # Discovery only needs one usable fact. Full deduplication of every large
    # module belongs to generation and can make the run picker take minutes.
    field = "facts" if "facts" in graph else "edges"
    records = graph.get(field)
    if isinstance(records, list):
        for record in records:
            try:
                extract_graph_facts({**graph, field: [record]})
                return
            except GraphInputError:
                continue
        raise GraphInputError("Saved graph contains no usable lesson facts.")
    extract_graph_facts(graph)


def valid_graph_checkpoint(path: Path, *, checked: bool = False) -> bool:
    try:
        graph = load_graph(path)
        _validate_checkpoint_facts(graph)
        return not checked or is_checked_graph(graph)
    except (GraphInputError, OSError, ValueError):
        return False


def _discover_graphs(run_root: Path) -> tuple[tuple[SavedGraphModule, ...], dict[Path, str]]:
    run_root = Path(run_root).resolve()
    workspaces = sorted({
        graph.parent.parent
        for directory in ("knowledge_graph", "unchecked_knowledge_graph")
        for graph in run_root.glob(f"pipeline_output/*/*/{directory}/knowledge_graph.json")
    })
    modules: list[SavedGraphModule] = []
    errors: dict[Path, str] = {}
    identities: set[str] = set()
    for workspace in workspaces:
        failures = []
        for directory in ("knowledge_graph", "unchecked_knowledge_graph"):
            source = workspace / directory / "knowledge_graph.json"
            if not source.is_file():
                continue
            try:
                if not _contained(source, run_root):
                    raise ValueError("Saved graph resolves outside its run.")
                graph = load_graph(source)
                _validate_checkpoint_facts(graph)
                identity = resolve_identity(graph, None, None)
                paths = _checkpoint_paths(workspace, run_root, identity.course_code, identity.module_number)
                if not all(_contained(path, run_root) for path in (
                    paths.flashcards, *paths.flashcard_parts, paths.flashcard_receipt,
                )):
                    raise ValueError("Saved flashcards resolve outside their run.")
                output_key = str(paths.flashcards).casefold()
                if output_key in identities:
                    raise ValueError("More than one saved workspace has this course and module number.")
                identities.add(output_key)
                checked = directory == "knowledge_graph" and is_checked_graph(graph)
                modules.append(SavedGraphModule(source, paths, identity.course_code,
                    identity.module_number, checked,
                    checked and valid_written_receipt(paths.flashcards, identity)))
                break
            except (GraphInputError, OSError, ValueError) as exc:
                failures.append(str(exc))
        else:
            errors[workspace] = "; ".join(failures)
    modules.sort(key=lambda module: (
        module.course_code.casefold(),
        tuple(int(part) if part.isdigit() else part.casefold()
              for part in re.split(r"(\d+)", module.module_number)),
    ))
    return tuple(modules), errors


def discover_graphs(run_root: Path) -> tuple[tuple[SavedGraphModule, ...], list[dict[str, str]]]:
    modules, errors = _discover_graphs(run_root)
    return modules, [{"file": workspace.name, "error": error} for workspace, error in errors.items()]


def discover_checkpoints(run_root: Path) -> tuple[tuple[SavedCheckpoint, ...], list[dict[str, str]]]:
    """Prefer usable graphs, falling back to structured text for each workspace."""
    run_root = Path(run_root).resolve()
    graphs, graph_errors = _discover_graphs(run_root)
    modules: list[SavedCheckpoint] = list(graphs)
    graph_workspaces = {module.paths.workspace for module in graphs}
    identities = {str(module.paths.flashcards).casefold() for module in graphs}
    errors: list[dict[str, str]] = []
    for source in sorted(run_root.glob("pipeline_output/*/*/structured_module.txt")):
        workspace = source.parent
        if workspace in graph_workspaces:
            continue
        graph_errors.pop(workspace, None)
        try:
            if not _contained(source, run_root):
                raise ValueError("Saved structured module resolves outside its run.")
            content = source.read_text(encoding="utf-8-sig")
            metadata = parse_module_metadata(content) or parse_slide_report_metadata(content)
            course = metadata.get("course_code", "").strip() or workspace.parent.name
            number = metadata.get("module_number", "").strip()
            # API workspaces retain the accepted upload identity, including TXT
            # filenames that deliberately overrode conflicting report metadata.
            canonical = re.fullmatch(re.escape(workspace.parent.name) + r"_M(.+)", workspace.name)
            if canonical:
                if safe_path_component(course, fallback="course") != workspace.parent.name:
                    course = workspace.parent.name
                if module_file_label(number) != canonical[1]:
                    number = canonical[1]
            if not number:
                raise ValueError("Saved structured module must declare a module number.")
            if not _valid_structured_text(source, identity_from_filename=bool(canonical)):
                raise ValueError("Saved structured module contains no valid readable learning content.")
            if not parse_module_metadata(content):
                validate_slide_report(content, course_code=course, module_number=number,
                                      identity_from_filename=bool(canonical))
            paths = _checkpoint_paths(workspace, run_root, course, number)
            output_key = str(paths.flashcards).casefold()
            if output_key in identities:
                raise ValueError("More than one saved workspace has this course and module number.")
            identities.add(output_key)
            modules.append(SavedStructuredModule(source, paths, course, number, bool(canonical)))
        except (OSError, ValueError) as exc:
            errors.append({"file": workspace.name, "error": str(exc)})
    errors.extend({"file": workspace.name, "error": error} for workspace, error in graph_errors.items())
    modules.sort(key=lambda module: (
        module.course_code.casefold(),
        tuple(int(part) if part.isdigit() else part.casefold()
              for part in re.split(r"(\d+)", module.module_number)),
    ))
    return tuple(modules), errors


def saved_run_summary(run_root: Path) -> dict[str, object] | None:
    modules, errors = discover_checkpoints(run_root)
    if not modules and not errors:
        return None
    modified = max(
        [run_root.stat().st_mtime]
        + [module.source.stat().st_mtime for module in modules]
        + [module.paths.flashcard_receipt.stat().st_mtime for module in modules if module.completed]
    )
    completed = sum(module.completed for module in modules)
    return {
        "runId": run_root.name,
        "moduleCount": len(modules),
        "completedCount": completed,
        "pendingCount": len(modules) - completed,
        "graphCount": sum(isinstance(module, SavedGraphModule) for module in modules),
        "structuredCount": sum(isinstance(module, SavedStructuredModule) for module in modules),
        "courses": sorted({module.course_code for module in modules}),
        "updatedAt": datetime.fromtimestamp(modified, timezone.utc).isoformat(),
        "errors": errors,
    }


def copy_graph_checkpoint(module: SavedGraphModule, destination_root: Path) -> PipelinePaths:
    """Make an isolated checkpoint, including already validated CSV pairs."""
    paths = pipeline_paths(Path(module.paths.workspace.name + ".json"), destination_root / "pipeline_output",
                           module.course_code, module.module_number)
    destination = paths.graph_json if module.checked else paths.unchecked_graph_json
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(module.graph, destination)
    if module.completed:
        paths.flashcards.parent.mkdir(parents=True, exist_ok=True)
        for source, target in zip(
            (*module.paths.flashcard_parts, module.paths.flashcard_receipt),
            (*paths.flashcard_parts, paths.flashcard_receipt),
        ):
            shutil.copy2(source, target)
    corpus = module.paths.flashcards.parent / "course_corpus.json"
    target_corpus = paths.flashcards.parent / "course_corpus.json"
    source_root = module.paths.workspace.parents[2]
    if corpus.is_file() and _contained(corpus, source_root) and not target_corpus.exists():
        target_corpus.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(corpus, target_corpus)
    return paths


def copy_checkpoint(module: SavedCheckpoint, destination_root: Path) -> PipelinePaths:
    if isinstance(module, SavedGraphModule):
        return copy_graph_checkpoint(module, destination_root)
    paths = pipeline_paths(Path(module.paths.workspace.name + ".txt"), destination_root / "pipeline_output",
                           module.course_code, module.module_number)
    stage_structured_module(module.source, paths.structured_text,
        course_code=module.course_code, module_number=module.module_number,
        identity_from_filename=module.identity_from_filename)
    return paths
