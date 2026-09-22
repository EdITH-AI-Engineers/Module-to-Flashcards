import json

from main import append_course_corpus, load_course_corpus


def test_course_corpus_merges_and_replaces_module_entries(tmp_path):
    append_course_corpus(tmp_path, ["Binary"], ["Which base?"], "1")
    append_course_corpus(tmp_path, ["Octal"], ["Which radix?"], "2")

    assert load_course_corpus(tmp_path) == (
        ["Binary", "Octal"],
        ["Which base?", "Which radix?"],
    )
    assert load_course_corpus(tmp_path, "1") == (["Octal"], ["Which radix?"])

    append_course_corpus(tmp_path, ["Binary revised"], ["Which base revised?"], "1")
    assert load_course_corpus(tmp_path, "1") == (["Octal"], ["Which radix?"])
    assert json.loads((tmp_path / "course_corpus.json").read_text())[
        "modules"
    ]["1"]["concept_names"] == ["Binary revised"]


def test_corrupt_course_corpus_is_ignored(tmp_path):
    (tmp_path / "course_corpus.json").write_text("not json", encoding="utf-8")
    assert load_course_corpus(tmp_path) == ([], [])