"""Small, non-destructive diagnostic for the corrected generation/scoring path.

This checks, with a fake provider and (optionally) the real local Ollama model:

  * complete generation error messages are exposed per attempt;
  * generated statements follow the <opinion> format and respect word budgets;
  * target_approval reaches the v3_very_high_detail system prompt;
  * Algorithm 1 qualification/selection and assignment behave correctly;
  * checkpoints are written during execution and can be read/resumed.

It never launches the full 500-budget experiment and writes only to a temporary
directory.

Run:
    PYTHONPATH=. python PROSE/test/diagnostic_generation_scoring.py            # fakes only
    PYTHONPATH=. python PROSE/test/diagnostic_generation_scoring.py --ollama   # + real qwen3:8b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

from PROSE.algorithm1_abortion import (
    CHECKPOINT_SCHEMA_V2,
    GenerationGroup,
    _load_checkpoint,
    run_algorithm1,
)
from PROSE.queries.abortion_query_contracts import AbortionGenQuery, GenQueryError
from PROSE.queries.fast_queries import FastGenerator
from PROSE.utils.helper_functions import count_words


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def call(self, messages, **params):
        self.calls.append((messages, params))
        response = self.responses.pop(0)
        return response, {}, {"provider": "fake"}


class FakeAgent:
    def __init__(self, identifier, description):
        self.identifier = identifier
        self.description = description

    def get_id(self):
        return self.identifier

    def get_description(self):
        return self.description


class FakeDisc:
    def __init__(self, scores):
        self.scores = scores

    def score(self, statement):
        return self.scores.get(statement, 1)


class FakeGen:
    def __init__(self, statement, *, interrupt_at=None):
        self.statement = statement
        self.interrupt_at = interrupt_at
        self.calls = 0

    def generate(self, participant_ids, target_approval, word_budget):
        self.calls += 1
        if self.interrupt_at is not None and self.calls == self.interrupt_at:
            raise KeyboardInterrupt("simulated interruption")
        from PROSE.queries.abortion_query_contracts import GeneratedStatement

        return GeneratedStatement(
            statement=self.statement,
            word_count=count_words(self.statement),
            target_approval=target_approval,
            word_budget=word_budget,
            participant_ids=tuple(participant_ids),
            attempts=1,
        )


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" :: {detail}" if detail else ""))
    if not condition:
        raise AssertionError(label)


def diagnostic_generation_errors():
    print("\n== 1. Full generation errors ==")
    responses = [
        "not tagged",                       # XML/format failure
        "<opinion>   </opinion>",           # empty output failure
        "<opinion>one two three</opinion>",  # word-count failure (budget=2)
    ]
    provider = FakeProvider(responses)
    events = []
    query = AbortionGenQuery(
        {"p1": "summary"},
        FastGenerator(text_provider=provider, prompt_type="v3_very_high_detail"),
        max_attempts=3,
        progress_callback=events.append,
    )
    try:
        query.generate(["p1"], target_approval=5, word_budget=2)
        raise AssertionError("expected generation to fail")
    except GenQueryError as exc:
        final_error = str(exc)

    for event in events:
        if event["event"] == "generation_result" and not event["success"]:
            print(
                f"  attempt={event['attempt']} "
                f"error_type={event['error_type']} "
                f"reason={event['error_message']}"
            )

    reasons = [event.get("error_message", "") for event in events]
    check("XML-format reason is exposed", any("exactly one" in r for r in reasons))
    check("empty-output reason is exposed", any("empty" in r for r in reasons))
    check("word-count reason is exposed", any("word budget is 2" in r for r in reasons))
    check("final error keeps full reason", "word budget is 2" in final_error, final_error)


def diagnostic_format_budget_and_target():
    print("\n== 2. Format, word budget and target approval ==")
    provider = FakeProvider(["<opinion>one two three</opinion>"])
    query = AbortionGenQuery(
        {"p1": "summary"},
        FastGenerator(text_provider=provider, prompt_type="v3_very_high_detail"),
    )
    result = query.generate(["p1"], target_approval=5, word_budget=3)
    system_prompt = provider.calls[0][0][0]["content"]
    print(f"  statement={result.statement!r} words={result.word_count}")
    print(f"  system_prompt excerpt: ...{system_prompt[-140:]}")
    check("statement has no XML tags", "<opinion>" not in result.statement)
    check("word budget respected", result.word_count <= 3)
    check("target approval reached prompt", "target approval level of 5.0" in system_prompt)
    check("word budget reached prompt", "at most 3 words" in system_prompt)
    check("no unrendered placeholders", "{target_approval}" not in system_prompt)


def diagnostic_checkpoint_and_selection(tmp_dir: Path):
    print("\n== 3. Algorithm 1 selection, checkpoints and resume ==")
    ids = ["p1", "p2", "p3", "p4"]
    groups = [GenerationGroup("g", tuple(ids))]
    discs = {
        "p1": FakeDisc({"one two": 6}),
        "p2": FakeDisc({"one two": 5}),
        "p3": FakeDisc({"one two": 4}),
        "p4": FakeDisc({"one two": 1}),
    }
    common = dict(
        participant_ids=ids,
        disc_queries=discs,
        generation_groups=groups,
        total_budget=4,
        utility_levels=[2, 1],
        cost_schedule=[2],
    )

    clean = run_algorithm1(gen_query=FakeGen("one two"), **common)
    check("clean run selects statements", len(clean["selected_statements"]) > 0)
    check(
        "qualification picks highest-utility supporters",
        clean["selected_statements"][0]["assigned_participant_ids"] == ["p1", "p2"],
    )
    check(
        "required support uses ceil(cost*n/B)",
        clean["selected_statements"][0]["required_support"] == 2,
    )

    checkpoint_path = tmp_dir / "budget_4.checkpoint.json"
    try:
        run_algorithm1(
            gen_query=FakeGen("one two", interrupt_at=2),
            checkpoint_path=checkpoint_path,
            **common,
        )
        raise AssertionError("expected simulated interruption")
    except KeyboardInterrupt:
        pass

    checkpoint = _load_checkpoint(checkpoint_path)
    check("checkpoint written during execution", checkpoint is not None)
    check("checkpoint schema is v2", checkpoint["schema_version"] == CHECKPOINT_SCHEMA_V2)
    for key in (
        "candidate_records",
        "selected_statements",
        "assigned_participant_ids",
        "remaining_participant_ids",
        "budget_remaining",
        "candidate_number",
        "cursor",
    ):
        check(f"checkpoint contains {key}", key in checkpoint)
    print(
        f"  checkpoint candidate_number={checkpoint['candidate_number']} "
        f"selected={len(checkpoint['selected_statements'])} "
        f"budget_remaining={checkpoint['budget_remaining']}"
    )

    resumed = run_algorithm1(
        gen_query=FakeGen("one two"),
        checkpoint_path=checkpoint_path,
        resume_state=checkpoint,
        **common,
    )
    for key in (
        "selected_statements",
        "assigned_participant_ids",
        "remaining_participant_ids",
        "budget_consumed",
        "budget_remaining",
        "termination_reason",
    ):
        check(f"resume matches clean run: {key}", resumed[key] == clean[key])


def diagnostic_real_ollama():
    print("\n== 4. Real local Ollama smoke (qwen3:8b, 2 participants) ==")
    import json as _json
    from PROSE.utils.ollama_provider import OllamaProvider

    endpoint = "http://172.21.112.1:11434"
    summaries_path = Path("calibration/participant_summaries_all_100.json")
    records = _json.loads(summaries_path.read_text(encoding="utf-8"))["records"]
    summaries = {record["participant_id"]: record["summary"] for record in records[:2]}

    provider = OllamaProvider(
        endpoint=endpoint,
        model="qwen3:8b",
        timeout=180,
        think=False,
        log_prompts=False,
        log_responses=False,
    )
    generator = FastGenerator(
        text_provider=provider, prompt_type="v3_very_high_detail", temperature=0
    )
    query = AbortionGenQuery(summaries, generator, max_attempts=3)
    result = query.generate(list(summaries), target_approval=5, word_budget=60)
    print(f"  statement={result.statement!r}")
    print(f"  words={result.word_count} attempts={result.attempts}")
    check("real statement non-empty", bool(result.statement))
    check("real statement has no XML tags", "<opinion>" not in result.statement)
    check("real word budget respected", result.word_count <= 60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="also run the small real qwen3:8b smoke test",
    )
    args = parser.parse_args()

    diagnostic_generation_errors()
    diagnostic_format_budget_and_target()
    with tempfile.TemporaryDirectory(prefix="gsc-prose-diagnostic-") as tmp:
        diagnostic_checkpoint_and_selection(Path(tmp))
    if args.ollama:
        diagnostic_real_ollama()
    print("\nALL DIAGNOSTIC CHECKS PASSED")


if __name__ == "__main__":
    main()
