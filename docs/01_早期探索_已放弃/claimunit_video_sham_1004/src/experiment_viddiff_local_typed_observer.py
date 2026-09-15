"""Local-only canary for typed VidDiff predicate observations.

The local VLM sees one video role at a time and emits no A/B direction. A
deterministic comparator runs after both unary records exist. Dataset labels are
loaded only after every selected case has been measured.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from jsonschema import Draft202012Validator
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from experiment_viddiff_local_qwen3vl import DEFAULT_MODEL
from pipeline import write_json
from viddiff_predicate_observer import (
    adjudicate_predicate_observations,
    observation_record_schema,
)
from viddiff_probe import select_cases_for_run


def _json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"local VLM returned no JSON object: {text[-500:]}")
    return json.loads(cleaned[start : end + 1])


class LocalTypedObserver:
    def __init__(self, model_path: Path) -> None:
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        ).eval()
        self.validator = Draft202012Validator(observation_record_schema())

    def observe(
        self,
        *,
        image_paths: list[Path],
        role: str,
        statement: str,
        action_description: str,
        protocol: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        prompt = (
            "You are a local label-blind unary video observer. All three images come from ONE "
            f"video role {role}: a chronological timeline, motion strips, and targeted adjacent "
            "frames. Do not compare with another video, choose A/B/C, guess a benchmark answer, "
            "or use generic skill. Fill exactly one typed observation record. Follow the "
            "proof_obligations. scalar_value is the target_variable; use null when it cannot be "
            "measured. measured_variable must exactly copy protocol.target_variable. event_count "
            "is only for count/persistent-event predicates and must equal the number of "
            "observed_units. Each observed_units description must satisfy every "
            "required_unit_keyword_groups group and the observation_unit_definition; do not "
            "substitute a nearby action. violation_count "
            "is only for visibility violations. Hidden force/tension is observable only through "
            "timestamped visible proxies. For universal 'never' claims, record observed violations; "
            "absence in sampled frames does not prove the universal. Return JSON only.\n"
            f"STATEMENT={statement!r}\nACTION={action_description!r}\n"
            f"PROTOCOL={json.dumps(protocol, ensure_ascii=False)}\n"
            f"ROLE_METADATA={json.dumps(metadata, ensure_ascii=False)}\n"
            f"JSON_SCHEMA={json.dumps(observation_record_schema(), ensure_ascii=False)}"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": image} for image in images),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
            )
        generated = generated[:, inputs.input_ids.shape[1] :]
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        record = _json_object(decoded)
        errors = sorted(
            self.validator.iter_errors(record),
            key=lambda error: (list(error.absolute_path), error.message),
        )
        if errors:
            details = "; ".join(
                f"{'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
                for error in errors[:8]
            )
            raise ValueError(f"typed observation schema failure: {details}")
        return record


def _asset_paths(
    protocol_report: Path,
    v2_report: Path,
    v3_report: Path,
    sample_key: str,
    role: str,
) -> list[Path]:
    suffix = role.lower()
    return [
        protocol_report.parent / sample_key / f"video_{suffix}_protocol_timeline.jpg",
        v2_report.parent / sample_key / f"video_{suffix}_motion.jpg",
        v3_report.parent / sample_key / f"video_{suffix}_targeted.jpg",
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_report = json.loads(args.protocol_report.read_text(encoding="utf-8"))
    selected = [
        item
        for item in protocol_report.get("cases") or []
        if not args.case_id or str(item.get("sample_key")) in set(args.case_id)
    ]
    observer = LocalTypedObserver(args.model)
    measured: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        records: dict[str, Any] = {}
        for role_index, role in enumerate(("A", "B")):
            paths = _asset_paths(
                args.protocol_report, args.v2_report, args.v3_report,
                str(item["sample_key"]), role,
            )
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"missing local derived assets: {missing}")
            records[role.lower()] = observer.observe(
                image_paths=paths,
                role=role,
                statement=str(item["candidate_difference"]),
                action_description=str(item["action_description"]),
                protocol=item["protocol"],
                metadata=item["protocol_assets"][role_index],
            )
        adjudication = adjudicate_predicate_observations(
            item["protocol"], records["a"], records["b"]
        )
        measured.append(
            {
                "sample_key": item["sample_key"],
                "comparison_statement": item["candidate_difference"],
                "protocol_family": item["protocol"]["family"],
                "records": records,
                "measurement": adjudication,
            }
        )
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(selected)}",
                    "sample_key": item["sample_key"],
                    "family": item["protocol"]["family"],
                    "measurement": adjudication["preferred_video"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    # Verifier phase begins only after every unary observation is frozen.
    local_rows = select_cases_for_run(
        seed=args.seed,
        sample_size=args.sample_size,
        selection_report=args.selection_report,
    )
    oracle = {
        str(row["sample_key"]): str(
            row["differences_gt"][str(row["selected_difference_key"])]
        )
        for row in local_rows
    }
    for item in measured:
        item["oracle_direction"] = oracle[str(item["sample_key"])]
        item["direction_correct"] = (
            item["measurement"]["preferred_video"] == item["oracle_direction"]
        )
    passed = sum(item["direction_correct"] for item in measured)
    return {
        "artifact": "viddiff_local_typed_predicate_observer_canary_v1",
        "external_service_calls": 0,
        "generator_label_visibility": False,
        "labels_loaded_only_after_all_observations": True,
        "cases": measured,
        "summary": {
            "cases": len(measured),
            "passed": passed,
            "accuracy": round(passed / len(measured), 4) if measured else 0.0,
            "schema_valid_observations": len(measured) * 2,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--protocol-report", type=Path, required=True)
    parser.add_argument("--v2-report", type=Path, required=True)
    parser.add_argument("--v3-report", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
