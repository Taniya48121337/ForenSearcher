import argparse
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from viddiff_batch import (
    DEFAULT_ACCEPTED,
    DEFAULT_ADAPTIVE,
    VidDiffAdaptiveMeasurementEnvironment,
    VidDiffRecordedMeasurementEnvironment,
    _candidate_set_reachability,
    reconcile_saved_run,
    VIDDIFF_SCHEMA,
    VIDDIFF_TOOL_REGISTRY,
    _manifest,
)
from interactive_harness import TaskContract, VideoForensicHarness
from viddiff_rejection_sampler import (
    DEFAULT_REPORTS,
    _candidate_is_admissible,
    _method_for_run_id,
    run,
)
from viddiff_probe import (
    _case_index_suffix,
    _resume_case_map,
    _validate_teacher_payload,
    bind_frozen_selection,
)
from viddiff_probe_v4 import compare_independent, predicate_family
from viddiff_v4_integrate import integrate as integrate_v4
from viddiff_consolidate import (  # noqa: E402
    DEFAULT_RUNS as CONSOLIDATION_RUNS,
    consolidate,
)


class VidDiffSelectionTest(unittest.TestCase):
    def test_frozen_dry_run_selection_controls_case_order_and_statement(self):
        rows = [
            {
                "sample_key": "surgery_20",
                "differences_annotated": {"2": {"description": "Statement 20"}},
            },
            {
                "sample_key": "surgery_73",
                "differences_annotated": {"1": {"description": "Statement 73"}},
            },
        ]
        report = {
            "seed": 20260821,
            "sample_size": 2,
            "cases": [
                {
                    "sample_key": "surgery_73",
                    "selected_difference_key": "1",
                    "candidate_difference": "Statement 73",
                },
                {
                    "sample_key": "surgery_20",
                    "selected_difference_key": "2",
                    "candidate_difference": "Statement 20",
                },
            ],
        }
        selected = bind_frozen_selection(
            report, rows, seed=20260821, sample_size=2
        )
        self.assertEqual(
            [item["sample_key"] for item in selected],
            ["surgery_73", "surgery_20"],
        )
        self.assertEqual(
            [item["selected_difference_key"] for item in selected], ["1", "2"]
        )

    def test_frozen_dry_run_selection_rejects_statement_drift(self):
        rows = [
            {
                "sample_key": "surgery_20",
                "differences_annotated": {"2": {"description": "Current statement"}},
            }
        ]
        report = {
            "seed": 20260821,
            "sample_size": 1,
            "cases": [
                {
                    "sample_key": "surgery_20",
                    "selected_difference_key": "2",
                    "candidate_difference": "Drifted statement",
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "does not match"):
            bind_frozen_selection(report, rows, seed=20260821, sample_size=1)

    def test_symmetric_candidate_requires_order_consistency(self):
        self.assertTrue(
            _candidate_is_admissible(
                {"oracle_match": True, "order_consistent": True}
            )
        )
        self.assertTrue(
            _candidate_is_admissible(
                {"oracle_match": True, "order_consistent": None}
            )
        )
        self.assertFalse(
            _candidate_is_admissible(
                {"oracle_match": True, "order_consistent": False}
            )
        )

    def test_large_index_panel_uses_windows_safe_directory_suffix(self):
        indices = list(range(30))
        suffix = _case_index_suffix(indices)
        self.assertEqual(suffix, "_subset30_range0-29")
        self.assertLessEqual(len(suffix), 80)

    def test_teacher_payload_boundary_accepts_only_student_visible_metadata(self):
        _validate_teacher_payload(
            action_description="Perform the shared suturing action.",
            candidate_difference="The right tool moves more radially.",
            metadata={
                "video_a": {
                    "role": "A",
                    "local_times_s": [0.0, 1.0],
                    "local_duration_s": 2.0,
                },
                "video_b": {
                    "role": "B",
                    "windows": [{"interval_s": [0.5, 1.5]}],
                },
            },
        )

    def test_teacher_payload_boundary_rejects_paths_urls_and_verifier_keys(self):
        safe = {
            "action_description": "Perform the shared suturing action.",
            "candidate_difference": "The right tool moves more radially.",
        }
        for metadata in (
            {"video_path": r"C:\\private\\clip.mp4"},
            {"nested": {"oracle_direction": "a"}},
            {"asset": "https://example.com/private.jpg"},
        ):
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    _validate_teacher_payload(metadata=metadata, **safe)

    def test_resume_reuses_only_completed_measurements_for_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "probe_report.json"
            report.write_text(
                json.dumps(
                    {
                        "cases": [
                            {"sample_key": "done", "status": "passed", "measurement": {}},
                            {"sample_key": "dry", "status": "ready"},
                            {"sample_key": "broken", "status": "execution_failed"},
                            {"sample_key": "failed", "status": "failed", "measurement": {}},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            cli = _resume_case_map(report, enabled=True, require_measurement=True)
            dry = _resume_case_map(report, enabled=True, require_measurement=False)
            self.assertEqual(set(cli), {"done", "failed"})
            self.assertEqual(set(dry), {"done", "dry", "failed"})
            self.assertEqual(
                _resume_case_map(report, enabled=False, require_measurement=True), {}
            )

    def test_method_resolution_accepts_indexed_scale_batches(self):
        self.assertEqual(
            _method_for_run_id("viddiff_v3_surgery_seed_20260818_n5_idx10-11-12-13-14")["tool_name"],
            "compare_action_refined_symmetric",
        )

    def test_consolidated_route_portfolio_exports_gold_and_preference_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            report = consolidate(CONSOLIDATION_RUNS, output_root)

            self.assertEqual(report["summary"]["cases"], 10)
            self.assertEqual(report["summary"]["passed"], 9)
            self.assertEqual(report["summary"]["oracle_unreachable_candidate_sets"], 1)
            self.assertEqual(report["route_preference_pair_count"], 4)
            self.assertEqual(report["tool_telemetry"]["selected_but_not_indispensable"], {})
            self.assertEqual(
                sum(bool(line.strip()) for line in (output_root / "sft_clean_accepted.jsonl").read_text(encoding="utf-8").splitlines()),
                61,
            )

    def test_best_of_tool_tree_requires_swap_consistency_without_student_oracle_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run(
                argparse.Namespace(
                    reports=DEFAULT_REPORTS,
                    output_root=Path(directory),
                )
            )
            self.assertEqual(report["summary"]["accepted"], 5)
            self.assertEqual(report["summary"]["rejected"], 5)
            self.assertEqual(report["summary"]["student_export_oracle_leak_tokens"], [])
            self.assertEqual(report["summary"]["student_export_private_leak_tokens"], [])
            self.assertFalse(report["summary"]["safe_to_build_harness_sft"])
            self.assertEqual(report["summary"]["adaptive_candidate_sets"], 10)
            self.assertTrue(report["summary"]["safe_to_build_adaptive_harness"])
            exported = (Path(directory) / report["run_id"] / "candidate_measurements.student.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertNotIn('"video_path"', exported)
            self.assertNotIn('"candidate_report"', exported)

    def test_recorded_environment_returns_only_selected_real_measurement(self):
        first = json.loads(DEFAULT_ACCEPTED.read_text(encoding="utf-8").splitlines()[0])
        environment = VidDiffRecordedMeasurementEnvironment(first)
        environment.execute("inspect_comparison_task", {"sample_ref": "current_sample"})
        observation = environment.execute(
            first["selected_tool"]["name"],
            first["selected_tool"]["arguments"],
        )
        serialized = str(observation)
        self.assertNotIn("oracle_direction", serialized)
        self.assertNotIn("differences_gt", serialized)
        self.assertEqual(
            observation["evidence_atoms"][0].observation["preferred_video"],
            first["tool_observation"]["preferred_video"],
        )

    def test_adaptive_environment_branches_only_from_visible_measurements(self):
        items = [
            json.loads(line)
            for line in DEFAULT_ADAPTIVE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        item = next(value for value in items if value["sample_key"] == "surgery_159")
        environment = VidDiffAdaptiveMeasurementEnvironment(item)
        environment.execute("inspect_comparison_task", {"sample_ref": "current_sample"})
        self.assertEqual(set(environment.eligible_tool_names()), set(environment.METHOD_NAMES))

        uniform = next(entry for entry in item["candidate_tools"] if entry["name"] == "compare_uniform_timeline")
        first = environment.execute(uniform["name"], uniform["arguments"])
        self.assertFalse(first["goal_satisfied"])
        self.assertIn("compare_global_motion_symmetric", environment.eligible_tool_names())

        global_tool = next(
            entry for entry in item["candidate_tools"] if entry["name"] == "compare_global_motion_symmetric"
        )
        second = environment.execute(global_tool["name"], global_tool["arguments"])
        self.assertTrue(second["goal_satisfied"])
        self.assertEqual(second["result"]["routing_state"]["reason_code"], "two_methods_agree")
        self.assertEqual(environment.eligible_tool_names(), [])
        serialized = str([first, second])
        self.assertNotIn("oracle", serialized.lower())

    def test_evidence_dropout_reopens_a_real_measurement_choice(self):
        items = [
            json.loads(line)
            for line in DEFAULT_ADAPTIVE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        item = next(value for value in items if value["sample_key"] == "surgery_155")
        environment = VidDiffAdaptiveMeasurementEnvironment(item)
        manifest = _manifest(item)
        harness = VideoForensicHarness(
            TaskContract(
                contract_id=f"CONTRACT-{manifest['case_id']}",
                episode_mode="clean",
                task=manifest["task"],
                visible_assets={"source": "video_a", "suspect": "video_b"},
                claim_schema_id=VIDDIFF_SCHEMA["schema_id"],
            ),
            environment,
            claim_schema=VIDDIFF_SCHEMA,
            tool_registry=VIDDIFF_TOOL_REGISTRY,
            target_fields=VIDDIFF_SCHEMA["target_fields"],
            invariant_fields=VIDDIFF_SCHEMA["invariant_fields"],
        )
        inspect = harness.execute_tool(
            "inspect_comparison_task", {"sample_ref": "current_sample"}, "inspect"
        )
        arguments = item["candidate_tools"][0]["arguments"]
        harness.execute_tool("compare_action_refined_symmetric", arguments, "action-clean")
        harness.execute_tool("compare_uniform_timeline", arguments, "uniform-clean")
        global_clean = harness.execute_tool(
            "compare_global_motion_symmetric", arguments, "global-clean"
        )
        global_atom = global_clean["created_evidence_ids"][0]
        global_values = global_clean["observation"]["result"]
        claim_values = {
            "sample_key": item["sample_key"],
            "comparison_statement": item["task"]["comparison_statement"],
            **{
                field: global_values[field]
                for field in VIDDIFF_SCHEMA["target_fields"]
            },
        }
        harness.execute_tool(
            "create_candidate_claim",
            {"claim_id": "claim_1", "field_values": claim_values},
            "create",
        )
        task_atom = inspect["created_evidence_ids"][0]
        harness.execute_tool(
            "bind_evidence",
            {
                "claim_id": "claim_1",
                "bindings": {
                    "sample_key": [task_atom],
                    "comparison_statement": [task_atom],
                    **{field: [global_atom] for field in VIDDIFF_SCHEMA["target_fields"]},
                },
            },
            "bind",
        )
        harness.execute_tool(
            "set_field_decision",
            {
                "claim_id": "claim_1",
                "fields": VIDDIFF_SCHEMA["required_fields"],
                "decision_status": "committed",
            },
            "commit",
        )
        harness.execute_tool("stop", {}, "stop")
        harness.inject_evidence_dropout(global_atom)
        harness.execute_tool(
            "set_field_decision",
            {
                "claim_id": "claim_1",
                "fields": VIDDIFF_SCHEMA["target_fields"],
                "decision_status": "open",
            },
            "reopen",
        )
        reopened = environment.eligible_tool_names()
        self.assertIn("compare_global_motion_symmetric", reopened)
        self.assertGreaterEqual(len(reopened), 2)

        alternative = harness.execute_tool(
            "compare_global_motion_symmetric", arguments, "global-repair"
        )
        compatibility = alternative["observation"]["claim_gap_compatibility"]
        self.assertTrue(compatibility["all_open_claim_values_supported"])
        self.assertEqual(compatibility["incompatible_claim_fields"], [])
        self.assertNotEqual(alternative["created_evidence_ids"][0], global_atom)
        self.assertEqual(harness.goals["G-DIFFERENCE"].status, "satisfied")
        self.assertEqual(environment.eligible_tool_names(), [])
        self.assertNotIn("oracle", json.dumps(alternative, default=str).lower())

    def test_label_blind_routing_variants_expose_different_visible_paths(self):
        items = [
            json.loads(line)
            for line in DEFAULT_ADAPTIVE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        item = next(value for value in items if value["sample_key"] == "surgery_159")
        environment = VidDiffAdaptiveMeasurementEnvironment(item, routing_variant="uniform_first")
        inspected = environment.execute("inspect_comparison_task", {"sample_ref": "current_sample"})
        contract = inspected["result"]["routing_contract"]
        self.assertEqual(contract["variant"], "uniform_first")
        self.assertFalse(contract["label_visible_to_router"])
        self.assertEqual(environment.eligible_tool_names(), ["compare_uniform_timeline"])

        uniform = next(entry for entry in item["candidate_tools"] if entry["name"] == "compare_uniform_timeline")
        first = environment.execute(uniform["name"], uniform["arguments"])
        self.assertFalse(first["goal_satisfied"])
        self.assertEqual(environment.eligible_tool_names(), ["compare_global_motion_symmetric"])
        self.assertNotIn("oracle", json.dumps([inspected, first], default=str).lower())

    def test_verifier_reachability_filters_only_impossible_candidate_sets(self):
        items = {
            item["sample_key"]: item
            for item in (
                json.loads(line)
                for line in DEFAULT_ADAPTIVE.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        unreachable = _candidate_set_reachability(items["surgery_159"], "a")
        reachable = _candidate_set_reachability(items["surgery_28"], "a")

        self.assertFalse(unreachable["reachable"])
        self.assertTrue(reachable["reachable"])
        self.assertEqual(unreachable["candidate_directions"], ["b", "b", "c"])
        self.assertNotIn("oracle", json.dumps([unreachable, reachable]).lower())

    def test_independent_predicate_comparison_is_calibrated_and_abstains(self):
        left = {"support_score": 23, "confidence": 0.75, "best_interval_s": [1, 2]}
        right = {"support_score": 34, "confidence": 0.63, "best_interval_s": [3, 4]}
        result = compare_independent(left, right)
        self.assertEqual(result["preferred_video"], "c")
        self.assertEqual(predicate_family("The tips are never out of view"), "universal_visibility")
        self.assertEqual(
            predicate_family("The needle is inserted more perpendicular to the incision"),
            "needle_insertion_angle",
        )
        self.assertEqual(predicate_family("The suturing thread has lower tension"), "suture_tension_proxy")
        self.assertEqual(predicate_family("The suturing thread tangles"), "suture_tangle_topology")
        self.assertEqual(
            predicate_family("The movement of the needle through the hoop is more radial"),
            "hoop_radial_trajectory",
        )
        self.assertEqual(
            predicate_family("Both graspers are used efficiently."),
            "dual_grasper_efficiency",
        )
        self.assertEqual(
            predicate_family("The needle is grasped closer to the tip"),
            "needle_tip_grasp_position",
        )
        self.assertEqual(
            predicate_family("The dot is more accurately hit"),
            "target_dot_accuracy",
        )

    def test_optional_predicate_tool_is_added_without_oracle_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "augmented.jsonl"
            summary = integrate_v4(output_path=output)
            self.assertEqual(sorted(summary["augmented_sample_keys"]), ["surgery_116", "surgery_159"])
            items = {
                item["sample_key"]: item
                for item in (json.loads(line) for line in output.read_text(encoding="utf-8").splitlines())
            }
            item = items["surgery_159"]
            environment = VidDiffAdaptiveMeasurementEnvironment(item, routing_variant="predicate_first")
            environment.execute("inspect_comparison_task", {"sample_ref": "current_sample"})
            self.assertEqual(environment.eligible_tool_names(), ["compare_predicate_independent"])
            candidate = next(
                entry for entry in item["candidate_tools"] if entry["name"] == "compare_predicate_independent"
            )
            observation = environment.execute(candidate["name"], candidate["arguments"])
            self.assertTrue(observation["goal_satisfied"])
            self.assertNotIn("oracle", json.dumps(item).lower())

    def test_predicate_measurement_preserves_typed_proof_facts_but_not_hidden_metadata(self):
        base_observation = {
            "preferred_video": "c",
            "video_a_interval_s": [0.0, 4.0],
            "video_b_interval_s": [0.0, 5.0],
            "confidence": 0.25,
            "order_consistent": True,
            "decision_rule": "universal_claim_no_observed_counterexample_abstain",
            "comparative_rationale": "No sampled counterexample; sparse sampling cannot prove never.",
        }
        candidates = []
        for name in VidDiffAdaptiveMeasurementEnvironment.METHOD_NAMES:
            candidates.append(
                {
                    "name": name,
                    "arguments": {
                        "video_a_ref": "video_a",
                        "video_b_ref": "video_b",
                        "comparison_statement": "The tips are never out of view",
                    },
                    "tool_observation": dict(base_observation),
                    "provenance": {"label_visible_to_tool": False},
                }
            )
        typed = {
            "name": "compare_predicate_independent",
            "arguments": {
                "video_a_ref": "video_a",
                "video_b_ref": "video_b",
                "comparison_statement": "The tips are never out of view",
            },
            "tool_observation": {
                **base_observation,
                "predicate_family": "visibility_coverage",
                "metric_a": 0.0,
                "metric_b": 0.0,
                "usable_a": True,
                "usable_b": True,
                "reasons_a": ["sparse_sampling_cannot_prove_never"],
                "reasons_b": ["sparse_sampling_cannot_prove_never"],
                "independent_typed_records": {
                    "a": {"valid_units": 48, "violation_count": 0},
                    "b": {"valid_units": 48, "violation_count": 0},
                },
                "oracle_direction": "a",
                "verifier_score": 1.0,
                "local_path": "C:/hidden/video.mp4",
            },
            "provenance": {"label_visible_to_tool": False},
        }
        candidates.append(typed)
        item = {
            "sample_key": "typed-test",
            "task": {"comparison_statement": "The tips are never out of view"},
            "candidate_tools": candidates,
        }
        environment = VidDiffAdaptiveMeasurementEnvironment(item, routing_variant="predicate_first")
        environment.execute("inspect_comparison_task", {"sample_ref": "current_sample"})
        result = environment.execute(typed["name"], typed["arguments"])
        visible = result["result"]
        atom_visible = result["evidence_atoms"][0].observation
        self.assertEqual(visible["predicate_family"], "visibility_coverage")
        self.assertEqual(visible["independent_typed_records"]["a"]["valid_units"], 48)
        self.assertEqual(atom_visible, visible)
        serialized = json.dumps(result, default=lambda value: value.__dict__)
        self.assertNotIn("oracle_direction", serialized)
        self.assertNotIn("verifier_score", serialized)
        self.assertNotIn("local_path", serialized)

    def test_saved_cli_execution_can_be_reaudited_without_teacher_calls(self):
        source_attempt = (
            ROOT
            / "outputs"
            / "utqr_v2_viddiff_canary"
            / "viddiff_adaptive_seed_20260818_n2_bon1"
            / "cases"
            / "surgery_80"
            / "attempts"
            / "01_balanced"
        )
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "saved_run"
            attempt_root = run_root / "cases" / "surgery_80" / "attempts" / "01_balanced"
            shutil.copytree(source_attempt, attempt_root)
            report = {
                "run_id": "saved_viddiff_test",
                "dataset": "VidDiffBench",
                "routing_mode": "adaptive",
                "accepted_input_count": 1,
                "cases": [
                    {
                        "case_id": "viddiff-surgery_80",
                        "status": "rubric_rejected",
                        "attempts": [
                            {
                                "attempt_index": 1,
                                "routing_variant": "balanced",
                                "status": "rubric_rejected",
                                "artifact_dir": str(attempt_root),
                            }
                        ],
                    }
                ],
            }
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "batch_report.json").write_text(
                json.dumps(report, ensure_ascii=False), encoding="utf-8"
            )

            reconciled = reconcile_saved_run(run_root)

            self.assertEqual(reconciled["summary"]["passed"], 1)
            self.assertTrue(reconciled["summary"]["safe_to_compile_sft"])
            self.assertEqual(reconciled["reconciliation"]["new_llm_calls"], 0)
            self.assertGreaterEqual(reconciled["cases"][0]["rubric_score"], 92)
            self.assertTrue((run_root / "sft_clean_accepted.jsonl").read_text(encoding="utf-8").strip())


if __name__ == "__main__":
    unittest.main()
