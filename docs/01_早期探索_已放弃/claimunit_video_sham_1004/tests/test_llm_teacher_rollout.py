from __future__ import annotations

import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_teacher_rollout import (  # noqa: E402
    OpenAIResponsesTeacher,
    TeacherAction,
    TeacherRolloutIncomplete,
    _accepted_turn_datums,
    _replay_trajectory_path,
    _write_bundle,
    build_llm_teacher_bundle,
    teacher_tool_schemas,
    validate_llm_teacher_bundle,
)
from cli_teacher_rollout import CodexCLITeacher, action_schema_for_tools, compact_teacher_context  # noqa: E402
from interactive_harness import (  # noqa: E402
    EvidenceAtom,
    TaskContract,
    VideoForensicHarness,
    evidence_interval_scope_issues,
)
from pipeline import EDITED, SOURCE, decode  # noqa: E402
from step_utility_auditor import (  # noqa: E402
    EVIDENCE_ID_PATTERN,
    _new_harness,
    _prepare_repair_harness,
    apply_step_utility_filter,
    audit_step_utility,
)
from sft_v2_sidecar import compile_transition_sidecar  # noqa: E402


REASONINGS = [
    "The media identities and neutral comparison scope are visible, so I will inspect both files before making any temporal claim.",
    "Metadata establishes the pair and scope but not the omitted content, so dense alignment is the necessary next measurement.",
    "The alignment observation exposes a contracted retained interval and boundary omissions, so I can now state one atomic grounded claim.",
    "The candidate values now exist but remain unsupported, so I will bind each field only to compatible visible evidence atoms.",
    "Every required field now has compatible active support, so committing the complete claim closes the remaining report obligations.",
    "All high priority goals and field obligations are visibly closed, so terminal validation is now the appropriate action.",
    "The dropout removed support from only five committed fields, so I will reopen exactly those fields while preserving independent decisions.",
    "The affected fields are open and the prior gap atom is unavailable, so rerunning alignment is required to obtain fresh support.",
    "Fresh contraction evidence is now active and compatible with the reopened fields, so I will bind only the damaged support slice.",
    "All reopened fields have fresh compatible support and invariant fields stayed intact, so I can recommit the repaired slice.",
    "The repair obligations are closed with fresh evidence and no independent field changed, so terminal validation should now succeed.",
]


def action(index: int, name: str, params: dict) -> TeacherAction:
    reasoning = REASONINGS[index]
    raw_arguments = json.dumps({"reasoning": reasoning, **params}, ensure_ascii=False, separators=(",", ":"))
    call_id = f"call_fake_{index + 1:02d}"
    replay_item = {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": raw_arguments,
    }
    return TeacherAction(
        response_id=f"resp_fake_{index + 1:02d}",
        call_id=call_id,
        tool_name=name,
        reasoning=reasoning,
        params=copy.deepcopy(params),
        raw_arguments=raw_arguments,
        usage={"input_tokens": 1, "output_tokens": 1},
        replay_items=[replay_item],
    )


class ScriptedTeacher:
    model = "fake_teacher"

    def __init__(self) -> None:
        fields = {
            "source_id": "vimeo-26432864",
            "claim_family": "selective_temporal_omission",
            "manipulation_type": "frame_dropping",
            "direction": "source_to_edited",
            "comparison_scope_s": [1.0, 12.0],
            "retained_source_interval_s": [2.369, 10.377],
            "omitted_source_intervals_s": [[1.0, 2.369], [10.377, 12.0]],
            "estimated_removed_time_s": 2.992,
        }
        all_fields = list(fields)
        target_fields = [
            "claim_family",
            "manipulation_type",
            "direction",
            "omitted_source_intervals_s",
            "estimated_removed_time_s",
        ]
        clean_bindings = {
            "source_id": ["EA-ALIGN-001"],
            "claim_family": ["EA-GAP-001"],
            "manipulation_type": ["EA-GAP-001"],
            "direction": ["EA-GAP-001"],
            "comparison_scope_s": ["EA-SCOPE-001"],
            "retained_source_interval_s": ["EA-ALIGN-001"],
            "omitted_source_intervals_s": ["EA-GAP-001"],
            "estimated_removed_time_s": ["EA-GAP-001"],
        }
        repair_bindings = {field: ["EA-GAP-002"] for field in target_fields}
        to_items = lambda bindings: [
            {"field": field, "support_ids": support_ids} for field, support_ids in bindings.items()
        ]
        self.actions = [
            action(0, "inspect_media", {"media_refs": ["source", "suspect"]}),
            action(1, "align_videos_dense", {"source_ref": "source", "suspect_ref": "suspect", "comparison_scope_s": [1.0, 12.0]}),
            action(2, "create_candidate_claim", {"claim_id": "CL-1004-001", "field_values": fields}),
            action(3, "bind_evidence", {"claim_id": "CL-1004-001", "binding_items": to_items(clean_bindings)}),
            action(4, "set_field_decision", {"claim_id": "CL-1004-001", "fields": all_fields, "decision_status": "committed"}),
            action(5, "stop", {}),
            action(6, "set_field_decision", {"claim_id": "CL-1004-001", "fields": target_fields, "decision_status": "open"}),
            action(7, "align_videos_dense", {"source_ref": "source", "suspect_ref": "suspect", "comparison_scope_s": [1.0, 12.0]}),
            action(8, "bind_evidence", {"claim_id": "CL-1004-001", "binding_items": to_items(repair_bindings)}),
            action(9, "set_field_decision", {"claim_id": "CL-1004-001", "fields": target_fields, "decision_status": "committed"}),
            action(10, "stop", {}),
        ]
        self.cursor = 0

    def next_action(self, input_items: list[dict], tools: list[dict]) -> TeacherAction:
        self.asserted_inputs = input_items
        if len(input_items) > 2:
            assert input_items[-1]["type"] == "function_call_output"
            assert json.loads(input_items[-1]["output"])["status"] in {"executed", "accepted"}
        selected = self.actions[self.cursor]
        self.cursor += 1
        return selected


class LLMTeacherRolloutTest(unittest.TestCase):
    def test_grounding_parser_keeps_lowercase_and_underscore_evidence_ids_whole(self):
        text = "Bind EA-VIDDIFF-compare_global_motion_symmetric-002 before committing."
        self.assertEqual(
            EVIDENCE_ID_PATTERN.findall(text),
            ["EA-VIDDIFF-compare_global_motion_symmetric-002"],
        )

    def test_teacher_prompt_requires_visible_conflict_adjudication(self):
        from llm_teacher_rollout import _teacher_system_prompt

        prompt = _teacher_system_prompt("clean")
        self.assertIn("never choose the newest atom", prompt)
        self.assertIn("order-consistency", prompt)
        self.assertIn("direct visual counterexamples", prompt)

    def test_repair_prompt_distinguishes_binding_loss_from_evidence_loss(self):
        from llm_teacher_rollout import _teacher_system_prompt

        prompt = _teacher_system_prompt("repair")
        self.assertIn("binding_invalidation removes only claim-to-evidence links", prompt)
        self.assertIn("same active atom remains valid", prompt)
        self.assertIn("decision_reopen preserves both evidence and bindings", prompt)

    def test_create_claim_disappears_after_the_single_claim_exists(self):
        harness, _ = _new_harness(self.bundle)
        for turn in self.bundle["clean_episode"]["turn_history"]:
            if turn["tool_name"] == "create_candidate_claim":
                self.assertIn(
                    "create_candidate_claim",
                    [item["name"] for item in harness.eligible_tool_registry()],
                )
            harness.execute_tool(turn["tool_name"], copy.deepcopy(turn["params"]), f"replay_{turn['turn_idx']}")
            if turn["tool_name"] == "create_candidate_claim":
                break
        self.assertNotIn("create_candidate_claim", [item["name"] for item in harness.eligible_tool_registry()])

    def test_duplicate_binding_is_rejected_as_a_noop(self) -> None:
        harness, _ = _new_harness(self.bundle)
        bind_turn = None
        for turn in self.bundle["clean_episode"]["turn_history"]:
            if turn["tool_name"] == "bind_evidence":
                bind_turn = turn
                break
            harness.execute_tool(
                turn["tool_name"], copy.deepcopy(turn["params"]), f"replay_{turn['turn_idx']}"
            )
        self.assertIsNotNone(bind_turn)
        partial_params = copy.deepcopy(bind_turn["params"])
        partial_params["bindings"].pop("estimated_removed_time_s")
        first = harness.execute_tool("bind_evidence", partial_params, "partial_bind")
        self.assertEqual(first["status"], "accepted")
        duplicate = harness.execute_tool(
            "bind_evidence", copy.deepcopy(partial_params), "duplicate_bind"
        )
        self.assertEqual(duplicate["status"], "rejected")
        self.assertTrue(duplicate["rejected_fields"])
        self.assertEqual(set(duplicate["rejected_fields"].values()), {"binding_already_active"})

    def test_action_mask_exposes_stop_only_after_obligations_close(self):
        harness, _ = _new_harness(self.bundle)
        self.assertEqual([item["name"] for item in harness.eligible_tool_registry()], ["inspect_media"])
        first_turn = self.bundle["clean_episode"]["turn_history"][0]
        harness.execute_tool(first_turn["tool_name"], copy.deepcopy(first_turn["params"]), "mask_inspect")
        self.assertEqual([item["name"] for item in harness.eligible_tool_registry()], ["align_videos_dense"])
        for turn in self.bundle["clean_episode"]["turn_history"][1:-1]:
            harness.execute_tool(turn["tool_name"], copy.deepcopy(turn["params"]), f"mask_{turn['turn_idx']}")
        self.assertEqual([item["name"] for item in harness.eligible_tool_registry()], ["stop"])
        final_turn = self.bundle["clean_episode"]["turn_history"][-1]
        harness.execute_tool(final_turn["tool_name"], copy.deepcopy(final_turn["params"]), "mask_stop")
        self.assertEqual(harness.eligible_tool_registry(), [])

    @classmethod
    def setUpClass(cls) -> None:
        manifest = json.loads((ROOT / "student_view" / "case_manifest.json").read_text(encoding="utf-8"))
        cls.manifest = manifest
        cls.source_frames = decode(SOURCE)
        cls.suspect_frames = decode(EDITED)
        teacher = ScriptedTeacher()
        cls.bundle = build_llm_teacher_bundle(
            manifest=manifest,
            source_frames=cls.source_frames,
            suspect_frames=cls.suspect_frames,
            teacher=teacher,
        )
        cls.teacher = teacher
        cls.step_report = audit_step_utility(cls.bundle)
        apply_step_utility_filter(cls.bundle, cls.step_report)
        oracle = json.loads((ROOT / "oracle" / "claim_oracle.json").read_text(encoding="utf-8"))
        cls.report = validate_llm_teacher_bundle(cls.bundle, oracle=oracle)

    def test_bounded_nonterminal_episode_carries_local_failure_artifact(self) -> None:
        with self.assertRaises(TeacherRolloutIncomplete) as caught:
            build_llm_teacher_bundle(
                manifest=self.manifest,
                source_frames=self.source_frames,
                suspect_frames=self.suspect_frames,
                teacher=ScriptedTeacher(),
                max_clean_turns=1,
            )
        artifact = caught.exception.failure_artifact
        self.assertEqual(artifact["failure_phase"], "clean")
        self.assertEqual(artifact["turns_executed"], 1)
        self.assertTrue(artifact["excluded_from_sft"])
        self.assertEqual(artifact["failed_episode"]["turn_history"][0]["tool_name"], "inspect_media")

    def test_out_of_scope_measurement_atom_is_an_explicit_integrity_failure(self) -> None:
        atom = EvidenceAtom(
            evidence_id="EA-SCOPE-BAD-001",
            kind="temporal_edit_measurement",
            supports_fields=["affected_suspect_interval_s"],
            observation={"affected_suspect_interval_s": [0.0, 10.2]},
            provenance={"generated_by": "EXEC-SCOPE-BAD-001"},
        )
        issues = evidence_interval_scope_issues([atom], [0.0, 4.04])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["reason"], "outside_visible_scope")

    def test_scope_check_uses_visible_suspect_coordinate_frame(self) -> None:
        atom = EvidenceAtom(
            evidence_id="EA-SUSPECT-TIMELINE-001",
            kind="temporal_edit_measurement",
            supports_fields=["affected_suspect_interval_s"],
            observation={
                "suspect_comparison_scope_s": [0.0, 15.2],
                "affected_suspect_interval_s": [0.0, 15.2],
            },
            provenance={"generated_by": "EXEC-SUSPECT-TIMELINE-001"},
        )
        self.assertEqual(evidence_interval_scope_issues([atom], [0.0, 12.0]), [])

        atom.observation["affected_suspect_interval_s"] = [0.0, 16.0]
        issues = evidence_interval_scope_issues([atom], [0.0, 12.0])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["coordinate_frame"], "suspect")
        self.assertEqual(issues[0]["comparison_scope_s"], [0.0, 15.2])

    def test_validation_reports_visible_interval_scope_check(self) -> None:
        self.assertIn("evidence_intervals_within_visible_scope", self.report["checks"])
        self.assertTrue(self.report["checks"]["evidence_intervals_within_visible_scope"])

    def test_replay_only_artifact_uses_short_name_near_windows_path_limit(self) -> None:
        short_root = Path("outputs")
        self.assertEqual(
            _replay_trajectory_path(short_root, "clean").name,
            "llm_teacher_sft_clean_full_trajectory.jsonl",
        )
        deep_root = Path(*(["segment0123456789"] * 20))
        self.assertEqual(
            _replay_trajectory_path(deep_root, "clean").name,
            "llm_teacher_clean_full.jsonl",
        )

    def test_strict_tool_schemas_require_reasoning(self) -> None:
        schemas = teacher_tool_schemas()
        self.assertEqual({schema["name"] for schema in schemas}, {
            "inspect_media", "align_videos_dense", "create_candidate_claim",
            "bind_evidence", "set_field_decision", "stop",
        })
        for schema in schemas:
            self.assertTrue(schema["strict"])
            self.assertFalse(schema["parameters"]["additionalProperties"])
            self.assertIn("reasoning", schema["parameters"]["required"])

    def test_responses_client_preserves_reasoning_state(self) -> None:
        reasoning = "The visible state lacks media metadata, so inspecting both referenced files is the grounded first action."
        arguments = json.dumps({"reasoning": reasoning, "media_refs": ["source", "suspect"]})
        body = {
            "id": "resp_live_shape",
            "output": [
                {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque", "summary": []},
                {"type": "function_call", "call_id": "call_1", "name": "inspect_media", "arguments": arguments},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        class FakeHTTPResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")

        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeHTTPResponse()

        client = OpenAIResponsesTeacher(api_key="test-key-not-real", model="gpt-5.4")
        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            selected = client.next_action([{"role": "user", "content": "task"}], teacher_tool_schemas())

        self.assertEqual(selected.tool_name, "inspect_media")
        self.assertEqual(len(selected.replay_items), 2)
        self.assertEqual(selected.replay_items[0]["encrypted_content"], "opaque")
        self.assertEqual(captured["payload"]["tool_choice"], "required")
        self.assertFalse(captured["payload"]["parallel_tool_calls"])
        self.assertFalse(captured["payload"]["store"])
        self.assertEqual(captured["payload"]["include"], ["reasoning.encrypted_content"])

    def test_teacher_controls_real_clean_and_repair_loop(self) -> None:
        clean = self.bundle["clean_episode"]
        repair = self.bundle["repair_episode"]
        self.assertTrue(clean["terminated"])
        self.assertTrue(repair["terminated"])
        self.assertEqual(self.teacher.cursor, 11)
        self.assertEqual(self.bundle["execution_summary"]["external_tool_call_counts"]["align_videos_dense"], 2)

    def test_every_step_survives_counterfactual_utility_audit(self) -> None:
        self.assertTrue(self.step_report["passed"])
        self.assertEqual(self.step_report["summary"]["total_steps"], 11)
        self.assertEqual(self.step_report["summary"]["accepted_for_sft"], 11)
        self.assertTrue(self.step_report["checks"]["repair_reopen_is_causally_required"])
        for episode in self.step_report["episodes"].values():
            for step in episode:
                self.assertFalse(step["removal_counterfactual"]["same_final_result"])
                self.assertTrue(step["accepted_for_sft"])

    def test_invalidated_committed_field_must_be_reopened_before_rebinding(self) -> None:
        reopen_step = next(
            item
            for item in self.step_report["episodes"]["repair"]
            if item["tool_name"] == "set_field_decision"
            and self.bundle["repair_episode"]["turn_history"][item["turn_idx"] - 1]["params"].get(
                "decision_status"
            )
            == "open"
        )
        self.assertTrue(reopen_step["causally_necessary"])
        self.assertFalse(reopen_step["removal_counterfactual"]["same_final_result"])
        self.assertTrue(reopen_step["removal_counterfactual"]["replay_failures"])

    def test_tool_measurements_do_not_disclose_semantic_answer(self) -> None:
        gap_atoms = [
            atom
            for event in self.bundle["clean_episode"]["events"]
            if event["action_type"] == "external_tool_execution"
            for atom in event["result"]["evidence_atoms"]
            if atom["kind"] == "source_time_contraction"
        ]
        self.assertTrue(gap_atoms)
        for atom in gap_atoms:
            self.assertFalse({"claim_family", "manipulation_type", "direction"}.intersection(atom["observation"]))
        self.assertTrue(self.report["checks"]["tool_measurements_do_not_emit_claim_answers"])
        self.assertTrue(self.report["checks"]["final_field_values_match_bound_evidence"])

    def test_binding_rejects_right_evidence_kind_with_wrong_value(self) -> None:
        contract = TaskContract(
            contract_id="test",
            episode_mode="paired_compare",
            task="test",
            visible_assets={"comparison_scope_s": [1.0, 12.0]},
            claim_schema_id="selective_temporal_omission_v1",
        )
        harness = VideoForensicHarness(contract, object())
        for goal_id in ("G-CONTEXT", "G-DIFFERENCE"):
            harness.goals[goal_id].status = "satisfied"
        fields = {
            "source_id": "source-1",
            "claim_family": "selective_temporal_omission",
            "manipulation_type": "frame_dropping",
            "direction": "source_to_edited",
            "comparison_scope_s": [1.0, 12.0],
            "retained_source_interval_s": [2.0, 10.0],
            "omitted_source_intervals_s": [[1.0, 2.0], [10.0, 12.0]],
            "estimated_removed_time_s": 3.0,
        }
        harness.execute_tool("create_candidate_claim", {"claim_id": "claim-1", "field_values": fields}, "call-create")
        harness.evidence_store["EA-GAP-TEST"] = EvidenceAtom(
            evidence_id="EA-GAP-TEST",
            kind="source_time_contraction",
            supports_fields=["estimated_removed_time_s"],
            observation={"estimated_removed_time_s": 2.0},
            provenance={"generated_by": "test"},
        )
        result = harness.execute_tool(
            "bind_evidence",
            {"claim_id": "claim-1", "bindings": {"estimated_removed_time_s": ["EA-GAP-TEST"]}},
            "call-bind",
        )
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(
            result["rejected_fields"]["estimated_removed_time_s"],
            "claim_value_not_supported_by_evidence",
        )

    def test_rejected_action_is_context_but_never_sft_target(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "A sufficiently long rejected reasoning message that explains the proposed action.", "tool_calls": [{"id": "bad", "type": "function", "function": {"name": "stop", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "bad", "name": "stop", "content": "{\"status\":\"rejected\"}"},
            {"role": "assistant", "content": "A sufficiently long recovery reasoning message that selects a valid next action.", "tool_calls": [{"id": "good", "type": "function", "function": {"name": "inspect_media", "arguments": "{}"}}]},
        ]
        history = [
            {"turn_idx": 1, "observation": {"status": "rejected"}},
            {"turn_idx": 2, "observation": {"status": "executed"}},
        ]
        datums, excluded = _accepted_turn_datums(
            messages,
            history,
            case_id="case",
            trajectory_id="trajectory",
            trajectory_type="test",
        )
        self.assertEqual(excluded, [1])
        self.assertEqual(len(datums), 1)
        self.assertEqual(datums[0]["metadata"]["source_turn_index"], 2)
        self.assertEqual(datums[0]["messages"][2]["tool_calls"][0]["id"], "bad")
        self.assertFalse(datums[0]["metadata"]["policy_choice_training_eligible"])

    def test_evidence_choice_audit_is_aligned_to_sft_turn_without_branch_results(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": "The audio and spatial explanations remain open, so this measurement discriminates them.",
                "tool_calls": [
                    {
                        "id": "choice",
                        "type": "function",
                        "function": {"name": "compare_audio_tracks", "arguments": "{}"},
                    }
                ],
            },
        ]
        history = [{"turn_idx": 1, "observation": {"status": "executed"}}]
        audits = [
            {
                "selected_action": {"tool_name": "compare_audio_tracks"},
                "control_plane_exposure": {
                    "semantic_measurement_tool_names": [
                        "compare_audio_tracks",
                        "compare_spatial_regions",
                    ],
                    "semantic_measurement_choice_count": 2,
                },
                "evidence_choice_search": {
                    "applicable": True,
                    "visible_choice_count": 2,
                    "proposed_choice_count": 2,
                    "proposal_coverage_passed": True,
                    "selection_ambiguous": False,
                    "selected_preaction_method_fit_score": 1.0,
                    "selected_counterfactual_tool_swap_passed": True,
                    "selection_frozen_before_branch_execution": True,
                    "hindsight_outcome_used_for_selection": False,
                    "selected_branch_execution_admissible": True,
                    "proposal_retry_outcome_feedback_used": False,
                    "committed_replay_checked": True,
                    "deterministic_replay_match": True,
                    "branches": [{"result": {"private_preview": True}}],
                },
            }
        ]
        datums, excluded = _accepted_turn_datums(
            messages,
            history,
            case_id="case",
            trajectory_id="trajectory",
            trajectory_type="test",
            turn_audits=audits,
        )
        self.assertEqual(excluded, [])
        metadata = datums[0]["metadata"]
        self.assertTrue(metadata["policy_choice_training_eligible"])
        self.assertTrue(
            metadata["evidence_choice_search"][
                "selected_counterfactual_tool_swap_passed"
            ]
        )
        self.assertEqual(metadata["policy_choice_audit_status"], "passed")
        self.assertNotIn("branches", metadata["evidence_choice_search"])

    def test_transient_api_failure_is_retried_without_losing_payload(self) -> None:
        reasoning = "The visible state lacks metadata, so inspecting both media files is the necessary grounded first action."
        arguments = json.dumps({"reasoning": reasoning, "media_refs": ["source", "suspect"]})
        body = {
            "id": "resp_after_retry",
            "output": [{"type": "function_call", "call_id": "call_retry", "name": "inspect_media", "arguments": arguments}],
            "usage": {},
        }

        class FakeHTTPResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")

        error = urllib.error.HTTPError(
            url="https://api.openai.com/v1/responses",
            code=429,
            msg="rate limited",
            hdrs={},
            fp=io.BytesIO(b'{"error":"retry"}'),
        )
        client = OpenAIResponsesTeacher(api_key="test-key-not-real", max_retries=1)
        with mock.patch("urllib.request.urlopen", side_effect=[error, FakeHTTPResponse()]) as urlopen:
            with mock.patch("time.sleep") as sleep:
                selected = client.next_action([{"role": "user", "content": "task"}], teacher_tool_schemas())
        self.assertEqual(selected.response_id, "resp_after_retry")
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(1.0)

    def test_cli_teacher_uses_structured_output_without_api_key(self) -> None:
        reasoning = (
            "The visible task has no media observations yet, so inspecting both declared assets is the necessary "
            "first evidence step."
        )
        cli_payload = {
            "reasoning": reasoning,
            "tool_name": "inspect_media",
            "arguments_json": json.dumps({"media_refs": ["source", "suspect"]}),
        }

        def fake_run(command, **_kwargs):
            output_index = command.index("--output-last-message") + 1
            Path(command[output_index]).write_text(json.dumps(cli_payload), encoding="utf-8")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout='{"usage":{"input_tokens":12,"output_tokens":4}}\n',
                stderr="",
            )

        client = CodexCLITeacher(model="gpt-5.4")
        with mock.patch("subprocess.run", side_effect=fake_run) as run:
            selected = client.next_action([{"role": "user", "content": "task"}], teacher_tool_schemas())
        self.assertEqual(selected.tool_name, "inspect_media")
        self.assertTrue(selected.response_id.startswith("cli_resp_"))
        self.assertEqual(selected.usage["total_tokens"], 16)
        command = run.call_args.args[0]
        self.assertIn("--ephemeral", command)
        self.assertIn("read-only", command)

    def test_cli_action_schema_uses_current_episode_tools(self) -> None:
        schema = action_schema_for_tools(
            ROOT / "tools" / "codex_cli" / "teacher_action.schema.json",
            [{"name": "inspect_media"}, {"name": "compare_spatial_regions"}],
        )
        self.assertEqual(
            schema["properties"]["tool_name"]["enum"],
            ["inspect_media", "compare_spatial_regions"],
        )

    def test_cli_context_compaction_keeps_latest_state_without_historical_duplicates(self) -> None:
        old_state = {"version": 1, "duplicate_payload": "old" * 2000}
        new_state = {"version": 2, "available_evidence_ids": ["EA-NEW"]}
        input_items = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": json.dumps({"task": "inspect", "harness_state": old_state})},
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "inspect_media",
                "arguments": json.dumps({
                    "reasoning": "No evidence exists yet, so inspecting the pair establishes identity before comparison."
                }),
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": json.dumps({
                    "status": "executed",
                    "created_evidence_ids": ["EA-NEW"],
                    "harness_state": new_state,
                }),
            },
        ]
        compact = compact_teacher_context(input_items)
        self.assertEqual(compact["current_harness_state"], new_state)
        self.assertNotIn("harness_state", compact["task"])
        self.assertEqual(compact["action_history"][0]["observation"]["created_evidence_ids"], ["EA-NEW"])
        self.assertIn("inspecting the pair", compact["action_history"][0]["reasoning"])
        self.assertLess(len(json.dumps(compact)), len(json.dumps(input_items)))

    def test_promotion_updates_all_training_and_validation_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            with mock.patch("llm_teacher_rollout.ROOT", temporary_root):
                _write_bundle(self.bundle, {"passed": True, "checks": {}}, promote=True)
            expected = {
                "llm_teacher_clean_episode.json",
                "llm_teacher_repair_episode.json",
                "llm_teacher_execution_summary.json",
                "llm_teacher_validation_report.json",
                "llm_teacher_step_utility_report.json",
                "llm_teacher_sft_clean.jsonl",
                "llm_teacher_sft_repair.jsonl",
                "llm_teacher_sft_clean_full_trajectory.jsonl",
                "llm_teacher_sft_repair_full_trajectory.jsonl",
                "llm_teacher_transition_sidecar.jsonl",
                "interactive_clean_episode.json",
                "interactive_repair_episode.json",
                "interactive_execution_summary.json",
                "interactive_validation_report.json",
                "interactive_step_utility_report.json",
                "interactive_sft_clean.jsonl",
                "interactive_sft_repair.jsonl",
                "interactive_sft_clean_full_trajectory.jsonl",
                "interactive_sft_repair_full_trajectory.jsonl",
                "interactive_transition_sidecar.jsonl",
            }
            self.assertEqual({path.name for path in (temporary_root / "outputs").iterdir()}, expected)

    def test_repair_uses_fresh_environment_evidence(self) -> None:
        summary = self.bundle["execution_summary"]
        repair = self.bundle["repair_episode"]
        dropped = summary["dropped_evidence_id"]
        self.assertNotIn(dropped, repair["final_state"]["available_evidence_ids"])
        for field in ["claim_family", "manipulation_type", "direction", "omitted_source_intervals_s", "estimated_removed_time_s"]:
            state = repair["final_state"]["candidate_claims"][0]["fields"][field]
            self.assertEqual(state["support_ids"], ["EA-GAP-002"])
            self.assertEqual(state["decision_status"], "committed")

    def test_harness1_style_target_excludes_observation(self) -> None:
        datums = self.bundle["clean_turn_sft"] + self.bundle["repair_turn_sft"]
        self.assertEqual(len(datums), 11)
        for datum in datums:
            self.assertEqual(datum["messages"][-1]["role"], "assistant")
            self.assertTrue(datum["messages"][-1]["tool_calls"])
            self.assertFalse(datum["metadata"]["target_includes_observation"])

    def test_sft_v2_sidecar_balances_each_episode_without_oracle_fields(self) -> None:
        rows = compile_transition_sidecar(self.bundle)
        self.assertEqual(len(rows), 11)
        for mode in ("clean", "repair"):
            total = sum(
                row["training_weights"]["episode_normalized"]
                for row in rows
                if row["episode_mode"] == mode
            )
            self.assertAlmostEqual(total, 1.0)
        for row in rows:
            self.assertIn(row["chosen_action"]["tool_name"], row["eligible_actions"])
            self.assertEqual(row["visibility"], "verifier_sidecar_not_rendered_to_student")
        serialized = json.dumps(rows)
        self.assertNotIn("oracle", serialized.lower())

    def test_repair_mutations_distinguish_evidence_binding_and_decision_failures(self) -> None:
        harness, _environment = _new_harness(self.bundle)
        for turn in self.bundle["clean_episode"]["turn_history"]:
            harness.execute_tool(turn["tool_name"], copy.deepcopy(turn["params"]), f"replay-{turn['turn_idx']}")
        evidence_id = self.bundle["execution_summary"]["mutation_evidence_id"]

        binding_harness = harness.clone_state()
        binding_result = binding_harness.inject_binding_invalidation(evidence_id)
        self.assertEqual(binding_result["operator"], "binding_invalidation")
        self.assertTrue(binding_harness.evidence_store[evidence_id].active)
        binding_state = binding_harness.render_state()
        self.assertIn(evidence_id, binding_state["available_evidence_ids"])
        self.assertTrue(binding_state["action_constraints"]["must_reopen_before_rebinding"])

        decision_harness = harness.clone_state()
        decision_result = decision_harness.inject_decision_reopen()
        self.assertEqual(decision_result["operator"], "decision_reopen")
        fields = decision_harness.render_state()["candidate_claims"][0]["fields"]
        for field_name in decision_harness.target_fields:
            self.assertEqual(fields[field_name]["evidence_status"], "supported")
            self.assertEqual(fields[field_name]["decision_status"], "open")

    def test_binding_invalidation_replay_remaps_one_equivalent_bound_atom(self) -> None:
        replay_bundle = copy.deepcopy(self.bundle)
        replay_bundle["execution_summary"]["repair_operator"] = "binding_invalidation"
        replay_bundle["execution_summary"]["mutation_evidence_id"] = "EA-STALE-REPLAY-ID"

        repair_harness = _prepare_repair_harness(replay_bundle)
        state = repair_harness.render_state()
        self.assertNotIn("EA-STALE-REPLAY-ID", state["available_evidence_ids"])
        self.assertTrue(state["action_constraints"]["must_reopen_before_rebinding"])
        active_target_atoms = [
            evidence_id
            for evidence_id, atom in repair_harness.evidence_store.items()
            if atom.active and set(atom.supports_fields).issuperset(repair_harness.target_fields)
        ]
        self.assertEqual(len(active_target_atoms), 1)


if __name__ == "__main__":
    unittest.main()
