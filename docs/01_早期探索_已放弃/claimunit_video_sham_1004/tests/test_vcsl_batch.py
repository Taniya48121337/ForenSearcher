import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vcsl_batch import (
    LOCALIZER_SPECS,
    VCSLFeatureEnvironment,
    VCSL_SCHEMA,
    vcsl_tool_registry_for_variant,
)
from vcsl_consolidate import DEFAULT_RUNS as CONSOLIDATION_RUNS, consolidate
from vcsl_probe import DEFAULT_FEATURE_ROOT, hv_localize, segment_precision_recall, select_cases


class VCSLBatchTest(unittest.TestCase):
    def test_portfolio_registry_exposes_all_executable_localizers(self):
        registry = vcsl_tool_registry_for_variant("portfolio")
        names = {item["name"] for item in registry}
        localizer_names = {spec["tool_name"] for spec in LOCALIZER_SPECS.values()}
        self.assertEqual(names.intersection(localizer_names), localizer_names)

    def test_fixed_route_exposes_only_its_executable_localizer(self):
        registry = vcsl_tool_registry_for_variant("tn_stability")
        names = {item["name"] for item in registry}
        localizer_names = {spec["tool_name"] for spec in LOCALIZER_SPECS.values()}

        self.assertEqual(
            names.intersection(localizer_names),
            {LOCALIZER_SPECS["tn_stability"]["tool_name"]},
        )
        self.assertIn("inspect_pair_features", names)
        self.assertIn("assess_evidence_candidate", names)
        self.assertIn("adjudicate_evidence", names)

    def test_portfolio_inspection_requires_policy_choice_without_recommendation(self):
        case = select_cases(20260818, 1)[0]
        environment = VCSLFeatureEnvironment(
            case=case,
            feature_root=DEFAULT_FEATURE_ROOT,
            routing_variant="portfolio",
        )
        inspection = environment.execute(
            "inspect_pair_features",
            {"query_ref": "query", "reference_ref": "reference"},
        )
        self.assertNotIn("recommended_localizer_tool", inspection["result"])
        diagnostics = inspection["result"]["localizer_routing_diagnostics"]
        self.assertNotIn("recommended_variant", diagnostics)
        self.assertNotIn("reason_code", diagnostics)
        self.assertTrue(diagnostics["policy_choice_required"])
        self.assertEqual(
            set(environment.eligible_tool_names()),
            {spec["tool_name"] for spec in LOCALIZER_SPECS.values()},
        )

    def test_empty_localizer_result_is_executed_negative_evidence(self):
        case = select_cases(20260818, 1)[0]
        environment = VCSLFeatureEnvironment(
            case=case,
            feature_root=DEFAULT_FEATURE_ROOT,
            routing_variant="portfolio",
        )
        environment.execute(
            "inspect_pair_features",
            {"query_ref": "query", "reference_ref": "reference"},
        )
        spec = LOCALIZER_SPECS["hv"]
        with patch("vcsl_batch.hv_localize", return_value=[]):
            result = environment.execute(
                spec["tool_name"],
                {
                    "query_ref": "query",
                    "reference_ref": "reference",
                    "method": spec["method"],
                },
            )
        self.assertFalse(result["goal_satisfied"])
        self.assertFalse(result["result"]["copied_segment_detected"])
        self.assertEqual(result["evidence_atoms"][0].supports_fields, [])
        self.assertNotIn(spec["tool_name"], environment.eligible_tool_names())

    def test_consolidated_export_deduplicates_repairs_and_keeps_only_gold_sft(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            report = consolidate(CONSOLIDATION_RUNS, output_root)

            self.assertEqual(report["summary"]["cases"], 22)
            self.assertEqual(report["summary"]["passed"], 21)
            self.assertEqual(report["summary"]["rubric_rejected"], 1)
            self.assertEqual(report["tool_telemetry"]["selected_but_not_indispensable"], {})
            self.assertEqual(
                sum(bool(line.strip()) for line in (output_root / "sft_clean_accepted.jsonl").read_text(encoding="utf-8").splitlines()),
                126,
            )

    def test_seeded_sample_is_reproducible(self):
        left = [item["pair_id"] for item in select_cases(20260818, 10)]
        right = [item["pair_id"] for item in select_cases(20260818, 10)]
        self.assertEqual(left, right)

    def test_official_features_localize_all_ten_random_positive_pairs(self):
        for case in select_cases(20260818, 10):
            environment = VCSLFeatureEnvironment(case=case, feature_root=DEFAULT_FEATURE_ROOT)
            similarity = environment.query @ environment.reference.T
            boxes = hv_localize(similarity, min_similarity=0.7, max_iou=0.9)
            metrics = segment_precision_recall(boxes, case["oracle_boxes"])
            self.assertGreaterEqual(metrics["recall"], 0.5)

    def test_environment_exposes_no_oracle_boxes(self):
        case = select_cases(20260818, 1)[0]
        environment = VCSLFeatureEnvironment(case=case, feature_root=DEFAULT_FEATURE_ROOT)
        environment.execute(
            "inspect_pair_features",
            {"query_ref": "query", "reference_ref": "reference"},
        )
        observation = environment.execute(
            "localize_copy_segment",
            {"query_ref": "query", "reference_ref": "reference", "method": "HV"},
        )
        self.assertNotIn("oracle_boxes", str(observation))
        self.assertEqual(
            observation["evidence_atoms"][0].supports_fields,
            VCSL_SCHEMA["target_fields"],
        )

    def test_official_spd_complementary_route_is_executable_without_oracle(self):
        pair_id = (
            "086c9b86da874985b45fe2db18ed8f9a-"
            "bf3e20cfd56d42a19eef0f872ebd0a51"
        )
        query_id, reference_id = pair_id.split("-", 1)
        environment = VCSLFeatureEnvironment(
            case={
                "pair_id": pair_id,
                "query_id": query_id,
                "reference_id": reference_id,
            },
            feature_root=DEFAULT_FEATURE_ROOT,
            routing_variant="spd_official",
        )
        environment.execute(
            "inspect_pair_features",
            {"query_ref": "query", "reference_ref": "reference"},
        )
        spec = LOCALIZER_SPECS["spd_official"]
        observation = environment.execute(
            spec["tool_name"],
            {
                "query_ref": "query",
                "reference_ref": "reference",
                "method": spec["method"],
            },
        )
        self.assertEqual(
            observation["result"]["copied_segments_frames"],
            [[41, 1, 68, 29]],
        )
        self.assertEqual(observation["result"]["localization_method"], "SPD-OFFICIAL-ISC")
        atom = observation["evidence_atoms"][0]
        self.assertNotIn("oracle", str((observation["result"], atom.provenance)).lower())

    def test_label_blind_localizer_portfolio_improves_seeded_canary(self):
        gold_cases = 0
        for case in select_cases(20260818, 10):
            case_gold = False
            for variant in ("dp", "tn_stability", "axis_saliency"):
                environment = VCSLFeatureEnvironment(
                    case=case,
                    feature_root=DEFAULT_FEATURE_ROOT,
                    routing_variant=variant,
                )
                environment.execute(
                    "inspect_pair_features",
                    {"query_ref": "query", "reference_ref": "reference"},
                )
                spec = LOCALIZER_SPECS[variant]
                try:
                    observation = environment.execute(
                        spec["tool_name"],
                        {
                            "query_ref": "query",
                            "reference_ref": "reference",
                            "method": spec["method"],
                        },
                    )
                except ValueError:
                    continue
                self.assertNotIn("oracle_boxes", str(observation))
                if observation.get("goal_satisfied") is False:
                    continue
                boxes = observation["result"]["copied_segments_frames"]
                metrics = segment_precision_recall(boxes, case["oracle_boxes"])
                minimum = min(metrics.values())
                if (
                    metrics["precision"] >= 0.5
                    and metrics["recall"] >= 0.5
                    and metrics["f1"] >= 0.6
                    and minimum >= 0.85
                ):
                    case_gold = True
                    break
            gold_cases += int(case_gold)
        self.assertEqual(gold_cases, 9)

    def test_state_externalized_auto_router_is_replayable_and_gold_on_canary(self):
        resolved = []
        reason_codes = []
        strict_gold = []
        for case in select_cases(20260818, 10):
            environment = VCSLFeatureEnvironment(
                case=case,
                feature_root=DEFAULT_FEATURE_ROOT,
                routing_variant="auto",
            )
            inspection = environment.execute(
                "inspect_pair_features",
                {"query_ref": "query", "reference_ref": "reference"},
            )
            self.assertFalse(
                inspection["result"]["localizer_routing_diagnostics"]["label_visible_to_router"]
            )
            self.assertNotIn("oracle", str(inspection).lower())
            reason_codes.append(
                inspection["result"]["localizer_routing_diagnostics"]["reason_code"]
            )
            spec = environment.localizer_spec
            measurement = environment.execute(
                spec["tool_name"],
                {
                    "query_ref": "query",
                    "reference_ref": "reference",
                    "method": spec["method"],
                },
            )
            metrics = segment_precision_recall(
                measurement["result"]["copied_segments_frames"],
                case["oracle_boxes"],
            )
            strict_gold.append(min(metrics.values()) >= 0.85)
            resolved.append(environment.routing_variant)
        self.assertEqual(sum(strict_gold), 9)
        self.assertFalse(strict_gold[8])
        self.assertEqual(
            resolved,
            [
                "dp",
                "tn_stability",
                "dp",
                "dp",
                "tn_stability",
                "tn_stability",
                "tn_stability",
                "tn_stability",
                "tn_stability",
                "axis_saliency",
            ],
        )
        self.assertEqual(reason_codes[4], "two_path_endpoint_uncertainty")
        self.assertEqual(reason_codes[7], "full_matrix_dp_overcoverage")
        self.assertEqual(reason_codes[9], "path_fragmentation_indicates_speed_change")


if __name__ == "__main__":
    unittest.main()
