"""以实际编码的视频验证连续片段、时间偏移和有限变换。"""

import cv2
import io
import json
import wave
import numpy as np
import pytest

from foren_searcher.inspect.factory import create_inspect_service
from foren_searcher.store import GlobalStore


def test_temporal_score_ranks_supported_paths_in_the_order_used_to_select_them():
    from foren_searcher.inspect.temporal import align_sequences
    similarity = np.zeros((24, 24))
    similarity[np.arange(16), np.arange(16)] = .95
    similarity[np.arange(20, 24), np.arange(20, 24)] = 1
    matches = align_sequences(similarity, top_k=2)
    assert [len(match.pairs) for match in matches] == [16, 4]
    assert matches[0].score > matches[1].score
    assert all(0 < match.score <= 1 for match in matches)


def test_visual_alignment_rejects_similar_scene_when_exact_correspondence_exists():
    from foren_searcher.inspect.temporal import align_sequences
    similarity = np.zeros((60, 20))
    similarity[np.arange(20), np.arange(20)] = .98
    similarity[np.arange(30, 50), np.arange(20)] = 1
    matches = align_sequences(similarity, top_k=5, relative_similarity=True)
    assert len(matches) == 1
    assert matches[0].pairs == tuple(zip(range(30, 50), range(20)))


def test_visual_alignment_keeps_continuity_across_a_degraded_frame():
    from foren_searcher.inspect.temporal import align_sequences, path_score
    similarity = np.zeros((25, 25))
    similarity[np.arange(20), np.arange(20)] = .99
    similarity[10, 10] = .97
    similarity[10, 24] = .99
    matches = align_sequences(similarity, top_k=2, relative_similarity=True)
    assert len(matches) == 1
    assert matches[0].pairs == tuple(zip(range(20), range(20)))
    assert matches[0].score == path_score(similarity, matches[0].pairs, relative_similarity=True)


def test_structural_comparison_removes_photometric_distractor_without_shortening_main_path():
    from foren_searcher.inspect.temporal import TemporalMatch, prune_dominated_paths

    structural = np.zeros((40, 20))
    main = TemporalMatch(tuple(zip(range(20), range(20))), .4)
    distractor = TemporalMatch(tuple(zip(range(20, 28), range(12, 20))), .02)
    structural[np.arange(20), np.arange(20)] = .99995
    structural[np.arange(20, 28), np.arange(12, 20)] = .999
    # 主路径内局部退化不改变其原有时间范围；只有整条次级路径均劣于已保留对应时才丢弃。
    structural[10, 10] = .9
    assert prune_dominated_paths((main, distractor), structural) == (main,)


@pytest.mark.parametrize("variant", ["repeated", "disjoint", "partly_new"])
def test_structural_comparison_keeps_repeated_or_new_correspondence(variant):
    from foren_searcher.inspect.temporal import TemporalMatch, prune_dominated_paths

    structural = np.zeros((30, 20))
    main = TemporalMatch(tuple(zip(range(10), range(10))), .4)
    structural[np.arange(10), np.arange(10)] = .99995
    columns = range(10) if variant == "repeated" else range(10, 20) if variant == "disjoint" else range(5, 15)
    other = TemporalMatch(tuple(zip(range(20, 30), columns)), .2)
    for i, j in other.pairs:
        structural[i, j] = .99995 if variant == "repeated" else .995
    assert prune_dominated_paths((main, other), structural) == (main, other)


def write_video(path, frames, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    assert writer.isOpened()
    for frame in frames:
        writer.write(cv2.resize(frame, (160, 120)))
    writer.release()


def scenes(seed):
    random = np.random.default_rng(seed)
    output = []
    for _ in range(8):
        scene = cv2.resize(random.integers(0, 256, (12, 16, 3), dtype=np.uint8), (160, 120))
        for _ in range(10):
            output.append(scene.copy())
    return output


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_short_clip_in_long_video_uses_a_common_time_step_with_bounded_work(tmp_path, reverse):
    rng = np.random.default_rng(9351)
    frames = []
    for _ in range(120):
        scene = cv2.resize(rng.integers(0, 256, (12, 16, 3), dtype=np.uint8), (160, 120))
        frames.extend([scene] * 10)
    files = [tmp_path / "source.avi", tmp_path / "clip.avi"]
    write_video(files[0], frames)
    write_video(files[1], frames[714:750])
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    evidence = [store.commit_ingested_evidence(source_ref=f"I{i}", media_type="video/x-msvideo",
        modalities=("video",), label=p.name, original_filename=p.name, path=p) for i, p in enumerate(files)]
    if reverse: evidence.reverse()
    long_side, short_side = ("b", "a") if reverse else ("a", "b")
    async with create_inspect_service(store=store) as service:
        for narrow in (False, True):
            params = {"mode": "visual"}
            if narrow: params[f"range_{long_side}_s"] = [71, 76]
            response = await service.execute({"targets": [e.evidence_id for e in evidence],
                "request": {"operation": "align_media", "params": params}}, workspace_id="W1-1")
            assert response.status == "ready", response.to_dict()
            assert any(f.code == "COUNT_REDUCED" and f.severity == "warning" for f in response.feedback) is not narrow
            saved = json.loads(store.evidence(response.result_evidence_id).content)
            assert saved["sampling_interval_s"][0] == saved["sampling_interval_s"][1]
            counts = [saved["coverage"][s]["processed"]["value"] for s in ("a", "b")]
            assert sum(counts) <= 512 and counts[0] * counts[1] <= 256 * 256
            best = saved["matches"][0]
            assert best[f"range_{long_side}_s"] == pytest.approx([71.4, 75], abs=.5)
            assert best[f"range_{short_side}_s"] == pytest.approx([0, 3.6], abs=.3)
            for anchor in best["anchors"]:
                offset = anchor[f"time_{long_side}_s"] - anchor[f"time_{short_side}_s"]
                assert offset == pytest.approx(71.4, abs=.3)


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["same", "trim", "prefix", "speed", "crop", "unrelated", "reordered", "repeated", "short_splice", "resampled", "partial_trim"])
async def test_continuous_visual_alignment(tmp_path, variant):
    frames = scenes(72)
    other, fps = frames, 10
    expected_a, expected_b = (0, 8), (0, 8)
    if variant == "trim":
        other = frames[20:60]
        expected_a, expected_b = (2, 6), (0, 4)
    elif variant == "prefix":
        other = scenes(31)[:10] + frames
        expected_b = (1, 9)
    elif variant == "speed":
        fps = 15
        expected_b = (0, 80 / 15)
    elif variant == "crop":
        other = [frame[9:111, 12:148] for frame in frames]
    elif variant == "unrelated":
        other = scenes(119)
    elif variant == "reordered":
        other = [frame for i in reversed(range(8)) for frame in frames[i * 10:(i + 1) * 10]]
    elif variant == "repeated":
        other = frames + scenes(31)[:10] + frames
    elif variant == "short_splice":
        other = frames[4:18] + frames[62:70]
    elif variant == "resampled":
        other, fps = [frame for frame in frames for _ in range(3)], 30
    elif variant == "partial_trim":
        other = frames[4:76]
        expected_a, expected_b = (.4, 7.6), (0, 7.2)
    first, second = tmp_path / "a.avi", tmp_path / "b.avi"
    write_video(first, frames)
    write_video(second, other, fps)
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    evidence = [store.commit_ingested_evidence(source_ref=f"I{i}", media_type="video/x-msvideo",
        modalities=("video",), label=path.name, original_filename=path.name, path=path) for i, path in enumerate((first, second))]
    async with create_inspect_service(store=store) as service:
        response = await service.execute({"targets": [e.evidence_id for e in evidence],
            "request": {"operation": "align_media", "params": {"mode": "visual"}}}, workspace_id="W1-1")
        if variant == "short_splice":
            from foren_searcher.contracts import WorkspaceTask
            from foren_searcher.workspace import WorkspaceHarness
            harness = WorkspaceHarness(task=WorkspaceTask("W1-2", "Compare", tuple(e.evidence_id for e in evidence)),
                store=store, inspect_service=service)
            reused = await harness.execute_inspect({"targets": [e.evidence_id for e in evidence],
                "request": {"operation": "align_media", "params": {"mode": "visual"}}})
            assert reused.result_evidence_id == response.result_evidence_id
            context = json.loads(harness.memory.latest_observation["items"][1])
            assert context["boundary_uncertainty_s"] is None
            assert context["sampling_interval_s"] == [.1, .1]
            assert context["limitations"]
    assert response.status == "ready"
    saved = json.loads(store.evidence(response.result_evidence_id).content)
    assert saved["boundary_uncertainty_s"] is None
    assert saved["sampling_interval_s"] == pytest.approx([.1, .1], abs=.005)
    assert saved["limitations"]
    matches = response.result.matches
    assert [m.score for m in matches] == sorted((m.score for m in matches), reverse=True)
    assert all(2 * len(match.anchors) <= 8 for match in matches)
    assert sum("image" in e.modalities for e in response.evidence) <= service.policy.maximum_visual_qa_images
    if variant == "unrelated":
        assert not matches
    elif variant == "reordered":
        assert all(match.range_a_s[1] - match.range_a_s[0] < 2 for match in matches)
    elif variant == "repeated":
        full_matches = sorted((m for m in matches if m.range_a_s[1] - m.range_a_s[0] > 7.5),
                              key=lambda m: m.range_b_s[0])
        assert len(full_matches) == 2
        assert full_matches[0].range_b_s == pytest.approx((0, 8), abs=0.3)
        assert full_matches[1].range_b_s == pytest.approx((9, 17), abs=0.3)
    elif variant == "short_splice":
        assert any(m.range_a_s[1] < 2.1 and m.range_b_s[0] < 0.3 for m in matches)
        short = [m for m in matches if m.range_a_s[0] >= 5.7 and m.range_b_s[0] >= 1.3]
        assert short, response.to_dict()
        assert short[0].range_b_s == pytest.approx((1.4, 2.2), abs=0.2)
        assert len(short[0].anchors) >= 4
    else:
        assert matches
        best = matches[0]
        if variant in {"same", "trim", "speed", "crop", "prefix"}:
            mapping = saved["matches"][0]["time_mapping"]
            assert mapping["scale"] == pytest.approx(2/3 if variant == "speed" else 1, abs=.05)
            expected_offset = -2 if variant == "trim" else 1 if variant == "prefix" else 0
            if variant != "speed":
                assert mapping["offset_s"] == pytest.approx(expected_offset, abs=.2)
                assert mapping["max_anchor_residual_s"] < .15
        assert best.range_a_s == pytest.approx(expected_a, abs=0.3)
        assert best.range_b_s == pytest.approx(expected_b, abs=0.3)
        assert len(best.anchors) >= 4
        assert all(a.time_a_s < b.time_a_s and a.time_b_s < b.time_b_s for a, b in zip(best.anchors, best.anchors[1:]))
        if variant == "partial_trim":
            assert all(anchor.time_a_s - anchor.time_b_s == pytest.approx(.4, abs=.11) for anchor in best.anchors)


def tone_sequence(seed, speed=1):
    rng = np.random.default_rng(seed)
    blocks = []
    for _ in range(10):
        time = np.arange(round(0.7 / speed * 8000)) / 8000
        frequencies = rng.uniform(180, 2400, 4)
        block = sum(np.sin(2 * np.pi * frequency * time + rng.uniform(0, 6)) for frequency in frequencies) / 8
        blocks.append(block * np.minimum(1, np.minimum(time * 40, (time[-1] - time) * 40)))
    return np.concatenate(blocks)


def wav_bytes(samples):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes((np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())
    return output.getvalue()


@pytest.mark.parametrize("variant", ["noise", "unrelated", "tempo", "trim_tempo"])
@pytest.mark.asyncio
async def test_audio_alignment_noise_and_tempo(tmp_path, variant):
    a = tone_sequence(71)
    if variant == "noise":
        b = np.concatenate([np.zeros(2400), a + np.random.default_rng(3).normal(0, 0.03, a.shape)])
    elif variant == "unrelated":
        b = np.random.default_rng(14).normal(0, 0.2, a.shape)
    else:
        b = tone_sequence(71, 1.25)
        if variant == "trim_tempo":
            b = b[round(1.4 / 1.25 * 8000):round(5.6 / 1.25 * 8000)]
    store = GlobalStore(tmp_path / "store.sqlite3", tmp_path / "artifacts")
    values = [store.commit_ingested_evidence(source_ref=f"I{i}", media_type="audio/wav",
        modalities=("audio",), label=f"audio {i}", original_filename=f"{i}.wav", data=wav_bytes(samples))
        for i, samples in enumerate((a, b))]
    async with create_inspect_service(store=store) as service:
        response = await service.execute({"targets":[item.evidence_id for item in values],
            "request":{"operation":"align_media", "params":{"mode":"audio"}}}, workspace_id="W1-1")
    assert response.status == "ready", response.to_dict()
    if variant == "unrelated":
        assert not response.result.matches
    else:
        assert response.result.matches
        match = response.result.matches[0]
        expected_a = (1.4, 5.6) if variant == "trim_tempo" else (0, 7)
        expected_b = (0.3, 7.3) if variant == "noise" else (0, 4.2 / 1.25 if variant == "trim_tempo" else 7 / 1.25)
        assert match.range_a_s == pytest.approx(expected_a, abs=0.25)
        assert match.range_b_s == pytest.approx(expected_b, abs=0.25)
        if "tempo" in variant:
            assert len(match.anchors) >= 4
