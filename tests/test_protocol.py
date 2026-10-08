"""v0.2 protocol pieces on synthetic inputs: splits, batching, heads, model
selection. CPU only; no audio, no upstream, no network."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from slsb.features import length_batches
from slsb.tasks._common import WeightedSum, fit_utterance_head, fit_with_dev, learning_rates
from slsb.tasks.asr import BLSTMCTCHead, greedy_decode
from slsb.tasks.speaker_verification import AMSoftmax, XVector
from slsb.utils.datasets import TaskSpec

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_make_splits():
    spec = importlib.util.spec_from_file_location("make_splits", REPO_ROOT / "data_prep" / "make_splits.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_splits = _load_make_splits()


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _spec(task_dir, name, task, kind, label_file):
    return TaskSpec(name=name, task=task, lang="x", kind=kind, task_dir=task_dir,
                    audio_dir=task_dir / "audio", label_path=task_dir / label_file)


def test_length_batches_cover_everything_within_budget():
    lengths = [5, 100, 30, 30, 7, 64, 1, 99]
    batches = length_batches(lengths, max_items=3, max_padded=150)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for b in batches:
        assert len(b) <= 3
        assert len(b) == 1 or max(lengths[i] for i in b) * len(b) <= 150


def test_weighted_sum_contracts_the_layer_axis():
    ws = WeightedSum(4)
    assert torch.allclose(ws.layer_weights().sum(), torch.tensor(1.0))
    x = torch.randn(2, 4, 8)  # (B, L, H)
    assert ws(x, dim=1).shape == (2, 8)
    assert torch.allclose(ws(x, dim=1), x.mean(dim=1), atol=1e-6)  # zero init = uniform mix


def test_fit_with_dev_keeps_the_best_epoch_and_lr():
    # A model whose "dev score" is a parameter we push around per epoch.
    def build():
        return torch.nn.Linear(1, 1, bias=False)

    def train_epoch(model, optimizer, rng):
        with torch.no_grad():
            model.weight += optimizer.param_groups[0]["lr"]

    def dev_score(model):  # peaks at init + 0.75: epoch 3 of lr 0.25; lr 0.1 can only get within 0.05
        return -abs(float(model.weight) - (init + 0.75))

    torch.manual_seed(0)
    init = float(build().weight.detach())
    fit = fit_with_dev(build, train_epoch, dev_score, lrs=[0.1, 0.25], max_epochs=10, patience=2, seed=0,
                       maximize=True)
    assert fit.lr == 0.25 and fit.best_epoch == 3
    assert float(fit.model.weight) == pytest.approx(init + 0.75)


def test_learning_rates_grid_then_reuse():
    cfg = {"lr_grid": [1e-2, 1e-3]}
    tuned = {}
    assert learning_rates(cfg, tuned) == [1e-2, 1e-3]
    tuned["lr"] = 1e-3
    assert learning_rates(cfg, tuned) == [1e-3]


def test_utterance_head_learns_separable_classes():
    torch.manual_seed(0)
    y = torch.arange(3).repeat(40)
    x = torch.randn(len(y), 4, 6) * 0.1
    x[torch.arange(len(y)), :, y] += 2.0  # class c lights up feature c in every layer
    cfg = {"batch_size": 16, "max_epochs": 50, "patience": 5}
    fit = fit_utterance_head(x[:90], y[:90], x[90:], y[90:], 3, cfg, [1e-2], seed=0)
    assert fit.dev_score == 1.0


def test_blstm_ctc_head_shapes_and_greedy_decode():
    head = BLSTMCTCHead(num_layers=3, input_size=8, vocab_size=5, hidden_size=4, rnn_layers=2, dropout=0.0)
    out = head(torch.randn(3, 2, 7, 8), torch.tensor([7, 4]))
    assert out.shape == (2, 7, 5)
    assert torch.allclose(out.exp().sum(-1), torch.ones(2, 7), atol=1e-5)
    vocab = {"<blank>": 0, "<unk>": 1, " ": 2, "a": 3, "b": 4}
    log_probs = torch.full((1, 6, 5), -10.0)
    for t, c in enumerate([3, 3, 0, 3, 4, 4]):  # a a _ a b b -> "aab"
        log_probs[0, t, c] = 0.0
    assert greedy_decode(log_probs, torch.tensor([6]), vocab) == ["aab"]


def test_xvector_and_am_softmax():
    model = XVector(input_size=8, channels=16, stats_channels=32, embedding_size=12)
    emb = model(torch.randn(4, 50, 8))
    assert emb.shape == (4, 12)
    loss_fn = AMSoftmax(12, num_classes=3, margin=0.4, scale=30.0)
    labels = torch.tensor([0, 1, 2, 0])
    with_margin = loss_fn(emb, labels)
    loss_fn.margin = 0.0
    assert with_margin > loss_fn(emb, labels)  # the margin only ever makes the target harder


def test_asr_split_is_speaker_disjoint_and_replaces_a_leaky_v1_split(tmp_path):
    task_dir = tmp_path / "asr_x"
    task_dir.mkdir()
    files = [f"u{i:03d}.wav" for i in range(200)]
    _write_csv(task_dir / "transcripts.csv", ["filename", "transcript"], [(f, "t") for f in files])
    _write_csv(task_dir / "speakers.csv", ["filename", "speaker"], [(f, f"s{i % 40}") for i, f in enumerate(files)])
    # v0.1-style random split: shares speakers, so it must not be reused.
    (task_dir / "split_transcripts.json").write_text(json.dumps({"train": files[:160], "test": files[160:]}))
    split = make_splits.asr_split(_spec(task_dir, "asr_x", "asr", "asr", "transcripts.csv"))
    speaker = {f: f"s{i % 40}" for i, f in enumerate(files)}
    parts = [{speaker[f] for f in split[p]} for p in ("train", "dev", "test")]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    assert sorted(split["train"] + split["dev"] + split["test"]) == files
    assert split["diagnostics"]["test_set"].startswith("new")


def test_er_folds_are_speaker_disjoint(tmp_path):
    task_dir = tmp_path / "er_x"
    task_dir.mkdir()
    rows = [(f"{s:02d}_{n:02d}_{e}.wav", e) for s in range(10) for n in range(3) for e in ("ang", "hap")]
    _write_csv(task_dir / "labels.csv", ["filename", "label"], rows)
    split = make_splits.kfold_split(_spec(task_dir, "er_x", "er", "classification", "labels.csv"))
    speakers = [{f.split("_")[0] for f in fold} for fold in split["folds"]]
    assert len(speakers) == 5
    assert all(not (a & b) for i, a in enumerate(speakers) for b in speakers[i + 1:])
    assert sum(len(f) for f in split["folds"]) == len(rows)


def test_asv_split_holds_out_dev_speakers_with_balanced_trials(tmp_path):
    asv_dir = tmp_path / "asv"
    asv_dir.mkdir()
    rows = [(f"{lang}/{lang[0]}{s}/c{c}.wav", f"{lang[0]}{s}")
            for lang in ("sinhala", "tamil") for s in range(20) for c in range(5)]
    _write_csv(asv_dir / "train_labels.csv", ["filename", "label"], rows)
    split = make_splits.asv_split(asv_dir)
    speaker = dict(rows)
    train_speakers = {speaker[f] for f in split["train"]}
    dev_speakers = {speaker[f] for f in split["dev"]}
    assert dev_speakers and not (train_speakers & dev_speakers)
    labels = [t[0] for t in split["dev_trials"]]
    assert labels.count(1) == labels.count(0)
    for label, a, b in split["dev_trials"]:
        assert a in split["dev"] and b in split["dev"]
        assert (speaker[a] == speaker[b]) == (label == 1)


def test_epochs_override_caps_every_head(monkeypatch):
    from slsb.utils.params import load_params
    monkeypatch.setenv("SLSB_EPOCHS_OVERRIDE", "2")
    params = load_params(REPO_ROOT / "params.yaml")
    assert params["utterance"]["max_epochs"] == 2
    assert params["asr"]["max_epochs"] == 2
    assert params["asv"]["max_epochs"] == 2 and params["asv"]["layer_mix"]["max_epochs"] == 2


def test_greedy_decode_skips_padding():
    vocab = {"<blank>": 0, "<unk>": 1, " ": 2, "a": 3}
    log_probs = torch.full((1, 4, 4), -10.0)
    log_probs[0, :, 3] = 0.0
    assert greedy_decode(log_probs, torch.tensor([1]), vocab) == ["a"]


def test_excluded_task_dirs_are_not_run(tmp_path):
    from slsb.runner import matching_dirs
    for name in ("asr_sinhala", "asr_tamil", "asr_omni_sinhala"):
        (tmp_path / name).mkdir()
    assert matching_dirs(tmp_path, "asr") == ["asr_sinhala", "asr_tamil"]


def test_stats_pooling_head():
    from slsb.tasks.speaker_verification import SpeakerModel
    model = SpeakerModel(8, num_speakers=3, cfg={"head": "stats", "embedding_size": 6, "margin": 0.4, "scale": 30.0})
    emb = model.encoder(torch.randn(4, 30, 8))
    assert emb.shape == (4, 6)
    assert model.loss(emb, torch.tensor([0, 1, 2, 0])).ndim == 0


def _wav(path, seed):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.random.RandomState(seed).randint(-3000, 3000, 1600).astype("int16"), 16000)


def test_shared_audio_is_caught_even_with_a_different_file(tmp_path):
    import soundfile as sf
    _wav(tmp_path / "a.wav", 1)
    _wav(tmp_path / "b.wav", 2)
    # The same samples re-written to another file must still count as the same audio.
    data, _ = sf.read(str(tmp_path / "a.wav"), dtype="int16")
    sf.write(str(tmp_path / "a_copy.wav"), data, 16000, subtype="PCM_16")
    make_splits.assert_no_shared_audio("ok", {"train": ["a.wav"], "test": ["b.wav"]}, tmp_path)
    with pytest.raises(AssertionError):
        make_splits.assert_no_shared_audio("leak", {"train": ["a.wav"], "test": ["a_copy.wav"]}, tmp_path)


def test_closed_set_split_dedupes_and_drops_conflicting_labels(tmp_path):
    task_dir = tmp_path / "sid"
    rows = []
    for spk in range(4):
        for c in range(10):
            _wav(task_dir / "audio" / f"s{spk}_{c}.wav", spk * 100 + c)
            rows.append((f"s{spk}_{c}.wav", f"s{spk}"))
    _wav(task_dir / "audio" / "dup.wav", 0)        # copy of s0_0, same label -> one copy kept
    _wav(task_dir / "audio" / "clash.wav", 101)    # copy of s1_1 under s2 -> both dropped
    rows += [("dup.wav", "s0"), ("clash.wav", "s2")]
    _write_csv(task_dir / "labels.csv", ["filename", "label"], rows)
    split = make_splits.closed_set_split(_spec(task_dir, "sid", "sid", "classification", "labels.csv"))
    kept = split["train"] + split["dev"] + split["test"]
    assert "clash.wav" not in kept and "s1_1.wav" not in kept
    assert ("dup.wav" in kept) != ("s0_0.wav" in kept)
    assert split["diagnostics"]["ambiguous_label_files_dropped"] == 2
    make_splits.assert_no_shared_audio("sid", {k: split[k] for k in ("train", "dev", "test")}, task_dir / "audio")


def test_intent_family_and_task_exclusions():
    from slsb.runner import family_of
    from slsb.tasks import EXCLUDED_TASKS, FAMILY_MODULES
    spec = TaskSpec(name="ic_health_tamil", task="ic_health", lang="tamil", kind="classification",
                    task_dir=Path("."), audio_dir=Path("."), label_path=Path("labels.csv"))
    assert family_of(spec) == "ic" and "ic" in FAMILY_MODULES
    assert {"asr_omni_sinhala", "asv_sinhala"} <= EXCLUDED_TASKS


def test_diarization_helpers():
    from slsb.tasks import speaker_diarization as sd
    turns = [(0.0, 4.0, "a"), (3.0, 6.0, "b"), (8.0, 9.0, "a")]
    assert sd.speech_regions(turns) == [(0.0, 6.0), (8.0, 9.0)]
    # 3-4 s is overlapped, so it is not a single-speaker stretch
    assert sd.single_speaker_spans(turns) == [(0.0, 3.0, "a"), (4.0, 6.0, "b"), (8.0, 9.0, "a")]
    wins = sd.windows(0.0, 4.0, 1.5, 0.75)
    assert wins[0] == (0.0, 1.5) and wins[-1] == (2.5, 4.0)
    assert sd.windows(0.0, 1.0, 1.5, 0.75) == [(0.0, 1.0)]
    ann = sd.annotation([(0, 1, 1), (1, 2, 1), (2, 3, 2)], "r")
    assert len(list(ann.itertracks())) == 2  # touching turns of one label are merged


def test_der_is_zero_for_a_perfect_clustering():
    from slsb.tasks import speaker_diarization as sd
    turns = [(0.0, 3.0, "a"), (3.0, 6.0, "b")]
    rec = sd.Recording("r", "x", turns)
    # Embeddings that separate the two speakers perfectly; window spans tile the timeline.
    spans = [(0.0, 1.5), (1.5, 3.0), (3.0, 4.5), (4.5, 6.0)]
    embs = np.array([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0], [0.01, 1.0]])
    metric = sd.der_components([rec], {"r": (embs, spans)}, threshold=0.5, collar=0.0)
    assert abs(metric) == pytest.approx(0.0)
    merged = sd.der_components([rec], {"r": (embs, spans)}, threshold=1.5, collar=0.0)  # one cluster
    assert abs(merged) == pytest.approx(0.5)


def test_rttm_turns_are_clipped_to_the_audio(tmp_path):
    from slsb.tasks import speaker_diarization as sd
    rttm = tmp_path / "r.rttm"
    rttm.write_text("SPEAKER r 1 0.0 5.0 <NA> <NA> a <NA> <NA>\n"
                    "SPEAKER r 1 8.0 4.0 <NA> <NA> b <NA> <NA>\n"
                    "SPEAKER r 1 11.0 2.0 <NA> <NA> a <NA> <NA>\n")
    assert sd.read_rttm(rttm, duration=10.0) == [(0.0, 5.0, "a"), (8.0, 10.0, "b")]
    assert len(sd.read_rttm(rttm)) == 3


def test_upstream_layer_norm_normalizes_every_layer(tmp_path):
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

    from slsb.upstream.loader import Upstream

    config = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                            conv_dim=(16, 16), conv_stride=(5, 4), conv_kernel=(10, 8), num_conv_pos_embeddings=16)
    torch.manual_seed(0)
    Wav2Vec2Model(config).save_pretrained(tmp_path)
    Wav2Vec2FeatureExtractor(return_attention_mask=True).save_pretrained(tmp_path)
    waveforms = [np.random.RandomState(0).randn(4000).astype(np.float32)]

    raw, _ = Upstream(str(tmp_path), torch.device("cpu")).extract(waveforms)
    normed, _ = Upstream(str(tmp_path), torch.device("cpu"), layer_norm=True).extract(waveforms)
    assert raw.shape == normed.shape == (3, 1, raw.shape[2], 32)
    assert torch.allclose(normed.mean(dim=-1), torch.zeros(1), atol=1e-4)
    assert torch.allclose(normed.std(dim=-1, unbiased=False), torch.ones(1), atol=1e-2)


def _load_log_results():
    spec = importlib.util.spec_from_file_location("log_results", REPO_ROOT / "scripts" / "log_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_log_results_summarizes_seeds_and_lets_later_folders_win(tmp_path):
    log_results = _load_log_results()

    def write(folder, entries, **extra):
        folder.mkdir()
        (folder / "results_x.json").write_text(json.dumps({"upstream": "m", **extra, "results": entries}))

    def entry(task, seed, value, status="ok"):
        return {"name": task, "seed": seed, "status": status, "metrics": {"wer": value}}

    write(tmp_path / "a", [entry("asr_x", 0, 0.5), entry("asr_x", 1, 0.7), entry("er_x", 0, 0.9, "error")],
          protocol="v0.3")
    write(tmp_path / "b", [entry("er_x", 0, 0.4), entry("er_x", 1, 0.6)])
    summary, by_task = log_results.load_folders([tmp_path / "a", tmp_path / "b"])
    metrics, rows, problems = log_results.summarize(by_task)
    assert summary["protocol"] == "v0.3"
    assert metrics["asr_x/wer"] == pytest.approx(0.6) and metrics["asr_x/wer_std"] == pytest.approx(0.1414, 1e-3)
    assert metrics["er_x/wer"] == pytest.approx(0.5)  # the failed seed in folder a was replaced by folder b
    assert problems == [] and len(rows) == 4


def test_sid_split_is_video_disjoint_with_every_speaker_in_every_part(tmp_path):
    task_dir = tmp_path / "sid"
    rows = []
    for s in range(3):
        for v in range(5):            # 5 videos per speaker, 4 clips each
            for c in range(4):
                name = f"id{s:05d}/interview/interview-{v:02d}-{c:03d}.wav"
                _wav(task_dir / "audio" / name, s * 1000 + v * 10 + c)
                rows.append((name, f"id{s:05d}"))
    _write_csv(task_dir / "labels.csv", ["filename", "label"], rows)
    split = make_splits.video_disjoint_split(_spec(task_dir, "sid", "sid", "classification", "labels.csv"))
    videos = {p: {make_splits.slceleb_video(f) for f in split[p]} for p in ("train", "dev", "test")}
    assert not (videos["train"] & videos["test"]) and not (videos["train"] & videos["dev"]) \
        and not (videos["dev"] & videos["test"])
    for p in ("train", "dev", "test"):
        assert {f.split("/")[0] for f in split[p]} == {"id00000", "id00001", "id00002"}
    assert sum(len(split[p]) for p in ("train", "dev", "test")) == len(rows)
    assert make_splits.slceleb_video("id10001/interview/interview-03-012.wav") == "id10001/interview-03"


def test_carry_over_never_copies_a_task_the_protocol_changed():
    log_results = _load_log_results()
    previous, changed = log_results.CARRY_OVER["v0.4"]
    assert previous == "v0.3" and changed == {"sid"}
