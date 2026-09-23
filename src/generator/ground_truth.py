import json
import os


def write(w, root):
    path = root / "ground_truth"
    path.mkdir(exist_ok=True)
    os.chmod(path, 0o700)
    episodes = []
    for e in w.ep.items:
        episodes.append(
            {
                **e,
                "exposure_counts": dict(w.ep.counts[e["episode_id"]]),
                "exposure_definition": "Rows matching an intervention scope at each application site; repeated exposure is possible. Placebo counts are not causal effects.",
            }
        )
    for filename, data in [
        ("benchmark_events.json", w.ep.benchmarks),
        ("latent_episodes.json", episodes),
        ("injection_counts.json", {k: dict(v) for k, v in w.ep.counts.items()}),
    ]:
        target = path / filename
        target.write_text(json.dumps(data, indent=2))
        os.chmod(target, 0o600)
