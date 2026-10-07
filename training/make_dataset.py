"""Generate the SYNTHETIC educational dataset for the wellbeing demo.

=============================================================
THIS DATASET IS 100% SYNTHETIC. IT IS NOT REAL STUDENT DATA.
=============================================================

Every number and every journal sentence below is produced by the random
generator in this file. No student was surveyed, no real journal was read,
and no real-world relationship is encoded here.

It is generated so that the usual ML workflow can be demonstrated
end-to-end (split -> fit -> evaluate -> save -> serve) on data the
reviewer can regenerate and inspect. It is deliberately built with
heavy overlap and label noise so that held-out accuracy lands in a
*mediocre* range. A model that scored 99% here would only be proving it
can memorise the generator, so the generator resists that.

Run with::

    python -m training.make_dataset
"""

from __future__ import annotations

import csv
import pathlib

import numpy as np

SEED = 20240917
N_ROWS = 1200
LABEL_NOISE_RATE = 0.08

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "synthetic_wellbeing.csv"

FEATURE_COLUMNS = (
    "marks",
    "attendance",
    "sleep_hours",
    "screen_time",
    "assignment_delay",
)

# --------------------------------------------------------------------------
# Journal phrase banks. Each band contributes vocabulary; sentences are
# assembled from 1-3 of them, so the vocabulary is far richer than a
# per-row single sentence and the TF-IDF model has real work to do.
# --------------------------------------------------------------------------
PHRASES = {
    "steady": [
        "i feel happy and excited about learning",
        "great motivation and energy this week",
        "loving the course, feeling great about it",
        "good balance between classes and free time",
        "enjoying the lectures and keeping on top of things",
        "my revision plan is working out well so far",
        "i've settled into a comfortable routine",
        "studying feels manageable and rewarding",
        "i slept well and focused fine in the library",
        "friendly group project, we finished early",
        "i'm managing my time much better now",
        "lectures are interesting and i take notes properly",
        "confident about the material so far",
        "i go for a run twice a week and it helps",
        "steady progress feels good to see",
    ],
    "strained": [
        "i feel okay but a bit tired most days",
        "confused but trying to cope with the workload",
        "managing fine although the assignments pile up",
        "busy week, deadlines are tight but i am coping",
        "a bit overwhelmed with the number of submissions",
        "some topics are tricky but i keep going",
        "sleep is okay, i just feel stretched thin",
        "lots to juggle but nothing feels unmanageable",
        "i skipped a few lectures to catch up on notes",
        "grades are fine, i am just stretched for time",
        "i keep thinking about the reading list",
        "quiet week, i am taking it one day at a time",
        "work is steady, i would just like more free hours",
        "i am tired but i am keeping up with things",
        "not the best week, still getting through it",
    ],
    "distressed": [
        "i am very stressed and cannot sleep",
        "depressed and feeling very lonely lately",
        "anxious about exams and worried constantly",
        "too much pressure, i want to quit",
        "feeling overwhelmed and isolated from friends",
        "i have been crying and cannot concentrate",
        "panic before every deadline, i am scared",
        "exhausted all the time and sleeping badly",
        "i feel worthless and hopeless about the year",
        "nobody understands how hard this has been",
        "i cannot stop worrying about failing",
        "feeling empty and unmotivated every morning",
        "i have been skipping meals and studying late",
        "the pressure is breaking me down honestly",
        "i feel trapped and completely burnt out",
    ],
}


def _build_numeric_features(rng: np.random.Generator):
    """Draw numeric features with a shared latent strain plus heavy noise.

    A single latent variable drives all five columns (so they are genuinely
    correlated, as the real ones plausibly are), but every column also gets
    independent noise. That correlation is much weaker than the original
    10-row dataset, whose features had pairwise rank correlation >= 0.99.
    """
    latent = rng.normal(0.0, 1.0, N_ROWS)

    marks = np.clip(78 - 13.0 * latent + rng.normal(0, 11.0, N_ROWS), 25, 100)
    attendance = np.clip(88 - 17.0 * latent + rng.normal(0, 10.0, N_ROWS), 30, 100)
    sleep_hours = np.clip(7.0 - 1.5 * latent + rng.normal(0, 1.2, N_ROWS), 2.0, 12.0)
    screen_time = np.clip(4.5 + 1.4 * latent + rng.normal(0, 1.8, N_ROWS), 0.0, 16.0)
    assignment_delay = np.clip(
        np.round(1.6 + 1.4 * latent + rng.normal(0, 2.0, N_ROWS)), 0, 14
    )

    numeric = {
        "marks": np.round(marks, 1),
        "attendance": np.round(attendance, 1),
        "sleep_hours": np.round(sleep_hours, 1),
        "screen_time": np.round(screen_time, 1),
        "assignment_delay": assignment_delay,
    }
    return numeric, latent


def _build_feedback(rng: np.random.Generator, text_latent: np.ndarray) -> list[str]:
    """Assemble a journal sentence from a phrase bank.

    ``text_latent`` is only loosely related to the numeric latent, and a
    quarter of sentences borrow from an adjacent band. Both choices stop the
    text branch from becoming a clean proxy for the label.
    """
    bands = ("steady", "strained", "distressed")
    thresholds = (-0.55, 0.55)

    sentences: list[str] = []
    for i in range(N_ROWS):
        z = text_latent[i]
        band_idx = 0 if z < thresholds[0] else (1 if z < thresholds[1] else 2)
        if rng.random() < 0.25:  # deliberate overlap between adjacent bands
            band_idx = int(np.clip(band_idx + rng.choice([-1, 1]), 0, 2))

        pool = PHRASES[bands[band_idx]]
        n_phrases = int(rng.integers(1, 4))
        chosen = rng.choice(len(pool), size=n_phrases, replace=False)
        parts = [pool[j].strip() for j in chosen]
        sentences.append(". ".join(parts) + ".")

    return sentences


def _assign_labels(rng: np.random.Generator, latent, text_latent) -> np.ndarray:
    """Blend numeric and text strain into a label, then corrupt it slightly.

    The blend is what makes the two branches partially complementary. The
    final 8% random relabelling guarantees a non-trivial error floor, so any
    reported accuracy is bounded well below 100%.
    """
    def zscore(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() or 1.0)

    blend = 0.55 * zscore(latent) + 0.45 * zscore(text_latent)
    edges = np.quantile(blend, [1 / 3, 2 / 3])

    labels = np.where(blend < edges[0], 0, np.where(blend < edges[1], 1, 2))
    flipped = rng.random(N_ROWS) < LABEL_NOISE_RATE
    labels = np.where(flipped, rng.integers(0, 3, N_ROWS), labels)
    return labels.astype(int)


def build_rows() -> list[dict]:
    rng = np.random.default_rng(SEED)
    numeric, latent = _build_numeric_features(rng)

    # Text strain is correlated with, but far from identical to, numeric strain.
    text_latent = latent + rng.normal(0.0, 1.35, N_ROWS)

    feedback = _build_feedback(rng, text_latent)
    labels = _assign_labels(rng, latent, text_latent)

    rows = []
    for i in range(N_ROWS):
        row = {c: float(numeric[c][i]) for c in FEATURE_COLUMNS}
        row["feedback"] = feedback[i]
        row["risk_label"] = int(labels[i])
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: pathlib.Path = CSV_PATH) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*FEATURE_COLUMNS, "feedback", "risk_label"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    rows = build_rows()
    path = write_csv(rows)
    counts = {lvl: sum(r["risk_label"] == lvl for r in rows) for lvl in (0, 1, 2)}
    print(f"Wrote {len(rows)} SYNTHETIC rows -> {path}")
    print(f"Label distribution: {counts}")
    print(
        "Reminder: synthetic data. These labels come from a random "
        "generator, not from any observation of a real student."
    )


if __name__ == "__main__":
    main()