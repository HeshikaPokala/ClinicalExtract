"""
Builds labeled (note_text -> diagnoses/medications) pairs from Synthea output.

Ground truth definition: for a given encounter note dated D, a diagnosis/medication
counts as a label if it was active on D per conditions.csv / medications.csv
(START <= D and (STOP is empty or STOP >= D)). Synthea's clinical notes narrate a
patient's full problem history in the "History of Present Illness" section on every
visit (not just new findings), so "active as of this visit" is the correct match to
what the note text actually describes.

Usage:
    python src/data_prep.py
"""

import json
import random
import re
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "dataset" / "notes"
CSV_DIR = ROOT / "dataset" / "csv"
OUT_DIR = ROOT / "dataset" / "processed"

DATE_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$", re.MULTILINE)
FILENAME_UUID_RE = re.compile(r"_([0-9a-f-]{36})\.txt$")

EVAL_FRACTION = 0.2
RANDOM_SEED = 42


def load_ground_truth():
    conditions = pd.read_csv(CSV_DIR / "conditions.csv", usecols=["START", "STOP", "PATIENT", "DESCRIPTION"])
    conditions["START"] = pd.to_datetime(conditions["START"]).dt.date
    conditions["STOP"] = pd.to_datetime(conditions["STOP"]).dt.date

    medications = pd.read_csv(CSV_DIR / "medications.csv", usecols=["START", "STOP", "PATIENT", "DESCRIPTION"])
    medications["START"] = pd.to_datetime(medications["START"]).dt.date
    medications["STOP"] = pd.to_datetime(medications["STOP"]).dt.date

    return conditions.groupby("PATIENT"), medications.groupby("PATIENT")


def active_descriptions(group_df, patient_id, as_of: date):
    if patient_id not in group_df.groups:
        return []
    rows = group_df.get_group(patient_id)
    active = rows[(rows["START"] <= as_of) & (rows["STOP"].isna() | (rows["STOP"] >= as_of))]
    return sorted(set(active["DESCRIPTION"].tolist()))


def split_note_into_blocks(text: str):
    """Split a consolidated patient note file into (date, block_text) per encounter."""
    matches = list(DATE_LINE_RE.finditer(text))
    blocks = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block_text = text[start:end].strip()
        note_date = date.fromisoformat(m.group())
        blocks.append((note_date, block_text))
    return blocks


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conditions_by_patient, medications_by_patient = load_ground_truth()

    note_files = sorted(NOTES_DIR.glob("*.txt"))
    print(f"Found {len(note_files)} note files")

    records = []
    for path in note_files:
        m = FILENAME_UUID_RE.search(path.name)
        if not m:
            continue
        patient_id = m.group(1)
        text = path.read_text(encoding="utf-8")

        for note_date, block_text in split_note_into_blocks(text):
            diagnoses = active_descriptions(conditions_by_patient, patient_id, note_date)
            medications = active_descriptions(medications_by_patient, patient_id, note_date)

            if not diagnoses and not medications:
                continue  # uninformative example, skip

            records.append(
                {
                    "patient_id": patient_id,
                    "note_date": note_date.isoformat(),
                    "note_text": block_text,
                    "diagnoses": diagnoses,
                    "medications": medications,
                }
            )

    print(f"Built {len(records)} labeled encounter examples from {len(note_files)} patients")

    # Split by PATIENT (not by record) so no patient appears in both train and eval
    patient_ids = sorted({r["patient_id"] for r in records})
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(patient_ids)

    n_eval_patients = max(1, int(len(patient_ids) * EVAL_FRACTION))
    eval_patients = set(patient_ids[:n_eval_patients])

    train_records = [r for r in records if r["patient_id"] not in eval_patients]
    eval_records = [r for r in records if r["patient_id"] in eval_patients]

    rng.shuffle(train_records)
    rng.shuffle(eval_records)

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    write_jsonl(OUT_DIR / "train.jsonl", train_records)
    write_jsonl(OUT_DIR / "eval.jsonl", eval_records)

    print(f"Train: {len(train_records)} examples ({len(patient_ids) - n_eval_patients} patients)")
    print(f"Eval:  {len(eval_records)} examples ({n_eval_patients} patients)")
    print(f"Written to {OUT_DIR}")


if __name__ == "__main__":
    main()
