# get_embedding.py

import pandas as pd
from datetime import timedelta


def prepare_patient_notes_data(patients_path, discharge_path, admissions_path):
    # Load the tables
    patients = pd.read_csv(patients_path)
    discharge = pd.read_csv(discharge_path)
    admissions = pd.read_csv(admissions_path)

    # Convert date columns to datetime
    patients["dod"] = pd.to_datetime(patients["dod"])
    discharge["charttime"] = pd.to_datetime(discharge["charttime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
    admissions["deathtime"] = pd.to_datetime(admissions["deathtime"])

    # Merge the tables
    merged = pd.merge(
        discharge, patients[["subject_id", "dod"]], on="subject_id", how="left"
    )
    merged = pd.merge(
        merged,
        admissions[["subject_id", "hadm_id", "dischtime", "deathtime"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )

    # Calculate last discharge time for each patient
    last_discharge = admissions.groupby("subject_id")["dischtime"].max().reset_index()
    merged = pd.merge(merged, last_discharge, on="subject_id", suffixes=("", "_last"))

    # Filter out in-hospital deaths
    merged = merged[merged["deathtime"].isna()]

    # Filter notes to include only those up to the last discharge time
    merged = merged[merged["charttime"] <= merged["dischtime_last"]]

    # Create censoring indicator for out-of-hospital deaths within one year
    merged["ind_death"] = (
        (~merged["dod"].isna())
        & (merged["dod"] <= merged["dischtime_last"] + timedelta(days=365))
    ).astype(int)

    # Function to get the 5 most recent notes
    def get_recent_notes(group):
        sorted_notes = sorted(
            list(zip(group["text"], group["charttime"])),
            key=lambda x: x[1],
            reverse=True,
        )
        recent_notes = sorted_notes[:5]
        # Pad with empty notes if less than 5
        recent_notes += [("", pd.NaT)] * (5 - len(recent_notes))
        return recent_notes

    # Group by patient and get recent notes
    grouped = (
        merged.groupby("subject_id")
        .agg(
            {
                "dod": "first",
                "ind_death": "first",
                "dischtime_last": "first",
                "text": lambda x: get_recent_notes(
                    pd.DataFrame(
                        {"text": x, "charttime": merged.loc[x.index, "charttime"]}
                    )
                ),
            }
        )
        .reset_index()
    )

    # Rename the 'text' column to 'notes'
    grouped = grouped.rename(columns={"text": "notes"})

    return grouped


patients = "/host/scratch/physionet.org/files/mimiciv/3.0/hosp/patients.csv"
discharge = "/host/scratch/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv"
admission = "/host/scratch/physionet.org/files/mimiciv/3.0/hosp/admissions.csv"
patient_data = prepare_patient_notes_data(patients, discharge, admission)
patient_data.to_pickle("data/patient_discharge_notes.pkl.gz")
