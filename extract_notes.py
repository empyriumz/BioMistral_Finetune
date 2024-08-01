import pandas as pd
from datetime import timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_patient_notes_data(patients_path, discharge_path, admissions_path):
    # Load the tables
    patients = pd.read_csv(patients_path)
    discharge = pd.read_csv(discharge_path)
    admissions = pd.read_csv(admissions_path)

    # Convert date columns to datetime
    for df in [patients, discharge, admissions]:
        for col in df.columns:
            if col.endswith("time") or col == "dod":
                df[col] = pd.to_datetime(df[col])

    # Merge discharge notes with admissions
    merged = pd.merge(
        discharge,
        admissions[["subject_id", "hadm_id", "admittime", "dischtime", "deathtime"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )

    # Filter out notes charted after discharge
    merged = merged[merged["charttime"] <= merged["dischtime"]]

    # Get the last discharge for each patient
    last_discharge = merged.groupby("subject_id")["dischtime"].max().reset_index()
    last_discharge = last_discharge.rename(columns={"dischtime": "last_dischtime"})

    # Merge with patient data and last discharge time
    merged = pd.merge(
        merged, patients[["subject_id", "dod"]], on="subject_id", how="left"
    )
    merged = pd.merge(merged, last_discharge, on="subject_id", how="left")

    # Filter out in-hospital deaths
    merged = merged[merged["deathtime"].isna()]

    # Create censoring indicator for out-of-hospital deaths within one year
    merged["ind_death"] = (
        (~merged["dod"].isna())
        & (merged["dod"] <= merged["last_dischtime"] + timedelta(days=365))
    ).astype(int)

    # Calculate time difference between note charttime and discharge time
    merged["time_to_discharge"] = (
        merged["dischtime"] - merged["charttime"]
    ).dt.total_seconds() / (24 * 60 * 60)

    # Function to get the 5 most recent notes with time to discharge
    def get_recent_notes(group):
        sorted_notes = sorted(
            zip(group["text"], group["time_to_discharge"]), key=lambda x: x[1]
        )
        recent_notes = sorted_notes[:5]
        recent_notes += [("", None)] * (5 - len(recent_notes))
        return recent_notes

    # Group by patient and get recent notes
    grouped = (
        merged.groupby("subject_id")
        .agg(
            {
                "dod": "first",
                "ind_death": "first",
                "last_dischtime": "first",
                "text": lambda x: get_recent_notes(merged.loc[x.index]),
            }
        )
        .reset_index()
    )

    # Calculate survival time
    grouped["survival_time"] = (
        grouped["dod"] - grouped["last_dischtime"]
    ).dt.total_seconds() / (24 * 60 * 60)
    grouped.loc[grouped["ind_death"] == 0, "survival_time"] = 365

    # Rename the 'text' column to 'notes'
    grouped = grouped.rename(columns={"text": "notes"})

    return grouped[grouped["survival_time"] >= 0]


# Load and prepare the data
patients = "/host/scratch/physionet.org/files/mimiciv/3.0/hosp/patients.csv"
discharge = "/host/scratch/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv"
admission = "/host/scratch/physionet.org/files/mimiciv/3.0/hosp/admissions.csv"
patient_data = prepare_patient_notes_data(patients, discharge, admission)

# Save the processed data
patient_data.to_pickle("data/patient_discharge_notes_processed.pkl.gz")

logger.info(
    f"Data processing completed. Saved {len(patient_data)} patients with valid data."
)
