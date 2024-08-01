import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
import h5py


class PatientDataset(Dataset):
    def __init__(
        self,
        embeddings,
        time_to_discharge,
        structured_data,
        survival_times,
        event_indicators,
    ):
        self.embeddings = embeddings
        self.time_to_discharge = time_to_discharge
        self.structured_data = structured_data
        self.survival_times = survival_times
        self.event_indicators = event_indicators

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings[idx], dtype=torch.float32),
            torch.tensor(self.time_to_discharge[idx], dtype=torch.float32),
            torch.tensor(self.structured_data[idx], dtype=torch.float32),
            torch.tensor(self.survival_times[idx], dtype=torch.float32),
            torch.tensor(self.event_indicators[idx], dtype=torch.float32),
        )


def load_data(embeddings_path, demographics_path):
    with h5py.File(embeddings_path, "r") as hf:
        patient_ids = hf["patient_ids"][()]
        embeddings = []
        time_to_discharge = []
        survival_times = []
        for i in range(len(patient_ids)):
            embeddings.append(hf[f"patient_{i}/embedding"][()])
            time_to_discharge.append(hf[f"patient_{i}/time_to_discharge"][()])
            survival_times.append(hf[f"patient_{i}/survival_time"][()])

    demographics = pd.read_pickle(demographics_path)

    data = pd.DataFrame(
        {
            "subject_id": patient_ids,
            "embeddings": embeddings,
            "time_to_discharge": time_to_discharge,
            "survival_time": survival_times,
        }
    )
    data = pd.merge(data, demographics, on="subject_id", how="inner")

    exclude_columns = [
        "subject_id",
        "survival_time",
        "embeddings",
        "dod",
        "dischtime",
        "time_to_discharge",
        "ind_death",
    ]
    structured_columns = [col for col in data.columns if col not in exclude_columns]
    X_structured = data[structured_columns]

    return {
        "embeddings": data["embeddings"].tolist(),
        "time_to_discharge": data["time_to_discharge"].tolist(),
        "structured": X_structured,
        "event_indicators": data["ind_death"].values,
        "survival_times": data["survival_time"].values,
    }


def prepare_data(data_dict, config):
    bool_columns = data_dict["structured"].select_dtypes(include=[bool]).columns
    data_dict["structured"][bool_columns] = data_dict["structured"][
        bool_columns
    ].astype(float)

    non_numeric_cols = (
        data_dict["structured"].select_dtypes(exclude=[np.number]).columns
    )
    if len(non_numeric_cols) > 0:
        raise ValueError(
            f"Non-numeric columns found in structured data: {non_numeric_cols}"
        )

    scaler = StandardScaler()
    structured_scaled = scaler.fit_transform(data_dict["structured"])
    num_structured_features = structured_scaled.shape[1]

    X = list(
        zip(data_dict["embeddings"], data_dict["time_to_discharge"], structured_scaled)
    )

    survival_times, event_indicators = add_censoring_variability(
        data_dict["survival_times"],
        data_dict["event_indicators"],
        variability=config["data"]["censoring_variability"],
    )

    y = Surv.from_arrays(event_indicators, survival_times)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    train_dataset = PatientDataset(
        [x[0] for x in X_train],
        [x[1] for x in X_train],
        [x[2] for x in X_train],
        y_train["time"],
        y_train["event"],
    )

    val_dataset = PatientDataset(
        [x[0] for x in X_test],
        [x[1] for x in X_test],
        [x[2] for x in X_test],
        y_test["time"],
        y_test["event"],
    )

    return train_dataset, val_dataset, y_train, y_test, num_structured_features


def custom_collate(batch):
    embeddings, time_to_discharge, structured_data, survival_times, event_indicators = (
        zip(*batch)
    )
    return (
        torch.tensor(np.array(embeddings), dtype=torch.float32),
        torch.tensor(np.array(time_to_discharge), dtype=torch.float32),
        torch.tensor(np.array(structured_data), dtype=torch.float32),
        torch.tensor(survival_times, dtype=torch.float32),
        torch.tensor(event_indicators, dtype=torch.float32),
    )


def add_censoring_variability(
    survival_times, event_indicators, max_time=365, variability=30
):
    for i in range(len(survival_times)):
        if event_indicators[i] == 0:  # Censored patient
            if survival_times[i] >= max_time:
                survival_times[i] = max_time + np.random.uniform(0, variability)
            else:
                pass
        else:  # Patient experienced the event
            if survival_times[i] > max_time:
                survival_times[i] = max_time + np.random.uniform(0, variability)
                event_indicators[i] = 0  # Change to censored

    return survival_times, event_indicators
