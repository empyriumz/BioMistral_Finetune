import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
import h5py
import numpy as np


class PatientDataset(Dataset):
    def __init__(self, embeddings, survival_times, event_indicators):
        self.embeddings = embeddings
        self.survival_times = survival_times
        self.event_indicators = event_indicators

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings[idx], dtype=torch.float32),
            torch.tensor(self.survival_times[idx], dtype=torch.float32),
            torch.tensor(self.event_indicators[idx], dtype=torch.float32),
        )


def get_max_survival_time(data_dict):
    return max(data_dict["survival_times"])


def load_data(embeddings_path):
    with h5py.File(embeddings_path, "r") as hf:
        patient_ids = hf["patient_ids"][()]
        embeddings = []
        survival_times = []
        event_indicators = []
        for i in range(len(patient_ids)):
            embeddings.append(hf[f"patient_{i}/embedding"][()])
            survival_times.append(hf[f"patient_{i}/survival_time"][()])
            event_indicators.append(
                hf[f"patient_{i}/death_indicator"][()]
            )  # Inverting death indicator

    return {
        "embeddings": embeddings,
        "survival_times": survival_times,
        "event_indicators": event_indicators,
    }


def add_censoring_variability(
    survival_times, event_indicators, max_time=365, variability=30
):
    for i in range(len(survival_times)):
        if event_indicators[i] == 0:  # Censored patient
            if survival_times[i] >= max_time:
                survival_times[i] = max_time + np.random.uniform(0, variability)

    return survival_times, event_indicators


def prepare_data(data_dict, config):
    X = list(data_dict["embeddings"])
    survival_times, event_indicators = add_censoring_variability(
        data_dict["survival_times"],
        data_dict["event_indicators"],
        variability=config["data"]["censoring_variability"],
    )
    y = Surv.from_arrays(event_indicators, survival_times)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["training"]["val_split"], random_state=42
    )

    train_dataset = PatientDataset(
        X_train,
        y_train["time"],
        y_train["event"],
    )

    val_dataset = PatientDataset(
        X_test,
        y_test["time"],
        y_test["event"],
    )

    return train_dataset, val_dataset, y_train, y_test


def custom_collate(batch):
    embeddings, survival_times, event_indicators = zip(*batch)
    return (
        torch.tensor(np.array(embeddings), dtype=torch.float32),
        torch.tensor(survival_times, dtype=torch.float32),
        torch.tensor(event_indicators, dtype=torch.float32),
    )
