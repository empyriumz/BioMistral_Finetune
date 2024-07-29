import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
import yaml
import h5py
from model import SurvivalModel
from loss import diffsurv_loss


class PatientDataset(Dataset):
    def __init__(
        self, notes, timestamps, structured_data, survival_times, event_indicators
    ):
        self.notes = notes
        self.timestamps = timestamps
        self.structured_data = structured_data
        self.survival_times = survival_times
        self.event_indicators = event_indicators

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, idx):
        return (
            self.notes[idx],
            self.timestamps[idx],
            self.structured_data[idx],
            self.survival_times[idx],
            self.event_indicators[idx],
        )


def load_data(embeddings_path, demographics_path):
    with h5py.File(embeddings_path, "r") as hf:
        embeddings = [
            torch.tensor(hf[f"patient_{i}/embedding"][()])
            for i in range(len(hf["patient_ids"]))
        ]
        timestamps = [
            torch.tensor(hf[f"patient_{i}/timestamp"][()])
            for i in range(len(hf["patient_ids"]))
        ]
        patient_ids = hf["patient_ids"][()]

    demographics = pd.read_pickle(demographics_path)

    # Merge embeddings with demographics
    data = pd.DataFrame(
        {"subject_id": patient_ids, "embeddings": embeddings, "timestamps": timestamps}
    )
    data = pd.merge(data, demographics, on="subject_id", how="inner")

    # Prepare the structured data
    exclude_columns = [
        "dod",
        "subject_id",
        "ind_death",
        "dischtime_last",
        "embeddings",
        "timestamps",
    ]
    structured_columns = [
        col for col in demographics.columns if col not in exclude_columns
    ]
    X_structured = data[structured_columns].values

    survival_times = (data["dod"] - data["dischtime_last"]).dt.total_seconds() / (
        24 * 3600
    )
    event_indicators = data["ind_death"].values

    # Create structured array for sksurv
    y = Surv.from_arrays(event_indicators, survival_times)

    return data["embeddings"].tolist(), data["timestamps"].tolist(), X_structured, y


def prepare_data(notes, timestamps, X_structured, y):
    scaler = StandardScaler()
    X_structured_scaled = scaler.fit_transform(X_structured)

    X = list(zip(notes, timestamps, X_structured_scaled))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = PatientDataset(
        [x[0] for x in X_train],
        [x[1] for x in X_train],
        torch.tensor([x[2] for x in X_train], dtype=torch.float32),
        torch.tensor(y_train["time"], dtype=torch.float32),
        torch.tensor(y_train["event"], dtype=torch.float32),
    )

    val_dataset = PatientDataset(
        [x[0] for x in X_test],
        [x[1] for x in X_test],
        torch.tensor([x[2] for x in X_test], dtype=torch.float32),
        torch.tensor(y_test["time"], dtype=torch.float32),
        torch.tensor(y_test["event"], dtype=torch.float32),
    )

    return train_dataset, val_dataset, y_train, y_test


def train_model(
    model, train_loader, val_loader, y_train, y_val, num_epochs, learning_rate, device
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for (
            document_embeddings,
            timestamps,
            structured_data,
            survival_times,
            event_indicators,
        ) in train_loader:
            document_embeddings = document_embeddings.to(device)
            timestamps = timestamps.to(device)
            structured_data = structured_data.to(device)
            survival_times = survival_times.to(device)
            event_indicators = event_indicators.to(device)

            _, perm_prediction = model(document_embeddings, timestamps, structured_data)
            loss = diffsurv_loss(perm_prediction, survival_times, event_indicators)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_c_index = evaluate_model(model, val_loader, y_train, y_val, device)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val C-index: {val_c_index:.4f}"
        )

    return model


def evaluate_model(model, data_loader, y_train, y_val, device):
    model.eval()
    all_risk_scores = []

    with torch.no_grad():
        for document_embeddings, timestamps, structured_data, _, _ in data_loader:
            document_embeddings = document_embeddings.to(device)
            timestamps = timestamps.to(device)
            structured_data = structured_data.to(device)

            risk_scores = model.get_risk_scores(
                document_embeddings, timestamps, structured_data
            )
            all_risk_scores.extend(risk_scores.cpu().numpy())

    all_risk_scores = np.array(all_risk_scores)

    c_index = concordance_index_ipcw(y_train, y_val, all_risk_scores)[0]
    return c_index


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    config_path = "config.yaml"
    config = load_config(config_path)

    embeddings_path = config["data"]["embeddings_path"]
    demographics_path = config["data"]["demographics_path"]

    notes, timestamps, X_structured, y = load_data(embeddings_path, demographics_path)
    train_dataset, val_dataset, y_train, y_val = prepare_data(
        notes, timestamps, X_structured, y
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=lambda x: (
            torch.stack([item[0] for item in x]),
            torch.stack([item[1] for item in x]),
            torch.stack([item[2] for item in x]),
            torch.stack([item[3] for item in x]),
            torch.stack([item[4] for item in x]),
        ),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=lambda x: (
            torch.stack([item[0] for item in x]),
            torch.stack([item[1] for item in x]),
            torch.stack([item[2] for item in x]),
            torch.stack([item[3] for item in x]),
            torch.stack([item[4] for item in x]),
        ),
    )

    embedding_dim = config["model"]["embedding_dim"]
    structured_features = X_structured.shape[1]

    survival_model = SurvivalModel(
        embedding_dim=embedding_dim,
        structured_features=structured_features,
        sorter_size=config["model"]["sorter_size"],
    )

    if config["training"]["use_gpu"] and torch.cuda.is_available():
        device = torch.device(f'cuda:{config["training"]["gpu_id"]}')
    else:
        device = torch.device("cpu")

    survival_model = survival_model.to(device)

    trained_model = train_model(
        survival_model,
        train_loader,
        val_loader,
        y_train,
        y_val,
        num_epochs=config["training"]["num_epochs"],
        learning_rate=config["training"]["learning_rate"],
        device=device,
    )

    final_c_index = evaluate_model(trained_model, val_loader, y_train, y_val, device)
    print(f"Final C-index (IPCW): {final_c_index:.3f}")

    # Save the model
    torch.save(trained_model.state_dict(), config["model"]["save_path"])
