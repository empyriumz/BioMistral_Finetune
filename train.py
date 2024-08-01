import torch
import argparse
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv
import yaml
import h5py
from model import SurvivalModel
from loss import diffsurv_loss
from utils import add_censoring_variability


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

    # Select structured data columns (all columns except the specified ones)
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
    # Convert boolean columns to float
    bool_columns = data_dict["structured"].select_dtypes(include=[bool]).columns
    data_dict["structured"][bool_columns] = data_dict["structured"][
        bool_columns
    ].astype(float)

    # Check for non-numeric columns (excluding bool which are now float)
    non_numeric_cols = (
        data_dict["structured"].select_dtypes(exclude=[np.number]).columns
    )
    if len(non_numeric_cols) > 0:
        raise ValueError(
            f"Non-numeric columns found in structured data: {non_numeric_cols}"
        )

    # Scale the numeric data
    scaler = StandardScaler()
    structured_scaled = scaler.fit_transform(data_dict["structured"])
    num_structured_features = structured_scaled.shape[1]

    # Combine all features
    X = list(
        zip(data_dict["embeddings"], data_dict["time_to_discharge"], structured_scaled)
    )

    # Apply censoring variability
    survival_times, event_indicators = add_censoring_variability(
        data_dict["survival_times"],
        data_dict["event_indicators"],
        variability=config["data"]["censoring_variability"],
    )

    # Create structured array for sksurv
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
            train_loss += loss.detach().cpu().numpy()

        model.eval()
        val_c_index, val_auc, val_mean_auc = evaluate_model(
            model, val_loader, y_train, y_val, device
        )

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.3f}, "
            f"Val C-index: {val_c_index:.3f}, Val Mean AUC: {val_mean_auc:.3f}"
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
            # Take the mean along the sorter_size dimension
            risk_scores = risk_scores.mean(dim=1)
            all_risk_scores.extend(risk_scores.cpu().numpy())

    all_risk_scores = np.array(all_risk_scores)

    c_index = concordance_index_ipcw(y_train, y_val, all_risk_scores)[0]

    # Calculate cumulative_dynamic_auc for 6 and 12 months
    times = np.array([180, 365])
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_val, all_risk_scores, times)

    return c_index, auc, mean_auc


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Train survival model with config file."
    )
    parser.add_argument(
        "--config", type=str, help="Path to the configuration YAML file"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    embeddings_path = config["data"]["embeddings_path"]
    demographics_path = config["data"]["demographics_path"]

    # Load all data
    data_dict = load_data(embeddings_path, demographics_path)
    # Prepare data with the updated function
    train_dataset, val_dataset, y_train, y_val, num_structured_features = prepare_data(
        data_dict, config
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,  # to ensure the sorter network works
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=custom_collate,
    )

    embedding_dim = config["model"]["embedding_dim"]

    survival_model = SurvivalModel(
        embedding_dim=embedding_dim,
        num_structured_features=num_structured_features,
        num_documents=config["model"]["num_documents"],
        sorter_size=config["training"]["batch_size"],
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

    final_c_index, final_auc, final_mean_auc = evaluate_model(
        trained_model, val_loader, y_train, y_val, device
    )
    print(f"Final C-index (IPCW): {final_c_index:.3f}")
    print(f"Final AUC at 6 months: {final_auc[0]:.3f}")
    print(f"Final AUC at 12 months: {final_auc[1]:.3f}")
    print(f"Final Mean AUC: {final_mean_auc:.3f}")
    # Save the model
    torch.save(trained_model.state_dict(), config["model"]["save_path"])
