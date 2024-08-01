import torch
import argparse
import yaml
import os
import numpy as np
from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from model import SurvivalModel
from data_processing import load_data, prepare_data, custom_collate
from utils import save_config
from loss import diffsurv_loss


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

    # Save the configuration
    save_config(config, os.path.dirname(config["model"]["save_path"]))
