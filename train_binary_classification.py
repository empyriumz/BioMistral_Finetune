import torch
import argparse
import yaml
import os
import numpy as np
import random
from torch.utils.data import DataLoader
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from model import BinaryClassificationModel
from data_processing import load_data, prepare_data, custom_collate
from utils import save_config


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False


def train_model(
    model,
    train_loader,
    val_loader,
    y_train,
    y_val,
    num_epochs,
    learning_rate,
    device,
    patience,
    save_dir,
    config,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCELoss()

    best_val_c_index = 0
    epochs_without_improvement = 0
    best_model_state = None

    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "best_model.pth")
    save_config(config, save_dir)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            if len(batch) == 5:  # embedding_structured
                (
                    document_embeddings,
                    timestamps,
                    structured_data,
                    _,
                    event_indicators,
                ) = [b.to(device) for b in batch]
                risk_scores = model(document_embeddings, timestamps, structured_data)
            else:  # embedding_only
                document_embeddings, timestamps, _, event_indicators = [
                    b.to(device) for b in batch
                ]
                risk_scores = model(document_embeddings, timestamps)

            loss = criterion(risk_scores, event_indicators)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().numpy()

        model.eval()
        val_c_index, val_auc, val_mean_auc = evaluate_model(
            model, val_loader, y_train, y_val, device, config["model"]["type"]
        )

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.3f}, "
            f"Val C-index: {val_c_index:.3f}, Val Mean AUC: {val_mean_auc:.3f}"
            f" (AUC at 6 months: {val_auc[0]:.3f}, AUC at 12 months: {val_auc[1]:.3f})"
        )

        if val_c_index > best_val_c_index:
            best_val_c_index = val_c_index
            epochs_without_improvement = 0
            best_model_state = model.state_dict()

            torch.save(best_model_state, model_save_path)
            print(f"New best model saved with C-index: {best_val_c_index:.3f}\n")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs\n")
            break

    if epoch == num_epochs - 1:
        print("Training completed without early stopping")

    model.load_state_dict(best_model_state)
    return model, best_val_c_index


def evaluate_model(model, data_loader, y_train, y_val, device, model_type):
    model.eval()
    all_risk_predictions = []

    with torch.no_grad():
        for batch in data_loader:
            if model_type == "embedding_structured":
                document_embeddings, timestamps, structured_data, _, _ = [
                    b.to(device) for b in batch
                ]
                risk_scores = model(document_embeddings, timestamps, structured_data)
            else:  # embedding_only
                document_embeddings, timestamps, _, _ = [b.to(device) for b in batch]
                risk_scores = model(document_embeddings, timestamps)
            all_risk_predictions.extend(risk_scores.cpu().numpy())

    all_risk_predictions = np.array(all_risk_predictions)

    c_index = concordance_index_ipcw(y_train, y_val, all_risk_predictions)[0]
    times = np.array([15, 30])
    auc, mean_auc = cumulative_dynamic_auc(y_train, y_val, all_risk_predictions, times)

    return c_index, auc, mean_auc


def main(config_path, output_dir):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    set_random_seed(config["training"]["random_seed"])

    embeddings_path = config["data"]["embeddings_path"]
    demographics_path = config["data"].get("demographics_path")
    # Override the output directory in the config
    config["training"]["output_dir"] = output_dir
    data_dict = load_data(embeddings_path, demographics_path)
    train_dataset, val_dataset, y_train, y_val, num_structured_features = prepare_data(
        data_dict, config
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=custom_collate,
    )

    embedding_dim = config["model"]["embedding_dim"]
    use_attention = config["model"].get("attention", True)
    use_time_weighting = config["model"].get("time_weighting", False)

    binary_model = BinaryClassificationModel(
        embedding_dim=embedding_dim,
        num_structured_features=num_structured_features,
        num_documents=config["model"]["num_documents"],
        use_attention=use_attention,
        use_time_weighting=use_time_weighting,
    )

    if config["training"]["use_gpu"] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['training']['gpu_id']}")
    else:
        device = torch.device("cpu")

    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    binary_model = binary_model.to(device)
    best_model, best_val_c_index = train_model(
        binary_model,
        train_loader,
        val_loader,
        y_train,
        y_val,
        num_epochs=config["training"]["num_epochs"],
        learning_rate=config["training"]["learning_rate"],
        device=device,
        patience=config["training"]["early_stopping_patience"],
        save_dir=output_dir,
        config=config,
    )

    final_c_index, _, final_mean_auc = evaluate_model(
        best_model, val_loader, y_train, y_val, device, config["model"]["type"]
    )
    print(f"Best Validation C-index: {best_val_c_index:.3f}")
    print(f"Final C-index (IPCW): {final_c_index:.3f}")
    print(f"Final Mean AUC: {final_mean_auc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train binary classification model with config file."
    )
    parser.add_argument(
        "--config", type=str, help="Path to the configuration YAML file"
    )
    parser.add_argument("--output_dir", type=str, help="Directory to save outputs")
    args = parser.parse_args()
    main(args.config, args.output_dir)
