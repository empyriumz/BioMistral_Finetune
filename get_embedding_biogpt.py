import torch
import pandas as pd
from transformers import BioGptTokenizer, BioGptForCausalLM
import h5py
import argparse
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def sliding_window_embedding(documents, model, tokenizer, device, max_length, stride):
    all_embeddings = []
    for document in documents:
        if not document:  # Handle empty notes
            all_embeddings.append(torch.zeros(model.config.hidden_size, device=device))
            continue

        tokens = tokenizer.encode(document, add_special_tokens=True)
        windows = [tokens[i : i + max_length] for i in range(0, len(tokens), stride)]
        window_embeddings = []
        for i in range(0, len(windows), 8):  # Process 8 windows at a time
            batch = windows[i : i + 8]
            padded_batch = [
                w + [tokenizer.pad_token_id] * (max_length - len(w)) for w in batch
            ]
            inputs = torch.tensor(padded_batch).to(device)
            attention_mask = (inputs != tokenizer.pad_token_id).long()

            with torch.no_grad():
                outputs = model(
                    inputs, attention_mask=attention_mask, output_hidden_states=True
                )
                # Use the last hidden state as the embedding
                embeddings = outputs.hidden_states[-1].mean(dim=1)
                window_embeddings.extend(embeddings)

        document_embedding = torch.mean(torch.stack(window_embeddings), dim=0)
        all_embeddings.append(document_embedding)

    return torch.stack(all_embeddings)


def pad_or_truncate(data, target_length, pad_value=0):
    data = list(data)  # Convert input to a list
    if len(data) > target_length:
        return data[:target_length]
    elif len(data) < target_length:
        padding = [pad_value] * (target_length - len(data))
        return data + padding
    return data


def process_patient_documents(
    patient_data,
    model,
    tokenizer,
    device,
    batch_size=32,
    max_documents=5,
    max_length=None,
    stride=None,
):
    logger.info(f"Processing {len(patient_data)} patients on device {device}")
    all_patient_embeddings = []
    all_patient_ids = []
    all_patient_time_to_discharge = []
    all_patient_survival_times = []
    all_patient_death_indicators = []

    for i in tqdm(range(0, len(patient_data), batch_size), position=1, leave=False):
        batch = patient_data.iloc[i : i + batch_size]
        batch_documents = []
        batch_time_to_discharge = []
        batch_ids = []
        batch_doc_indices = []

        for _, row in batch.iterrows():
            notes_list = row["notes"]
            subject_id = row["subject_id"]
            patient_documents = []
            patient_time_to_discharge = []
            patient_doc_indices = []

            for doc_idx, (note, time_to_discharge) in enumerate(notes_list):
                if note:  # Only process non-empty notes
                    patient_documents.append(note)
                    patient_time_to_discharge.append(
                        time_to_discharge if pd.notna(time_to_discharge) else -1
                    )
                    patient_doc_indices.append(doc_idx)

            # Sort documents, time_to_discharge, and indices based on doc_idx
            sorted_data = sorted(
                zip(patient_doc_indices, patient_documents, patient_time_to_discharge)
            )
            patient_doc_indices, patient_documents, patient_time_to_discharge = map(
                list, zip(*sorted_data)
            )

            # Pad or truncate documents, time_to_discharge, and indices
            patient_documents = pad_or_truncate(patient_documents, max_documents, "")
            patient_time_to_discharge = pad_or_truncate(
                patient_time_to_discharge, max_documents, -1
            )
            patient_doc_indices = pad_or_truncate(
                patient_doc_indices, max_documents, -1
            )

            batch_documents.extend(patient_documents)
            batch_time_to_discharge.extend(patient_time_to_discharge)
            batch_ids.extend([subject_id] * max_documents)
            batch_doc_indices.extend(patient_doc_indices)

        if batch_documents:
            # Process the batch of documents
            batch_embeddings = sliding_window_embedding(
                batch_documents, model, tokenizer, device, max_length, stride
            )
            # Reshape embeddings to group by patient
            batch_embeddings = batch_embeddings.view(
                -1, max_documents, batch_embeddings.size(-1)
            )

            all_patient_embeddings.extend(batch_embeddings.cpu().numpy())
            all_patient_time_to_discharge.extend(
                np.array(batch_time_to_discharge).reshape(-1, max_documents)
            )
            all_patient_ids.extend(batch_ids[::max_documents])
            all_patient_survival_times.extend(
                [row["survival_time"] for _, row in batch.iterrows()]
            )
            all_patient_death_indicators.extend(
                [row["ind_death"] for _, row in batch.iterrows()]
            )

    logger.info(f"Processed {len(all_patient_ids)} patients")
    return (
        all_patient_embeddings,
        all_patient_ids,
        all_patient_time_to_discharge,
        all_patient_survival_times,
        all_patient_death_indicators,
    )


def generate_embeddings_gpu(patient_data, gpu_id):
    logger.info(f"Initializing embedding generation on GPU {gpu_id}")
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/BioGPT-large")
    model = BioGptForCausalLM.from_pretrained("microsoft/BioGPT-large")
    device = torch.device(f"cuda:{gpu_id}" if gpu_id >= 0 else "cpu")
    model = model.to(device)
    logger.info(f"Model loaded on {'GPU ' + str(gpu_id) if gpu_id >= 0 else 'CPU'}")

    # Obtain the model's maximum context length
    max_length = model.config.max_position_embeddings
    stride = max_length // 2
    logger.info(f"Model max context length: {max_length}, Using stride: {stride}")
    (
        patient_embeddings,
        patient_ids,
        patient_time_to_discharge,
        survival_times,
        death_indicators,
    ) = process_patient_documents(
        patient_data, model, tokenizer, device, max_length=max_length, stride=stride
    )

    logger.info(
        f"Embedding generation completed on {'GPU ' + str(gpu_id) if gpu_id >= 0 else 'CPU'}"
    )
    return (
        patient_embeddings,
        patient_ids,
        patient_time_to_discharge,
        survival_times,
        death_indicators,
    )


def save_embeddings(
    output_path,
    patient_embeddings,
    patient_ids,
    patient_time_to_discharge,
    survival_times,
    death_indicators,
):
    logger.info(f"Saving embeddings to {output_path}")
    with h5py.File(output_path, "w") as hf:
        for i, (
            embedding,
            subject_id,
            time_to_discharge,
            survival_time,
            death_indicator,
        ) in enumerate(
            zip(
                patient_embeddings,
                patient_ids,
                patient_time_to_discharge,
                survival_times,
                death_indicators,
            )
        ):
            hf.create_dataset(f"patient_{i}/embedding", data=embedding)
            hf.create_dataset(f"patient_{i}/subject_id", data=subject_id)
            hf.create_dataset(f"patient_{i}/time_to_discharge", data=time_to_discharge)
            hf.create_dataset(f"patient_{i}/survival_time", data=survival_time)
            hf.create_dataset(f"patient_{i}/death_indicator", data=death_indicator)
        hf.create_dataset("patient_ids", data=patient_ids)
    logger.info(f"Embeddings saved for {len(patient_ids)} patients")


def generate_and_save_embeddings(patient_data, output_path, use_gpu):
    start_time = datetime.now()
    logger.info(f"Starting embedding generation at {start_time}")

    if use_gpu and torch.cuda.device_count() > 1:
        num_gpus = min(2, torch.cuda.device_count())  # Use up to 2 GPUs
        logger.info(f"Using {num_gpus} GPUs for parallel processing")
        split_data = np.array_split(patient_data, num_gpus)

        with mp.Pool(num_gpus) as pool:
            results = pool.starmap(
                generate_embeddings_gpu,
                [(split, i) for i, split in enumerate(split_data)],
            )

        all_embeddings = []
        all_ids = []
        all_time_to_discharge = []
        all_survival_times = []
        all_death_indicators = []
        for emb, ids, times, survival_times, death_indicators in results:
            all_embeddings.extend(emb)
            all_ids.extend(ids)
            all_time_to_discharge.extend(times)
            all_survival_times.extend(survival_times)
            all_death_indicators.extend(death_indicators)
    else:
        logger.info("Using single GPU or CPU for processing")
        (
            all_embeddings,
            all_ids,
            all_time_to_discharge,
            all_survival_times,
            all_death_indicators,
        ) = generate_embeddings_gpu(
            patient_data, 0 if use_gpu and torch.cuda.is_available() else -1
        )

    save_embeddings(
        output_path,
        all_embeddings,
        all_ids,
        all_time_to_discharge,
        all_survival_times,
        all_death_indicators,
    )

    end_time = datetime.now()
    logger.info(f"Embedding generation completed at {end_time}")
    logger.info(f"Total time taken: {end_time - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and save embeddings for patient notes"
    )
    parser.add_argument(
        "--patient_data", type=str, required=True, help="Path to patient data"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output H5 file"
    )
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU for computation if available"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    logger.info(f"Loading patient data from {args.patient_data}")
    patient_data = pd.read_pickle(args.patient_data)
    logger.info(f"Loaded data for {len(patient_data)} patients")

    generate_and_save_embeddings(patient_data, args.output, args.use_gpu)
