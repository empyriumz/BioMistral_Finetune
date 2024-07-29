import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
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


def sliding_window_embedding(
    document, model, tokenizer, device, max_length=2048, stride=1024
):
    tokens = tokenizer.encode(document, add_special_tokens=False)
    windows = [tokens[i : i + max_length] for i in range(0, len(tokens), stride)]
    window_embeddings = []
    for window in windows:
        inputs = torch.tensor([window]).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)
            window_embeddings.append(embedding)
    document_embedding = torch.mean(torch.cat(window_embeddings), dim=0)
    return document_embedding.cpu()


def process_patient_documents(patient_data, model, tokenizer, device, batch_size=64):
    logger.info(f"Processing {len(patient_data)} patients on device {device}")
    all_patient_embeddings = []
    all_patient_ids = []
    all_patient_timestamps = []
    for i in tqdm(range(0, len(patient_data), batch_size), position=1, leave=False):
        batch = patient_data.iloc[i : i + batch_size]
        batch_embeddings = []
        batch_timestamps = []
        for _, row in batch.iterrows():
            notes_list = row["notes"]
            subject_id = row["subject_id"]
            patient_embeddings = []
            patient_timestamps = []
            for note_idx, (note, timestamp) in enumerate(notes_list):
                if note:
                    logger.debug(
                        f"Processing note {note_idx + 1} for patient {subject_id}"
                    )
                    doc_embedding = sliding_window_embedding(
                        note, model, tokenizer, device
                    )
                    patient_embeddings.append(doc_embedding)
                    patient_timestamps.append(
                        timestamp.timestamp() if pd.notna(timestamp) else -1
                    )
                else:
                    logger.debug(f"Empty note {note_idx + 1} for patient {subject_id}")
                    patient_embeddings.append(torch.zeros(model.config.hidden_size))
                    patient_timestamps.append(-1)
            batch_embeddings.append(torch.stack(patient_embeddings))
            batch_timestamps.append(patient_timestamps)
            all_patient_ids.append(subject_id)
        all_patient_embeddings.extend(batch_embeddings)
        all_patient_timestamps.extend(batch_timestamps)
    logger.info(f"Processed {len(all_patient_ids)} patients")
    return all_patient_embeddings, all_patient_ids, all_patient_timestamps


def generate_embeddings_gpu(patient_data, gpu_id):
    logger.info(f"Initializing embedding generation on GPU {gpu_id}")
    tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B")
    model = AutoModel.from_pretrained("BioMistral/BioMistral-7B")
    device = torch.device(f"cuda:{gpu_id}")
    model = model.to(device)
    logger.info(f"Model loaded on GPU {gpu_id}")

    patient_embeddings, patient_ids, patient_timestamps = process_patient_documents(
        patient_data, model, tokenizer, device
    )

    logger.info(f"Embedding generation completed on GPU {gpu_id}")
    return patient_embeddings, patient_ids, patient_timestamps


def save_embeddings(output_path, patient_embeddings, patient_ids, patient_timestamps):
    logger.info(f"Saving embeddings to {output_path}")
    with h5py.File(output_path, "w") as hf:
        for i, (embedding, subject_id, timestamps) in enumerate(
            zip(patient_embeddings, patient_ids, patient_timestamps)
        ):
            hf.create_dataset(f"patient_{i}/embedding", data=embedding.numpy())
            hf.create_dataset(f"patient_{i}/subject_id", data=subject_id)
            hf.create_dataset(f"patient_{i}/timestamp", data=timestamps)
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
        all_timestamps = []
        for emb, ids, timestamps in results:
            all_embeddings.extend(emb)
            all_ids.extend(ids)
            all_timestamps.extend(timestamps)

        save_embeddings(output_path, all_embeddings, all_ids, all_timestamps)
    else:
        logger.info("Using single GPU or CPU for processing")
        generate_embeddings_gpu(patient_data, 0 if use_gpu else -1)

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
