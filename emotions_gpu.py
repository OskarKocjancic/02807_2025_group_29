import argparse
import polars as pl
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Emotion classification for reviews.")
    parser.add_argument("--csv_path", type=str, default="data/bgg-26m-reviews.csv",
                        help="Path to input CSV file")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for prediction")
    parser.add_argument("--output_path", type=str,
                        default="data/bgg-26m-reviews-with-emotions.csv",
                        help="Where to save output CSV")
    return parser.parse_args()


def batched_predict_emotions(data, tokenizer, model, labels, batch_size, device):
    comments = data["comment"].to_list()
    probs_dict = {f"prob_{label}": [] for label in labels}

    with torch.no_grad():
        for i in tqdm(range(0, len(comments), batch_size)):
            batch = comments[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(device)

            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu()

            for j, label in enumerate(labels):
                probs_dict[f"prob_{label}"].extend(probs[:, j].tolist())

            del inputs, outputs, probs
            torch.cuda.empty_cache()

    df_probs = pl.DataFrame(probs_dict)
    return data.with_columns(df_probs)


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data = pl.read_csv(args.csv_path)
    data = data.filter(pl.col("comment").is_not_null())
    data = data.sample(fraction=.0001)

    print(f"Processing {data.shape[0]} reviews")

    # Load model
    model_name = "SamLowe/roberta-base-go_emotions"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        problem_type="multi_label_classification",
    ).to(device)

    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]

    # Predict
    data_with_emotions = batched_predict_emotions(
        data,
        tokenizer=tokenizer,
        model=model,
        labels=labels,
        batch_size=args.batch_size,
        device=device,
    )

    # Save output
    data_with_emotions.write_csv(args.output_path)
    print(f"Saved output to {args.output_path}")


if __name__ == "__main__":
    main()
