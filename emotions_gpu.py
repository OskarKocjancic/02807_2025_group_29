import polars as pl
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pl.read_csv("data/bgg-26m-reviews.csv")
data = data.filter(pl.col("comment").is_not_null())
data = data.sample(fraction=1) 

print(f"Processing {data.shape[0]} reviews")

model_name = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    problem_type="multi_label_classification",
).to(device)

labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]

def batched_predict_emotions(data, batch_size=128):
    comments = data["comment"].to_list()

    probs_dict = {f"prob_{label}": [] for label in labels}

    with torch.no_grad():  
        for i in tqdm(range(0, len(comments), batch_size)):
            batch = comments[i : i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(device)

            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu()

            for j, label in enumerate(labels):
                probs_dict[f"prob_{label}"].extend(probs[:, j].tolist())

            del inputs, outputs, probs
            torch.cuda.empty_cache()

    df_probs = pl.DataFrame(probs_dict)
    return data.with_columns(df_probs)

data_with_emotions = batched_predict_emotions(data, batch_size=64)
data_with_emotions.write_csv('data/bgg-26m-reviews-with-emotions.csv')
