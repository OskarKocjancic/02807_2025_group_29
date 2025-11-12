import polars as pl
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pl.read_csv("data/bgg-26m-reviews.csv")
print("All reviews:", data.shape[0])
data = data.filter(pl.col("comment").is_not_null())
data = data.sample(fraction=0.1)

print("Reviews with text:", data.shape[0])

model_name = "SamLowe/roberta-base-go_emotions"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# specify that this is a multi‑label classification problem
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    problem_type="multi_label_classification",
).to(device)

def batched_predict_emotions(data, batch_size=32):
    comments = data["comment"].to_list()
    all_probs = []
    for i in tqdm(range(0, len(comments), batch_size)):
        batch = comments[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)
        all_probs.extend(probs.cpu().tolist())

    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    df_probs = pl.DataFrame({f"prob_{labels[i]}": [p[i] for p in all_probs] for i in range(len(labels))})

    return data.with_columns(df_probs)

data_with_emotions = batched_predict_emotions(data, batch_size=512)
data_with_emotions.write_csv('data/bgg-26m-reviews-with-emotions.csv')
