import polars as pl
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pl.read_csv("data/bgg-26m-reviews.csv")
print("All reviews:", data.shape[0])
data = data.filter(pl.col("comment").is_not_null())
data = data.sample(fraction=1)

print("Reviews with text:", data.shape[0])
# print all reviews that have a low score
print("Reviews with text and score <= 5:", data.filter(pl.col("rating") <= 5).shape[0])

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)


def batched_predict_sentiment(data, batch_size=32):
    comments = data["comment"].to_list()
    sentiments = []

    for i in tqdm(range(0, len(comments), batch_size)):
        batch = comments[i : i + batch_size]
        # tokenize a batch
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(device)
        # predict
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu() * 10
        sentiments.extend(preds.tolist())

    return data.with_columns(pl.Series("sentiment", sentiments))


data_with_sentiment = batched_predict_sentiment(data, batch_size=64)
data_with_sentiment.write_csv('data/bgg-26m-reviews-with-nnet-sentiment.csv')