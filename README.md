# DTU 02807 Group 29 - Board Games Recommendation System

## Introduction
The objective of this project is to uncover relationships between users and the board games they have positively reviewed, with the ultimate goal of building a recommendation system that proposes games tailored to individual preferences.

In this project, we design, evaluate and compare three recommendation system approaches:
1.  **Association Rules**: Utilizes user review data.
2.  **Nearest Neighbor Search**: Relies on board game descriptions (TF-IDF and Transformer embeddings).
3.  **Spectral Clustering**: Relies on board game descriptions (TF-IDF and Shingling).

## Installation
To install the required dependencies run:
```bash
pip install -r requirements.txt
```

## Data Retrieval
To download and extract the dataset from Kaggle, run the following notebook:
- `data.ipynb`

## Usage
The project is divided into several notebooks, each implementing a different part of the recommendation system pipeline.

### Baseline & Association Rules
- `baseline_system.ipynb`: Implements the baseline model.
- `frequent_itemsets.ipynb`: Implements the association rules-based recommendation system.

### Nearest Neighbor Search
- `tf_idf_recommendation.ipynb`: Implements the TF-IDF based Nearest Neighbor search.
- `transformer_recommendation.ipynb`: Implements the Transformer-based Nearest Neighbor search.

### Clustering-based Approach
- `graphing.ipynb`: Implements the shingleing and LSH of the clustering-based recommendation system and explores sentenceTransformer embeddings. 
- `clustering_recommendation_system.ipynb`: Main notebook for the clustering-based recommendation system.
- `spectral_clustering.ipynb`, `spectral_clustering_matrix1.ipynb`, `spectral_clustering_matrix2.ipynb`: Notebooks for spectral clustering analysis.

## Contributions

| Member | Student ID | Contribution |
| :--- | :--- | :--- |
| Oskar Kocjancic | s253070 | TF-IDF implementation of Nearest Neighbour search, TF-IDF similarity matrix, embedding pipeline design, evaluation. Writing and formatting of the report |
| Raquel Pascual | s254628 | Spectral clustering implementation; clustering-based recommendation system implementation and evaluation; writing of the report |
| Marcel Skumantz | s253732 | Shingling, LSH, graph construction and visualization; sentenceTransformer embedding pipeline; recommendation system implementation; writing and documentation |
| Magdalena Zydorczak | s253712 | Baseline model implementation and evaluation; association rules based recommendation system implementation and evaluation; writing of the report |
