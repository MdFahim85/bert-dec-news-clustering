📝 Project Description

Unsupervised News Topic Clustering using Deep Embedded Clustering (DEC) and BERT

This project applies Deep Embedded Clustering (DEC) to extract meaningful topic clusters from the AG News dataset using sentence-level BERT embeddings. It integrates UMAP for dimensionality reduction and visualizes the results with PCA. The final model achieves a Silhouette Score of 60.74%, demonstrating strong unsupervised clustering performance.


📂 Features

✅ Pretrained MiniLM (BERT) embeddings via SentenceTransformers

✅ Dimensionality reduction using UMAP

✅ Autoencoder-based latent space learning

✅ Student’s t-distribution based clustering layer

✅ DEC training loop with KL divergence and cluster stabilization

✅ Clean PCA visualization of clusters

✅ Evaluation using Silhouette Score


📊 Results

Silhouette Score: 60.74%

PCA Clustering Visualization included

DEC outperforms KMeans and Agglomerative Clustering on BERT embeddings


⚙️ Requirements

pip install datasets sentence-transformers umap-learn scikit-learn matplotlib seaborn torch


