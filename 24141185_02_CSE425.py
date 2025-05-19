

# Required Libraries
!pip install -q datasets sentence-transformers umap-learn scikit-learn matplotlib seaborn torch

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Data loading
print("Loading AG News dataset")
dataset = load_dataset("ag_news")["train"].select(range(10000))
texts = [sample["text"] for sample in dataset]

print("Generating BERT embeddings using all-MiniLM-L6-v2...")
model_st = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model_st.encode(texts, batch_size=64, show_progress_bar=True)
umap_model = umap.UMAP(n_components=64, random_state=42)
X_umap = umap_model.fit_transform(embeddings)
X_tensor = torch.tensor(X_umap, dtype=torch.float32)

# Model define
class DEC(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256, embed_dim=32, n_clusters=4):
        super(DEC, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        self.n_clusters = n_clusters
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, embed_dim))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_centers)**2, dim=2))
        q = q ** 1  # (1 + dof) / 2 with dof=1
        q = (q.t() / torch.sum(q, dim=1)).t()
        return x_hat, q, z

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

# Model calls
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DEC(input_dim=64, hidden_dim=256, embed_dim=32, n_clusters=4).to(device)
loader = DataLoader(TensorDataset(X_tensor), batch_size=256, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Pretraining of encoder
print("Pretraining autoencoder...")
model.train()
for epoch in range(30):
    total_loss = 0
    for batch in loader:
        data = batch[0].to(device)
        optimizer.zero_grad()
        x_hat, _, _ = model(data)
        loss = F.mse_loss(x_hat, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Recon Loss = {total_loss / len(loader):.4f}")

# Kmeans algorithm
print("Initializing cluster centers with KMeans...")
model.eval()
with torch.no_grad():
    z_init = model.encoder(X_tensor.to(device)).cpu().numpy()

kmeans = KMeans(n_clusters=4, n_init=20).fit(z_init)
y_pred_last = kmeans.labels_
model.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

# Training part
print("Training DEC with clustering loss...")
max_iter = 100
update_interval = 10
tol = 1e-3

for iteration in range(max_iter):
    if iteration % update_interval == 0:
        model.eval()
        with torch.no_grad():
            _, q_all, _ = model(X_tensor.to(device))
        p = target_distribution(q_all).detach()
        y_pred = q_all.cpu().numpy().argmax(axis=1)
        delta = np.sum(y_pred != y_pred_last) / len(y_pred)
        print(f"Iter {iteration}: Label change rate = {delta:.4f}")
        if delta < tol:
            print("Converged.")
            break
        y_pred_last = y_pred

    model.train()
    total_loss = 0
    for batch in loader:
        data = batch[0].to(device)
        optimizer.zero_grad()
        x_hat, q_batch, _ = model(data)
        p_batch = p[:data.size(0)]
        recon_loss = F.mse_loss(x_hat, data)
        kl_loss = F.kl_div(q_batch.log(), p_batch.to(device), reduction='batchmean')
        loss = recon_loss + kl_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if iteration % update_interval == 0:
        print(f"Iter {iteration}: Loss = {total_loss / len(loader):.4f}")

# Saving the Model
torch.save(model.state_dict(), "dec_model.pt")
np.save("X_umap.npy", X_umap)
print("Model saved to 'dec_model.pt' and embeddings to 'X_umap.npy'")

# Evaluation Process
model.eval()
with torch.no_grad():
    _, q_final, z_final = model(X_tensor.to(device))
    cluster_labels = q_final.cpu().numpy().argmax(axis=1)
    z_np = z_final.cpu().numpy()

sil_score = silhouette_score(z_np, cluster_labels)
print(f"Silhouette Score: {sil_score * 100:.2f}%")

# Visualization
print("\n Visualizing with PCA...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(z_np)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=cluster_labels, palette="tab10")
plt.title("DEC Clustering")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
