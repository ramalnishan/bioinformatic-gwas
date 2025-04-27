import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain

# Step 1: Load the full dataset
data = pd.read_csv("snp_data.csv", sep="\t")  # use sep="\t" if it's tab-separated, else remove sep

# Step 2: Select the columns from dataset
selected_snps = data.columns  # All SNP columns
subset = data[selected_snps]

# Step 3: Calculate correlation matrix
corr_matrix = subset.corr()

# Step 4: Create the SNP network
threshold = 0.7  # Threshold for strong correlation

G = nx.Graph()

# Add all SNPs as nodes
for snp in selected_snps:
    G.add_node(snp)

# Add edges based on correlation
for i in range(len(selected_snps)):
    for j in range(i+1, len(selected_snps)):
        if corr_matrix.iloc[i, j] > threshold:
            G.add_edge(selected_snps[i], selected_snps[j], weight=corr_matrix.iloc[i, j])

# Step 5: Perform Louvain clustering
partition = community_louvain.best_partition(G)

# Step 6: Visualize the SNP network
pos = nx.spring_layout(G, seed=42)
colors = [partition[node] for node in G.nodes()]

plt.figure(figsize=(12, 10))
nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.Set3, node_size=300)
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=7)
plt.title("SNP Network Clustering (Louvain Method)")
plt.show()

# Step 7: Print clustering results
num_clusters = len(set(partition.values()))
print(f"\nNumber of clusters found: {num_clusters}\n")

clusters = {}
for node, cluster_id in partition.items():
    clusters.setdefault(cluster_id, []).append(node)

for cluster_id, snp_list in clusters.items():
    print(f"Cluster {cluster_id} ({len(snp_list)} SNPs): {snp_list}\n")
