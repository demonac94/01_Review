# Description: This file contains the functions to perform clustering using BERTopic model

#================================================================================================
# Importing libraries
#================================================================================================
# Import tools for clustering
import umap.umap_ as umap
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import PartOfSpeech
from bertopic.representation import TextGeneration
import hdbscan
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

# Import transformers and others
from transformers import pipeline
from sentence_transformers import SentenceTransformer, SimilarityFunction, util
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Import other libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#================================================================================================
# Function to rescale the embeddings (fromg BERTopic documentation)
#================================================================================================

def rescale(x, inplace=False):
  """ Rescale an embedding so optimization will not have convergence issues.
  """
  if not inplace:
      x = np.array(x, copy=True)
  x /= np.std(x[:, 0]) * 10000

  return x

#================================================================================================
# Define BERTopic model function
#================================================================================================

def BERTopic_model(docs, embeddings, n_comp=20, n_neigh=25, minS_cluster=5):
  # Initialize and rescale PCA embeddings
  pca_embeddings = rescale(PCA(n_components=n_comp).fit_transform(embeddings))

  # Embedding model
  embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
  #embedding_model = None

  # Reduce dimensionality
  umap_model      = umap.UMAP(n_neighbors=n_neigh, 
                              n_components=n_comp, 
                              metric='cosine',
                              random_state=42,
                              init=pca_embeddings,
                              )

  # Cluster the reduced data
  hdbscan_model   = hdbscan.HDBSCAN(min_cluster_size=minS_cluster, 
                                      metric='euclidean', 
                                      cluster_selection_method='eom',
                                      prediction_data = True)

  # Vectorizer
  vectorizer_model = CountVectorizer(stop_words="english",
                                      ngram_range=(1, 2))

  # Topic representation
  ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

  # Representation model
  KeyBert = KeyBERTInspired()
  mmr     = MaximalMarginalRelevance(diversity=0.3)
  representation_model = {
                      "MMR":    mmr,
                      "KeyBert": KeyBert,
                      } 
  # All steps together
  topic_model = BERTopic(
    embedding_model=embedding_model,          # Step 1 - Extract embeddings
    umap_model=umap_model,                    # Step 2 - Reduce dimensionality
    hdbscan_model=hdbscan_model,              # Step 3 - Cluster reduced embeddings
    vectorizer_model=vectorizer_model,        # Step 4 - Tokenize topics
    ctfidf_model=ctfidf_model,                # Step 5 - Extract topic words
    representation_model=representation_model,# Step 6 - (Optional) Fine-tune topic representations
    calculate_probabilities=True
  )

  # Fit the BERTopic model
  topics, probs = topic_model.fit_transform(docs,  embeddings)

  return topics, probs, topic_model

#================================================================================================
# Function to calculate silhouette score of the clustering
#================================================================================================

def silhouette_score_calc(embeddings, topics):
  # Print the silhouette score
  non_noise_indices   = [i for i, topic in enumerate(topics) if topic != -1]
  filtered_embeddings = embeddings[non_noise_indices]
  filtered_topics     = [topics[i] for i in non_noise_indices]

  silhouette_vals     = silhouette_samples(filtered_embeddings, filtered_topics)
  silhouette_avg      = silhouette_score(filtered_embeddings, filtered_topics)

  # Create a DataFrame to hold the results
  results = pd.DataFrame({'topic': filtered_topics, 'silhouette_score': silhouette_vals})

  # Group by topic and calculate the average Silhouette score per topic
  avg_silhouette_per_topic = results.groupby('topic')['silhouette_score'].mean().reset_index()

  # Count how many topics are negative
  negative_topics = avg_silhouette_per_topic[avg_silhouette_per_topic['silhouette_score'] < 0].shape[0]

  return silhouette_avg, negative_topics

#================================================================================================
# Function to find best parameters for BERTopic model
#================================================================================================

def BERTopic_best_params(docs, embeddings, n_ClusterSize, n_comp_matrix, n_neigh_matrix):
  results = []
  for k in n_ClusterSize:
      for i in n_comp_matrix:
          for j in n_neigh_matrix:
              topics, probs, topic_model = BERTopic_model(
                                              docs,
                                              embeddings, 
                                              n_comp        = i, 
                                              n_neigh       = j, 
                                              minS_cluster  = k,
                                              )
              
              silhouette_avg, negative_topics = silhouette_score_calc(embeddings, topics)

              # Count how many documents area noise (topic == -1)
              noise_docs = len([topic for topic in topics if topic == -1])

              # Store the results
              results.append({
                  'n_minCluster':     k,
                  'n_components':     i,
                  'n_neighbors':      j,
                  'silhouette_score': silhouette_avg,
                  'negative_topics':  negative_topics,
                  'noise_docs':       noise_docs
                  })

  # Create a DataFrame with the results
  results = pd.DataFrame(results)
  results = results.sort_values('silhouette_score', ascending=False)

  return results

#================================================================================================
# Function to plot silhouette score
#================================================================================================

def plot_silhouette_score(embedings_x, topics):
  non_noise_indices   = [i for i, topic in enumerate(topics) if topic != -1]
  filtered_embeddings = embedings_x[non_noise_indices]
  filtered_topics     = [topics[i] for i in non_noise_indices]

  silhouette_vals     = silhouette_samples(filtered_embeddings, filtered_topics)

  # Number of clusters
  n_clusters = len(np.unique(filtered_topics))

  # Initialize the plot
  fig, ax = plt.subplots()
  y_lower, y_upper = 0, 0
  yticks = []

  # Iterate over clusters to plot silhouette scores
  for i in range(n_clusters):
      cluster_silhouette_vals = silhouette_vals[np.array(filtered_topics) == i]
      cluster_silhouette_vals.sort()
      y_upper += len(cluster_silhouette_vals)
      ax.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1.0)
      yticks.append((y_lower + y_upper) / 2)
      y_lower += len(cluster_silhouette_vals)

  # Styling the plot
  ax.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
  ax.set_yticks(yticks)
  ax.set_yticklabels(range(n_clusters))
  ax.set_xlabel("Silhouette Coefficient")
  ax.set_ylabel("Cluster")
  ax.set_title("Silhouette Plot")

  return plt.show()