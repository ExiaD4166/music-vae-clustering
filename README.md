# Unsupervised & Multimodal Music Clustering with VAEs

This project explores VAE-based latent representations for clustering a custom hybrid music dataset (English + Bangla clips).  
We implement three stages: Easy, Medium, and Hard tasks, and compare VAE clustering against classical baselines.

## Dataset (Custom)
- Created by collecting mp3 songs from multiple genres
- Trimmed into 20-second clips
- Lyrics generated per clip and stored with metadata (clip_name, lyrics, genre)
- Total clips: 3368

Dataset link : https://drive.google.com/drive/folders/1WqemBds8_gHtxfVfOY9dvTxxUfxZ9jXF

## Project Structure
- `notebooks/`  
  Colab notebooks for Easy/Medium/Hard tasks and evaluations
- `results/`
  - `results_easy/` metrics + t-SNE plots
  - `results_medium/` metrics + DBSCAN diagnostics + t-SNE plots
  - `results_hard/` metrics + baseline comparisons + t-SNE plots


## Tasks Implemented
### Easy Task
- Basic VAE on MFCC features
- K-Means clustering on latent space
- t-SNE visualization
- Baseline: PCA + KMeans (Silhouette, Calinski–Harabasz)

### Medium Task
- Multimodal ConvVAE (audio spectrogram + lyric embeddings)
- Clustering: KMeans, Agglomerative, DBSCAN
- Metrics: Silhouette, Davies–Bouldin, ARI
- DBSCAN eps selection via k-distance curve

### Hard Task
- Beta-VAE for disentangled latent representations
- Multi-modal clustering (audio + lyrics + genre info)
- Metrics: Silhouette, NMI, ARI, Purity
- Baselines: PCA+KMeans, AE+KMeans, Direct MFCC+KMeans
- Visualizations: t-SNE plots, cluster composition, recon examples

## How to Run
1. Open notebooks in Google Colab
2. Install dependencies if needed:
   ```bash
   pip install numpy pandas scikit-learn torch librosa matplotlib

