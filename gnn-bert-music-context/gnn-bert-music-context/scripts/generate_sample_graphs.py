"""
Preprocessed Graph Samples Generator
Generates and exports 20 sample .pt and .json music structure graphs for assignment deliverable #2.
"""

import os
import torch
import numpy as np
from src.graph_builder import MusicGraphBuilder

def generate_samples(output_dir: str = "data/processed", num_samples: int = 20):
    os.makedirs(output_dir, exist_ok=True)
    builder = MusicGraphBuilder(similarity_threshold=0.65, add_self_loops=True)
    
    genres = ["Electronic", "Folk", "Rock", "Pop", "Classical", "Jazz", "Hip-Hop", "Instrumental"]
    print(f"Generating {num_samples} sample music graphs in {output_dir}...")
    
    for i in range(1, num_samples + 1):
        track_id = f"track_{i:04d}"
        num_segments = np.random.randint(8, 15) # 8 to 14 temporal segments
        feat_dim = 39 # 12 chroma + 20 mfcc + 7 spectral contrast
        
        # Simulate structured music features with repeating sections (e.g. verse-chorus-verse)
        base_verse = np.random.randn(feat_dim).astype(np.float32)
        base_chorus = np.random.randn(feat_dim).astype(np.float32)
        
        node_features = []
        for s in range(num_segments):
            if s % 2 == 0:
                feat = base_verse + np.random.randn(feat_dim) * 0.1
            else:
                feat = base_chorus + np.random.randn(feat_dim) * 0.1
            node_features.append(feat)
            
        node_features = np.array(node_features, dtype=np.float32)
        genre_id = i % len(genres)
        label = torch.tensor([genre_id], dtype=torch.long)
        
        # Build PyG Data object
        graph = builder.build_graph_from_features(node_features, label=label, track_id=track_id)
        
        # Save .pt and .json
        pt_path = os.path.join(output_dir, f"{track_id}.pt")
        json_path = os.path.join(output_dir, f"{track_id}.json")
        builder.save_graph_pt(graph, pt_path)
        builder.export_graph_json(graph, json_path)
        
    print(f"Successfully generated {num_samples} .pt and .json graphs.")

if __name__ == "__main__":
    generate_samples()
