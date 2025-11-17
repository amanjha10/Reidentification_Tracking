#!/usr/bin/env python3
"""
ReID Similarity Analysis Tool
Helps analyze and tune similarity thresholds for the ReID system
"""

import numpy as np
import cv2
import sys
import time
from person_reid import PersonReIDManager

def create_test_images():
    """Create a set of test images with known relationships"""
    images = {}
    
    # Same person - slight variations
    base_image = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    images['person1_frame1'] = base_image.copy()
    
    # Add some noise for person1_frame2
    noise = np.random.randint(-20, 20, base_image.shape, dtype=np.int16)
    images['person1_frame2'] = np.clip(base_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Different person - completely different
    images['person2_frame1'] = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    
    # Very different person
    person3 = np.zeros((256, 128, 3), dtype=np.uint8)
    person3[:, :, 0] = 255  # Pure red
    images['person3_frame1'] = person3
    
    # Another different person
    person4 = np.zeros((256, 128, 3), dtype=np.uint8)
    person4[:, :, 2] = 255  # Pure blue
    images['person4_frame1'] = person4
    
    return images

def analyze_similarities():
    """Analyze similarity patterns to help tune threshold"""
    print("🔍 ReID Similarity Analysis")
    print("=" * 50)
    
    reid_manager = PersonReIDManager(
        embedding_dim=512,
        similarity_threshold=0.5,  # Low threshold for analysis
        ttl_seconds=300
    )
    
    images = create_test_images()
    bbox = [20, 20, 108, 236]  # Standard bbox
    
    # Extract embeddings
    embeddings = {}
    for name, image in images.items():
        emb = reid_manager.extract_embedding(image, bbox)
        if emb is not None:
            embeddings[name] = emb
            print(f"✓ Extracted embedding for {name}")
        else:
            print(f"✗ Failed to extract embedding for {name}")
    
    print("\nSimilarity Matrix:")
    print("-" * 70)
    print(f"{'Image Pair':<30} {'Similarity':<12} {'Expected':<15} {'Status'}")
    print("-" * 70)
    
    # Calculate all pairwise similarities
    similarities = {}
    names = list(embeddings.keys())
    
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i <= j:  # Avoid duplicates
                emb1 = embeddings[name1]
                emb2 = embeddings[name2]
                
                # Calculate cosine similarity
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarities[(name1, name2)] = sim
                
                # Determine expected relationship
                if name1 == name2:
                    expected = "Identical"
                elif name1.split('_')[0] == name2.split('_')[0]:
                    expected = "Same Person"
                else:
                    expected = "Different Person"
                
                # Status based on similarity
                if expected == "Same Person" and sim > 0.8:
                    status = "✓ Good"
                elif expected == "Different Person" and sim < 0.7:
                    status = "✓ Good"
                elif expected == "Identical" and sim > 0.95:
                    status = "✓ Perfect"
                else:
                    status = "⚠ Check"
                
                pair_name = f"{name1} vs {name2}"
                print(f"{pair_name:<30} {sim:<12.3f} {expected:<15} {status}")
    
    print("\nThreshold Recommendations:")
    print("-" * 40)
    
    # Analyze threshold performance
    same_person_sims = [sim for (n1, n2), sim in similarities.items() 
                       if n1 != n2 and n1.split('_')[0] == n2.split('_')[0]]
    diff_person_sims = [sim for (n1, n2), sim in similarities.items() 
                       if n1.split('_')[0] != n2.split('_')[0]]
    
    if same_person_sims and diff_person_sims:
        min_same = min(same_person_sims)
        max_diff = max(diff_person_sims)
        
        print(f"Min similarity (same person): {min_same:.3f}")
        print(f"Max similarity (diff person): {max_diff:.3f}")
        
        if min_same > max_diff:
            optimal_threshold = (min_same + max_diff) / 2
            print(f"✓ Good separation! Optimal threshold: {optimal_threshold:.3f}")
        else:
            print("⚠ Poor separation - consider improving feature extraction")
            optimal_threshold = 0.8
        
        print(f"Conservative threshold (fewer false positives): {max_diff + 0.1:.3f}")
        print(f"Aggressive threshold (fewer false negatives): {min_same - 0.1:.3f}")
    else:
        print("Insufficient data for analysis")
    
    reid_manager.cleanup()
    return similarities

if __name__ == "__main__":
    try:
        similarities = analyze_similarities()
        print("\n🎯 Analysis complete!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
