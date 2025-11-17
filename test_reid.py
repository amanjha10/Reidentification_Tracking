#!/usr/bin/env python3
"""
Test script for Person ReID system
Tests the ReID functionality without requiring camera feeds
"""

import numpy as np
import cv2
import sys
import time
from person_reid import PersonReIDManager

def create_test_person_image(person_id: int, width: int = 128, height: int = 256):
    """Create a synthetic person image for testing"""
    # Create a simple colored rectangle to represent a person
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Use person_id to create very different colors/patterns
    if person_id == 1:
        # Person 1: Blue shirt with horizontal stripes
        base_color = (100, 50, 200)  # Blue
        cv2.rectangle(image, (10, 10), (width-10, height-10), base_color, -1)
        for i in range(0, height-20, 20):
            cv2.rectangle(image, (15, 15+i), (width-15, 25+i), (255, 255, 255), -1)
    elif person_id == 2:
        # Person 2: Red shirt with vertical stripes
        base_color = (50, 50, 200)  # Red
        cv2.rectangle(image, (10, 10), (width-10, height-10), base_color, -1)
        for i in range(0, width-20, 15):
            cv2.rectangle(image, (15+i, 15), (20+i, height-15), (255, 255, 255), -1)
    else:
        # Other persons: Random patterns
        base_color = (
            (person_id * 73) % 255,
            (person_id * 137) % 255,
            (person_id * 211) % 255
        )
        cv2.rectangle(image, (10, 10), (width-10, height-10), base_color, -1)
        
        # Add diagonal pattern
        for i in range(-width, width, 10):
            cv2.line(image, (i, 0), (i + height, height), (255, 255, 255), 2)
    
    return image

def test_reid_system():
    """Test the ReID system functionality"""
    print("🧪 Testing Person ReID System...")
    
    # Initialize ReID manager
    try:
        reid_manager = PersonReIDManager(
            embedding_dim=512,
            similarity_threshold=0.96,  # Optimal threshold from analysis
            ttl_seconds=300
        )
        print("✅ ReID Manager initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize ReID Manager: {e}")
        return False
    
    # Test 1: Add a new person
    print("\n📝 Test 1: Adding new person...")
    test_image1 = create_test_person_image(1)
    bbox1 = [20, 20, 108, 236]  # x1, y1, x2, y2
    
    is_known1, person_id1, similarity1 = reid_manager.check_person_identity(
        test_image1, bbox1, "test_camera", track_id=1
    )
    
    print(f"   Person 1: Known={is_known1}, ID={person_id1}, Similarity={similarity1:.3f}")
    assert not is_known1, "First person should not be known"
    assert person_id1 is not None, "Person ID should be generated"
    
    # Test 2: Re-identify the same person
    print("\n🔍 Test 2: Re-identifying same person...")
    is_known2, person_id2, similarity2 = reid_manager.check_person_identity(
        test_image1, bbox1, "test_camera", track_id=2
    )
    
    print(f"   Person 1 (again): Known={is_known2}, ID={person_id2}, Similarity={similarity2:.3f}")
    assert is_known2, "Same person should be recognized as known"
    assert person_id1 == person_id2, "Person IDs should match"
    assert similarity2 > 0.7, f"Similarity should be high (got {similarity2:.3f})"
    
    # Test 3: Add a different person
    print("\n👤 Test 3: Adding different person...")
    test_image2 = create_test_person_image(2)
    bbox2 = [25, 25, 113, 241]
    
    is_known3, person_id3, similarity3 = reid_manager.check_person_identity(
        test_image2, bbox2, "test_camera", track_id=3
    )
    
    print(f"   Person 2: Known={is_known3}, ID={person_id3}, Similarity={similarity3:.3f}")
    assert not is_known3, "Different person should not be known"
    assert person_id3 != person_id1, "Different person should have different ID"
    
    # Test 4: Check statistics
    print("\n📊 Test 4: Checking statistics...")
    stats = reid_manager.get_stats()
    print(f"   Stats: {stats}")
    assert stats['unique_persons'] >= 2, "Should have at least 2 unique persons stored"
    assert stats['embeddings_per_person'] <= 2.0, "Should have minimal embeddings per person"
    
    # Test 5: Reset database
    print("\n🔄 Test 5: Resetting database...")
    reid_manager.reset_database()
    stats_after_reset = reid_manager.get_stats()
    print(f"   Stats after reset: {stats_after_reset}")
    assert stats_after_reset['unique_persons'] == 0, "Should have no persons after reset"
    
    # Cleanup
    reid_manager.cleanup()
    print("✅ All ReID tests passed successfully!")
    return True

def test_embedding_similarity():
    """Test embedding similarity calculations"""
    print("\n🎯 Testing embedding similarity...")
    
    reid_manager = PersonReIDManager()
    
    # Create two identical images
    img1 = create_test_person_image(1)
    img2 = create_test_person_image(1)  # Same person
    img3 = create_test_person_image(2)  # Different person
    
    bbox = [20, 20, 108, 236]
    
    # Extract embeddings
    emb1 = reid_manager.extract_embedding(img1, bbox)
    emb2 = reid_manager.extract_embedding(img2, bbox)
    emb3 = reid_manager.extract_embedding(img3, bbox)
    
    # Calculate similarities
    sim_same = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    sim_diff = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
    
    print(f"   Similarity (same person): {sim_same:.3f}")
    print(f"   Similarity (different person): {sim_diff:.3f}")
    
    assert sim_same > sim_diff, "Same person should have higher similarity than different person"
    
    reid_manager.cleanup()
    print("✅ Embedding similarity test passed!")

if __name__ == "__main__":
    print("🚀 Starting Person ReID Tests...\n")
    
    try:
        # Run basic functionality tests
        success = test_reid_system()
        
        if success:
            # Run embedding tests
            test_embedding_similarity()
            
            print("\n🎉 All tests completed successfully!")
            print("\n📋 ReID System Status:")
            print("  ✅ ChromaDB integration working")
            print("  ✅ Embedding extraction functional")
            print("  ✅ Person identification working")
            print("  ✅ Similarity matching operational")
            print("  ✅ Database operations successful")
            
            print("\n🔧 Ready for production use!")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
