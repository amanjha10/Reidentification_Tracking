# person_reid.py
"""
Person Re-Identification Module
- Lightweight ReID model using ResNet backbone
- ChromaDB for embedding storage and similarity search
- TTL mechanism for embedding expiry
- Production-ready with proper error handling
"""

import time
import logging
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import chromadb
from chromadb.config import Settings
from PIL import Image
import threading

logger = logging.getLogger("person_reid")

class LightweightReIDModel(nn.Module):
    """Lightweight Person ReID model based on ResNet18"""
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Add custom ReID head
        self.reid_head = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.reid_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.reid_head(features)
        # L2 normalize embeddings
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

class PersonReIDManager:
    """Manages person re-identification with ChromaDB backend"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.55,
                 embedding_dim: int = 512,
                 ttl_seconds: int = 300,  # 5 minutes default TTL
                 collection_name: str = "person_embeddings",
                 db_path: str = "./chroma_db"):
        
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = embedding_dim
        self.ttl_seconds = ttl_seconds
        self.collection_name = collection_name
        self.db_path = db_path
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LightweightReIDModel(embedding_dim).to(self.device)
        self.model.eval()
        
        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((128, 64)),  # Standard ReID input size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Initialize ChromaDB
        self.chroma_client = None
        self.collection = None
        self._init_chromadb()
        
        # Thread lock for database operations
        self.db_lock = threading.Lock()
        
        logger.info(f"Person ReID initialized - Device: {self.device}, TTL: {ttl_seconds}s")
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(allow_reset=True)
            )
            
            # Create or get collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            logger.info(f"ChromaDB initialized - Collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def extract_embedding(self, person_crop: np.ndarray, bbox: Optional[List[float]] = None) -> Optional[np.ndarray]:
        """Extract embedding from person crop"""
        try:
            if person_crop is None or person_crop.size == 0:
                return None
            
            # Ensure minimum size for meaningful features
            h, w = person_crop.shape[:2]
            if h < 32 or w < 16:
                return None
            
            # Convert BGR to RGB
            if len(person_crop.shape) == 3:
                person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(person_crop)
            
            # Apply transforms
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model(input_tensor)
                embedding = embedding.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return None
    
    def crop_person_from_bbox(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Crop person from frame using bounding box"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure valid coordinates
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Crop person
            person_crop = frame[y1:y2, x1:x2]
            
            # Ensure minimum size
            if person_crop.shape[0] < 32 or person_crop.shape[1] < 16:
                return None
                
            return person_crop
            
        except Exception as e:
            logger.error(f"Failed to crop person: {e}")
            return None
    
    def search_similar_person(self, embedding: np.ndarray, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar persons in the database"""
        try:
            with self.db_lock:
                # Clean expired embeddings first
                self._cleanup_expired_embeddings()
                
                # Get collection count
                count = self.collection.count()
                if count == 0:
                    return []
                
                # Search for similar embeddings
                results = self.collection.query(
                    query_embeddings=[embedding.tolist()],
                    n_results=min(max_results, count),
                    include=['metadatas', 'distances']
                )
                
                # Process results
                similar_persons = []
                if results['metadatas'] and results['metadatas'][0]:
                    for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
                        # ChromaDB uses cosine distance (0 = identical, 2 = opposite)
                        # Convert to cosine similarity: similarity = 1 - (distance / 2)
                        similarity = 1.0 - (distance / 2.0)
                        
                        if similarity >= self.similarity_threshold:
                            similar_persons.append({
                                'person_id': metadata.get('person_id'),
                                'similarity': similarity,
                                'timestamp': metadata.get('timestamp'),
                                'camera': metadata.get('camera'),
                                'track_id': metadata.get('track_id')
                            })
                
                return similar_persons
                
        except Exception as e:
            logger.error(f"Failed to search similar persons: {e}")
            return []
    
    def add_person_embedding(self, 
                           embedding: np.ndarray,
                           person_id: str,
                           camera: str,
                           track_id: int,
                           additional_metadata: Optional[Dict] = None) -> bool:
        """Add person embedding to the database"""
        try:
            with self.db_lock:
                current_time = time.time()
                
                metadata = {
                    'person_id': person_id,
                    'timestamp': current_time,
                    'camera': camera,
                    'track_id': track_id,
                    'expires_at': current_time + self.ttl_seconds
                }
                
                if additional_metadata:
                    metadata.update(additional_metadata)
                
                # Add to ChromaDB
                self.collection.add(
                    embeddings=[embedding.tolist()],
                    metadatas=[metadata],
                    ids=[f"{person_id}_{current_time}_{camera}_{track_id}"]
                )
                
                logger.debug(f"Added embedding for person {person_id} from camera {camera}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to add person embedding: {e}")
            return False
    
    def _cleanup_expired_embeddings(self):
        """Remove expired embeddings from the database"""
        try:
            current_time = time.time()
            
            # Get all embeddings
            all_data = self.collection.get(include=['metadatas'])
            
            if not all_data['metadatas']:
                return
            
            # Find expired IDs
            expired_ids = []
            for i, metadata in enumerate(all_data['metadatas']):
                expires_at = metadata.get('expires_at', 0)
                if current_time > expires_at:
                    expired_ids.append(all_data['ids'][i])
            
            # Delete expired embeddings
            if expired_ids:
                self.collection.delete(ids=expired_ids)
                logger.debug(f"Cleaned up {len(expired_ids)} expired embeddings")
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired embeddings: {e}")
    
    def check_person_identity(self, 
                            frame: np.ndarray, 
                            bbox: List[float],
                            camera: str,
                            track_id: int) -> Tuple[bool, Optional[str], float]:
        """
        Check if person has been seen before
        
        Returns:
            (is_known_person, person_id, max_similarity)
        """
        try:
            # Crop person from frame
            person_crop = self.crop_person_from_bbox(frame, bbox)
            if person_crop is None:
                return False, None, 0.0
            
            # Extract embedding
            embedding = self.extract_embedding(person_crop)
            if embedding is None:
                return False, None, 0.0
            
            # Search for similar persons
            similar_persons = self.search_similar_person(embedding)
            
            if similar_persons:
                # Get the most similar person
                best_match = max(similar_persons, key=lambda x: x['similarity'])
                person_id = best_match['person_id']
                similarity = best_match['similarity']
                
                logger.info(f"Found similar person: {person_id} (similarity: {similarity:.3f})")
                
                # Add current embedding to the database for this person
                self.add_person_embedding(embedding, person_id, camera, track_id)
                
                return True, person_id, similarity
            else:
                # New person - generate unique ID
                person_id = f"person_{int(time.time())}_{camera}_{track_id}"
                
                # Add to database
                self.add_person_embedding(embedding, person_id, camera, track_id)
                
                logger.info(f"New person detected: {person_id}")
                return False, person_id, 0.0
                
        except Exception as e:
            logger.error(f"Failed to check person identity: {e}")
            return False, None, 0.0
    
    def store_or_match_embedding(self, 
                               frame: np.ndarray, 
                               bbox: List[float],
                               camera: str,
                               track_id: int) -> Tuple[bool, Optional[str], float]:
        """
        OPTIMIZED: Smart embedding storage with minimal database growth
        
        Returns:
            (is_known_person, person_id, max_similarity)
        """
        try:
            # Crop person from frame
            person_crop = self.crop_person_from_bbox(frame, bbox)
            if person_crop is None:
                return False, None, 0.0
            
            # Check detection quality - only process high quality detections
            quality_score = self._calculate_detection_quality(person_crop)
            if quality_score < 0.6:  # Skip low quality detections
                logger.debug(f"Skipping low quality detection (quality: {quality_score:.2f})")
                return False, None, 0.0
            
            # Extract embedding
            embedding = self.extract_embedding(person_crop, bbox)
            if embedding is None:
                return False, None, 0.0
            
            # STEP 1: Search for similar persons FIRST (before storing anything)
            similar_persons = self.search_similar_person(embedding, max_results=3)
            
            if similar_persons:
                # Found similar person(s) - return the best match
                best_match = similar_persons[0]  # Highest similarity first
                person_id = best_match['person_id']
                similarity = best_match['similarity']
                
                logger.info(f"Person matched: {person_id} (similarity: {similarity:.3f})")
                
                # DON'T store additional embeddings for known persons
                # Just extend TTL without adding new embeddings
                self._extend_person_ttl(person_id)
                
                return True, person_id, similarity
            else:
                # STEP 2: No similar person found - create NEW identity
                person_id = f"person_{int(time.time())}_{camera}_{track_id}"
                
                # Store ONLY ONE high-quality embedding per person
                success = self.add_person_embedding(embedding, person_id, camera, track_id, {
                    'bbox_width': bbox[2] - bbox[0],
                    'bbox_height': bbox[3] - bbox[1],
                    'detection_quality': quality_score,
                    'is_canonical': True  # Mark as the canonical embedding for this person
                })
                
                if success:
                    logger.info(f"New person stored: {person_id} (quality: {quality_score:.2f})")
                    return False, person_id, 0.0
                else:
                    logger.error(f"Failed to store new person embedding")
                    return False, None, 0.0
                
        except Exception as e:
            logger.error(f"Failed in store_or_match_embedding: {e}")
            return False, None, 0.0
    
    def _extend_person_ttl(self, person_id: str):
        """Extend TTL for existing person WITHOUT adding new embeddings"""
        try:
            with self.db_lock:
                current_time = time.time()
                new_expiry = current_time + self.ttl_seconds
                
                # Get all embeddings for this person
                results = self.collection.get(
                    where={"person_id": person_id},
                    include=['metadatas']
                )
                
                if results['metadatas']:
                    # Update expiry time for existing embeddings
                    for i, metadata in enumerate(results['metadatas']):
                        # Update the metadata in place
                        metadata['expires_at'] = new_expiry
                        metadata['last_seen'] = current_time
                        
                        # ChromaDB requires delete and re-add for updates
                        embedding_id = results['ids'][i]
                        
                        # Get the actual embedding
                        embedding_data = self.collection.get(
                            ids=[embedding_id],
                            include=['embeddings']
                        )
                        
                        if embedding_data['embeddings']:
                            # Delete old entry
                            self.collection.delete(ids=[embedding_id])
                            
                            # Re-add with updated metadata
                            self.collection.add(
                                embeddings=[embedding_data['embeddings'][0]],
                                metadatas=[metadata],
                                ids=[embedding_id]
                            )
                
                logger.debug(f"Extended TTL for person {person_id}")
                
        except Exception as e:
            logger.error(f"Failed to extend person TTL: {e}")
    
    def _calculate_detection_quality(self, person_crop: np.ndarray) -> float:
        """Calculate quality score for person detection (0-1)"""
        try:
            if person_crop is None or person_crop.size == 0:
                return 0.0
            
            # Simple quality metrics
            h, w = person_crop.shape[:2]
            size_score = min(1.0, (h * w) / (128 * 256))  # Normalize to standard size
            
            # Check for blur (Laplacian variance)
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY) if len(person_crop.shape) == 3 else person_crop
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0  # Normalize
            blur_score = min(1.0, blur_score)
            
            # Brightness check
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Prefer mid-range brightness
            
            # Combined quality score
            quality = (size_score * 0.4 + blur_score * 0.4 + brightness_score * 0.2)
            return min(1.0, max(0.0, quality))
            
        except Exception as e:
            logger.error(f"Error calculating detection quality: {e}")
            return 0.5  # Default middle quality
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.db_lock:
                count = self.collection.count()
                
                # Clean expired first to get accurate count
                self._cleanup_expired_embeddings()
                active_count = self.collection.count()
                
                return {
                    'total_embeddings': count,
                    'active_embeddings': active_count,
                    'expired_cleaned': count - active_count,
                    'similarity_threshold': self.similarity_threshold,
                    'ttl_seconds': self.ttl_seconds
                }
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def reset_database(self):
        """Reset the entire database (for testing)"""
        try:
            with self.db_lock:
                self.chroma_client.reset()
                self._init_chromadb()
                logger.info("Database reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
    
    def cleanup(self):
        """Cleanup resources and close database connections"""
        try:
            with self.db_lock:
                # Clean expired embeddings one last time
                self._cleanup_expired_embeddings()
                
                # Close ChromaDB client if needed
                if self.chroma_client:
                    # ChromaDB doesn't need explicit closing, but we can log
                    logger.info("ReID Manager cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_unique_person_count(self) -> int:
        """Get count of unique persons (not total embeddings)"""
        try:
            with self.db_lock:
                # Clean expired embeddings first
                self._cleanup_expired_embeddings()
                
                # Get all active embeddings
                results = self.collection.get(include=['metadatas'])
                
                if not results['metadatas']:
                    return 0
                
                # Count unique person IDs
                unique_persons = set()
                current_time = time.time()
                
                for metadata in results['metadatas']:
                    expires_at = metadata.get('expires_at', 0)
                    if current_time <= expires_at:  # Still active
                        person_id = metadata.get('person_id')
                        if person_id:
                            unique_persons.add(person_id)
                
                return len(unique_persons)
                
        except Exception as e:
            logger.error(f"Failed to get unique person count: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive ReID statistics"""
        try:
            db_stats = self.get_database_stats()
            unique_count = self.get_unique_person_count()
            
            return {
                'unique_persons': unique_count,  # This is the important metric
                'total_embeddings': db_stats.get('active_embeddings', 0),
                'active_persons': unique_count,  # Same as unique_persons
                'similarity_threshold': self.similarity_threshold,
                'ttl_seconds': self.ttl_seconds,
                'embedding_dim': self.embedding_dim,
                'embeddings_per_person': db_stats.get('active_embeddings', 0) / max(1, unique_count)
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                'unique_persons': 0,
                'total_embeddings': 0,
                'active_persons': 0,
                'similarity_threshold': self.similarity_threshold,
                'ttl_seconds': self.ttl_seconds,
                'embedding_dim': self.embedding_dim,
                'embeddings_per_person': 0
            }
