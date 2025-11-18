# person_reid.py
"""
Person Re-Identification Module
- Lightweight ReID model using ResNet backbone
- ChromaDB for embedding storage and similarity search
- TTL mechanism for embedding expiry
- Production-ready with proper error handling
- Updated with debug logs and relaxed thresholds for better ReID
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
logger.setLevel(logging.DEBUG)  # Enable debug logs

class LightweightReIDModel(nn.Module):
    """Lightweight Person ReID model based on ResNet18"""
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove final classification layer
        self.reid_head = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.reid_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.reid_head(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

class PersonReIDManager:
    """Manages person re-identification with ChromaDB backend"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.80,
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
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((128, 64)),
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
        
        # Thread lock for DB and person ID counter
        self.db_lock = threading.Lock()
        
        # Sequential person ID counter (starts from 1)
        self.person_counter = self._get_next_person_id()
        
        logger.info(f"Person ReID initialized - Device: {self.device}, TTL: {ttl_seconds}s, Threshold: {similarity_threshold}, Next ID: {self.person_counter}")
    
    def _init_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(allow_reset=True)
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB initialized - Collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _get_next_person_id(self) -> int:
        """Get the next sequential person ID by checking existing IDs in ChromaDB"""
        try:
            with self.db_lock:
                # Get all person IDs from the database
                results = self.collection.get(include=['metadatas'])
                if not results['metadatas']:
                    return 1  # Start from 1 if no persons exist
                
                # Extract numeric person IDs and find the highest
                max_id = 0
                for metadata in results['metadatas']:
                    person_id = metadata.get('person_id', '')
                    if person_id.startswith('person_'):
                        try:
                            # Handle both old format (person_timestamp_camera_track) and new format (person_X)
                            parts = person_id.split('_')
                            if len(parts) == 2:
                                # New format: person_X
                                id_num = int(parts[1])
                            elif len(parts) >= 4:
                                # Old format: person_timestamp_camera_track - skip these
                                continue
                            else:
                                continue
                            max_id = max(max_id, id_num)
                        except (ValueError, IndexError):
                            continue
                
                return max_id + 1
        except Exception as e:
            logger.error(f"Failed to get next person ID: {e}")
            return 1  # Fallback to 1
    
    def save_person_snapshot(self, person_crop: np.ndarray, person_id: str, media_config: Dict) -> bool:
        """Save person snapshot to media directory"""
        try:
            if not media_config.get('enable_snapshots', False):
                return False
                
            import os
            
            # Create media directory if it doesn't exist
            media_dir = media_config.get('media_directory', './media/persons')
            os.makedirs(media_dir, exist_ok=True)
            
            # Check minimum size requirements
            min_size = media_config.get('min_snapshot_size', (64, 128))
            if person_crop.shape[1] < min_size[0] or person_crop.shape[0] < min_size[1]:
                logger.debug(f"Person crop too small for snapshot: {person_crop.shape}")
                return False
            
            # Save with timestamp for uniqueness
            timestamp = int(time.time())
            filename = f"{person_id}_{timestamp}.jpg"
            filepath = os.path.join(media_dir, filename)
            
            # Save image with specified quality
            quality = media_config.get('snapshot_quality', 95)
            cv2.imwrite(filepath, person_crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            logger.info(f"Saved person snapshot: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save person snapshot: {e}")
            return False
    
    def crop_person_from_bbox(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Crop person from frame using bounding box"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(x2, w), min(y2, h)
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.shape[0] < 32 or person_crop.shape[1] < 16:
                return None
            return person_crop
        except Exception as e:
            logger.error(f"Failed to crop person: {e}")
            return None
    
    def extract_embedding(self, person_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding from person crop"""
        try:
            if person_crop is None or person_crop.size == 0:
                return None
            if len(person_crop.shape) == 3:
                person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(person_crop)
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model(input_tensor).cpu().numpy().flatten()
            return embedding
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return None
    
    def _calculate_detection_quality(self, person_crop: np.ndarray) -> float:
        """Calculate quality score for detection"""
        try:
            if person_crop is None or person_crop.size == 0:
                return 0.0
            h, w = person_crop.shape[:2]
            size_score = min(1.0, (h * w) / (128 * 256))
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY) if len(person_crop.shape) == 3 else person_crop
            blur_score = min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0)
            brightness_score = 1.0 - abs(np.mean(gray)/255.0 - 0.5)*2
            return min(1.0, max(0.0, size_score*0.4 + blur_score*0.4 + brightness_score*0.2))
        except Exception as e:
            logger.error(f"Error calculating detection quality: {e}")
            return 0.5
    
    def add_person_embedding(self, embedding: np.ndarray, person_id: str, camera: str, track_id: int, additional_metadata: Optional[Dict] = None) -> bool:
        """Add embedding to ChromaDB"""
        try:
            with self.db_lock:
                current_time = time.time()
                metadata = {
                    "person_id": person_id,
                    "timestamp": current_time,
                    "camera": camera,
                    "track_id": track_id,
                    "expires_at": current_time + self.ttl_seconds
                }
                if additional_metadata:
                    metadata.update(additional_metadata)
                self.collection.add(
                    embeddings=[embedding.tolist()],
                    metadatas=[metadata],
                    ids=[f"{person_id}_{current_time}_{camera}_{track_id}"]
                )
            logger.debug(f"Added embedding for person {person_id} (camera {camera})")
            return True
        except Exception as e:
            logger.error(f"Failed to add embedding: {e}")
            return False
    
    def search_similar_person(self, embedding: np.ndarray, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar persons"""
        try:
            with self.db_lock:
                self._cleanup_expired_embeddings()
                count = self.collection.count()
                if count == 0:
                    return []
                results = self.collection.query(
                    query_embeddings=[embedding.tolist()],
                    n_results=min(max_results, count),
                    include=['metadatas', 'distances']
                )
                similar_persons = []
                if results['metadatas'] and results['metadatas'][0]:
                    for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                        similarity = 1.0 - (distance/2.0)
                        if similarity >= self.similarity_threshold:
                            similar_persons.append({
                                "person_id": metadata.get("person_id"),
                                "similarity": similarity,
                                "timestamp": metadata.get("timestamp"),
                                "camera": metadata.get("camera"),
                                "track_id": metadata.get("track_id")
                            })
                return similar_persons
        except Exception as e:
            logger.error(f"Failed to search similar persons: {e}")
            return []
    
    def _cleanup_expired_embeddings(self):
        """Remove expired embeddings"""
        try:
            current_time = time.time()
            all_data = self.collection.get(include=['metadatas'])
            if not all_data['metadatas']:
                return
            expired_ids = []
            for i, metadata in enumerate(all_data['metadatas']):
                if current_time > metadata.get("expires_at", 0):
                    expired_ids.append(all_data['ids'][i])
            if expired_ids:
                self.collection.delete(ids=expired_ids)
                logger.debug(f"Cleaned {len(expired_ids)} expired embeddings")
        except Exception as e:
            logger.error(f"Failed cleanup expired embeddings: {e}")
    
    def store_or_match_embedding(self, frame: np.ndarray, bbox: List[float], camera: str, track_id: int, media_config: Optional[Dict] = None) -> Tuple[bool, Optional[str], float]:
        """Main ReID function - returns (is_known, person_id, similarity)"""
        try:
            logger.debug(f"🔄 ReID processing track {track_id} on {camera}")
            person_crop = self.crop_person_from_bbox(frame, bbox)
            if person_crop is None:
                logger.debug("❌ Person crop invalid, skipping ReID")
                return False, None, 0.0
            quality_score = self._calculate_detection_quality(person_crop)
            logger.debug(f"📊 Detection quality: {quality_score:.3f}")
            if quality_score < 0.35:  # Lowered for more sensitive quality filtering
                logger.debug(f"❌ Skipping low quality detection (quality: {quality_score:.2f})")
                return False, None, 0.0
            embedding = self.extract_embedding(person_crop)
            if embedding is None:
                logger.debug("❌ Failed to extract embedding")
                return False, None, 0.0
            logger.debug(f"✅ Extracted embedding successfully")
            similar_persons = self.search_similar_person(embedding, max_results=3)
            if similar_persons:
                best_match = similar_persons[0]
                person_id = best_match["person_id"]
                similarity = best_match["similarity"]
                logger.info(f"Person matched: {person_id} (similarity {similarity:.3f})")
                self._extend_person_ttl(person_id)
                return True, person_id, similarity
            else:
                # Use sequential person ID (1, 2, 3, ...)
                with self.db_lock:
                    person_id = f"person_{self.person_counter}"
                    self.person_counter += 1
                
                self.add_person_embedding(embedding, person_id, camera, track_id, {"detection_quality": quality_score, "is_canonical": True})
                
                # Save person snapshot if enabled
                if media_config:
                    snapshot_saved = self.save_person_snapshot(person_crop, person_id, media_config)
                    logger.debug(f"📸 Snapshot saved: {snapshot_saved}")
                
                logger.info(f"🟢 NEW person stored: {person_id} (quality {quality_score:.2f})")
                return False, person_id, 0.0
        except Exception as e:
            logger.error(f"Failed store_or_match_embedding: {e}")
            return False, None, 0.0
    
    def _extend_person_ttl(self, person_id: str):
        """Extend TTL without adding new embeddings"""
        try:
            with self.db_lock:
                current_time = time.time()
                new_expiry = current_time + self.ttl_seconds
                # Get IDs separately to avoid ChromaDB API issues
                results_with_ids = self.collection.get(where={"person_id": person_id})
                results = self.collection.get(where={"person_id": person_id}, include=['metadatas', 'embeddings'])
                
                if results['metadatas'] and results_with_ids.get('ids'):
                    for i, metadata in enumerate(results['metadatas']):
                        metadata['expires_at'] = new_expiry
                        metadata['last_seen'] = current_time
                        embedding_data = results['embeddings'][i]
                        embedding_id = results_with_ids['ids'][i]
                        self.collection.delete(ids=[embedding_id])
                        self.collection.add(embeddings=[embedding_data], metadatas=[metadata], ids=[embedding_id])
                logger.debug(f"Extended TTL for person {person_id}")
        except Exception as e:
            logger.error(f"Failed to extend TTL: {e}")
    
    def get_unique_person_count(self) -> int:
        """Get count of unique persons (not total embeddings)"""
        try:
            with self.db_lock:
                self._cleanup_expired_embeddings()
                results = self.collection.get(include=['metadatas'])
                
                if not results['metadatas']:
                    return 0
                
                unique_persons = set()
                current_time = time.time()
                
                for metadata in results['metadatas']:
                    expires_at = metadata.get('expires_at', 0)
                    if current_time <= expires_at:
                        person_id = metadata.get('person_id')
                        if person_id:
                            unique_persons.add(person_id)
                
                return len(unique_persons)
        except Exception as e:
            logger.error(f"Failed to get unique person count: {e}")
            return 0
    
    def get_database_stats(self) -> dict:
        """Get database statistics"""
        try:
            with self.db_lock:
                count = self.collection.count()
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
    
    def get_stats(self) -> dict:
        """Get comprehensive ReID statistics"""
        try:
            db_stats = self.get_database_stats()
            unique_count = self.get_unique_person_count()
            
            return {
                'unique_persons': unique_count,
                'total_embeddings': db_stats.get('active_embeddings', 0),
                'active_persons': unique_count,
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
    
    def reset_database(self):
        """Reset the entire database"""
        try:
            with self.db_lock:
                self.chroma_client.reset()
                self._init_chromadb()
                logger.info("Database reset successfully")
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            with self.db_lock:
                self._cleanup_expired_embeddings()
                logger.info("ReID Manager cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
