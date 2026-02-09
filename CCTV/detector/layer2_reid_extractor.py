import torch
import torchreid
import cv2
import numpy as np


class ReIDExtractor:
    def __init__(self, model_name='osnet_x0_25'):
        """
        Initialize ReID model for 128D feature extraction
        
        FIXED: Changed from osnet_x1_0 (512D) to osnet_x0_25 (128D)
        
        Benefits of 128D:
        - Faster similarity comparison
        - Lower database storage cost
        - Sufficient accuracy for body-based ReID in indoor environments
        
        model_name: Default 'osnet_x0_25' produces 128D features
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ReID] Using device: {self.device}")
        
        # Load pretrained model - MUST use osnet_x0_25 for 128D features
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,  # dummy, we only need features
            pretrained=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Verify feature dimension is 128D
        if 'x0_25' in model_name:
            feature_dim = 128
        elif 'x0_5' in model_name:
            feature_dim = 256
        else:
            feature_dim = 512
        
        if feature_dim != 128:
            print(f"[ReID] ⚠️ WARNING: Model {model_name} produces {feature_dim}D features, expected 128D!")
            print(f"[ReID] ⚠️ Please use 'osnet_x0_25' for 128D features")
        
        print(f"[ReID] ✓ Loaded model: {model_name} (Feature dimension: {feature_dim}D)")

    def extract_features(self, frame, bbox):
        """
        Extract 128-D feature vector from person crop (using osnet_x0_25)
        
        Args:
            frame: Full frame (numpy array)
            bbox: [x1, y1, x2, y2]
        
        Returns:
            feature: 128-D numpy array (normalized)
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Crop person from frame
        person_crop = frame[y1:y2, x1:x2]
        
        if person_crop.size == 0:
            return None
        
        # Resize to 256x128 (standard ReID input)
        person_crop = cv2.resize(person_crop, (128, 256))
        
        # Normalize (ImageNet stats)
        person_crop = person_crop.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        person_crop = (person_crop - mean) / std
        
        # Convert to tensor (C, H, W) - ensure float32
        person_tensor = torch.from_numpy(person_crop).permute(2, 0, 1).unsqueeze(0).float()
        person_tensor = person_tensor.to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(person_tensor)
            features = features.cpu().numpy().flatten()
        
        # Normalize to unit vector
        features = features / (np.linalg.norm(features) + 1e-12)
        
        return features

    def compare_features(self, feat1, feat2):
        """
        Compute cosine similarity between two feature vectors
        Returns similarity score (0 to 1, higher = more similar)
        """
        if feat1 is None or feat2 is None:
            return 0.0
        
        similarity = np.dot(feat1, feat2)
        return float(similarity)
