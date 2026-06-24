import numpy as np
import os
import json

class SmartOverrideEngine:
    def __init__(self, storage_path="custom_memory.json", threshold=0.35):
        self.storage_path = storage_path
        self.threshold = threshold  # Strictness: lower means it must be a closer match
        self.memory_bank = self._load_memory()

    def _load_memory(self):
        """Loads saved manual overrides from the JSON file."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                return {label: [np.array(v) for v in vectors] for label, vectors in data.items()}
        return {}

    def _save_memory(self):
        """Saves custom overrides to disk so they don't erase when you turn off the app."""
        serializable_bank = {label: [v.tolist() for v in vectors] for label, vectors in self.memory_bank.items()}
        with open(self.storage_path, 'w') as f:
            json.dump(serializable_bank, f)

    def extract_embeddings(self, frame):
        """
        Extracts the unique mathematical fingerprint of the image.
        Placeholder: Replace this line with your actual model's feature extraction layer.
        """
        rng = np.random.default_rng(seed=hash(frame.tobytes()) % (2**32))
        mock_vector = rng.random(128)
        return mock_vector / np.linalg.norm(mock_vector)

    def learn_override(self, frame, user_assigned_label):
        """Memorizes the current visual blueprint and attaches it to your manual label."""
        embedding = self.extract_embeddings(frame)
        
        if user_assigned_label not in self.memory_bank:
            self.memory_bank[user_assigned_label] = []
            
        self.memory_bank[user_assigned_label].append(embedding)
        self._save_memory()
        return f"Successfully locked in '{user_assigned_label}' blueprint."

    def classify(self, frame, base_model_prediction):
        """Checks the manual memory bank first. Falls back to base model if no match."""
        current_embedding = self.extract_embeddings(frame)
        best_match_label = None
        smallest_distance = float('inf')

        for label, saved_embeddings in self.memory_bank.items():
            for saved_vector in saved_embeddings:
                distance = np.linalg.norm(current_embedding - saved_vector)
                if distance < smallest_distance:
                    smallest_distance = distance
                    best_match_label = label

        # If the object perfectly matches something you manually taught it, override!
        if smallest_distance < self.threshold:
            return {
                "prediction": best_match_label,
                "is_override": True,
                "confidence_score": float(1.0 - smallest_distance)
            }

        # Otherwise, let your regular model run completely normally
        return {
            "prediction": base_model_prediction,
            "is_override": False,
            "confidence_score": 0.85
        }
