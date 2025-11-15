"""behavior_classifier.py - Behavior classification for person tracking"""


class BehaviorClassifier:
    """Placeholder classifier for person behavior (walking, running, standing, etc).
    
    Implement your own trained model here.
    """
    
    def __init__(self):
        pass

    def predict(self, seq):
        """Predict behavior from sequence of crops or keypoints.
        
        Args:
            seq: sequence of frames/keypoints/features
            
        Returns:
            str: behavior label (e.g., 'walking', 'running', 'standing')
        """
        return 'unknown'
