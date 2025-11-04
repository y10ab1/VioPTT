"""
Modular Technique Processing for Piano Transcription

This module provides a clean, modular interface for integrating technique predictions
with piano transcription systems.
"""

import numpy as np
from typing import Optional, Dict, List, Union, Tuple
from enum import Enum


class TechniqueStrategy(Enum):
    """Technique assignment strategies."""
    MAJORITY = "majority"
    PEAK = "peak" 
    ONSET = "onset"


class TechniqueProcessor:
    """
    Modular processor for handling technique predictions in piano transcription.
    
    This class provides a clean interface for:
    1. Processing frame-wise technique predictions
    2. Assigning techniques to note events
    3. Converting between different technique formats
    """
    
    def __init__(self, 
                 strategy: TechniqueStrategy = TechniqueStrategy.MAJORITY,
                 confidence_threshold: float = 0.6,
                 technique_labels: Optional[Dict[int, str]] = None):
        """
        Initialize technique processor.
        
        Args:
            strategy: Technique assignment strategy
            confidence_threshold: Minimum confidence for technique assignment
            technique_labels: Optional mapping of technique IDs to names
        """
        self.strategy = strategy
        self.confidence_threshold = confidence_threshold
        self.technique_labels = technique_labels or self._default_technique_labels()
        
    def _default_technique_labels(self) -> Dict[int, str]:
        """Default technique label mapping."""
        return {
            0: "legato",
            1: "pizzicato", 
            2: "spiccato",
            3: "sustain",
            -1: "unknown"
        }
    
    def process_technique_output(self, technique_output: np.ndarray) -> np.ndarray:
        """
        Normalize technique output to probabilities.
        
        Args:
            technique_output: Raw technique predictions (logits or probabilities)
                Shape: (frames_num, num_techniques) or (frames_num, classes_num, num_techniques)
                
        Returns:
            Normalized probabilities with same shape as input
        """
        if technique_output is None:
            return None
            
        # Convert logits to probabilities if needed
        if technique_output.max() > 1.0 or technique_output.min() < 0.0:
            # Likely logits, apply softmax along last dimension
            axis = -1
            exp_scores = np.exp(technique_output - np.max(technique_output, axis=axis, keepdims=True))
            probabilities = exp_scores / np.sum(exp_scores, axis=axis, keepdims=True)
        else:
            # Already probabilities
            probabilities = technique_output
            
        return probabilities
    
    def assign_technique_to_note(self, 
                                bgn_frame: int, 
                                fin_frame: int,
                                technique_probs: Optional[np.ndarray]) -> int:
        """
        Assign technique label to a single note based on frame-wise predictions.
        
        Args:
            bgn_frame: Note onset frame
            fin_frame: Note offset frame  
            technique_probs: Technique probabilities 
                Shape: (frames_num, num_techniques)
                
        Returns:
            technique_id: Predicted technique class (-1 if unknown/low confidence)
        """
        if technique_probs is None:
            return -1
            
        # Extract predictions for note duration
        note_duration_probs = technique_probs[bgn_frame:fin_frame+1]
        
        if note_duration_probs.shape[0] == 0:
            return -1
            
        # Apply assignment strategy
        if self.strategy == TechniqueStrategy.MAJORITY:
            return self._majority_voting(note_duration_probs)
        elif self.strategy == TechniqueStrategy.PEAK:
            return self._peak_assignment(note_duration_probs)
        elif self.strategy == TechniqueStrategy.ONSET:
            return self._onset_assignment(note_duration_probs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _majority_voting(self, probs: np.ndarray) -> int:
        """Majority voting: most frequent prediction across frames."""
        frame_predictions = np.argmax(probs, axis=1)
        technique_counts = np.bincount(frame_predictions)
        technique_id = np.argmax(technique_counts)
        
        # Check confidence
        confidence = technique_counts[technique_id] / len(frame_predictions)
        return int(technique_id) if confidence >= self.confidence_threshold else -1
    
    def _peak_assignment(self, probs: np.ndarray) -> int:
        """Peak assignment: technique with highest average probability."""
        avg_probs = np.mean(probs, axis=0)
        technique_id = np.argmax(avg_probs)
        
        # Check confidence
        confidence = avg_probs[technique_id]
        return int(technique_id) if confidence >= self.confidence_threshold else -1
    
    def _onset_assignment(self, probs: np.ndarray) -> int:
        """Onset-based: use technique prediction at onset frame only."""
        onset_probs = probs[0]
        technique_id = np.argmax(onset_probs)
        
        # Check confidence
        confidence = onset_probs[technique_id]
        return int(technique_id) if confidence >= self.confidence_threshold else -1
    
    def add_techniques_to_note_events(self, 
                                     note_events: List[Dict],
                                     technique_assignments: List[int]) -> List[Dict]:
        """
        Add technique labels to existing note events.
        
        Args:
            note_events: List of note event dictionaries
            technique_assignments: List of technique IDs (same length as note_events)
            
        Returns:
            Enhanced note events with technique field
        """
        if len(note_events) != len(technique_assignments):
            raise ValueError("Mismatch between note_events and technique_assignments length")
            
        enhanced_events = []
        for event, technique_id in zip(note_events, technique_assignments):
            enhanced_event = event.copy()
            enhanced_event['technique'] = technique_id
            enhanced_events.append(enhanced_event)
            
        return enhanced_events
    
    def get_technique_statistics(self, note_events: List[Dict]) -> Dict:
        """
        Compute statistics about technique usage.
        
        Args:
            note_events: List of note events with technique field
            
        Returns:
            Dictionary with technique statistics
        """
        if not note_events:
            return {}
            
        techniques = [event.get('technique', -1) for event in note_events]
        technique_counts = {}
        
        for technique_id in techniques:
            label = self.technique_labels.get(technique_id, f"Unknown_{technique_id}")
            technique_counts[label] = technique_counts.get(label, 0) + 1
            
        total_notes = len(note_events)
        technique_percentages = {
            label: (count / total_notes) * 100 
            for label, count in technique_counts.items()
        }
        
        return {
            'total_notes': total_notes,
            'technique_counts': technique_counts,
            'technique_percentages': technique_percentages,
            'most_common_technique': max(technique_counts.items(), key=lambda x: x[1])[0] if technique_counts else None
        }


def create_technique_enhanced_note_detection():
    """
    Factory function to create technique-enhanced note detection function.
    
    Returns:
        Function that can be used as a drop-in replacement for 
        note_detection_with_onset_offset_regress with technique support.
    """
    
    def note_detection_with_technique(frame_output: np.ndarray,
                                    onset_output: np.ndarray, 
                                    onset_shift_output: np.ndarray,
                                    offset_output: np.ndarray,
                                    offset_shift_output: np.ndarray,
                                    velocity_output: np.ndarray,
                                    frame_threshold: float,
                                    technique_output: Optional[np.ndarray] = None,
                                    technique_processor: Optional[TechniqueProcessor] = None) -> List[List]:
        """
        Enhanced note detection with technique assignment.
        
        Args:
            Standard note detection parameters plus:
            technique_output: Optional technique predictions (frames_num, num_techniques)
            technique_processor: Optional TechniqueProcessor instance
            
        Returns:
            List of [bgn, fin, onset_shift, offset_shift, velocity, technique_id]
        """
        # Initialize technique processor if not provided
        if technique_processor is None:
            technique_processor = TechniqueProcessor()
            
        # Process technique output
        technique_probs = technique_processor.process_technique_output(technique_output)
        
        # Standard note detection logic
        output_tuples = []
        bgn = None
        frame_disappear = None
        offset_occur = None

        for i in range(onset_output.shape[0]):
            if onset_output[i] == 1:
                """Onset detected"""
                if bgn:
                    """Consecutive onsets"""
                    fin = max(i - 1, 0)
                    technique_id = technique_processor.assign_technique_to_note(bgn, fin, technique_probs)
                    output_tuples.append([bgn, fin, onset_shift_output[bgn], 0, velocity_output[bgn], technique_id])
                    frame_disappear, offset_occur = None, None
                bgn = i

            if bgn and i > bgn:
                """Search for offset"""
                if frame_output[i] <= frame_threshold and not frame_disappear:
                    frame_disappear = i

                if offset_output[i] == 1 and not offset_occur:
                    offset_occur = i

                if frame_disappear:
                    if offset_occur and offset_occur - bgn > frame_disappear - offset_occur:
                        fin = offset_occur
                    else:
                        fin = frame_disappear
                    technique_id = technique_processor.assign_technique_to_note(bgn, fin, technique_probs)
                    output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                                        offset_shift_output[fin], velocity_output[bgn], technique_id])
                    bgn, frame_disappear, offset_occur = None, None, None

                if bgn and (i - bgn >= 600 or i == onset_output.shape[0] - 1):
                    """Offset not detected"""
                    fin = i
                    technique_id = technique_processor.assign_technique_to_note(bgn, fin, technique_probs)
                    output_tuples.append([bgn, fin, onset_shift_output[bgn], 
                                        offset_shift_output[fin], velocity_output[bgn], technique_id])
                    bgn, frame_disappear, offset_occur = None, None, None

        # Sort by onsets
        output_tuples.sort(key=lambda pair: pair[0])
        return output_tuples
    
    return note_detection_with_technique


class TechniqueEnhancedPostProcessor:
    """
    Enhanced post-processor that wraps existing RegressionPostProcessor with technique support.
    """
    
    def __init__(self, 
                 base_post_processor,
                 technique_processor: Optional[TechniqueProcessor] = None):
        """
        Initialize enhanced post-processor.
        
        Args:
            base_post_processor: Existing RegressionPostProcessor instance
            technique_processor: Optional TechniqueProcessor instance
        """
        self.base_post_processor = base_post_processor
        self.technique_processor = technique_processor or TechniqueProcessor()
        
        # Create technique-enhanced note detection function
        self.note_detection_fn = create_technique_enhanced_note_detection()
    
    def output_dict_to_midi_events(self, output_dict: Dict) -> Tuple[List[Dict], Optional[List[Dict]]]:
        """
        Enhanced version that adds technique support to existing post-processor.
        
        Args:
            output_dict: Model outputs including optional 'technique_output'
            
        Returns:
            (enhanced_note_events, pedal_events) with technique labels
        """
        # Process technique output
        technique_output = output_dict.get('technique_output', None)
        technique_probs = self.technique_processor.process_technique_output(technique_output)
        
        # Get standard note detection results
        note_events, pedal_events = self.base_post_processor.output_dict_to_midi_events(output_dict)
        
        # If we have technique predictions, enhance the note events
        if technique_probs is not None and note_events:
            # Extract frame information from base processing for technique assignment
            # This is a simplified approach - in practice you'd want to modify the base processor
            technique_assignments = []
            
            for event in note_events:
                # Convert time to frames (approximate)
                onset_frame = int(event['onset_time'] * self.base_post_processor.frames_per_second)
                offset_frame = int(event['offset_time'] * self.base_post_processor.frames_per_second)
                
                technique_id = self.technique_processor.assign_technique_to_note(
                    onset_frame, offset_frame, technique_probs)
                technique_assignments.append(technique_id)
            
            # Add techniques to note events
            note_events = self.technique_processor.add_techniques_to_note_events(
                note_events, technique_assignments)
        
        return note_events, pedal_events
    
    def get_technique_statistics(self, note_events: List[Dict]) -> Dict:
        """Get technique usage statistics."""
        return self.technique_processor.get_technique_statistics(note_events)


# Convenience functions for easy integration
def enhance_existing_post_processor(base_post_processor, 
                                  technique_strategy: TechniqueStrategy = TechniqueStrategy.MAJORITY,
                                  confidence_threshold: float = 0.6) -> TechniqueEnhancedPostProcessor:
    """
    Enhance an existing post-processor with technique support.
    
    Args:
        base_post_processor: Existing RegressionPostProcessor
        technique_strategy: Technique assignment strategy
        confidence_threshold: Confidence threshold for technique assignment
        
    Returns:
        Enhanced post-processor with technique support
    """
    technique_processor = TechniqueProcessor(
        strategy=technique_strategy,
        confidence_threshold=confidence_threshold
    )
    
    return TechniqueEnhancedPostProcessor(base_post_processor, technique_processor)


def process_technique_predictions_simple(output_dict: Dict,
                                       base_post_processor,
                                       technique_strategy: str = "majority",
                                       confidence_threshold: float = 0.6) -> Tuple[List[Dict], Optional[List[Dict]]]:
    """
    Simple function to add technique support to any existing post-processor.
    
    Args:
        output_dict: Model outputs with optional 'technique_output'
        base_post_processor: Existing post-processor instance
        technique_strategy: "majority", "peak", or "onset"
        confidence_threshold: Minimum confidence for technique assignment
        
    Returns:
        (enhanced_note_events, pedal_events) with technique labels
    """
    strategy_map = {
        "majority": TechniqueStrategy.MAJORITY,
        "peak": TechniqueStrategy.PEAK, 
        "onset": TechniqueStrategy.ONSET
    }
    
    enhanced_processor = enhance_existing_post_processor(
        base_post_processor,
        technique_strategy=strategy_map[technique_strategy],
        confidence_threshold=confidence_threshold
    )
    
    return enhanced_processor.output_dict_to_midi_events(output_dict)
