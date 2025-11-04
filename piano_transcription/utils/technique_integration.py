"""
Simple Technique Integration Helper

This module provides the simplest possible interface for adding technique
prediction support to existing piano transcription systems.

Usage:
    from piano_transcription.utils.technique_integration import add_technique_support
    
    enhanced_note_events = add_technique_support(
        model_output=output_dict,
        base_post_processor=your_existing_processor
    )
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from technique_processor import (
    TechniqueProcessor,
    TechniqueStrategy, 
    process_technique_predictions_simple
)


def add_technique_support(model_output: Dict,
                         base_post_processor,
                         strategy: str = "majority",
                         confidence_threshold: float = 0.6,
                         technique_labels: Optional[Dict[int, str]] = None) -> Tuple[List[Dict], Optional[List[Dict]]]:
    """
    Add technique support to any existing piano transcription system.
    
    This is the main entry point for technique integration. Simply replace your
    existing post-processing call with this function.
    
    Args:
        model_output: Dictionary containing model predictions, including:
            - All standard outputs (reg_onset_output, frame_output, etc.)
            - technique_output: (frames_num, num_techniques) - NEW
        base_post_processor: Your existing RegressionPostProcessor instance
        strategy: Technique assignment strategy ("majority", "peak", "onset")
        confidence_threshold: Minimum confidence for technique assignment (0.0-1.0)
        technique_labels: Optional custom technique ID to name mapping
        
    Returns:
        (enhanced_note_events, pedal_events) where note_events now include 'technique' field
        
    Example:
        # Before:
        note_events, pedal_events = post_processor.output_dict_to_midi_events(output_dict)
        
        # After (with technique support):
        note_events, pedal_events = add_technique_support(output_dict, post_processor)
        
        # Each note event now has: {'midi_note', 'onset_time', 'offset_time', 'velocity', 'technique'}
    """
    return process_technique_predictions_simple(
        output_dict=model_output,
        base_post_processor=base_post_processor,
        technique_strategy=strategy,
        confidence_threshold=confidence_threshold
    )


def create_technique_processor(strategy: str = "majority",
                             confidence_threshold: float = 0.6,
                             technique_labels: Optional[Dict[int, str]] = None) -> TechniqueProcessor:
    """
    Create a standalone technique processor for manual processing.
    
    Args:
        strategy: "majority", "peak", or "onset"
        confidence_threshold: Minimum confidence for technique assignment
        technique_labels: Optional custom technique labels
        
    Returns:
        TechniqueProcessor instance
        
    Example:
        processor = create_technique_processor("majority", 0.7)
        technique_id = processor.assign_technique_to_note(onset_frame, offset_frame, technique_probs)
    """
    strategy_map = {
        "majority": TechniqueStrategy.MAJORITY,
        "peak": TechniqueStrategy.PEAK,
        "onset": TechniqueStrategy.ONSET
    }
    
    return TechniqueProcessor(
        strategy=strategy_map.get(strategy, TechniqueStrategy.MAJORITY),
        confidence_threshold=confidence_threshold,
        technique_labels=technique_labels
    )


def assign_technique_to_notes(note_events: List[Dict],
                             technique_predictions: np.ndarray,
                             frames_per_second: int = 100,
                             strategy: str = "majority",
                             confidence_threshold: float = 0.6) -> List[Dict]:
    """
    Assign techniques to existing note events using frame-wise predictions.
    
    Use this if you want to add technique labels to already-detected notes.
    
    Args:
        note_events: List of note event dicts (without technique field)
        technique_predictions: (frames_num, num_techniques) technique predictions
        frames_per_second: Frames per second for time-to-frame conversion
        strategy: Technique assignment strategy
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Enhanced note events with technique field added
        
    Example:
        # You already have note events from somewhere
        note_events = [{'midi_note': 60, 'onset_time': 1.0, 'offset_time': 1.5, 'velocity': 80}, ...]
        
        # Add technique labels
        enhanced_events = assign_technique_to_notes(note_events, technique_predictions)
    """
    processor = create_technique_processor(strategy, confidence_threshold)
    technique_probs = processor.process_technique_output(technique_predictions)
    
    enhanced_events = []
    for event in note_events:
        # Convert times to frames
        onset_frame = int(event['onset_time'] * frames_per_second)
        offset_frame = int(event['offset_time'] * frames_per_second)
        
        # Assign technique
        technique_id = processor.assign_technique_to_note(onset_frame, offset_frame, technique_probs)
        
        # Add to enhanced event
        enhanced_event = event.copy()
        enhanced_event['technique'] = technique_id
        enhanced_events.append(enhanced_event)
    
    return enhanced_events


def get_technique_name(technique_id: int, custom_labels: Optional[Dict[int, str]] = None) -> str:
    """
    Convert technique ID to human-readable name.
    
    Args:
        technique_id: Technique class ID
        custom_labels: Optional custom label mapping
        
    Returns:
        Human-readable technique name
        
    Example:
        name = get_technique_name(2)  # Returns "Legato"
        name = get_technique_name(-1)  # Returns "Unknown"
    """
    if custom_labels:
        return custom_labels.get(technique_id, f"Unknown_{technique_id}")
    
    default_labels = {
        0: "Normal",
        1: "Staccato",
        2: "Legato", 
        3: "Accent",
        4: "Tenuto",
        -1: "Unknown"
    }
    
    return default_labels.get(technique_id, f"Technique_{technique_id}")


def analyze_technique_usage(note_events: List[Dict], 
                           custom_labels: Optional[Dict[int, str]] = None) -> Dict:
    """
    Analyze technique usage in note events.
    
    Args:
        note_events: List of note events with technique field
        custom_labels: Optional custom technique labels
        
    Returns:
        Dictionary with technique usage statistics
        
    Example:
        stats = analyze_technique_usage(note_events)
        print(f"Most common technique: {stats['most_common']}")
        print(f"Technique distribution: {stats['percentages']}")
    """
    if not note_events:
        return {}
    
    # Count techniques
    technique_counts = {}
    for event in note_events:
        technique_id = event.get('technique', -1)
        technique_name = get_technique_name(technique_id, custom_labels)
        technique_counts[technique_name] = technique_counts.get(technique_name, 0) + 1
    
    # Calculate percentages
    total = len(note_events)
    technique_percentages = {name: (count / total) * 100 for name, count in technique_counts.items()}
    
    # Find most common
    most_common = max(technique_counts.items(), key=lambda x: x[1])[0] if technique_counts else None
    
    return {
        'total_notes': total,
        'counts': technique_counts,
        'percentages': technique_percentages,
        'most_common': most_common,
        'num_unique_techniques': len(technique_counts)
    }


# Backward compatibility aliases
process_with_techniques = add_technique_support
enhance_note_events = assign_technique_to_notes


def quick_demo():
    """Quick demonstration of the integration functions."""
    print("🎵 Quick Technique Integration Demo")
    
    # Simulate some basic note events (without techniques)
    note_events = [
        {'midi_note': 60, 'onset_time': 0.5, 'offset_time': 1.0, 'velocity': 80},
        {'midi_note': 64, 'onset_time': 1.0, 'offset_time': 1.4, 'velocity': 75},
        {'midi_note': 67, 'onset_time': 1.5, 'offset_time': 2.0, 'velocity': 85}
    ]
    
    # Simulate technique predictions (3 seconds, 4 techniques)
    frames_num = 300  # 3 seconds * 100 fps
    num_techniques = 4
    technique_predictions = np.random.randn(frames_num, num_techniques)
    
    # Add techniques to existing note events
    enhanced_events = assign_technique_to_notes(
        note_events=note_events,
        technique_predictions=technique_predictions,
        frames_per_second=100,
        strategy="majority"
    )
    
    print(f"Enhanced {len(enhanced_events)} notes with technique labels:")
    for i, event in enumerate(enhanced_events):
        technique_name = get_technique_name(event['technique'])
        print(f"  Note {i+1}: MIDI {event['midi_note']}, Technique: {technique_name}")
    
    # Analyze usage
    stats = analyze_technique_usage(enhanced_events)
    print(f"\nTechnique Analysis: {stats}")
    
    return enhanced_events


if __name__ == "__main__":
    quick_demo()
