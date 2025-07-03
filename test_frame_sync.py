#!/usr/bin/env python3
"""Test script to verify frame-based synchronization fixes."""

import sys
from core_clips_manager import CoreClipsManager
from vad_torch import AudioSegment

def test_frame_selection():
    """Test the frame-based clip selection logic."""
    print("Testing frame-based clip selection...")
    
    # Initialize manager
    manager = CoreClipsManager("LS1")
    
    # Test scenarios
    test_cases = [
        ("5s segment with 4s clip", 5.0, "talk"),
        ("3s segment with 4s clip", 3.0, "talk"),
        ("10s segment with 4s clip", 10.0, "talk"),
        ("2s segment with 1s clip", 2.0, "silence"),
    ]
    
    for desc, duration, clip_type in test_cases:
        print(f"\n{desc}:")
        print(f"  Segment duration: {duration}s")
        
        try:
            selections = manager.select_clips_for_segment(duration, clip_type, fps=25)
            
            for clip, start_frame, end_frame, padding_frames in selections:
                clip_frames = end_frame - start_frame
                total_frames = clip_frames + padding_frames
                total_duration = total_frames / 25.0
                
                print(f"  Selected: {clip.path}")
                print(f"    Clip duration: {clip.duration}s")
                print(f"    Frames: {start_frame}-{end_frame} ({clip_frames} frames)")
                print(f"    Padding: {padding_frames} frames")
                print(f"    Total frames: {total_frames} ({total_duration:.2f}s)")
                
                # Verify the selection covers the requested duration
                expected_frames = int(duration * 25)
                if total_frames == expected_frames:
                    print(f"    ✓ Correct: {total_frames} frames = {expected_frames} expected")
                else:
                    print(f"    ✗ Error: {total_frames} frames != {expected_frames} expected")
                    
        except Exception as e:
            print(f"  Error: {e}")

def test_edl_creation():
    """Test the EDL creation with frame-based approach."""
    print("\n\nTesting EDL creation...")
    
    from core_clips_processor import CoreClipsProcessor
    
    # Create a processor
    processor = CoreClipsProcessor("LS1")
    
    # Create test segments
    segments = [
        AudioSegment(start_time=0.0, end_time=5.0, label="speech"),
        AudioSegment(start_time=5.0, end_time=6.5, label="silence"),
        AudioSegment(start_time=6.5, end_time=10.0, label="speech"),
    ]
    
    # Create EDL
    edl = processor._create_edit_decision_list(segments)
    
    print(f"\nCreated {len(edl)} EDL items:")
    for i, item in enumerate(edl):
        print(f"\nItem {i+1}:")
        print(f"  Timeline: {item.start_time:.2f}s - {item.end_time:.2f}s")
        print(f"  Duration: {item.duration:.2f}s")
        print(f"  Clip: {item.clip.path}")
        print(f"  Frames: {item.clip_start_frame}-{item.clip_end_frame}")
        print(f"  Padding: {item.padding_frames} frames")
        print(f"  Total frames: {item.total_frames}")
        print(f"  Needs lipsync: {item.needs_lipsync}")

if __name__ == "__main__":
    test_frame_selection()
    test_edl_creation()