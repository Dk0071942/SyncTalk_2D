#!/usr/bin/env python3
"""Test the final looping logic"""

def create_frame_sequence_final(total_frames, available_frames, start_frame, loop_back=True):
    """Final logic from the updated app"""
    frame_sequence = []
    
    if total_frames <= available_frames:
        # Audio is shorter than video: use frames from start_frame forward
        for i in range(total_frames):
            frame_sequence.append((start_frame + i) % available_frames)
    else:
        # Audio is longer than video: create a looping pattern
        # Create a proper forward-backward cycle
        # Forward: 0, 1, 2, ..., available_frames-1
        # Backward: available_frames-2, ..., 1 (excluding 0 and available_frames-1 to avoid repetition)
        forward_indices = list(range(available_frames))
        backward_indices = list(range(available_frames - 2, 0, -1))
        
        # Full cycle of indices
        full_cycle = forward_indices + backward_indices
        cycle_length = len(full_cycle)
        
        # Fill the sequence
        for i in range(total_frames):
            # Get the index in the cycle
            cycle_pos = i % cycle_length
            # Get the actual frame number (offset by start_frame)
            frame_idx = (start_frame + full_cycle[cycle_pos]) % available_frames
            frame_sequence.append(frame_idx)
    
    # Ensure we end with the starting frame for seamless loop (if enabled)
    if loop_back and total_frames > 1 and frame_sequence[-1] != start_frame:
        # Adjust the last few frames to smoothly return to start
        transition_frames = min(15, total_frames // 10)  # Use up to 15 frames for transition
        
        # Force the last frames to be the start frame
        # This ensures we always end exactly where we started
        for i in range(transition_frames):
            idx = total_frames - transition_frames + i
            if idx < total_frames and idx >= 0:
                # For the last half of the transition, use start_frame
                if i >= transition_frames // 2:
                    frame_sequence[idx] = start_frame
    
    return frame_sequence

# Test various scenarios
test_cases = [
    (50, 100, 0, "Short audio, start at 0"),
    (50, 100, 25, "Short audio, start at 25"),
    (200, 100, 10, "Medium audio, start at 10"),
    (1000, 100, 50, "Long audio, start at 50"),
    (2000, 200, 75, "Very long audio, start at 75"),
]

for total_frames, available_frames, start_frame, description in test_cases:
    print(f"\n{'='*60}")
    print(f"Test: {description}")
    print(f"Audio frames: {total_frames}, Available frames: {available_frames}, Start: {start_frame}")
    
    seq = create_frame_sequence_final(total_frames, available_frames, start_frame)
    
    print(f"First 10 frames: {seq[:10]}")
    print(f"Last 10 frames: {seq[-10:]}")
    print(f"Start frame: {seq[0]}, End frame: {seq[-1]}")
    print(f"✓ Loops back correctly: {seq[0] == seq[-1]}")
    
    # Check that all frames are within valid range
    invalid_frames = [f for f in seq if f < 0 or f >= available_frames]
    if invalid_frames:
        print(f"❌ ERROR: Invalid frames found: {invalid_frames[:10]}...")
    else:
        print(f"✓ All frames are valid (0-{available_frames-1})")