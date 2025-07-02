#!/usr/bin/env python3
"""Detailed test of the looping logic to find the bug"""

def create_frame_sequence_original(total_frames, available_frames, start_frame, loop_back=True):
    """Original logic from the app"""
    frame_sequence = []
    
    if total_frames <= available_frames:
        # Audio is shorter than video: use frames from start_frame forward
        for i in range(total_frames):
            frame_sequence.append((start_frame + i) % available_frames)
    else:
        # Audio is longer than video: create a looping pattern
        # First, go forward through all frames
        forward_frames = list(range(start_frame, start_frame + available_frames))
        # Then go backward (excluding first and last to avoid repetition)
        backward_frames = list(range(start_frame + available_frames - 2, start_frame, -1))
        
        # Create a full cycle
        full_cycle = forward_frames + backward_frames
        cycle_length = len(full_cycle)
        
        # Fill the sequence by repeating the cycle
        for i in range(total_frames):
            frame_sequence.append(full_cycle[i % cycle_length] % available_frames)
    
    # Ensure we end with the starting frame for seamless loop (if enabled)
    if loop_back and total_frames > 1 and frame_sequence[-1] != start_frame:
        # Adjust the last few frames to smoothly return to start
        transition_frames = min(10, total_frames // 4)  # Use up to 10 frames for transition
        for i in range(transition_frames):
            idx = total_frames - transition_frames + i
            if idx < total_frames:
                # Linear interpolation back to start frame
                progress_ratio = i / transition_frames
                # Gradually move back to start frame
                if progress_ratio > 0.5:
                    frame_sequence[idx] = start_frame
    
    return frame_sequence

def create_frame_sequence_fixed(total_frames, available_frames, start_frame, loop_back=True):
    """Fixed logic"""
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
        transition_frames = min(10, total_frames // 4)  # Use up to 10 frames for transition
        # Simply set the last few frames to gradually approach start_frame
        for i in range(transition_frames):
            idx = total_frames - transition_frames + i
            if idx < total_frames:
                # For the last half of transition, use start_frame
                if i >= transition_frames // 2:
                    frame_sequence[idx] = start_frame
    
    return frame_sequence

# Test with problematic case
print("=== Testing with long audio ===")
print("\nCase: 1000 frames audio, 100 frames video, start at frame 10")

print("\nORIGINAL LOGIC:")
seq_orig = create_frame_sequence_original(1000, 100, 10)
print(f"First 10 frames: {seq_orig[:10]}")
print(f"Last 10 frames: {seq_orig[-10:]}")
print(f"Start frame: {seq_orig[0]}, End frame: {seq_orig[-1]}")
print(f"Does it loop back? {seq_orig[0] == seq_orig[-1]}")

# Check the forward_frames issue
forward_frames = list(range(10, 10 + 100))  # This goes from 10 to 109!
print(f"\nProblem: forward_frames goes from {forward_frames[0]} to {forward_frames[-1]}")
print(f"But we only have frames 0-99! So {forward_frames[-1]} % 100 = {forward_frames[-1] % 100}")

print("\n\nFIXED LOGIC:")
seq_fixed = create_frame_sequence_fixed(1000, 100, 10)
print(f"First 10 frames: {seq_fixed[:10]}")
print(f"Last 10 frames: {seq_fixed[-10:]}")
print(f"Start frame: {seq_fixed[0]}, End frame: {seq_fixed[-1]}")
print(f"Does it loop back? {seq_fixed[0] == seq_fixed[-1]}")

# Test the cycle pattern
print("\n=== Checking cycle pattern ===")
print("Available frames: 100 (0-99)")
print("Start frame: 10")
print("\nFixed cycle pattern (first 200 frames):")
for i in range(0, 200, 20):
    print(f"Frames {i:3d}-{i+19:3d}: {seq_fixed[i:i+20]}")