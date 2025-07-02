#!/usr/bin/env python3
"""Test the improved looping logic"""

def create_frame_sequence(total_frames, available_frames, start_frame, loop_back=True):
    """Create frame sequence that starts and ends with the same frame"""
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

# Test cases
print("Test 1: Short audio (10 frames), Long video (100 frames), start at 0")
seq = create_frame_sequence(10, 100, 0)
print(f"Sequence: {seq}")
print(f"Start: {seq[0]}, End: {seq[-1]}")
print()

print("Test 2: Long audio (200 frames), Short video (50 frames), start at 10")
seq = create_frame_sequence(200, 50, 10)
print(f"Sequence length: {len(seq)}")
print(f"First 20: {seq[:20]}")
print(f"Last 20: {seq[-20:]}")
print(f"Start: {seq[0]}, End: {seq[-1]}")
print()

print("Test 3: Medium audio (75 frames), Medium video (100 frames), start at 25")
seq = create_frame_sequence(75, 100, 25)
print(f"Sequence length: {len(seq)}")
print(f"First 10: {seq[:10]}")
print(f"Last 10: {seq[-10:]}")
print(f"Start: {seq[0]}, End: {seq[-1]}")
print()

print("Test 4: Same test without loop_back")
seq = create_frame_sequence(75, 100, 25, loop_back=False)
print(f"Without loop_back - Start: {seq[0]}, End: {seq[-1]}")