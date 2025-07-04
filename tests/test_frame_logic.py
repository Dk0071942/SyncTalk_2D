#!/usr/bin/env python3
"""Test the frame-based logic without dependencies."""

def test_frame_calculations():
    """Test frame calculation logic."""
    fps = 25
    
    test_cases = [
        ("5s segment with 4s clip", 5.0, 4.0),
        ("3s segment with 4s clip", 3.0, 4.0),
        ("10s segment with 4s clip", 10.0, 4.0),
        ("2s segment with 1s clip", 2.0, 1.0),
    ]
    
    print("Frame-based calculation tests:")
    print("=" * 60)
    
    for desc, segment_duration, clip_duration in test_cases:
        print(f"\n{desc}:")
        print(f"  Segment: {segment_duration}s ({int(segment_duration * fps)} frames)")
        print(f"  Clip: {clip_duration}s ({int(clip_duration * fps)} frames)")
        
        # Calculate frames
        total_frames_needed = int(segment_duration * fps)
        clip_frames = int(clip_duration * fps)
        
        if clip_frames >= total_frames_needed:
            # Clip is longer than needed
            frames_to_use = total_frames_needed
            padding_frames = 0
            print(f"  → Use first {frames_to_use} frames of clip")
        else:
            # Clip is shorter, need padding
            frames_to_use = clip_frames
            padding_frames = total_frames_needed - clip_frames
            print(f"  → Play full clip ({frames_to_use} frames)")
            print(f"  → Pad with last frame ({padding_frames} times)")
        
        # Verify
        total_output = frames_to_use + padding_frames
        print(f"  Total output: {total_output} frames ({total_output/fps:.2f}s)")
        
        if total_output == total_frames_needed:
            print(f"  ✓ Correct synchronization")
        else:
            print(f"  ✗ ERROR: {total_output} != {total_frames_needed}")

def simulate_frame_playback():
    """Simulate frame playback with padding."""
    print("\n\nFrame playback simulation:")
    print("=" * 60)
    
    # Simulate 5s segment with 4s clip at 25fps
    fps = 25
    segment_duration = 5.0
    clip_duration = 4.0
    
    total_frames = int(segment_duration * fps)  # 125 frames
    clip_frames = int(clip_duration * fps)      # 100 frames
    padding_frames = total_frames - clip_frames # 25 frames
    
    print(f"Simulating {segment_duration}s segment with {clip_duration}s clip:")
    print(f"  Total frames needed: {total_frames}")
    print(f"  Clip has: {clip_frames} frames")
    print(f"  Padding needed: {padding_frames} frames")
    print("\n  Frame sequence:")
    
    # Show first 10 frames
    for i in range(min(10, total_frames)):
        if i < clip_frames:
            frame_idx = i
            print(f"    Frame {i:3d}: Play clip frame {frame_idx}")
        else:
            frame_idx = clip_frames - 1  # Last frame
            print(f"    Frame {i:3d}: Repeat last frame ({frame_idx})")
    
    if total_frames > 10:
        print("    ...")
        
        # Show transition point
        for i in range(max(95, 0), min(105, total_frames)):
            if i < clip_frames:
                frame_idx = i
                print(f"    Frame {i:3d}: Play clip frame {frame_idx}")
            else:
                frame_idx = clip_frames - 1
                print(f"    Frame {i:3d}: Repeat last frame ({frame_idx})")
        
        if total_frames > 105:
            print("    ...")
            
            # Show last 5 frames
            for i in range(max(total_frames - 5, 105), total_frames):
                frame_idx = clip_frames - 1
                print(f"    Frame {i:3d}: Repeat last frame ({frame_idx})")

if __name__ == "__main__":
    test_frame_calculations()
    simulate_frame_playback()