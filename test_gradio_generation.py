#!/usr/bin/env python3
"""Test script to verify Gradio app video generation works"""

import os
import sys
from app_gradio import SyncTalkGradio

def test_generation():
    app = SyncTalkGradio()
    
    # Test parameters
    model_name = "AD2.2"
    audio_file = "demo/talk_hb.wav"
    start_frame = 0
    asr_mode = "ave"
    
    print(f"Testing video generation with model: {model_name}")
    print(f"Audio file: {audio_file}")
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file {audio_file} not found")
        return False
    
    # Mock progress function
    def mock_progress(value, desc=""):
        print(f"Progress: {value:.1%} - {desc}")
    
    try:
        video_path, message = app.generate_video(
            model_name=model_name,
            audio_file=audio_file,
            start_frame=start_frame,
            asr_mode=asr_mode,
            progress=mock_progress
        )
        
        if video_path and os.path.exists(video_path):
            print(f"Success! Video generated at: {video_path}")
            print(f"File size: {os.path.getsize(video_path) / 1024 / 1024:.2f} MB")
            return True
        else:
            print(f"Error: {message}")
            return False
            
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_generation()
    sys.exit(0 if success else 1)