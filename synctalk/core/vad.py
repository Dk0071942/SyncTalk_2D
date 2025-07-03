"""
Voice Activity Detection (VAD) module for SyncTalk 2D.

This module provides functionality for detecting speech and silence segments
in audio files using the Silero VAD model or fallback energy-based detection.
"""

import os
import torch
import torchaudio
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import warnings


@dataclass
class AudioSegment:
    """Represents a segment of audio with speech/silence label."""
    start_time: float
    end_time: float
    label: str  # "speech" or "silence"
    
    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end_time - self.start_time
        
    def __repr__(self) -> str:
        return f"AudioSegment({self.label}, {self.start_time:.2f}-{self.end_time:.2f}s)"


class SileroVAD:
    """Voice Activity Detection using Silero VAD model."""
    
    def __init__(self, 
                 threshold: float = 0.5,
                 min_speech_duration_ms: int = 250,
                 min_silence_duration_ms: int = 100,
                 speech_pad_ms: int = 30):
        """
        Initialize Silero VAD.
        
        Args:
            threshold: Speech probability threshold (0-1)
            min_speech_duration_ms: Minimum speech segment duration in milliseconds
            min_silence_duration_ms: Minimum silence segment duration in milliseconds
            speech_pad_ms: Padding around speech segments in milliseconds
        """
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        
        # Load Silero VAD model
        self.model = None
        self.utils = None
        self.get_speech_timestamps = None
        self.read_audio = None
        
        try:
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            self.model.eval()
            
            # Get utility functions
            self.get_speech_timestamps = self.utils[0]
            self.read_audio = self.utils[2]
            
            print("Silero VAD model loaded successfully")
            
        except Exception as e:
            print(f"Failed to load Silero VAD: {e}")
            print("Falling back to energy-based VAD")
            self.model = None
            
    def process_audio(self, audio_path: str) -> Tuple[List[AudioSegment], float]:
        """
        Process audio file and detect speech/silence segments.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (segments list, total duration in seconds)
        """
        if self.model is not None:
            return self._process_with_silero(audio_path)
        else:
            return self._process_with_energy(audio_path)
            
    def _process_with_silero(self, audio_path: str) -> Tuple[List[AudioSegment], float]:
        """Process audio using Silero VAD model."""
        # Read audio at 16kHz (Silero requirement)
        wav = self.read_audio(audio_path, sampling_rate=16000)
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            wav, 
            self.model,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            return_seconds=True
        )
        
        # Calculate total duration
        duration = len(wav) / 16000.0
        
        # Convert timestamps to segments
        segments = []
        last_end = 0.0
        
        for timestamp in speech_timestamps:
            start = timestamp['start']
            end = timestamp['end']
            
            # Add silence segment if there's a gap
            if start > last_end:
                segments.append(AudioSegment(last_end, start, "silence"))
                
            # Add speech segment
            segments.append(AudioSegment(start, end, "speech"))
            last_end = end
            
        # Add final silence if needed
        if last_end < duration:
            segments.append(AudioSegment(last_end, duration, "silence"))
            
        # If no speech detected, entire audio is silence
        if not segments:
            segments = [AudioSegment(0.0, duration, "silence")]
            
        return segments, duration
        
    def _process_with_energy(self, audio_path: str) -> Tuple[List[AudioSegment], float]:
        """Fallback energy-based VAD."""
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
            
        # Calculate energy in windows
        window_size = int(0.02 * sample_rate)  # 20ms windows
        hop_size = int(0.01 * sample_rate)  # 10ms hop
        
        energy = []
        for i in range(0, len(waveform[0]) - window_size, hop_size):
            window = waveform[0, i:i+window_size]
            energy.append(torch.sum(window ** 2).item())
            
        energy = np.array(energy)
        
        # Normalize and threshold
        if len(energy) > 0:
            energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy) + 1e-8)
            is_speech = energy > self.threshold
            
            # Convert to segments
            segments = []
            duration = len(waveform[0]) / sample_rate
            
            in_speech = False
            start_time = 0.0
            
            for i, speech in enumerate(is_speech):
                current_time = i * hop_size / sample_rate
                
                if speech and not in_speech:
                    start_time = current_time
                    in_speech = True
                elif not speech and in_speech:
                    segments.append(AudioSegment(start_time, current_time, "speech"))
                    in_speech = False
                    
            # Handle last segment
            if in_speech:
                segments.append(AudioSegment(start_time, duration, "speech"))
                
            # Fill gaps with silence
            filled_segments = []
            last_end = 0.0
            
            for seg in segments:
                if seg.start_time > last_end:
                    filled_segments.append(AudioSegment(last_end, seg.start_time, "silence"))
                filled_segments.append(seg)
                last_end = seg.end_time
                
            if last_end < duration:
                filled_segments.append(AudioSegment(last_end, duration, "silence"))
                
            return filled_segments if filled_segments else [AudioSegment(0.0, duration, "silence")], duration
        else:
            duration = len(waveform[0]) / sample_rate
            return [AudioSegment(0.0, duration, "silence")], duration
            
    def merge_short_segments(self, segments: List[AudioSegment], 
                           min_duration: float = 0.5) -> List[AudioSegment]:
        """
        Merge short segments to avoid choppy transitions.
        
        Args:
            segments: List of audio segments
            min_duration: Minimum duration for a segment in seconds
            
        Returns:
            Merged segments
        """
        if not segments:
            return segments
            
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            # If current segment is too short and same type as next, merge
            if current.duration < min_duration and current.label == next_seg.label:
                current = AudioSegment(current.start_time, next_seg.end_time, current.label)
            else:
                merged.append(current)
                current = next_seg
                
        merged.append(current)
        
        # Second pass: convert short silence to speech to avoid choppy video
        final_segments = []
        for i, seg in enumerate(merged):
            if seg.label == "silence" and seg.duration < min_duration:
                # Convert to speech
                final_segments.append(AudioSegment(seg.start_time, seg.end_time, "speech"))
            else:
                final_segments.append(seg)
                
        return final_segments
        
    def visualize_vad(self, audio_path: str, segments: List[AudioSegment], 
                     output_path: Optional[str] = None) -> None:
        """
        Visualize VAD results on audio waveform.
        
        Args:
            audio_path: Path to audio file
            segments: List of audio segments
            output_path: Optional path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
        except ImportError:
            print("Matplotlib not available for visualization")
            return
            
        # Load audio for visualization
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        else:
            waveform = waveform[0]
            
        # Create time axis
        time = np.arange(len(waveform)) / sample_rate
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time, waveform.numpy(), 'b-', alpha=0.6, linewidth=0.5)
        
        # Add segment rectangles
        for seg in segments:
            if seg.label == "speech":
                color = 'lightgreen'
                alpha = 0.3
            else:
                color = 'lightcoral'
                alpha = 0.2
                
            rect = patches.Rectangle(
                (seg.start_time, -1), 
                seg.duration, 
                2,
                linewidth=0,
                facecolor=color,
                alpha=alpha
            )
            ax.add_patch(rect)
            
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Voice Activity Detection Results')
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        speech_patch = patches.Patch(color='lightgreen', alpha=0.3, label='Speech')
        silence_patch = patches.Patch(color='lightcoral', alpha=0.2, label='Silence')
        ax.legend(handles=[speech_patch, silence_patch])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            print(f"VAD visualization saved to {output_path}")
        else:
            plt.show()
            
        plt.close()


def test_vad():
    """Test VAD functionality."""
    vad = SileroVAD()
    
    # Test with a sample audio file
    test_audio = "pause_audio_test_23s.wav"
    if os.path.exists(test_audio):
        segments, duration = vad.process_audio(test_audio)
        
        print(f"\nVAD Results for {test_audio}")
        print(f"Total duration: {duration:.2f}s")
        print(f"Number of segments: {len(segments)}")
        
        for seg in segments:
            print(f"  {seg}")
            
        # Test merging
        merged = vad.merge_short_segments(segments, min_duration=0.75)
        print(f"\nAfter merging short segments:")
        print(f"Number of segments: {len(merged)}")
        for seg in merged:
            print(f"  {seg}")


if __name__ == "__main__":
    test_vad()