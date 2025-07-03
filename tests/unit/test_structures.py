"""Unit tests for core data structures."""

import unittest
from synctalk.core.structures import CoreClip, EditDecisionItem, FrameBasedClipSelection


class TestCoreClip(unittest.TestCase):
    """Test CoreClip data structure."""
    
    def test_creation(self):
        """Test CoreClip creation."""
        clip = CoreClip(
            path="/path/to/clip.mp4",
            clip_type="talk",
            duration=2.5
        )
        
        self.assertEqual(clip.path, "/path/to/clip.mp4")
        self.assertEqual(clip.clip_type, "talk")
        self.assertEqual(clip.duration, 2.5)
        self.assertEqual(clip.fps, 25)
        self.assertEqual(clip.frame_count, 0)
    
    def test_equality(self):
        """Test CoreClip equality comparison."""
        clip1 = CoreClip("/path/clip.mp4", "talk", 2.5)
        clip2 = CoreClip("/path/clip.mp4", "talk", 2.5)
        clip3 = CoreClip("/path/clip.mp4", "silence", 2.5)
        
        self.assertEqual(clip1, clip2)
        self.assertNotEqual(clip1, clip3)
    
    def test_repr(self):
        """Test string representation."""
        clip = CoreClip("/path/clip.mp4", "talk", 2.5)
        repr_str = repr(clip)
        
        self.assertIn("talk", repr_str)
        self.assertIn("2.50s", repr_str)
        self.assertIn("/path/clip.mp4", repr_str)


class TestEditDecisionItem(unittest.TestCase):
    """Test EditDecisionItem data structure."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.clip = CoreClip("/path/clip.mp4", "talk", 2.0)
        self.clip.frame_count = 50
    
    def test_creation(self):
        """Test EditDecisionItem creation."""
        edl = EditDecisionItem(
            start_time=0.0,
            end_time=1.0,
            clip=self.clip,
            clip_start_frame=0,
            clip_end_frame=25,
            padding_frames=0,
            needs_lipsync=True,
            fps=25
        )
        
        self.assertEqual(edl.start_time, 0.0)
        self.assertEqual(edl.end_time, 1.0)
        self.assertEqual(edl.duration, 1.0)
        self.assertEqual(edl.total_frames, 25)
        self.assertTrue(edl.needs_lipsync)
    
    def test_validation(self):
        """Test validation method."""
        # Valid EDL
        edl = EditDecisionItem(
            start_time=0.0,
            end_time=1.0,
            clip=self.clip,
            clip_start_frame=0,
            clip_end_frame=25,
            padding_frames=0,
            needs_lipsync=True,
            fps=25
        )
        self.assertTrue(edl.validate())
        
        # Invalid time range
        edl_invalid_time = EditDecisionItem(
            start_time=1.0,
            end_time=0.0,  # End before start
            clip=self.clip,
            clip_start_frame=0,
            clip_end_frame=25,
            padding_frames=0,
            needs_lipsync=True,
            fps=25
        )
        self.assertFalse(edl_invalid_time.validate())
        
        # Invalid frame range
        edl_invalid_frames = EditDecisionItem(
            start_time=0.0,
            end_time=1.0,
            clip=self.clip,
            clip_start_frame=25,
            clip_end_frame=0,  # End before start
            padding_frames=0,
            needs_lipsync=True,
            fps=25
        )
        self.assertFalse(edl_invalid_frames.validate())
    
    def test_with_padding(self):
        """Test EDL with padding frames."""
        edl = EditDecisionItem(
            start_time=0.0,
            end_time=1.2,  # 30 frames at 25fps
            clip=self.clip,
            clip_start_frame=0,
            clip_end_frame=25,
            padding_frames=5,  # 5 frames padding
            needs_lipsync=False,
            fps=25
        )
        
        self.assertEqual(edl.total_frames, 30)
        self.assertTrue(edl.validate())


class TestFrameBasedClipSelection(unittest.TestCase):
    """Test FrameBasedClipSelection data structure."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.clip = CoreClip("/path/clip.mp4", "silence", 3.0)
        self.clip.frame_count = 75
    
    def test_creation(self):
        """Test FrameBasedClipSelection creation."""
        selection = FrameBasedClipSelection(
            clip=self.clip,
            start_frame=10,
            end_frame=40,
            output_start_frame=0,
            output_end_frame=30,
            padding_frames=0
        )
        
        self.assertEqual(selection.clip_frame_count, 30)
        self.assertEqual(selection.total_output_frames, 30)
        self.assertFalse(selection.needs_padding)
    
    def test_with_padding(self):
        """Test selection with padding."""
        selection = FrameBasedClipSelection(
            clip=self.clip,
            start_frame=0,
            end_frame=20,
            output_start_frame=0,
            output_end_frame=30,
            padding_frames=10
        )
        
        self.assertEqual(selection.clip_frame_count, 20)
        self.assertEqual(selection.total_output_frames, 30)
        self.assertTrue(selection.needs_padding)
        self.assertTrue(selection.validate())
    
    def test_validation(self):
        """Test validation method."""
        # Valid selection
        selection_valid = FrameBasedClipSelection(
            clip=self.clip,
            start_frame=0,
            end_frame=25,
            output_start_frame=0,
            output_end_frame=25,
            padding_frames=0
        )
        self.assertTrue(selection_valid.validate())
        
        # Invalid - output frames don't match
        selection_invalid = FrameBasedClipSelection(
            clip=self.clip,
            start_frame=0,
            end_frame=25,
            output_start_frame=0,
            output_end_frame=30,  # Should be 25
            padding_frames=0
        )
        self.assertFalse(selection_invalid.validate())


if __name__ == '__main__':
    unittest.main()