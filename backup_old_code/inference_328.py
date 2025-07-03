import argparse
import os
from inference_module import SyncTalkInference

# Command line argument parsing
parser = argparse.ArgumentParser(description='Train',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--asr', type=str, default="ave")
parser.add_argument('--name', type=str, default="May")
parser.add_argument('--audio_path', type=str, default="demo/talk_hb.wav")
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--parsing', action='store_true', help="Use parsing if available")
parser.add_argument('--loop_back', action='store_true', help="Loop video back to start frame (default: True)")
parser.add_argument('--no_loop_back', dest='loop_back', action='store_false', help="Disable loop back to start frame")
parser.set_defaults(loop_back=True)

def main():
    args = parser.parse_args()
    
    # Create inference instance
    inference = SyncTalkInference(args.name)
    
    # Run CLI mode
    output_path = inference.run_cli(
        audio_path=args.audio_path,
        start_frame=args.start_frame,
        loop_back=args.loop_back,
        use_parsing=args.parsing,
        asr_mode=args.asr
    )
    
    print(f"Video saved to: {output_path}")

if __name__ == "__main__":
    main()