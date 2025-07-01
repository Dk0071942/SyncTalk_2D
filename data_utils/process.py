import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
def extract_audio(path, out_path, sample_rate=16000):
    
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    cmd = f'ffmpeg -y -v error -nostats -i {path} -f wav -ar {sample_rate} {out_path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio =====')
    
def extract_images(path):
    
    
    full_body_dir = path.replace(path.split("/")[-1], "full_body_img")
    os.makedirs(full_body_dir, exist_ok=True)
    
    # Determine the correct video path (handling 25fps conversion)
    video_to_process = path
    cap = cv2.VideoCapture(video_to_process)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps != 25:
        cap.release()
        # High quality conversion to 25fps using ffmpeg
        converted_path = video_to_process.replace(".mp4", "_25fps.mp4")
        cmd = f'ffmpeg -y -v error -nostats -i "{video_to_process}" -vf "fps=25" -c:v libx264 -c:a aac "{converted_path}"'
        os.system(cmd)
        video_to_process = converted_path
        cap = cv2.VideoCapture(video_to_process)

    # Check if images are already extracted and complete
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_existing_images = len([f for f in os.listdir(full_body_dir) if f.endswith('.jpg')])

    if num_existing_images >= total_frames and total_frames > 0:
        print(f'[INFO] ===== Images for {path} already extracted. Skipping. =====')
        cap.release()
        return

    # This check is important to ensure the final video is 25fps.
    if cap.get(cv2.CAP_PROP_FPS) != 25:
        raise ValueError("Your video fps should be 25!!!")
        
    print("extracting images...")
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(full_body_dir, str(counter)+'.jpg'), frame)
        counter += 1
    cap.release()
    
def get_audio_feature(wav_path):
    
    print("extracting audio feature...")
    os.system("python ./data_utils/ave/test_w2l_audio.py --wav_path "+wav_path)
    
def get_landmark(path, landmarks_dir):
    print("detecting landmarks...")
    full_img_dir = path.replace(path.split("/")[-1], "full_body_img")
    
    from get_landmark import Landmark
    landmark = Landmark()
    
    for img_name in tqdm(os.listdir(full_img_dir)):
        if not img_name.endswith(".jpg"):
            continue
        img_path = os.path.join(full_img_dir, img_name)
        lms_path = os.path.join(landmarks_dir, img_name.replace(".jpg", ".lms"))
        if os.path.exists(lms_path):
            continue
        pre_landmark, x1, y1 = landmark.detect(img_path)
        if pre_landmark is None:
            continue
        with open(lms_path, "w") as f:
            for p in pre_landmark:
                x, y = p[0]+x1, p[1]+y1
                f.write(str(x))
                f.write(" ")
                f.write(str(y))
                f.write("\n")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file")
    opt = parser.parse_args()

    base_dir = os.path.dirname(opt.path)
    wav_path = os.path.join(base_dir, 'aud.wav')
    landmarks_dir = os.path.join(base_dir, 'landmarks')

    os.makedirs(landmarks_dir, exist_ok=True)
    
    extract_audio(opt.path, wav_path)
    extract_images(opt.path)
    get_landmark(opt.path, landmarks_dir)
    get_audio_feature(wav_path)
    
    