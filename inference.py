# inference.py
import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from models import UNetGenerator
from dataset import transform # Import the *same* transform
from utils import denorm, calculate_psnr, degrade_frame # Import our helpers

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load Trained Generator ---
    if not os.path.exists(args.weights_path):
        print(f"Error: Cannot find weights file at {args.weights_path}")
        return
        
    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load(args.weights_path, map_location=device))
    generator.eval() # Set model to evaluation mode
    print("Generator model loaded successfully.")

    # --- 2. Setup Input and Output Paths ---
    if not os.path.exists(args.degraded_video):
        print(f"Error: Cannot find degraded video at {args.degraded_video}")
        return
    if args.original_video and not os.path.exists(args.original_video):
        print(f"Warning: Cannot find original video at {args.original_video}. Will not calculate PSNR.")
        args.original_video = None

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.degraded_video))[0]
    output_filename = f"{base_name}_restored.avi"
    OUTPUT_VIDEO_PATH = os.path.join(args.output_dir, output_filename)
    print(f"Will save restored video to: {OUTPUT_VIDEO_PATH}")

    # --- 3. Open Video Streams ---
    cap_deg = cv2.VideoCapture(args.degraded_video)
    frame_count_deg = int(cap_deg.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_deg.get(cv2.CAP_PROP_FPS)
    
    cap_orig = None
    if args.original_video:
        cap_orig = cv2.VideoCapture(args.original_video)
        
    # --- 4. Prepare Output Video Writer ---
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (128, 128))

    psnr_values = []
    
    # --- 5. Process Video Frame-by-Frame ---
    with torch.no_grad():
        progress_bar = tqdm(range(frame_count_deg), desc="Restoring video")
        for _ in progress_bar:
            ret_deg, frame_deg = cap_deg.read()
            if not ret_deg:
                break

            # A. Pre-process the degraded frame
            frame_deg_rgb = cv2.cvtColor(frame_deg, cv2.COLOR_BGR2RGB)
            degraded_pil = Image.fromarray(frame_deg_rgb)
            degraded_tensor = transform(degraded_pil).to(device).unsqueeze(0) # Add batch dim

            # B. Run the Generator
            restored_tensor = generator(degraded_tensor).squeeze(0) # Remove batch dim

            # C. Post-process the restored frame
            restored_np_0_1 = denorm(restored_tensor)
            restored_np_0_255 = (restored_np_0_1 * 255).astype(np.uint8)
            restored_bgr = cv2.cvtColor(restored_np_0_255, cv2.COLOR_RGB2BGR)
            
            # D. Write to output video
            out_writer.write(restored_bgr)

            # E. Calculate PSNR (if original is provided)
            if cap_orig:
                ret_orig, frame_orig = cap_orig.read()
                if ret_orig:
                    # We must resize the original frame to 128x128 for a fair PSNR comparison
                    frame_orig_resized = cv2.resize(frame_orig, (128, 128))
                    psnr = calculate_psnr(frame_orig_resized, restored_bgr)
                    psnr_values.append(psnr)
                    progress_bar.set_postfix(PSNR=f"{psnr:.2f} dB")

    # --- 6. Cleanup and Final Report ---
    cap_deg.release()
    out_writer.release()
    if cap_orig:
        cap_orig.release()
        
    print(f"\n--- Video saved successfully to {OUTPUT_VIDEO_PATH} ---")
    if psnr_values:
        avg_psnr = np.mean(psnr_values)
        print(f"--- Average PSNR for the video: {avg_psnr:.2f} dB ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore a degraded video using a trained GAN.")
    
    parser.add_argument('--degraded_video', type=str, required=True, help='Path to the input degraded video file')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to the trained generator.pth weights file')
    parser.add_argument('--original_video', type=str, default=None, help='(Optional) Path to the original video for PSNR calculation')
    parser.add_argument('--output_dir', type=str, default='restored_videos', help='Directory to save the restored video')

    args = parser.parse_args()
    main(args)
