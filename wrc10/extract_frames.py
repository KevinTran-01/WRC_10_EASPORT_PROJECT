"""
Extract frames from WRC gameplay video
"""

import cv2
import os

VIDEO_PATH = '../wrc_10_gameplay.mp4'
OUTPUT_DIR = 'data/images'
EVERY_N_FRAMES = 30 # save 1 frame every 15 frames

os.makedirs(OUTPUT_DIR, exist_ok = True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: could not open {VIDEO_PATH}")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = total_frames/fps

print(f"Video has {total_frames} frames, {fps:.1f} fps, {duration:.1f} seconds")

count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break


    if count % EVERY_N_FRAMES == 0:
        filename = f'{OUTPUT_DIR}/frame_{count:06d}.jpg'
        cv2.imwrite(filename, frame)
        saved += 1

        if saved % 50 == 0:
            print(f'  Saved {saved} frames so far...')

    count += 1

cap.release()
print(f'Saved {saved} frames to {OUTPUT_DIR}/')
