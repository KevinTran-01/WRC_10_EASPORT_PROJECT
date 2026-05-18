"""
WRC Road Detection — Inference Script
Runs the trained model on a video and saves the result as output_detection.mp4
Usage: python inference.py <video_path>
"""

import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import cv2
import os
import sys

# Config
VIDEO_PATH   = sys.argv[1] if len(sys.argv) > 1 else '../wrc_10_gameplay.mp4'
MODEL_PATH   = 'road_segmenter.pth'
OUTPUT_PATH  = 'output_detection.mp4'
IMG_SIZE     = (256, 256)
DISPLAY_SIZE = (960, 540)
THRESHOLD    = 0.15
SAVE_DIR     = 'output'

os.makedirs(SAVE_DIR, exist_ok=True)

# Model
class RoadSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights='DEFAULT')
        self.encoder = mobilenet.features
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),  # add this

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),  # add this

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0)
    return img


def draw_overlay(display_frame, mask_bin):
    # green road overlay
    overlay = display_frame.copy()
    overlay[mask_bin > 0] = (0, 180, 0)
    result = cv2.addWeighted(overlay, 0.35, display_frame, 0.65, 0)

    # find boundary contours
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(result, [largest], -1, (0, 255, 100), 2)

    # left and right boundary lines
    h, w = mask_bin.shape
    left_pts  = []
    right_pts = []
    for y in range(int(h * 0.4), h, 6):
        row     = mask_bin[y, :]
        road_px = np.where(row > 0)[0]
        if len(road_px) > 20:
            left_pts.append((road_px[0], y))
            right_pts.append((road_px[-1], y))

    for i in range(len(left_pts) - 1):
        cv2.line(result, left_pts[i],  left_pts[i+1],  (255, 80, 0),  2)
        cv2.line(result, right_pts[i], right_pts[i+1], (0,  80, 255), 2)

    return result


def run():
    # check files
    if not os.path.exists(MODEL_PATH):
        print(f'ERROR: Model not found at {MODEL_PATH}')
        return

    if not os.path.exists(VIDEO_PATH):
        print(f'ERROR: Video not found at {VIDEO_PATH}')
        return

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading model on {device}...')
    model = RoadSegmenter().to(device)
    model.load_state_dict(torch.load(
        MODEL_PATH, map_location=device, weights_only=True
    ))
    model.eval()
    print('Model loaded!')

    # open video
    cap   = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f'Video: {total} frames at {fps:.1f} fps')

    # saves to file
    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        DISPLAY_SIZE
    )

    # side-by-side version (original | mask)
    out_compare = cv2.VideoWriter(
        'output_compare.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (DISPLAY_SIZE[0] * 2, DISPLAY_SIZE[1])
    )

    kernel    = np.ones((15, 15), np.uint8)
    frame_num = 0
    dw, dh    = DISPLAY_SIZE

    print(f'Processing {total} frames...')
    print(f'Saving detection to: {OUTPUT_PATH}')
    print(f'Saving comparison to: output_compare.mp4')


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # black out inset camera (bottom left corner)
        frame[int(h * 0.82):, :int(w * 0.18)] = 0

        # run model on downscaled frame
        frame_small = cv2.resize(frame, IMG_SIZE)
        with torch.no_grad():
            inp  = preprocess(frame_small).to(device)
            pred = model(inp).squeeze().cpu().numpy()

        # resize mask to display size
        mask_np  = cv2.resize(pred, (dw, dh))
        mask_bin = (mask_np > THRESHOLD).astype(np.uint8) * 255

        # clean up noise
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN,  kernel)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)

        # resize frame for display
        display_frame = cv2.resize(frame, (dw, dh))

        # draw overlay
        result = draw_overlay(display_frame, mask_bin)

        # add info text
        cv2.putText(result,
                   f'Frame {frame_num}/{total}  |  threshold: {THRESHOLD}',
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(result,
                   'GREEN = detected road  BLUE = left boundary  RED = right boundary',
                   (10, dh - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # write detection output
        out.write(result)

        # write side-by-side comparison (original left, detection right)
        mask_color = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
        compare    = np.hstack([display_frame, mask_color])
        out_compare.write(compare)

        frame_num += 1

        # progress update every 300 frames
        if frame_num % 300 == 0:
            pct = (frame_num / total) * 100
            print(f'  {frame_num}/{total} frames done ({pct:.1f}%)')

    cap.release()
    out.release()
    out_compare.release()

if __name__ == '__main__':
    run()