"""
WRC Road Detection — Inference Script
Runs the trained model on a video and saves the result as output_detection.mp4
Usage: python inference.py <video_path>
"""

import torch
import torch.nn as nn
from torchvision import models
from collections import deque
import numpy as np
import cv2
import os
import sys

# Config
VIDEO_PATH   = sys.argv[1] if len(sys.argv) > 1 else '../rawvid.mp4'
MODEL_PATH   = 'road_segmenter.pth'
OUTPUT_PATH  = 'output_rawvid.mp4'
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
            nn.Dropout2d(0.3),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),

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


def draw_overlay(display_frame, mask_bin, left_history, right_history):
    overlay = display_frame.copy()
    overlay[mask_bin > 0] = (0, 180, 0)
    result = cv2.addWeighted(overlay, 0.35, display_frame, 0.65, 0)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(result, [largest], -1, (0, 255, 100), 2)

    h, w = mask_bin.shape
    left_pts  = []
    right_pts = []

    for y in range(int(h * 0.4), h, 6):
        row     = mask_bin[y, :]
        road_px = np.where(row > 0)[0]
        if len(road_px) > 20:
            left_pts.append((road_px[0], y))
            right_pts.append((road_px[-1], y))

    if left_pts:
        left_history.append(left_pts)
    if right_pts:
        right_history.append(right_pts)

    # average left boundary
    avg_left = []
    if len(left_history) > 0:
        min_len = min(len(pts) for pts in left_history)
        for i in range(min_len):
            avg_x = int(np.mean([pts[i][0] for pts in left_history]))
            avg_y = int(np.mean([pts[i][1] for pts in left_history]))
            avg_left.append((avg_x, avg_y))
        for i in range(len(avg_left) - 1):
            cv2.line(result, avg_left[i], avg_left[i+1], (255, 80, 0), 2)

    # average right boundary
    avg_right = []
    if len(right_history) > 0:
        min_len = min(len(pts) for pts in right_history)
        for i in range(min_len):
            avg_x = int(np.mean([pts[i][0] for pts in right_history]))
            avg_y = int(np.mean([pts[i][1] for pts in right_history]))
            avg_right.append((avg_x, avg_y))
        for i in range(len(avg_right) - 1):
            cv2.line(result, avg_right[i], avg_right[i+1], (0, 80, 255), 2)
    
    return result, avg_left, avg_right

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

    mask_history  = deque(maxlen=5)
    left_history  = deque(maxlen=5)
    right_history = deque(maxlen=5)
    mask_bin      = None
    

        
    # @@@@@Handling logic for steering@@@@@
    # The car is always at the same place every frame => hardcode
    # take the car as the center 0. Left will be -x, right will be +x
    # boundary distance = distance between the detected road to the center
    # Note: It is impossible to keep the car in the center of the road all the time since it is in a race

    def get_car_position(display_frame):
        h, w = display_frame.shape[:2]
        car_x = w // 2
        car_y = int(h * 0.75)
        return car_x, car_y
    def get_boundary_distances_from_avg(avg_left, avg_right, car_x, car_y):
        if not avg_left or not avg_right:
            return None, None
       
        left_x  = min(avg_left,  key=lambda pt: abs(pt[1] - car_y))[0]
        right_x = min(avg_right, key=lambda pt: abs(pt[1] - car_y))[0]


        dist_left  = car_x - left_x
        dist_right = right_x - car_x

        # car detected outside road mask:
        # dist_left < 0  = left boundary is to the RIGHT of car
        # dist_right < 0 = right boundary is to the LEFT of car
        # return None, None → steering = 0.0
        if dist_left < 0 or dist_right < 0:
            return None, None
        
        # normal detection:
        # dist_left > 0 and dist_right > 0
        # compute offset normally
        return dist_left, dist_right

    def get_steering(dist_left, dist_right):
        # only real failure case to handle
        if dist_left is None or dist_right is None:
            return 0.0  # no detection — go straight

        total = dist_left + dist_right
        offset = (dist_right - dist_left) / total
        return offset  # -1.0 = far left, 0 = centered, +1.0 = far right


    # Note: Cant just steer every time car is out of road detection region
    # Set a threshold of number of frames before reacting
    OUT_OF_BOUNDS_THRESHOLD = 10
    out_of_bounds_counter   = 0
    steering_active         = False


    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_small = cv2.resize(frame, (256, 256))
        with torch.no_grad():
            inp  = preprocess(frame_small).to(device)
            pred = model(inp).squeeze().cpu().numpy()
        
        mask_np = cv2.resize(pred, (dw, dh))
        # add history and average
        mask_history.append(mask_np)
        smooth_mask = np.mean(mask_history, axis=0)

        # threshold average mask
        mask_bin = (smooth_mask > THRESHOLD).astype(np.uint8) * 255

        #cleanup
        kernel   = np.ones((25, 25), np.uint8)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN,  kernel)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)

        # always draw overlay using current mask_bin
        display_frame = cv2.resize(frame, (dw, dh))
        result, avg_left, avg_right = draw_overlay(display_frame, mask_bin, left_history, right_history)

        car_x, car_y   = get_car_position(display_frame)
        dist_l, dist_r = get_boundary_distances_from_avg(avg_left, avg_right, car_x, car_y)
        steering       = get_steering(dist_l, dist_r)

        if dist_l is None or dist_l < 0 or dist_r < 0:
            out_of_bounds_counter += 1
        else:
            out_of_bounds_counter = 0
            steering_active       = False

        # steering after passing threshold
        if out_of_bounds_counter >= OUT_OF_BOUNDS_THRESHOLD:
            steering_active = True

        cv2.circle(result, (car_x, car_y), 8, (0, 255, 255), -1)
        if dist_l and dist_r:
            cv2.line(result, (car_x - dist_l, car_y), (car_x, car_y), (255, 80, 0), 2)
            cv2.line(result, (car_x, car_y), (car_x + dist_r, car_y), (0, 80, 255), 2)

        # display
        status = f'CORRECTING: {steering:.2f}' if steering_active else 'ON ROAD'
        color  = (0, 0, 255) if steering_active else (0, 255, 0)
        cv2.putText(result, status, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(result, f'Out of bounds: {out_of_bounds_counter} frames',
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        out.write(result)

        mask_color = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
        compare    = np.hstack([display_frame, mask_color])
        out_compare.write(compare)

        frame_num += 1

    cap.release()
    out.release()
    out_compare.release()

if __name__ == '__main__':
    run()