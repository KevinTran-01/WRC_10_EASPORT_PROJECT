"""
STEP 2 (updated) — Convert CVAT XML annotations to binary mask PNGs
Run this after exporting annotations from CVAT
"""

import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os
from glob import glob

ANNOTATION_DIR = 'cvat_export_xml'   # XML files
IMAGE_DIR      = 'data/images'   # original frames
MASK_DIR       = 'data/masks'    # masks will be saved here

os.makedirs(MASK_DIR, exist_ok=True)

xml_files = glob(f'{ANNOTATION_DIR}/*.xml')

if len(xml_files) == 0:
    print(f'ERROR: No XML files found in {ANNOTATION_DIR}/')
    print('Make sure you unzipped the CVAT export there.')
    exit()

print(f'Found {len(xml_files)} annotation files. Converting to masks...')

converted = 0
skipped   = 0

for xml_path in sorted(xml_files):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # get image filename and size from XML
    filename = root.find('filename').text
    nrows    = int(root.find('imagesize/nrows').text)
    ncols    = int(root.find('imagesize/ncols').text)

    # blank black mask
    mask = np.zeros((nrows, ncols), dtype=np.uint8)

    road_found = False
    for obj in root.findall('object'):
        label = obj.find('name').text
        if label != 'road':
            continue

        # extract polygon points
        pts = []
        for pt in obj.findall('polygon/pt'):
            x = float(pt.find('x').text)
            y = float(pt.find('y').text)
            pts.append([int(x), int(y)])

        if len(pts) < 3:
            continue

        poly = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [poly], 255)
        road_found = True

    if not road_found:
        print(f'  WARNING: No road polygon in {os.path.basename(xml_path)} — skipping')
        skipped += 1
        continue

    # save mask with matching image filename
    base      = os.path.splitext(filename)[0]
    mask_path = f'{MASK_DIR}/{base}_mask.png'
    cv2.imwrite(mask_path, mask)
    converted += 1

    if converted % 100 == 0:
        print(f'  {converted} converted so far...')

print(f'Converted {converted} ')
print(f'Skipped {skipped}')
