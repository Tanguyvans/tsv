# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Railway surface defect dataset and analysis tools. The project deals with classifying rail surface defects into 7 classes (Flakings, Squats, Spallings, Shellings, Cracks, Joints, Grooves). The dataset is heavily imbalanced — Flakings+Squats account for ~91% of images.

## Setup

- Python 3.11 with virtualenv at `./venv`
- Activate: `source venv/bin/activate`
- Dependencies: `opencv-python` (cv2), `numpy`

## Running

```bash
# Play color video (default)
python read_video.py

# Play specific video
python read_video.py Images/depth_0.mkv
```

## Data Layout

- `data/surface/` — 5,153 images across 7 defect class subdirectories
- `Images/` — video files (color_0.mkv, depth_0.mkv) and timing data (time_0.time)

## Language

Project documentation and code comments are in French.
