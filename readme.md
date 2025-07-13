**README.md**

# Player Re-Identification in Sports Footage

This project is a submission for the AI Internship Assignment by **Liat.ai**. The objective is to track football players in a video and consistently assign the same ID to each player, even when they leave and re-enter the frame.

---

## 📊 Project Description

We implemented a re-identification and tracking pipeline using:

- YOLOv8 for object (player) detection
- Color clustering (KMeans) for jersey identification
- Cosine similarity and heuristics for player tracking

The system processes the video frame-by-frame and overlays ID and team info on each detected player.

---

## 📂 Folder Contents

- `player_tracker.py` - main tracking logic
- `model.pt` - pre-trained YOLOv8 model (not included due to size, see below)
- `input_video.mp4` - input test video (replace with your own)
- `resultant_output.mp4` - output with bounding boxes, IDs, and team stats
- `report.md` - documentation of approach, learnings, and reflection

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the YOLOv8 Model

Download from:
[Model Link](https://drive.google.com/file/d/1-510SHOSB9UXYPenOoZNAMScrePVcMD/view)
Save as `model.pt` in the same folder.

### 3. Run the Code

```bash
python Tracking.py
```

Make sure you have:

- `input_video.mp4` in the same folder (or change the path in code)
- The `model.pt` file loaded correctly

---

## 🌐 Output

The output video will be saved as `resultant_output.mp4` and shows:

- Bounding boxes with Player ID and Team
- Frame-wise team distribution
- Total players tracked
- FPS and processing stats

---

## 🌐 Notes

- Processing might be slow on lower-end systems or without GPU.
- The current version processes frame-by-frame but only visualizes 1 frame unless fully debugged.
- Designed for learning, extendability, and experimentation.

---

## 📘 Author

Built and debugged by **\Abhishek Sharma** as part of Liat.ai Internship Application.

---
