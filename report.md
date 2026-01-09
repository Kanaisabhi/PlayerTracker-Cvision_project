**report.md**

# Player Re-Identification Assignment Report

## 🔍 Overview

This project was part to test real-world problem-solving in computer vision. The objective was to identify and track football players using object detection, ensuring consistent IDs even if they leave and re-enter the frame.

---

## 🔧 My Approach

- I started with theoretical understanding of tracking, detection, and re-identification.
- Explored YOLOv8 for detection and wrote logic to extract each player.
- Used jersey color and cosine similarity to reassign IDs.
- Built a frame-by-frame tracking system with visualization and stats overlay.

---

## 🧮 Learnings

- This was my **first major real-world computer vision project**.
- I learned how **frame-by-frame processing, tracking, and ID assignment** actually work.
- Understood how difficult **re-identification** can be due to movement, occlusion, and camera angles.
- Got deep into **OpenCV**, **KMeans clustering**, and **cosine similarity**.

Before this, I had mostly theoretical understanding. This project grounded my learning into practice and helped me understand how all these parts come together.

---

## ⚡ What I Tried

- Manually writing a tracker with stats, color analysis, frame loop
- Debugging with OpenCV display and JSON report summaries
- Using public code (e.g., Claude's version) to learn and compare modular implementation
- Merged concepts and attempted multiple refactors to fix bugs

---

## ❌ What Didn't Work

- My personal + resourced implementation processed only 1 frame (issue with video writer loop or condition breaking early).
- Tried debugging frame loop, `out.write()`, and FPS count but still stuck.

Despite this, I **understood every function's role deeply**, and I believe I can now rebuild it end-to-end from scratch.

---

## 🚀 What I'd Do Next

Given more time, I would:

- Fully debug the `VideoWriter` and ensure multi-frame tracking
- Add a **Kalman Filter** or **DeepSORT** style tracker
- Try **cross-camera player re-ID** using visual + spatial embedding
- Improve FPS performance with frame skipping or GPU acceleration

---

## ✨ Final Reflection

This project has:

- Strengthened my understanding of **real-time CV pipelines**
- Made me confident in combining **AI models with custom tracking** logic
- Taught me how to **think like an engineer** and not just a user of models

Even if the full video output isn’t working yet, I know the logic, and I now have the skills to build more complex systems like **F1 car tracking**, **match analytics**, and **sports predictions**.

Thanks for reviewing my work. I’d love to discuss my learnings, improvements, or ideas in a follow-up!

---

**Submitted by:** Abhishek
