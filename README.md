# PreviewVideoMonitorPro is a ComfyUI custom node for video preview on multiple monitors. Lets you select one of up to six secondary monitors to display generated videos in fullscreen!

<img width="1921" height="1049" alt="pvmp3" src="https://github.com/user-attachments/assets/97fea8ac-985b-4e15-aa97-1d189942771d" />

<img width="1088" height="559" alt="pvmp2" src="https://github.com/user-attachments/assets/0f21c302-c20f-4358-830e-05793f474410" />

#

📰 NEWS📰

🖥️ Preview Video Monitor Pro 🖥️ — Version 2.0 is out!!

🚀 Major Update — Real-Time RAM-Cached Playback

This update delivers a **massive performance leap** for video playback in ComfyUI, transforming the node into a fully RAM-accelerated, 8-bit optimized video preview system.

✨ Highlights

⚡ Full RAM Caching
- Videos are now decoded once and fully cached in memory as 8-bit RGB frames.  
- Playback no longer depends on OpenCV’s disk streaming per frame.  
- Re-runs of the same video reuse cache instantly.

🎨 8-Bit Color Pipeline (was 32-bit Float)
- Replaced float32 frame storage with compact 8-bit RGB surfaces.  
- Cuts memory use by **~75%** and eliminates costly float↔uint8 conversions.  
- No visible quality loss for MP4/H.264 previews.

🧮 Pre-Scaling on Load
- All frames are pre-scaled to the target monitor resolution and fit mode during extraction.  
- Eliminates per-frame OpenCV resize calls during playback.

🧠 Smart Global Cache
- Added a shared global cache keyed by video path + resolution + fit mode.  
- Cached videos persist across prompts until ComfyUI restarts.  
- Automatically bypasses re-decoding for repeated runs.

🧱 Memory-Safe Design
- Built-in memory threshold (default: **2 GiB**) to prevent overload.  
- Large videos gracefully fall back to streaming mode with a log warning.  
- Limit adjustable via environment variable:
  ```bash
  set PREVIEWVM_MAX_CACHE_BYTES=4294967296

#

☕︎ Support: A small donation helps! ☕︎ https://buymeacoffee.com/nicolaslandajo

⭐ Star this repository — it’s quick, free, and helps others discover the project

🌍 Share this tool with anyone you think would enjoy it! 🌍

🪲 Report any bugs you come across

📄 Contribute improvements

#

PLEASE READ THE LICENSE FILE

#

Please note that at the moment this tool is developed and tested over ComfyUI Desktop only, not Portable one. But it should work fine if you want to try it. I will go over testing on Portable soon.

