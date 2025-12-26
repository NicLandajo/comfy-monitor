<img width="1543" height="595" alt="pvmp" src="https://github.com/user-attachments/assets/f1117b5a-9688-4714-8a0b-a8af15919acc" />

<div align="center">

# PreviewVideoMonitor 🖥️🖥️ Multimonitor Fullscreen

PreviewVideoMonitor is a ComfyUI custom node for video preview on multiple physical monitors.

Lets you select one of up to six secondary monitors to display generations instantly for both video and image-batch.

#

# 🔥 NEWS 🔥

### V5.0 - NEW PREVIEW EXPERIENCE
### 🚀 RELEASE JANUARY 5 2026 🚀

Watch the Teaser Demo here> https://www.youtube.com/watch?v=yEAaxmaA-CE

</div>

#
<img width="1920" height="1080" alt="001" src="https://github.com/user-attachments/assets/86f49d41-8d19-46bd-bdeb-5a031cee1713" />

The Monitor Interface

#
<img width="1920" height="1080" alt="003" src="https://github.com/user-attachments/assets/dca8fbe3-ba97-48b3-bbfd-6b8657ad10e2" />

In/Out Marking System. Set Range Instantly - Press I for IN point, O for OUT point at current frame, reset with P; or use the "In" - "Reset" - "Out" red circles

#
<img width="1920" height="1080" alt="002" src="https://github.com/user-attachments/assets/00be4ec5-ab4b-4fea-981a-234ebe9bb10d" />

Generations cycling menu - Display your creative history instantly to the monitor. Rename generations as you go.

#
<img width="1920" height="1080" alt="005" src="https://github.com/user-attachments/assets/fec4bbfd-ab30-4471-9aba-4a036a2030d7" />

ClearCache. "Deletion confirmation" and "Explore" buttons.

#
<img width="1920" height="1080" alt="006" src="https://github.com/user-attachments/assets/1c4079cb-8fec-477a-b4a1-0acd8cda5d0d" />

Take Snapshots of your creative process anytime. Whatever is on screen at that moment will be saved to a PNG image with your comfyui workflow injected in it.
Share the PNG or open it later in comfy and continue working from there. The Snapshots and Generations work orchestrated together maintaining naming sync.
Explore the dir and grab your history work and continue creating.

#
<img width="1920" height="1080" alt="009" src="https://github.com/user-attachments/assets/45f861d6-9a68-40f2-8753-0b1d376ca2d5" />

Wipe/Compare/A/B - You can wipe/compare the current generation on screen against any other gen in the Generations dropup menu.
As soon as you click and hold, the yellow wipe interface dissapears and you see a clean wipe that you can move side to side.
As soon as you release the left click, the yellow wipe interface comes back right where your pointer is.
Deactivate wipe with W shortcut.

#
<img width="1921" height="977" alt="007" src="https://github.com/user-attachments/assets/872f7cd5-36fa-4974-b751-417ef567b8b5" />

The Node

#

- 🏭 Currently working on

- In addition to the already existing wipe compare, "side by side" will be added

- 🕰️ Long term development for 2026 for version V6 is a 3D system in the monitor to visualize 3D geometry and gaussian splats

#

<div align="center">

# FEATURES

</div>

-🎬 Professional Playback Control Suite

Intuitive Timeline Scrubbing - Click and drag anywhere above the timeline for precise frame control

Frame-Perfect Navigation - Step through frames with keyboard arrows (← →)

Smart Playback Modes - Forward, Backward, and Ping-Pong playback with single-click switching

Instant Play/Pause - Spacebar toggles playback instantly, right from your keyboard home row

-🖥️ Immersive Viewing Experience

One-Click Fullscreen - Press 5 to enter true fullscreen mode, ESC or 5 again to exit

Smart Fit Modes - Instant switching using numbers 1:1 (1), Fit (2), Width (3), Height (4), and Fullscreen (5). Press again and you retun to Fit

Keyboard-Optimized Shortcuts - All controls under your left hand without looking at the keyboard

Multi-Monitor Ready - Display on any monitor with automatic resolution detection

-🎯 Professional In/Out Marking System

Set Range Instantly - Press I for IN point, O for OUT point at current frame

Visual Timeline Feedback - Blue highlighted areas with red marker lines show your selection

Smart Playback Bounds - All playback modes respect your marked range automatically

Quick Reset - Press P to instantly reset to full media length

-🎮 Total Interactive Control

Dynamic Real-Time Interface - All controls moved from static nodes to interactive monitor

Intuitive Zoom & Pan - Mouse wheel zooms anchored to cursor, right-click drag pans

Instant Fit Modes - Click buttons or use number keys 12345 for instant view adjustments

Live Visual Feedback - Buttons light up with color-coded active states

-🧠 Smart Persistent Generations Vault Cache

Cross-Session Preservation - Your creative generations survive ComfyUI restarts and system reboots

Unlimited Version History - Store every iteration of your work without worrying about limits during sessions

Rename any Generation instantly to keep your creative sessions organized and highlight the moments that inspired you!

Instant Time Travel - Dropup menu with keyboard navigation (↑ ↓ ENTER) to revisit any point in your creative journey
Visual Generation Gallery - Hover and click to instantly resurrect any previous creation

Take Snapshots of your creative process anytime. Whatever is on screen at that moment will be saved to a PNG image with
your comfyui workflow injected in it.
The Snapshots and Generations work orchestrated together maintaining naming sync.
PreviewVideoMonitor saves snapshos data inside its own default folder custom_nodes/PreviewVideoMonitorPro/,
but you can change this to your own custom directory directly from the node with one click, super easy,
and giving you full control over where Snaps are kept.

Smart Memory Management - Configurable archive limits with intelligent oldest-first curation

Your Creative Legacy - Everything persists until you consciously choose to clear the cache

-⚡ Performance Engine

GPU-Accelerated Processing - Automatic CUDA detection with OpenCV & PyTorch backend fallbacks

Smart Caching System - Dynamic frame caching with memory management

Real-Time Resolution Handling - Automatic scaling with maintained aspect ratios

Custom FPS Control - Playback at your exact desired framerate

-🎨 Enhanced Visual Experience

Live Frame Counter - Real-time display with custom starting frame numbers

Resolution Overlay - Always-visible original media resolution

Color-Coded Interface - Intuitive button states with meaningful color schemes

Professional Timeline - Extended scrubbing area for easy frame selection

-🔄 Universal Compatibility

Multi-Format Support - Videos, image sequences, tensors, and all ComfyUI video formats

Flexible Input Sources - Direct files, node outputs, or generated content

Seamless Workflow Integration - Works alongside your existing ComfyUI pipelines

Cross-Platform Ready - Windows, Linux, macOS with automatic fallback handling

-🎮 Quick Start Controls:

Space = Play/Pause

← → = Frame stepping

I/O = Set IN/OUT points

P = Reset to full length

1-5 = Fit modes (1:1, Fit, Width, Height, Fullscreen)

ESC = Exit fullscreen

Q = Close viewer

ENTER = Toggle generations menu


-📺 Cycling the RGB channels: Use "r" "g" and "b" shortcuts to see the red green and blue channels in the monitor

#

Important Notes

PreviewVideoMonitor is an advanced video/image viewer with powerful caching and workflow-saving features. However, it is not designed to be your primary file management system.
Key Points:

Always use Save nodes (e.g., Save Image, Save Video, etc.) to permanently store your final work in your regular working directory or custom pipelines.
PreviewVideoMonitor should currently be treated as a volatile preview system — ideal for creative sessions and quick iterations, but not for long-term storage.
PreviewVideoMonitor can save files to disk, its internal session data and cached files are stored inside its own default folder
custom_nodes/PreviewVideoMonitorPro/, but you can change this to your own custom working directory directly from the node, super easy, and giving you full control over where files are kept.
This internal storage may be cleared by users (or accidentally by you) or changed between sessions or updates, so do not rely on it for keeping your work 100% safe.

#
<div align="center">
  
# PreviewVideoMonitor V3.3
# current version soon to be deprecated

</div>

<img width="1848" height="925" alt="3 2interface_01" src="https://github.com/user-attachments/assets/85badbaa-de59-4eb8-a7ad-d4c264aae90c" />

<img width="1851" height="929" alt="3 2interface_02" src="https://github.com/user-attachments/assets/adf23364-5856-4b66-86cd-1ec1290459b1" />

<img width="1920" height="756" alt="Banner_github_guy" src="https://github.com/user-attachments/assets/c0f3719d-92d9-434d-b562-aefa56d00d10" />

#

📼 WATCH THE TUTORIAL: https://www.youtube.com/watch?v=q6JLcth2LnY

#

Please read the install instructions.

#

🚀 Preview Video Monitor Pro v3.3 — Faster, Cleaner, and More Precise Than Ever

Preview Video Monitor Pro v3.3 brings a smoother, more intuitive experience with accurate previews.

✨ What’s New

🎚️ workflow_fps Moved to the Top
Your main timing control is now the first setting you see — making setup quicker and more natural.

🎞️ Improved preview_fps Control now accepts both

"Smart" automatic detection when you want the tool to choose the best framerate, or
Exact custom values like 24, 29.97, 48, or 60 when you need exact animation timing.

⚡ Smoother Playback, Same Rock-Solid Performance

v3.3 builds on the fast, stable playback you already know:

Frames load instantly thanks to optimized caching

Previews feel snappy and lightweight

📁🚂 Preview Video Monitor Pro automatically creates two folders inside /custom_nodes/PreviewVideoMonitorPro/

/cache/: stores temporary optimized video files and pre-scaled frames

/__pycache__/: standard Python bytecode compiled at runtime

Both folders are safe, automatic, and self-maintaining.

#

☕︎ HEY!!!👋 Please support, just $1 can make a real difference ☕︎

☕︎ https://buymeacoffee.com/nicolaslandajo ☕︎

⭐Can’t or don’t feel like donating? No worries at all! Just giving the repo a Star helps others discover it, and that support means the world

🌍 Share this tool with anyone you think would enjoy it! 🌍

🪲 Report any bugs you come across

📄 Contribute improvements

#

🐍 Special Thanks to PyGame 🐍 Preview Image Monitor relies on the powerful Pygame library to handle cross-platform window creation, display management, and image rendering. We extend our sincere thanks to the Pygame developers and contributors for their invaluable work. https://github.com/pygame

#

PLEASE READ THE LICENSE FILE

#

Please note that at the moment this tool is developed and tested over Windows 10 and ComfyUI Desktop only, not Portable one. But it should work fine if you want to try it. I will go over testing on Portable soon.

