# CrateTools PreviewVideoMonitorPro v5.2 - Generation Tracking System and Viewer - January 5 2026
# GPU-accelerated

import os
import time
import uuid
import glob
import sys
import traceback
import hashlib
import json
import random
from threading import Thread, Lock, Event, RLock
import weakref

import numpy as np
from PIL import Image
import io
import struct

# Optional libs: guarded so module import is safe
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False

try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception:
    pygame = None
    PYGAME_AVAILABLE = False

try:
    from screeninfo import get_monitors
    SCREENINFO_AVAILABLE = True
except Exception:
    get_monitors = None
    SCREENINFO_AVAILABLE = False

# CUPY is optional — we don't rely on it as primary backend here.
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

import tempfile
import shutil
import subprocess  # NEW: For opening explorer

# Detect GPU capability (prefer OpenCV CUDA; else PyTorch CUDA)


def _detect_gpu():
    cv2_cuda = False
    torch_cuda = False
    try:
        if CV2_AVAILABLE and hasattr(cv2, 'cuda'):
            try:
                # getCudaEnabledDeviceCount may throw if OpenCV not built with CUDA
                cnt = cv2.cuda.getCudaEnabledDeviceCount()
                cv2_cuda = bool(cnt and cnt > 0)
            except Exception:
                cv2_cuda = False
    except Exception:
        cv2_cuda = False
    try:
        if TORCH_AVAILABLE:
            torch_cuda = torch.cuda.is_available()
    except Exception:
        torch_cuda = False
    return {'cv2_cuda': cv2_cuda, 'torch_cuda': torch_cuda}


_GPU_INFO = _detect_gpu()
CV2_CUDA_AVAILABLE = _GPU_INFO['cv2_cuda']
TORCH_CUDA_AVAILABLE = _GPU_INFO['torch_cuda']
GPU_AVAILABLE = CV2_CUDA_AVAILABLE or TORCH_CUDA_AVAILABLE

# Constants
BASE_DIR = os.path.dirname(__file__)
RUNS_CACHE_DIR = os.path.join(BASE_DIR, "runs_cache")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")  # NEW: Separate snapshots directory
RUNS_METADATA_FILE = os.path.join(BASE_DIR, "runs_metadata.json")
LOGOS_DIR = os.path.join(BASE_DIR, "logos")
# NEW: Temp workflow directory
TEMP_WORKFLOW_DIR = os.path.join(RUNS_CACHE_DIR, "temp_workflow")
TEMP_WORKFLOW_FILE = os.path.join(TEMP_WORKFLOW_DIR, "workflow.json")

os.makedirs(RUNS_CACHE_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)  # NEW: Create snapshots directory
os.makedirs(LOGOS_DIR, exist_ok=True)
# NEW: Create temp workflow directory
os.makedirs(TEMP_WORKFLOW_DIR, exist_ok=True)

# default maximum bytes allowed for a single video cache (2 GiB)
DEFAULT_MAX_CACHE_BYTES = int(os.environ.get(
    "PREVIEWVM_MAX_CACHE_BYTES", 2 * 1024 ** 3))


class PreviewVideoMonitorPro:
    """
    Node implementing v5.2 with generation tracking system and GPU-accelerated frame ops.
    """

    @classmethod
    def INPUT_TYPES(cls):
        monitor_list = cls.get_monitors()
        return {
            "required": {
                "source": (["video", "images"], {"default": "video"}),
                # REMOVED: workflow_fps and preview_fps (replaced by UI FPS control)
                "monitor": (monitor_list, {"default": monitor_list[0]}),
                # CHANGED: Boolean switch (True = On, False = Off)
                "power_state": ("BOOLEAN", {"default": True}),
                "target_resolution": (["1920x1080", "3840x2160"], {"default": "1920x1080"}),
                # REMOVED: first_frame_is (moved to FPS dropup UI control)
                # REMOVED: max_generations (storage management is user's responsibility)
                # Parameter name matches UI terminology (Generations dropdown)
                # The value is used internally for cache/file naming
                "generations_name": ("STRING", {"default": "Generation"}),
                # NEW: Snapshot workflow boolean
                "snapshot_workflow": ("BOOLEAN", {"default": True}),
                # NEW: Snapshot path - "smart" for default, or custom directory path
                "snapshot_path": ("STRING", {"default": "smart"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "video": ("VIDEO",),
            },
            # CRITICAL FIX: Add hidden inputs for ComfyUI workflow data
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "display_video"
    OUTPUT_NODE = True
    CATEGORY = "video/preview"

    # Shared global cache keyed by (video/path/res/fit)
    _global_cache = {}
    _global_cache_lock = Lock()

    def __init__(self):
        # playback / thread
        self.running = False
        self.thread = None
        self.thread_lock = Lock()
        self.cache_lock = Lock()
        self.new_content_ready = Event()  # Signal when new content is ready
        self.pending_content = None  # Store pending content to load

        # caches
        # pre-scaled frames for fast path (numpy [T,H,W,3])
        self.cached_scaled_frames = None
        self.cached_original_frames = None  # original-resolution frames for dynamic path
        self.cached_surfaces = None         # list of pygame.Surface objects for fast blit
        
        # NEW: Color space transformation cache
        self.cached_color_transformed_frames = None  # Cache for color-transformed original frames
        self.cached_color_space = "sRGB"  # Track which color space the cache is for
        
        self.cache_key = None
        self.current_video_fps = 24.0
        # Store original resolution for display
        self.original_resolution = (0, 0)
        self.target_res = (1920, 1080)

        # viewer interactive state
        self.viewer_zoom = 1.0
        self.viewer_offset = [0, 0]
        self.viewer_fit_mode = "fit"  # Start with fit mode
        self.viewer_playback_mode = "Forward"  # Internal default
        self.viewer_paused = True   # start paused by default; Play will set False
        self.channel_mode = "RGB"  # Can be "RGB", "Red", "Green", "Blue"
        self.dragging = False
        self.drag_button = None
        self.last_mouse_pos = (0, 0)
        self.current_frame = 0
        self.direction = 1
        
        # NEW: FPS Control (replaces workflow_fps/preview_fps)
        self.user_fps = 24.0  # User-controlled playback FPS (default 24)
        self.fps_dropup_open = False  # FPS dropup state
        self.fps_dropup_hover_index = -1  # Hover index in FPS dropup
        self.fps_custom_editing = False  # Is user editing custom FPS field
        self.fps_custom_text = "24.000"  # Custom FPS input text
        self.fps_presets = [60.0, 59.94, 50.0, 48.0, 30.0, 29.97, 25.0, 24.0, 23.976, 15.0, 12.0, 8.0]  # High to low
        
        # NEW: First Frame control (moved from node parameter to UI)
        self.first_frame_editing = False  # Is user editing first frame field
        self.first_frame_text = "1"  # First frame input text (default 1)
        self.first_frame_offset = 0  # Actual offset (first_frame_text - 1)

        # toolbar config - INCREASED HEIGHT to accommodate timeline
        # Increased from 44 to fit timeline (70px total)
        self.toolbar_height = 70
        self.timeline_height = 30  # Increased height for better visibility
        self.buttons_height = 40   # Height for buttons area
        self.active_button = None
        self.button_hover = None

        # timeline scrubber state
        self.scrubbing = False
        self.timeline_extended_rect = None  # Extended area for easier scrubbing

        # Track counter update - simple approach
        self.counter_update_frame = 0

        # frame counter optimization
        self.last_display_frame = -1
        self.last_total_frames = -1
        self.force_counter_update = True  # Force update on first draw

        # Generation tracking system
        self.current_generation_id = None
        self.generations_metadata = self._load_generations_metadata()
        
        # Check for old cache format and migrate
        self._check_and_migrate_cache()
        
        self.dropup_open = False
        self.dropup_hover_index = -1
        self.dropup_scroll_offset = 0
        self.dropup_selected_index = -1  # Track keyboard selection in dropup

        # NEW: In-place editing state
        # Which item is being edited (-1 = none)
        self.dropup_editing_index = -1
        self.dropup_edit_text = ""  # Current edit text
        # When editing started (for cursor blink)
        self.dropup_edit_start_time = 0
        self.dropup_edit_cursor_visible = True  # Cursor blink state
        # Track if editing started via keyboard
        self.dropup_edit_initiated_by_keyboard = False
        # Track cursor position within edit text
        self.dropup_edit_cursor_pos = 0
        # NEW: Text selection
        self.dropup_edit_selection_start = -1  # -1 = no selection
        self.dropup_edit_selection_end = -1
        self.dropup_edit_mouse_dragging = False  # Track if mouse is dragging for selection
        # Track if backspace is being held down
        self.dropup_backspace_held = False
        
        # NEW: Delete confirmation state
        self.dropup_delete_confirm_index = -1  # Which generation is pending deletion (-1 = none)
        self.dropup_delete_confirm_gen_id = None  # Generation ID pending deletion
        self.dropup_backspace_start_time = 0
        
        # NEW: Clear cache confirmation states
        self.clearcache_clear_confirm = False  # Confirm before clearing generations
        self.snapshot_clear_confirm = False    # Confirm before clearing snapshots

        # Fullscreen state
        self.fullscreen_mode = False
        self.pre_fullscreen_fit_mode = "fit"
        self.pre_fullscreen_window_size = (1920, 1080)

        # In/Out marking system
        self.original_in_point = 0
        self.original_out_point = 0
        self.user_in_point = 0
        self.user_out_point = 0
        self.user_marks_active = False

        # Session state for logo screens
        self.first_run_of_session = True
        self.current_source_type = None  # Track if current source is video or images
        # Track if we're showing cache clear screen
        self.showing_cache_clear_screen = False
        self.cache_clear_start_time = 0  # NEW: Track when cache clear screen started
        self.showing_wait_screen = False  # Track if we're showing wait screen
        self.processing_new_content = False  # Track if we're processing new content (before MP4 is ready)
        
        # Snapshot path management
        self.current_snapshot_path = SNAPSHOTS_DIR  # Default, can be overridden by user

        # GPU info for this instance - read from module globals
        self.cv2_cuda = CV2_CUDA_AVAILABLE
        self.torch_cuda = TORCH_CUDA_AVAILABLE
        self.gpu_available = GPU_AVAILABLE
        if self.gpu_available:
            self.log(
                f"GPU acceleration enabled: cv2_cuda={self.cv2_cuda}, torch_cuda={self.torch_cuda}")
        else:
            self.log("GPU acceleration not detected — running CPU-only mode")

        # =========== OPTIMIZATION CACHES ===========
        self._ui_cache = {}  # Cache for rendered UI elements
        self._ui_cache_dirty = True  # Flag to indicate UI needs re-rendering
        self._frame_cache_dirty = True  # NEW: Separate flag for frame cache
        self._last_mouse_pos = (0, 0)  # Track mouse movement for hover updates
        self._last_ui_state_hash = 0  # Hash of UI state to detect changes
        self._cached_toolbar_surface = None  # Pre-rendered toolbar
        self._cached_timeline_surface = None  # Pre-rendered timeline
        self._cached_counter_surfaces = None  # Pre-rendered counter text
        self._last_frame_rendered = -1  # Last frame index rendered
        self._last_fit_mode = None  # Last fit mode used
        self._last_zoom = 1.0  # Last zoom level
        self._last_offset = (0, 0)  # Last offset
        
        # Multi-frame surface cache - cache ALL frames for instant replay!
        self._frame_surface_cache = {}  # {state_hash: (surface, position)}
        self._frame_surface_lock = RLock()  # NEW: Lock for frame surface access

        # Performance monitoring
        self._render_time_total = 0
        self._render_time_ui = 0
        self._render_time_video = 0
        self._frame_count = 0

        # NEW: Mouse click debouncing (prevent multiple triggers from held button)
        self._prev_mouse_pressed = False  # Previous frame mouse button state

        # NEW: Snapshot dropup state
        self.snapshot_dropup_open = False
        self.snapshot_button_hover = False
        self.snapshot_button_active = False
        self.last_snapshot_time = 0  # For visual feedback

        # NEW: ClearCache dropup state
        self.clearcache_dropup_open = False
        self.clearcache_button_hover = False
        self.clearcache_button_active = False

        # =========== COLOR SPACE SYSTEM - LOCKED TO sRGB ===========
        # Color space is ALWAYS sRGB (gamut support removed for performance)
        self.selected_color_space = "sRGB"  # Locked to sRGB only
        
        # Dummy variables for removed gamut UI (prevents crashes)
        self.color_space_dropup_open = False
        self.color_spaces = ["sRGB"]
        self.color_space_hover_index = -1
        self.color_space_selected_index = 0
        self.gamuts_hover = False
        self.gamuts_active = False

        # NEW: Workflow snapshot state
        self.snapshot_workflow_enabled = True  # Default state
        self.workflow_metadata = None  # Store current workflow metadata
        self.snapshot_saving = False  # Track when snapshot is being saved
        
        # NEW: Camera iris animation for snapshot visual feedback
        self.iris_animation_active = False  # True when iris animation is playing
        self.iris_animation_start_time = 0  # When animation started (milliseconds)
        self.iris_animation_duration = 2000  # 2 seconds total duration
        
        # =========== VISION MODULE ===========
        # Vision module: Color grading + analysis tools (covers timeline when active)
        self.vision_module_open = False  # True when Vision module is visible
        
        # Color grading sliders (exposure stops model)
        self.vision_gain = 0.0  # Range: -6.0 to +6.0 stops (0 = neutral)
        self.vision_gamma = 1.0  # Range: 0.1 - 3.0 (or as needed)
        self.vision_saturation = 1.0  # Range: 0.0 - 3.0 (or as needed)
        
        # Slider dragging state
        self.vision_dragging_slider = None  # "gain", "gamma", "saturation", or None
        
        # Gamma LUT optimization (pre-calculated lookup table for speed)
        self.vision_gamma_lut = None  # Cached gamma correction lookup table
        self.vision_gamma_lut_value = None  # The gamma value used to build current LUT
        
        # Analysis overlays (toggles)
        self.vision_zebra_active = False  # Zebra pattern for clipping
        self.vision_grid_active = False  # Rule of thirds grid
        self.vision_guides_active = False  # Broadcast guides overlay
        
        # Aspect ratio masks (cycle through 10 ratios)
        self.vision_mask_index = 0  # 0 = off, 1-10 = different ratios
        self.vision_mask_ratios = [
            ("OFF", None),
            ("2.39:1", 2.39),
            ("2.35:1", 2.35),
            ("2.00:1", 2.00),
            ("1.85:1", 1.85),
            ("16:9", 16/9),
            ("3:2", 3/2),
            ("4:3", 4/3),
            ("1:1", 1.0),
            ("9:16", 9/16),
            ("4:5", 4/5)
        ]
        
        # Persistence: Guides and Masks can stay visible when Vision module closed
        self.vision_guides_persist = False  # Keep guides visible after closing Vision
        self.vision_mask_persist = False    # Keep mask visible after closing Vision
        
        # Flip/Flop mode (0=normal, 1=h-flip, 2=v-flip, 3=both)
        self.vision_flip_mode = 0
        
        # Color Picker (P button)
        self.vision_picker_active = False  # Picker enabled/disabled
        self.vision_picker_pos = None  # (x, y) in screen coordinates, None = center
        self.vision_picker_dragging = False  # Currently dragging picker
        
        # Scopes (only one active at a time)
        self.vision_active_scope = None  # "vector", "histogram", "parade", "waveform", or None
        
        # NEW: Instance ID for unique workflow storage
        self.instance_id = None
        self.current_workflow_data = None  # Store workflow data for this instance
        
        # Clean up old workflow files
        self._cleanup_old_workflow_files()

        # =========== WIPE COMPARISON SYSTEM ===========
        self.wipe_active = False  # True when wipe is visible
        self.wipe_comparison_gen_id = None  # Which generation is selected for wipe
        self.wipe_comparison_folder = None  # Folder/file path of selected generation
        self.wipe_position = 0.5  # Video-relative position (0.0-1.0) - FIXED: Now relative to video, not screen
        self.wipe_dragging = False  # Whether user is dragging wipe line
        self.wipe_comparison_frames = None  # Cache for loaded comparison frames
        self.wipe_comparison_frame_idx = -1  # Last loaded frame index
        # NEW: Store current video display rectangle for wipe calculations
        self.current_video_rect = None  # (x, y, width, height) of video on screen
        # NEW: Store comparison video dimensions
        self.comparison_video_size = (0, 0)  # (width, height) of comparison video
        
        # =========== SIDE-BY-SIDE COMPARISON SYSTEM ===========
        self.sbs_active = False  # True when side-by-side is visible
        self.sbs_comparison_gen_id = None  # Which generation is selected for SBS
        self.sbs_comparison_folder = None  # Folder/file path of selected generation for SBS
        self.sbs_comparison_frames = None  # Cache for loaded SBS comparison frames
        self.sbs_comparison_frame_idx = -1  # Last loaded SBS frame index
        self.sbs_comparison_video_size = (0, 0)  # (width, height) of SBS comparison video

    def log(self, *args):
        print("[PreviewVideoMonitorPro]", *args)
        sys.stdout.flush()

    # =========== WORKFLOW FILE CLEANUP ===========
    def _cleanup_old_workflow_files(self):
        """Clean up old instance workflow files to prevent disk bloat"""
        try:
            max_age_hours = 24  # Keep files for 24 hours
            
            if not os.path.exists(TEMP_WORKFLOW_DIR):
                return
            
            for item in os.listdir(TEMP_WORKFLOW_DIR):
                item_path = os.path.join(TEMP_WORKFLOW_DIR, item)
                
                # Check if it's an instance directory
                if os.path.isdir(item_path) and item.startswith('instance_'):
                    try:
                        # Check modification time
                        mtime = os.path.getmtime(item_path)
                        age_hours = (time.time() - mtime) / 3600
                        
                        if age_hours > max_age_hours:
                            shutil.rmtree(item_path)
                            self.log(f"Cleaned up old workflow dir: {item}")
                    except Exception as e:
                        self.log(f"Error cleaning up {item}: {e}")
                        
        except Exception as e:
            self.log(f"Error in workflow file cleanup: {e}")

    # =========== IMMEDIATE WORKFLOW DATA SAVING ===========
    def _save_workflow_data_immediately(self, prompt, extra_pnginfo):
        """Save workflow data immediately when node executes - CRITICAL FIX"""
        if not self.snapshot_workflow_enabled:
            return False
        
        try:
            # Create instance-specific temp file to avoid conflicts
            instance_temp_dir = os.path.join(TEMP_WORKFLOW_DIR, f"instance_{self.instance_id}")
            os.makedirs(instance_temp_dir, exist_ok=True)
            
            instance_workflow_file = os.path.join(instance_temp_dir, "workflow.json")
            
            workflow_data = {
                'saved_at': time.time(),
                'instance_id': self.instance_id,
                'prompt': None,
                'extra_pnginfo': None,
                'workflow': None
            }
            
            # Store prompt (main workflow structure)
            if prompt is not None:
                if isinstance(prompt, dict):
                    workflow_data['prompt'] = prompt
                elif hasattr(prompt, '__dict__'):
                    # Handle ComfyUI prompt objects
                    try:
                        workflow_data['prompt'] = prompt.__dict__
                    except:
                        workflow_data['prompt'] = str(prompt)
                else:
                    try:
                        # Try to parse as JSON if it's a string
                        workflow_data['prompt'] = json.loads(prompt)
                    except:
                        workflow_data['prompt'] = str(prompt)
            
            # Store extra_pnginfo
            if extra_pnginfo is not None:
                if isinstance(extra_pnginfo, dict):
                    workflow_data['extra_pnginfo'] = extra_pnginfo
                elif hasattr(extra_pnginfo, '__dict__'):
                    try:
                        workflow_data['extra_pnginfo'] = extra_pnginfo.__dict__
                    except:
                        workflow_data['extra_pnginfo'] = str(extra_pnginfo)
                else:
                    try:
                        workflow_data['extra_pnginfo'] = json.loads(extra_pnginfo)
                    except:
                        workflow_data['extra_pnginfo'] = str(extra_pnginfo)
            
            # Try to extract workflow from extra_pnginfo (ComfyUI standard location)
            if extra_pnginfo and isinstance(extra_pnginfo, dict) and 'workflow' in extra_pnginfo:
                workflow_data['workflow'] = extra_pnginfo['workflow']
            elif prompt and isinstance(prompt, dict) and 'workflow' in prompt:
                workflow_data['workflow'] = prompt['workflow']
            
            # Save to instance-specific file
            with open(instance_workflow_file, 'w', encoding='utf-8') as f:
                json.dump(workflow_data, f, indent=2, ensure_ascii=False)
            
            # Store in instance variable for immediate access
            self.current_workflow_data = workflow_data
            
            self.log(f"✓ Workflow data saved for instance {self.instance_id}")
            self.log(f"  Prompt keys: {len(workflow_data.get('prompt', {})) if isinstance(workflow_data.get('prompt'), dict) else 'N/A'}")
            self.log(f"  Extra PNG info: {bool(workflow_data.get('extra_pnginfo'))}")
            
            return True
            
        except Exception as e:
            self.log(f"✗ Error saving workflow data: {e}")
            traceback.print_exc()
            return False

    # =========== COLOR SPACE INITIALIZATION ===========
    def _init_color_space_matrices(self):
        """Initialize color space transformation matrices"""
        # Note: These are simplified matrices for preview purposes.
        # For production use, more accurate matrices or ICC profiles would be needed.

        # sRGB (identity - no transformation needed)
        self.color_matrices = {
            "sRGB": np.array([
                [1.0000, 0.0000, 0.0000],
                [0.0000, 1.0000, 0.0000],
                [0.0000, 0.0000, 1.0000]
            ]),

            # sRGB to Linear approximation (simplified gamma 2.2)
            "sRGB ↔ Linear": np.array([
                [1.0000, 0.0000, 0.0000],
                [0.0000, 1.0000, 0.0000],
                [0.0000, 0.0000, 1.0000]
            ]),

            # Adobe RGB to sRGB (simplified)
            "Adobe RGB": np.array([
                [0.7152, 0.2848, 0.0000],
                [0.0000, 1.0000, 0.0000],
                [0.0000, 0.0000, 1.0000]
            ]),

            # DCI-P3 to sRGB (simplified)
            "DCI-P3": np.array([
                [0.8225, 0.1775, 0.0000],
                [0.0332, 0.9668, 0.0000],
                [0.0171, 0.0724, 0.9105]
            ]),

            # Rec.2020 to sRGB (simplified)
            "Rec. 2020": np.array([
                [0.6274, 0.3293, 0.0433],
                [0.0691, 0.9195, 0.0114],
                [0.0164, 0.0880, 0.8956]
            ]),

            # Rec.709 (very close to sRGB)
            "Rec.709": np.array([
                [0.9578, 0.0422, 0.0000],
                [0.0000, 1.0000, 0.0000],
                [0.0000, 0.0000, 1.0000]
            ]),

            # Rec.709 to Linear (same as sRGB for simplicity)
            "Rec.709 ↔ Linear": np.array([
                [1.0000, 0.0000, 0.0000],
                [0.0000, 1.0000, 0.0000],
                [0.0000, 0.0000, 1.0000]
            ]),

            # PQ/HLG will be handled specially
            "PQ (ST2084) → sRGB": None,  # Special handling
            "HLG → sRGB": None  # Special handling
        }

        # Store inverse matrices for conversion back (if needed)
        self.inverse_matrices = {}
        for name, matrix in self.color_matrices.items():
            if matrix is not None:
                try:
                    self.inverse_matrices[name] = np.linalg.inv(matrix)
                except np.linalg.LinAlgError:
                    # Fallback to identity
                    self.inverse_matrices[name] = matrix

    # =========== COLOR SPACE TRANSFORMATION METHODS ===========
    def _apply_color_space_transform(self, frame_rgb):
        """
        Apply the selected color space transformation to a frame.
        frame_rgb: numpy array (H,W,3) uint8 [0-255] in sRGB color space
        Returns: transformed frame in the selected color space (still as sRGB for display)
        
        PERFORMANCE CRITICAL: This function is called EVERY FRAME during playback!
        Uses caching to avoid expensive matrix operations on every frame.
        """
        if frame_rgb is None:
            return frame_rgb
        
        # ============================================================
        # PERFORMANCE OPTIMIZATION: Skip transformation for sRGB
        # sRGB is the default and requires no computation
        # This early return means ZERO work for sRGB playback!
        # ============================================================
        # TEMPORARY: DISABLE ALL COLOR TRANSFORMATIONS
        # Just return the original frame like sRGB does
        # This will make ALL gamuts perform identically to sRGB
        # ============================================================
        
        # ============================================================
        # CHANNEL EXTRACTION (R/G/B keys in Vision)
        # ============================================================
        if self.channel_mode != "RGB":
            try:
                if self.channel_mode == "Red":
                    channel_data = frame_rgb[:, :, 0]
                    frame_rgb = np.stack([channel_data, channel_data, channel_data], axis=2)
                elif self.channel_mode == "Green":
                    channel_data = frame_rgb[:, :, 1]
                    frame_rgb = np.stack([channel_data, channel_data, channel_data], axis=2)
                elif self.channel_mode == "Blue":
                    channel_data = frame_rgb[:, :, 2]
                    frame_rgb = np.stack([channel_data, channel_data, channel_data], axis=2)
            except Exception as e:
                self.log(f"Channel extraction error ({self.channel_mode}):", e)
        
        # sRGB ONLY - NO COLOR SPACE TRANSFORMATIONS
        return frame_rgb
    def _pre_transform_all_frames(self):
        """
        Pre-transform ALL frames to the current color space.
        This is called when color space changes, NOT during playback.
        
        PERFORMANCE: Transform once (2-3 seconds) instead of every frame (laggy).
        Result: Full-speed playback for ALL gamuts!
        """
        if self.cached_original_frames is None:
            return
        
        # If sRGB, just reference original frames (no transformation needed)
        if self.selected_color_space == "sRGB":
            self.cached_color_transformed_frames = self.cached_original_frames
            self.cached_color_space = "sRGB"
            self.log("Color space: sRGB (no transformation needed)")
            return
        
        # Check if already transformed to this color space
        if (self.cached_color_transformed_frames is not None and 
            self.cached_color_space == self.selected_color_space):
            self.log(f"Frames already transformed to {self.selected_color_space}")
            return
        
        # Transform all frames
        total = len(self.cached_original_frames)
        self.log(f"Pre-transforming {total} frames to {self.selected_color_space}...")
        
        import time
        start_time = time.time()
        
        transformed_frames = []
        for i, frame in enumerate(self.cached_original_frames):
            # Show progress every 10 frames
            if i > 0 and i % 10 == 0:
                elapsed = time.time() - start_time
                fps_rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / fps_rate if fps_rate > 0 else 0
                self.log(f"  Transforming: {i}/{total} frames ({fps_rate:.1f} fps, ETA {eta:.1f}s)")
            
            # Transform this frame
            transformed = self._apply_color_space_transform(frame)
            transformed_frames.append(transformed)
        
        # Store as numpy array
        self.cached_color_transformed_frames = np.array(transformed_frames)
        self.cached_color_space = self.selected_color_space
        
        elapsed = time.time() - start_time
        self.log(f"✓ Transformed {total} frames to {self.selected_color_space} in {elapsed:.2f}s")
        self.log(f"  Playback will now be full-speed!")

    def _apply_linear_transform(self, frame_rgb, to_linear=True):
        """
        Apply gamma linearization or delinearization.
        Simplified gamma 2.2 transform for preview purposes.
        """
        if frame_rgb is None:
            return frame_rgb

        # Convert to float
        frame_float = frame_rgb.astype(np.float32) / 255.0

        if to_linear:
            # sRGB to Linear (inverse gamma)
            # Simplified: use power 2.2 instead of proper piecewise sRGB
            mask = frame_float > 0.04045
            frame_float = np.where(
                mask,
                ((frame_float + 0.055) / 1.055) ** 2.4,
                frame_float / 12.92
            )
        else:
            # Linear to sRGB (apply gamma)
            mask = frame_float > 0.0031308
            frame_float = np.where(
                mask,
                1.055 * (frame_float ** (1/2.4)) - 0.055,
                12.92 * frame_float
            )

        # Clip and convert back
        frame_float = np.clip(frame_float, 0.0, 1.0)
        return (frame_float * 255.0).astype(np.uint8)

    def _pq_to_srgb(self, frame_rgb):
        """
        Convert from PQ (ST2084) HDR to sRGB SDR with tone mapping.
        Simplified version for preview - proper implementation would need actual PQ metadata.
        """
        if frame_rgb is None:
            return frame_rgb

        # Convert to float
        frame_float = frame_rgb.astype(np.float32) / 255.0

        # Simplified PQ inverse EOTF (assuming content is already tone-mapped)
        # In reality, PQ uses a complex perceptual quantizer curve
        # For preview, we'll just apply a simple tone mapping curve

        # Simple Reinhard tone mapping
        luma = 0.2126 * frame_float[:, :, 0] + 0.7152 * \
            frame_float[:, :, 1] + 0.0722 * frame_float[:, :, 2]
        max_luma = np.max(luma)

        if max_luma > 1.0:  # HDR content
            scale = 1.0 / max_luma
            frame_float = frame_float * scale

        # Clip and convert back
        frame_float = np.clip(frame_float, 0.0, 1.0)
        return (frame_float * 255.0).astype(np.uint8)

    def _hlg_to_srgb(self, frame_rgb):
        """
        Convert from HLG HDR to sRGB SDR.
        Simplified version - proper HLG needs OOTF/OETF handling.
        """
        if frame_rgb is None:
            return frame_rgb

        # Convert to float
        frame_float = frame_rgb.astype(np.float32) / 255.0

        # Simplified HLG to sRGB conversion
        # HLG uses a hybrid log-gamma curve
        # For preview, apply a simple gamma correction

        # Simple gamma adjustment
        gamma = 1.2  # Approximate HLG to sRGB adjustment
        frame_float = frame_float ** gamma

        # Clip and convert back
        frame_float = np.clip(frame_float, 0.0, 1.0)
        return (frame_float * 255.0).astype(np.uint8)

    def _check_gamut_warning(self, original, transformed):
        """
        Check for out-of-gamut colors and show warning.
        Simplified check for preview purposes.
        """
        # Check if any colors are out of [0, 1] range after transformation
        out_of_gamut = np.any((transformed < 0.0) | (transformed > 1.0))

        if out_of_gamut:
            current_time = time.time()
            if current_time - self.last_gamut_warning_time > 5.0:  # Throttle warnings
                self.log(
                    f"Warning: Colors in {self.selected_color_space} may be out of sRGB gamut")
                self.last_gamut_warning_time = current_time

    # =========== COLOR WHEEL BUTTON RENDERING ===========
    def _draw_gamuts_button(self, screen, position, is_active=False, is_hover=False):
        """Draw vertical rectangle gamuts button with RGB lines - REDESIGNED"""
        # Get Pong button width as reference (it's about 50px wide based on text width)
        # Gamuts button is half width of Pong = ~25px wide
        btn_width = 12  # Half of typical Pong button width (~25px / 2 for thinner look)
        btn_height = 30  # Standard button height
        btn_rect = pygame.Rect(position[0], position[1], btn_width, btn_height)

        # Background color based on state
        if is_active or self.color_space_dropup_open:
            bg_color = (70, 70, 70)  # Medium gray for active/open
        elif is_hover:
            bg_color = (70, 70, 70)  # Medium gray for hover
        else:
            bg_color = (60, 60, 60)  # Default dark gray

        # Border color - GREEN outline when selected (like Pong button)
        if self.color_space_dropup_open:
            border_color = (100, 220, 100)  # Green when dropup is open
        else:
            border_color = (120, 120, 120)  # Gray when not open

        # Draw rectangular button
        pygame.draw.rect(screen, bg_color, btn_rect, border_radius=3)
        pygame.draw.rect(screen, border_color, btn_rect, width=2, border_radius=3)

        # Draw 3 vertical RGB lines inside the button
        # Thin lines with gaps between them
        line_width = 2
        gap = 2
        
        # Calculate positions for 3 evenly distributed lines
        usable_width = btn_width - 4  # Leave 2px padding on each side
        total_line_width = 3 * line_width + 2 * gap  # 3 lines + 2 gaps
        start_x = position[0] + (btn_width - total_line_width) // 2
        
        # Red line (left)
        red_x = start_x
        pygame.draw.line(screen, (255, 80, 80), 
                        (red_x, position[1] + 6), 
                        (red_x, position[1] + btn_height - 6), 
                        line_width)
        
        # Green line (middle)
        green_x = red_x + line_width + gap
        pygame.draw.line(screen, (80, 255, 80), 
                        (green_x, position[1] + 6), 
                        (green_x, position[1] + btn_height - 6), 
                        line_width)
        
        # Blue line (right)
        blue_x = green_x + line_width + gap
        pygame.draw.line(screen, (80, 80, 255), 
                        (blue_x, position[1] + 6), 
                        (blue_x, position[1] + btn_height - 6), 
                        line_width)

        return btn_rect

    # =========== COLOR SPACE DROPUP MENU ===========
    def _draw_color_space_dropup(self, screen, mouse_pos, gamuts_button_rect=None):
        """Draw the color space/gamut dropUP menu positioned above the gamuts button - FIXED TO SHOW ALL ITEMS"""
        if not self.color_space_dropup_open:
            return None

        # Menu dimensions - SHOW ALL ITEMS (9 items total)
        menu_width = 250
        item_height = 28
        total_items = len(self.color_spaces)
        # Show all items at once - increased from 8 to show all 9
        menu_height = total_items * item_height + 10

        # Position above gamuts button if provided, otherwise default position
        if gamuts_button_rect:
            # DON'T center - button is at far right, so align menu's right edge
            # Position menu so its right edge is a few pixels from screen edge
            vw = screen.get_width()
            padding_from_edge = 10  # Distance from right edge
            menu_x = vw - menu_width - padding_from_edge
            menu_y = gamuts_button_rect.top - menu_height - 5
        else:
            # Fallback position
            vw, vh = screen.get_size()
            menu_x = vw - menu_width - 10
            menu_y = vh - self.toolbar_height - menu_height - 5

        # Ensure menu stays on screen (left edge check)
        menu_x = max(10, menu_x)
        menu_y = max(10, menu_y)

        # Cache font
        font_key = f"font_18"
        if font_key not in self._ui_cache:
            self._ui_cache[font_key] = pygame.font.SysFont(None, 18)
        font = self._ui_cache[font_key]

        # Background
        menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
        pygame.draw.rect(screen, (40, 40, 40), menu_rect, border_radius=4)
        pygame.draw.rect(screen, (80, 80, 80), menu_rect,
                         width=2, border_radius=4)

        # No scroll needed since we show all items
        self.dropup_scroll_offset = 0

        # Draw items with caching
        item_rects = []

        for i in range(total_items):
            color_space_name = self.color_spaces[i]

            item_y = menu_y + 5 + i * item_height
            item_rect = pygame.Rect(
                menu_x + 5, item_y, menu_width - 10, item_height - 2)
            item_rects.append((color_space_name, item_rect, i))

            # Check if this is the selected color space
            is_selected = (color_space_name == self.selected_color_space)
            is_hover = (i == self.color_space_hover_index)

            # Highlight selected in VIOLET
            if is_selected:
                # Violet background for selected color space
                pygame.draw.rect(screen, (100, 60, 140),
                                 item_rect, border_radius=3)
                pygame.draw.rect(screen, (180, 100, 220),
                                 item_rect, width=2, border_radius=3)
            elif is_hover:
                # Gray background for hover
                pygame.draw.rect(screen, (70, 70, 70),
                                 item_rect, border_radius=3)

            # Draw text - cached
            text_key = f"colorspace_{color_space_name}_{is_selected}"
            if text_key not in self._ui_cache:
                if is_selected:
                    text_color = (220, 180, 255)  # Light violet for selected
                else:
                    text_color = (220, 220, 220)  # Normal white

                text_surf = font.render(color_space_name, True, text_color)
                self._ui_cache[text_key] = text_surf
            else:
                text_surf = self._ui_cache[text_key]

            screen.blit(text_surf, (item_rect.x + 5, item_rect.y + 5))

        return (menu_rect, item_rects)

    # =========== WEBP EXIF METADATA CREATION - ComfyUI Compatible ===========
    def _create_webp_exif(self, pil_image):
        """
        Create EXIF metadata for WebP with workflow data in ComfyUI compatible format.
        
        WebP uses EXIF tags which handle large metadata much better than PNG text chunks.
        This allows saving workflows with many more nodes without size limitations.
        
        ComfyUI convention (used by SaveImgAdv and similar extensions):
        - Tag 270 (ImageDescription) = "Workflow:" + workflow JSON
        - Tag 271 (Make) = "Prompt:" + prompt JSON
        """
        if not self.snapshot_workflow_enabled:
            return None
        
        try:
            # Try to get workflow data from instance variable first
            workflow_data = getattr(self, 'current_workflow_data', None)
            
            # If not available, try to load from instance-specific file
            if not workflow_data and hasattr(self, 'instance_id'):
                instance_temp_dir = os.path.join(TEMP_WORKFLOW_DIR, f"instance_{self.instance_id}")
                instance_workflow_file = os.path.join(instance_temp_dir, "workflow.json")
                
                if os.path.exists(instance_workflow_file):
                    with open(instance_workflow_file, 'r', encoding='utf-8') as f:
                        workflow_data = json.load(f)
            
            if not workflow_data:
                self.log("Warning: No workflow data available for snapshot")
                return None
            
            # Get EXIF object from image
            exif = pil_image.getexif()
            total_size = 0
            
            # Tag 271 (Make) = Prompt data
            # ComfyUI convention: prefix with "Prompt:"
            prompt_data = workflow_data.get('prompt')
            if prompt_data:
                if isinstance(prompt_data, dict):
                    prompt_json = json.dumps(prompt_data, ensure_ascii=False, separators=(',', ':'))
                elif isinstance(prompt_data, str):
                    prompt_json = prompt_data
                else:
                    prompt_json = str(prompt_data)
                exif[271] = "Prompt:" + prompt_json
                total_size += len(prompt_json)
            
            # Tag 270 (ImageDescription) = Workflow data
            # ComfyUI convention: prefix with "Workflow:"
            # Try to get workflow from extra_pnginfo first (ComfyUI standard location)
            workflow_direct = None
            extra_data = workflow_data.get('extra_pnginfo')
            if extra_data and isinstance(extra_data, dict) and 'workflow' in extra_data:
                workflow_direct = extra_data['workflow']
            elif workflow_data.get('workflow'):
                workflow_direct = workflow_data['workflow']
            
            if workflow_direct:
                if isinstance(workflow_direct, dict):
                    workflow_json = json.dumps(workflow_direct, ensure_ascii=False, separators=(',', ':'))
                elif isinstance(workflow_direct, str):
                    workflow_json = workflow_direct
                else:
                    workflow_json = str(workflow_direct)
                exif[270] = "Workflow:" + workflow_json
                total_size += len(workflow_json)
            
            self.log(f"Created WebP EXIF metadata: {total_size/1024:.1f} KB")
            return exif
            
        except Exception as e:
            self.log(f"Error creating WebP EXIF metadata: {e}")
            traceback.print_exc()
            return None

    # =========== WEBP SNAPSHOT SYSTEM - ComfyUI Compatible ===========
    def _take_snapshot(self):
        """
        Take a snapshot of current frame and save as WebP with workflow metadata.
        
        WebP format advantages:
        - Smaller file sizes than PNG
        - Better support for large workflow metadata via EXIF
        - Full ComfyUI compatibility (drag & drop to load workflow)
        - Native thumbnail support on Windows 10+, macOS Big Sur+
        """
        try:
            # Set saving flag for visual feedback
            self.snapshot_saving = True
            self._mark_ui_dirty()  # Update UI to show cyan button
            
            if not self.current_generation_id:
                self.log("No current generation loaded - cannot take snapshot")
                self.snapshot_saving = False
                return False

            # Get current generation metadata
            metadata = self.generations_metadata.get(
                self.current_generation_id)
            if not metadata:
                self.log("Current generation metadata not found")
                self.snapshot_saving = False
                return False

            # Get the video file path (folder for image sequences, file for legacy)
            video_path = metadata.get('folder_path', metadata.get('file_path'))
            if not video_path or not os.path.exists(video_path):
                self.log("Generation folder/file not found for current generation")
                self.snapshot_saving = False
                return False

            # Create WebP filename with same base name as generation
            base_name = os.path.basename(video_path)
            # Remove extension if it's a file, keep name if it's a folder
            if os.path.isdir(video_path):
                webp_filename = base_name + ".webp"
            else:
                webp_filename = os.path.splitext(base_name)[0] + ".webp"
            webp_path = os.path.join(self.current_snapshot_path, webp_filename)

            # Check if we have the current frame
            if self.cached_original_frames is None or len(self.cached_original_frames) <= self.current_frame:
                self.log("No frame data available for snapshot")
                self.snapshot_saving = False
                return False

            # Get current frame (with color space transformation applied)
            frame = self.cached_original_frames[self.current_frame]
            if frame is None:
                self.log("Current frame is None")
                self.snapshot_saving = False
                return False

            # Apply color space transformation if needed
            frame_rgb = self._apply_color_space_transform(frame)
            if frame_rgb is None:
                frame_rgb = frame  # Fallback to original if transformation failed

            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # Create WebP EXIF metadata with workflow data
            exif_data = self._create_webp_exif(pil_image)
            
            # Save as WebP with EXIF metadata
            # quality=95 gives excellent quality with good compression
            workflow_embedded = False
            if exif_data:
                pil_image.save(webp_path, "WEBP", exif=exif_data, quality=95)
                workflow_embedded = True
                self.log(f"✓ Snapshot with workflow metadata saved: {webp_filename}")
            else:
                pil_image.save(webp_path, "WEBP", quality=95)
                self.log(f"✓ Snapshot saved (no workflow data): {webp_filename}")

            # Log snapshot details
            display_name = metadata.get('display_name', 'Unknown')
            frame_display = self.current_frame + self.first_frame_offset + 1
            total_frames = len(self.cached_original_frames)
            total_display = total_frames + self.first_frame_offset

            self.log(f"Snapshot saved: {webp_filename}")
            self.log(f"  Generation: {display_name}")
            self.log(f"  Frame: {frame_display}/{total_display}")
            self.log(f"  Color Space: {self.selected_color_space}")
            self.log(f"  Workflow embedded: {workflow_embedded}")
            self.log(f"  Path: {webp_path}")

            # Update last snapshot time for visual feedback
            self.last_snapshot_time = pygame.time.get_ticks()
            self.snapshot_saving = False  # Reset saving flag
            
            # Trigger camera iris animation
            self.iris_animation_active = True
            self.iris_animation_start_time = pygame.time.get_ticks()
            
            self._mark_ui_dirty()  # Update UI
            
            return True

        except Exception as e:
            self.log(f"Error taking snapshot: {e}")
            traceback.print_exc()
            self.snapshot_saving = False
            self._mark_ui_dirty()
            return False

    def _draw_camera_iris_animation(self, screen):
        """
        Draw camera aperture iris animation (6 blades closing to center)
        Duration: 2 seconds
        - 0.0s - 0.5s: Iris closes from full circle to center point
        - 0.5s - 0.8s: Fully closed (tiny point)
        - 0.8s - 2.0s: Fade out
        """
        if not self.iris_animation_active:
            return
        
        current_time = pygame.time.get_ticks()
        elapsed = current_time - self.iris_animation_start_time
        
        # Check if animation is complete
        if elapsed >= self.iris_animation_duration:
            self.iris_animation_active = False
            return
        
        # Get screen center
        screen_width, screen_height = screen.get_size()
        center_x = screen_width // 2
        center_y = screen_height // 2
        
        # Animation parameters
        num_blades = 6
        max_radius = min(screen_width, screen_height) // 3  # Max iris size
        
        # Calculate animation progress
        progress = elapsed / self.iris_animation_duration
        
        # Phase 1: Closing (0.0 - 0.5s) → 0-0.25 progress
        # Phase 2: Closed (0.5 - 0.8s) → 0.25-0.4 progress
        # Phase 3: Fade out (0.8 - 2.0s) → 0.4-1.0 progress
        
        if progress < 0.25:  # Closing phase (first 0.5s)
            # Iris closes from full to tiny point
            close_progress = progress / 0.25  # 0 to 1
            radius = max_radius * (1.0 - close_progress)
            alpha = 255
        elif progress < 0.4:  # Closed phase (0.5-0.8s)
            # Fully closed, tiny point
            radius = 0
            alpha = 255
        else:  # Fade out phase (0.8-2.0s)
            # Fade out
            fade_progress = (progress - 0.4) / 0.6  # 0 to 1
            radius = 0
            alpha = int(255 * (1.0 - fade_progress))
        
        # Only draw if visible
        if alpha > 0 and radius >= 0:
            # Create surface with per-pixel alpha
            iris_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
            
            # Draw 6 iris blades as triangular segments
            blade_angle = 360 / num_blades
            
            for i in range(num_blades):
                # Calculate blade rotation
                start_angle = i * blade_angle
                end_angle = (i + 1) * blade_angle
                
                # Convert to radians
                start_rad = np.radians(start_angle)
                end_rad = np.radians(end_angle)
                
                # Blade points (triangular shape from center)
                if radius > 5:  # Only draw if iris is open enough
                    points = [
                        (center_x, center_y),  # Center point
                        (center_x + radius * np.cos(start_rad), 
                         center_y + radius * np.sin(start_rad)),
                        (center_x + radius * np.cos(end_rad), 
                         center_y + radius * np.sin(end_rad))
                    ]
                    
                    # Draw blade with gradient effect (lighter in center)
                    blade_color = (200, 200, 200, alpha)  # Light grey with alpha
                    pygame.draw.polygon(iris_surface, blade_color, points)
                    
                    # Draw blade edges (darker outline)
                    edge_color = (100, 100, 100, alpha)  # Dark grey with alpha
                    pygame.draw.line(iris_surface, edge_color,
                                   (center_x, center_y),
                                   (center_x + radius * np.cos(start_rad), 
                                    center_y + radius * np.sin(start_rad)), 2)
            
            # Blit the iris surface onto the screen
            screen.blit(iris_surface, (0, 0))

    # =========== VISION MODULE ===========
    def _draw_vision_module(self, screen, mouse_pos):
        """
        Draw Vision module: Color grading sliders + analysis tools
        Covers timeline area when active
        Layout: 3/4 width for sliders (Gain, Gamma, Saturation) + 1/4 width for buttons/scopes
        """
        if not self.vision_module_open:
            return None
        
        screen_width, screen_height = screen.get_size()
        
        # Module dimensions (same size as timeline)
        module_height = 50  # Same as timeline height
        module_y = screen_height - self.toolbar_height - module_height
        module_rect = pygame.Rect(0, module_y, screen_width, module_height)
        
        # Create semi-transparent surface (80% opaque = 204 alpha)
        module_surface = pygame.Surface((screen_width, module_height), pygame.SRCALPHA)
        
        # Background - darker grey with 80% opacity
        bg_color_with_alpha = (35, 35, 35, 204)  # RGB + Alpha (204 = 80% of 255)
        pygame.draw.rect(module_surface, bg_color_with_alpha, (0, 0, screen_width, module_height))
        
        # Top border (fully opaque)
        pygame.draw.line(module_surface, (60, 60, 60, 255), (0, 0), (screen_width, 0), 2)
        
        # Divide into sections: 3/4 for sliders, 1/4 for buttons
        slider_section_width = int(screen_width * 0.75)
        button_section_width = screen_width - slider_section_width
        
        # Vertical divider between sections (on surface)
        divider_x = slider_section_width
        pygame.draw.line(module_surface, (60, 60, 60, 255), (divider_x, 0), (divider_x, module_height), 2)
        
        # === LEFT SECTION: Sliders (3/4 width) ===
        # Divide slider section into 3 equal parts (Gain, Gamma, Saturation)
        slider_width = slider_section_width // 3
        
        # Cache font
        font_key = "font_16"
        if font_key not in self._ui_cache:
            self._ui_cache[font_key] = pygame.font.SysFont(None, 16)
        font = self._ui_cache[font_key]
        
        # Draw GAIN slider (first third) - now on module_surface
        gain_x = 10
        gain_rect = self._draw_vision_slider(
            module_surface, font, "GAIN", self.vision_gain, 
            gain_x, 0, slider_width, module_height, mouse_pos, 
            range_type="exposure"  # -6 to +6
        )
        
        # Draw GAMMA slider (second third)
        gamma_x = 10 + slider_width
        gamma_rect = self._draw_vision_slider(
            module_surface, font, "GAMMA", self.vision_gamma,
            gamma_x, 0, slider_width, module_height, mouse_pos,
            range_type="gamma"  # 0.0 to 4.0, center 1.0
        )
        
        # Draw SATURATION slider (third third)
        sat_x = 10 + slider_width * 2
        sat_rect = self._draw_vision_slider(
            module_surface, font, "SATURATION", self.vision_saturation,
            sat_x, 0, slider_width, module_height, mouse_pos,
            range_type="saturation"  # 0.0 to 4.0, center 1.0
        )
        
        # === RIGHT SECTION: Analysis Tools Buttons (1/4 width) ===
        button_section_x = slider_section_width + 10
        button_section_width = screen_width - slider_section_width
        
        # Draw scope toggle buttons: H (Histogram), V (Vector), W (Waveform)
        scope_buttons = ["H", "V", "W"]
        button_size = 30
        button_spacing = 10
        button_y_start = (module_height - button_size) // 2
        
        scope_button_rects = []
        for i, scope_label in enumerate(scope_buttons):
            button_x = button_section_x + i * (button_size + button_spacing)
            button_rect = pygame.Rect(button_x, button_y_start, button_size, button_size)
            
            # Check if this scope is active
            scope_type_map = {"H": "histogram", "V": "vector", "W": "waveform"}
            is_active = (self.vision_active_scope == scope_type_map[scope_label])
            
            # Button colors
            if is_active:
                bg_color = (60, 100, 140, 255)  # Blue when active
                border_color = (100, 160, 220, 255)  # Bright blue border
            else:
                bg_color = (50, 50, 50, 255)  # Dark grey when inactive
                border_color = (100, 100, 100, 255)  # Grey border
            
            # Draw button
            pygame.draw.rect(module_surface, bg_color, 
                           (button_rect.x, button_rect.y, button_rect.width, button_rect.height), 
                           border_radius=4)
            pygame.draw.rect(module_surface, border_color,
                           (button_rect.x, button_rect.y, button_rect.width, button_rect.height),
                           width=2, border_radius=4)
            
            # Draw label
            label_text = font.render(scope_label, True, (220, 220, 220, 255))
            label_rect = label_text.get_rect(center=(button_rect.centerx, button_rect.centery))
            module_surface.blit(label_text, label_rect)
            
            scope_button_rects.append((scope_label, button_rect))
        
        # Draw Zebra toggle button (Z) after scope buttons
        zebra_button_x = button_section_x + 3 * (button_size + button_spacing)
        zebra_button_rect = pygame.Rect(zebra_button_x, button_y_start, button_size, button_size)
        
        # Button colors for Zebra
        if self.vision_zebra_active:
            bg_color = (60, 100, 140, 255)  # Blue when active
            border_color = (100, 160, 220, 255)  # Bright blue border
        else:
            bg_color = (50, 50, 50, 255)  # Dark grey when inactive
            border_color = (100, 100, 100, 255)  # Grey border
        
        # Draw Zebra button
        pygame.draw.rect(module_surface, bg_color,
                       (zebra_button_rect.x, zebra_button_rect.y, zebra_button_rect.width, zebra_button_rect.height),
                       border_radius=4)
        pygame.draw.rect(module_surface, border_color,
                       (zebra_button_rect.x, zebra_button_rect.y, zebra_button_rect.width, zebra_button_rect.height),
                       width=2, border_radius=4)
        
        # Draw Z label
        zebra_label_text = font.render("Z", True, (220, 220, 220, 255))
        zebra_label_rect = zebra_label_text.get_rect(center=(zebra_button_rect.centerx, zebra_button_rect.centery))
        module_surface.blit(zebra_label_text, zebra_label_rect)
        
        # Draw Picker toggle button (P) after Zebra
        picker_button_x = button_section_x + 4 * (button_size + button_spacing)
        picker_button_rect = pygame.Rect(picker_button_x, button_y_start, button_size, button_size)
        
        # Button colors for Picker
        if self.vision_picker_active:
            bg_color = (60, 100, 140, 255)  # Blue when active
            border_color = (100, 160, 220, 255)  # Bright blue border
        else:
            bg_color = (50, 50, 50, 255)  # Dark grey when inactive
            border_color = (100, 100, 100, 255)  # Grey border
        
        # Draw Picker button
        pygame.draw.rect(module_surface, bg_color,
                       (picker_button_rect.x, picker_button_rect.y, picker_button_rect.width, picker_button_rect.height),
                       border_radius=4)
        pygame.draw.rect(module_surface, border_color,
                       (picker_button_rect.x, picker_button_rect.y, picker_button_rect.width, picker_button_rect.height),
                       width=2, border_radius=4)
        
        # Draw P label
        picker_label_text = font.render("P", True, (220, 220, 220, 255))
        picker_label_rect = picker_label_text.get_rect(center=(picker_button_rect.centerx, picker_button_rect.centery))
        module_surface.blit(picker_label_text, picker_label_rect)
        
        # Draw RGB channel buttons (R, G, B) after Picker
        channel_buttons = [
            ("R", "Red", (255, 80, 80, 255), 5),      # Red button
            ("G", "Green", (80, 255, 80, 255), 6),    # Green button
            ("B", "Blue", (80, 80, 255, 255), 7)      # Blue button
        ]
        
        for label, mode, color, offset in channel_buttons:
            button_x = button_section_x + offset * (button_size + button_spacing)
            button_rect = pygame.Rect(button_x, button_y_start, button_size, button_size)
            
            # Check if this channel is active
            is_active = (self.channel_mode == mode)
            
            # Button colors
            if is_active:
                bg_color = (60, 60, 60, 255)  # Darker grey when active
                border_color = color  # Bright colored border when active
                text_color = color  # Colored text
            else:
                bg_color = (50, 50, 50, 255)  # Dark grey when inactive
                border_color = color  # Colored border (dimmer)
                text_color = color  # Colored text
            
            # Draw button
            pygame.draw.rect(module_surface, bg_color,
                           (button_rect.x, button_rect.y, button_rect.width, button_rect.height),
                           border_radius=4)
            pygame.draw.rect(module_surface, border_color,
                           (button_rect.x, button_rect.y, button_rect.width, button_rect.height),
                           width=2, border_radius=4)
            
            # Draw label in channel color
            label_text = font.render(label, True, text_color)
            label_rect = label_text.get_rect(center=(button_rect.centerx, button_rect.centery))
            module_surface.blit(label_text, label_rect)
        
        # Draw Guides button (custom 4-line grid icon) after RGB buttons
        guides_button_x = button_section_x + 8 * (button_size + button_spacing)
        guides_button_rect = pygame.Rect(guides_button_x, button_y_start, button_size, button_size)
        
        # Button colors for Guides
        if self.vision_guides_active:
            bg_color = (60, 100, 140, 255)  # Blue when active
            border_color = (100, 160, 220, 255)  # Bright blue border
        else:
            bg_color = (50, 50, 50, 255)  # Dark grey when inactive
            border_color = (100, 100, 100, 255)  # Grey border
        
        # Draw Guides button
        pygame.draw.rect(module_surface, bg_color,
                       (guides_button_rect.x, guides_button_rect.y, guides_button_rect.width, guides_button_rect.height),
                       border_radius=4)
        pygame.draw.rect(module_surface, border_color,
                       (guides_button_rect.x, guides_button_rect.y, guides_button_rect.width, guides_button_rect.height),
                       width=2, border_radius=4)
        
        # Draw custom 4-line grid icon (not touching edges)
        icon_color = (220, 220, 220, 255)
        icon_padding = 8
        # Vertical lines
        v_left = guides_button_rect.x + icon_padding
        v_right = guides_button_rect.x + button_size - icon_padding
        v_top = guides_button_rect.y + icon_padding
        v_bottom = guides_button_rect.y + button_size - icon_padding
        
        # Draw 2 vertical lines
        pygame.draw.line(module_surface, icon_color, (v_left, v_top), (v_left, v_bottom), 1)
        pygame.draw.line(module_surface, icon_color, (v_right, v_top), (v_right, v_bottom), 1)
        
        # Draw 2 horizontal lines
        pygame.draw.line(module_surface, icon_color, (v_left, v_top), (v_right, v_top), 1)
        pygame.draw.line(module_surface, icon_color, (v_left, v_bottom), (v_right, v_bottom), 1)
        
        # Draw Mask button (M) - last slot!
        mask_button_x = button_section_x + 9 * (button_size + button_spacing)
        mask_button_rect = pygame.Rect(mask_button_x, button_y_start, button_size, button_size)
        
        # Button colors for Mask
        is_mask_active = (self.vision_mask_index > 0)
        if is_mask_active:
            bg_color = (60, 100, 140, 255)  # Blue when active
            border_color = (100, 160, 220, 255)  # Bright blue border
        else:
            bg_color = (50, 50, 50, 255)  # Dark grey when inactive
            border_color = (100, 100, 100, 255)  # Grey border
        
        # Draw Mask button
        pygame.draw.rect(module_surface, bg_color,
                       (mask_button_rect.x, mask_button_rect.y, mask_button_rect.width, mask_button_rect.height),
                       border_radius=4)
        pygame.draw.rect(module_surface, border_color,
                       (mask_button_rect.x, mask_button_rect.y, mask_button_rect.width, mask_button_rect.height),
                       width=2, border_radius=4)
        
        # Draw M label
        mask_label_text = font.render("M", True, (220, 220, 220, 255))
        mask_label_rect = mask_label_text.get_rect(center=(mask_button_rect.centerx, mask_button_rect.centery))
        module_surface.blit(mask_label_text, mask_label_rect)
        
        # Draw Flip/Flop button - split triangles (mirror icon) - slot 10!
        flip_button_x = button_section_x + 10 * (button_size + button_spacing)
        flip_button_rect = pygame.Rect(flip_button_x, button_y_start, button_size, button_size)
        
        # Button colors for Flip
        is_flip_active = (self.vision_flip_mode > 0)
        if is_flip_active:
            bg_color = (60, 100, 140, 255)  # Blue when active
            border_color = (100, 160, 220, 255)  # Bright blue border
        else:
            bg_color = (50, 50, 50, 255)  # Dark grey when inactive
            border_color = (100, 100, 100, 255)  # Grey border
        
        # Draw Flip button
        pygame.draw.rect(module_surface, bg_color,
                       (flip_button_rect.x, flip_button_rect.y, flip_button_rect.width, flip_button_rect.height),
                       border_radius=4)
        pygame.draw.rect(module_surface, border_color,
                       (flip_button_rect.x, flip_button_rect.y, flip_button_rect.width, flip_button_rect.height),
                       width=2, border_radius=4)
        
        # Draw split triangles icon (mirror/reflection symbol)
        icon_color = (220, 220, 220, 255)
        icon_padding = 7
        center_x = flip_button_rect.centerx
        center_y = flip_button_rect.centery
        tri_size = 6
        
        # Left triangle (pointing right)
        left_tri = [
            (center_x - tri_size - 1, center_y - tri_size),
            (center_x - tri_size - 1, center_y + tri_size),
            (center_x - 1, center_y)
        ]
        pygame.draw.polygon(module_surface, icon_color, left_tri)
        
        # Right triangle (pointing left) - mirror of left
        right_tri = [
            (center_x + tri_size + 1, center_y - tri_size),
            (center_x + tri_size + 1, center_y + tri_size),
            (center_x + 1, center_y)
        ]
        pygame.draw.polygon(module_surface, icon_color, right_tri)
        
        # Blit the semi-transparent module surface to screen
        screen.blit(module_surface, (0, module_y))
        
        return module_rect

    def _draw_vision_slider(self, screen, font, label, value, x, y, width, height, mouse_pos, range_type="exposure"):
        """
        Draw a single Vision slider with red manipulator (like timeline indicator)
        range_type: "exposure" (-6 to +6), "gamma" (0 to 4, center 1), "saturation" (0 to 4, center 1)
        Returns the slider rect for interaction
        """
        # Determine range based on type
        if range_type == "exposure":
            min_val = -6.0
            max_val = 6.0
        elif range_type == "gamma" or range_type == "saturation":
            min_val = 0.0
            max_val = 4.0
        
        padding = 10
        label_height = 15
        
        # Calculate display value based on range type
        if range_type == "exposure":
            # GAIN: Display as exposure multiplier (2^value) so 0 shows as 1.0
            display_value = pow(2.0, value)
        else:
            # GAMMA/SATURATION: Display actual value
            display_value = value
        
        # Draw label and value on same line: "GAIN   1.0"
        label_text = font.render(label, True, (180, 180, 180, 255))
        value_text = font.render(f"   {display_value:.1f}", True, (220, 220, 220, 255))  # 3 spaces before value
        
        # Calculate combined width to center both together
        combined_width = label_text.get_width() + value_text.get_width()
        start_x = x + (width - combined_width) // 2
        
        # Draw label
        screen.blit(label_text, (start_x, y + 5))
        
        # Draw value right after label
        screen.blit(value_text, (start_x + label_text.get_width(), y + 5))
        
        # Slider bar dimensions
        bar_y = y + label_height + 15  # Adjusted for single-line label
        bar_height = 4
        bar_x_start = x + padding
        bar_x_end = x + width - padding
        bar_width = bar_x_end - bar_x_start
        
        # Draw slider bar background
        bar_rect = pygame.Rect(bar_x_start, bar_y, bar_width, bar_height)
        pygame.draw.rect(screen, (60, 60, 60, 255), bar_rect, border_radius=2)
        
        # Draw 20 tick marks along the slider (75% height of main manipulator)
        # Main manipulator size = 8, so ticks = 6 (75% of 8)
        tick_height = 6
        num_ticks = 20
        
        for i in range(num_ticks + 1):  # +1 to include both ends
            tick_x = bar_x_start + int((i / num_ticks) * bar_width)
            # Draw small triangular tick (without red tip)
            tick_points = [
                (tick_x, bar_y - 2),  # Top point
                (tick_x - tick_height // 3, bar_y - tick_height - 2),  # Left point
                (tick_x + tick_height // 3, bar_y - tick_height - 2)   # Right point
            ]
            pygame.draw.polygon(screen, (90, 90, 90, 255), tick_points)  # Grey ticks
        
        # Calculate manipulator position based on value (using range from range_type)
        normalized = (value - min_val) / (max_val - min_val)
        manipulator_x = bar_x_start + int(normalized * bar_width)
        
        # Draw red manipulator (triangle pointing down, like timeline indicator)
        # This one is DOMINANT - 100% size
        manipulator_size = 8
        manipulator_points = [
            (manipulator_x, bar_y - 2),  # Top point
            (manipulator_x - manipulator_size // 2, bar_y - manipulator_size - 2),  # Left point
            (manipulator_x + manipulator_size // 2, bar_y - manipulator_size - 2)   # Right point
        ]
        pygame.draw.polygon(screen, (220, 60, 60, 255), manipulator_points)  # Red - on top of ticks!
        
        # Return slider interaction rect (for mouse detection)
        interaction_rect = pygame.Rect(bar_x_start, bar_y - 15, bar_width, 30)
        return interaction_rect

    def _draw_histogram_scope(self, screen, frame_surface):
        """
        Draw RGB histogram scope overlay
        - Analyzes ORIGINAL cached frames (before zoom/pan/fit transformations)
        - 80% opacity (see-through like Vision bar)
        - Vertical deployment above right section
        - Downsampled for speed (every 4th pixel)
        - RGB channels overlapping
        """
        # Check if we have original frames to analyze
        if self.cached_original_frames is None or self.current_frame >= len(self.cached_original_frames):
            return
        
        screen_width, screen_height = screen.get_size()
        
        # Scope dimensions (covers 1/4 width on right side)
        scope_width = int(screen_width * 0.25)
        scope_height = screen_height - self.toolbar_height - 50  # Minus Vision bar height
        scope_x = screen_width - scope_width
        scope_y = 0
        
        # Create semi-transparent surface (80% opacity like Vision bar)
        scope_surface = pygame.Surface((scope_width, scope_height), pygame.SRCALPHA)
        bg_color = (20, 20, 20, 204)  # Dark background, 80% opaque
        pygame.draw.rect(scope_surface, bg_color, (0, 0, scope_width, scope_height))
        
        # Border
        pygame.draw.rect(scope_surface, (80, 80, 80, 255), (0, 0, scope_width, scope_height), width=2)
        
        # Calculate histogram from ORIGINAL cached frame (FAST: downsample by 4x4 = 16x fewer pixels)
        try:
            # Get original frame directly from cache (before any transformations!)
            original_frame = self.cached_original_frames[self.current_frame]
            
            # Downsample: every 4th pixel in both dimensions
            downsampled = original_frame[::4, ::4, :]
            
            # Calculate histograms for each channel (256 bins: 0-255)
            hist_r = np.histogram(downsampled[:,:,0], bins=256, range=(0, 256))[0]
            hist_g = np.histogram(downsampled[:,:,1], bins=256, range=(0, 256))[0]
            hist_b = np.histogram(downsampled[:,:,2], bins=256, range=(0, 256))[0]
            
            # Normalize to fit scope height (with padding)
            max_count = max(hist_r.max(), hist_g.max(), hist_b.max())
            if max_count > 0:
                graph_height = scope_height - 40  # Leave padding for labels
                hist_r_norm = (hist_r / max_count * graph_height).astype(np.int32)
                hist_g_norm = (hist_g / max_count * graph_height).astype(np.int32)
                hist_b_norm = (hist_b / max_count * graph_height).astype(np.int32)
                
                # Draw histograms (from bottom up, overlapping)
                graph_bottom = scope_height - 20
                bin_width = scope_width / 256
                
                # Draw each channel as vertical lines
                for i in range(256):
                    x = int(i * bin_width)
                    
                    # Red channel
                    if hist_r_norm[i] > 0:
                        pygame.draw.line(scope_surface, (255, 80, 80, 180),
                                       (x, graph_bottom), 
                                       (x, graph_bottom - hist_r_norm[i]), 1)
                    
                    # Green channel
                    if hist_g_norm[i] > 0:
                        pygame.draw.line(scope_surface, (80, 255, 80, 180),
                                       (x, graph_bottom),
                                       (x, graph_bottom - hist_g_norm[i]), 1)
                    
                    # Blue channel
                    if hist_b_norm[i] > 0:
                        pygame.draw.line(scope_surface, (80, 80, 255, 180),
                                       (x, graph_bottom),
                                       (x, graph_bottom - hist_b_norm[i]), 1)
            
            # Draw labels
            font_key = "font_14"
            if font_key not in self._ui_cache:
                self._ui_cache[font_key] = pygame.font.SysFont(None, 14)
            small_font = self._ui_cache[font_key]
            
            label_text = small_font.render("HISTOGRAM", True, (180, 180, 180, 255))
            scope_surface.blit(label_text, (10, 10))
            
            # Mark 0 and 255 (black and white points)
            zero_text = small_font.render("0", True, (120, 120, 120, 255))
            max_text = small_font.render("255", True, (120, 120, 120, 255))
            scope_surface.blit(zero_text, (5, scope_height - 15))
            scope_surface.blit(max_text, (scope_width - 25, scope_height - 15))
            
        except Exception as e:
            # If histogram calculation fails, just show label
            font_key = "font_14"
            if font_key not in self._ui_cache:
                self._ui_cache[font_key] = pygame.font.SysFont(None, 14)
            small_font = self._ui_cache[font_key]
            error_text = small_font.render("Histogram Error", True, (200, 100, 100, 255))
            scope_surface.blit(error_text, (10, 10))
        
        # Blit scope to screen
        screen.blit(scope_surface, (scope_x, scope_y))

    def _draw_vectorscope(self, screen, frame_surface):
        """
        Draw vectorscope overlay (circular chrominance plot)
        - Analyzes ORIGINAL cached frames (before zoom/pan/fit transformations)
        - 80% opacity (see-through like Vision bar)
        - SQUARE aspect ratio (size determined by 1/4 screen width)
        - Shows color distribution on UV color wheel
        - Downsampled for speed (every 8th pixel)
        """
        # Check if we have original frames to analyze
        if self.cached_original_frames is None or self.current_frame >= len(self.cached_original_frames):
            return
        
        screen_width, screen_height = screen.get_size()
        
        # Scope dimensions - SQUARE (size = 1/4 screen width)
        scope_size = int(screen_width * 0.25)
        scope_x = screen_width - scope_size
        scope_y = 0
        
        # Create semi-transparent square surface (80% opacity)
        scope_surface = pygame.Surface((scope_size, scope_size), pygame.SRCALPHA)
        bg_color = (20, 20, 20, 204)  # Dark background, 80% opaque
        scope_surface.fill(bg_color)
        
        # Border
        pygame.draw.rect(scope_surface, (80, 80, 80, 255), (0, 0, scope_size, scope_size), width=2)
        
        # Vectorscope parameters
        center_x = scope_size // 2
        center_y = scope_size // 2
        radius = int(scope_size * 0.4)  # Max radius for plotting
        
        # Draw graticule (target circles)
        graticule_color = (60, 60, 60, 255)
        for i in range(1, 4):
            circle_radius = int(radius * i / 3)
            pygame.draw.circle(scope_surface, graticule_color, (center_x, center_y), circle_radius, 1)
        
        # Draw crosshair
        pygame.draw.line(scope_surface, graticule_color, (center_x, 10), (center_x, scope_size - 10), 1)
        pygame.draw.line(scope_surface, graticule_color, (10, center_y), (scope_size - 10, center_y), 1)
        
        # Draw color target boxes at broadcast standard angles
        # Rotated 90° clockwise: Red at 8:00, then clockwise
        # U is NEGATED for correct horizontal flip
        # VIBRANT PRIMARY COLORS for better visibility
        color_targets = [
            (240, "R", (255, 60, 60)),     # Red at 8 o'clock - VIBRANT!
            (300, "M", (255, 60, 255)),    # Magenta at 10 o'clock - VIBRANT!
            (0, "B", (60, 60, 255)),       # Blue at 12 o'clock - VIBRANT!
            (60, "C", (60, 255, 255)),     # Cyan at 2 o'clock - VIBRANT!
            (120, "G", (60, 255, 60)),     # Green at 4 o'clock - VIBRANT!
            (180, "Y", (255, 255, 60))     # Yellow at 6 o'clock - VIBRANT!
        ]
        
        box_size = 12  # 1.5x larger (was 8)
        for angle_deg, label, color in color_targets:
            angle_rad = np.radians(angle_deg)
            # NEGATED cos for horizontal flip (U coordinate)
            target_x = int(center_x - radius * np.cos(angle_rad))
            target_y = int(center_y - radius * np.sin(angle_rad))  # Negative because screen Y is inverted
            
            # Draw colored box (1.5x larger, half luminance)
            box_rect = pygame.Rect(target_x - box_size//2, target_y - box_size//2, box_size, box_size)
            pygame.draw.rect(scope_surface, color, box_rect)
            pygame.draw.rect(scope_surface, (200, 200, 200, 255), box_rect, 1)
        
        # Calculate vectorscope from ORIGINAL frame (FAST: downsample by 8x8)
        try:
            # Get original frame directly from cache
            original_frame = self.cached_original_frames[self.current_frame]
            
            # Downsample: every 8th pixel for half density (was 4th)
            downsampled = original_frame[::8, ::8, :]
            
            # Convert RGB to UV (chrominance)
            # Standard BT.601 conversion
            R = downsampled[:,:,0].astype(np.float32) / 255.0
            G = downsampled[:,:,1].astype(np.float32) / 255.0
            B = downsampled[:,:,2].astype(np.float32) / 255.0
            
            U = -0.147 * R - 0.289 * G + 0.436 * B
            V = 0.615 * R - 0.515 * G - 0.100 * B
            
            # Flatten for plotting
            U_flat = U.flatten()
            V_flat = V.flatten()
            
            # Convert UV to screen coordinates
            # Scale UV range [-0.5, 0.5] to radius
            # Reduced scale to keep dots inside outer circle
            scale = radius * 4.5  # Reduced from 6 to stay inside
            
            # NEGATE U for horizontal flip (broadcast standard orientation)
            screen_x = (center_x - U_flat * scale).astype(np.int32)  # NEGATED U!
            screen_y = (center_y - V_flat * scale).astype(np.int32)  # Negative because screen Y is inverted
            
            # Calculate angle for each UV point (for color wheel)
            # Angle determines hue (rotated 90° clockwise to match our vectorscope)
            angles = np.arctan2(V_flat, -U_flat) + np.pi / 2  # Rotate 90° clockwise
            
            # Clip to scope bounds
            valid_mask = (
                (screen_x >= 2) & (screen_x < scope_size - 2) &
                (screen_y >= 2) & (screen_y < scope_size - 2)
            )
            screen_x = screen_x[valid_mask]
            screen_y = screen_y[valid_mask]
            angles = angles[valid_mask]
            
            # Draw colored dots based on color wheel position
            # 4x4 pixel dots for excellent visibility (16x area vs 1 pixel)
            for x, y, angle in zip(screen_x, screen_y, angles):
                # Convert angle to hue (0-360°)
                hue = (np.degrees(angle) % 360) / 360.0
                
                # Simple HSV to RGB conversion (fast!)
                # H = hue, S = 0.8 (saturated), V = 0.9 (bright)
                h_i = int(hue * 6)
                f = hue * 6 - h_i
                q = 0.9 * (1 - 0.8 * f)
                t = 0.9 * (1 - 0.8 * (1 - f))
                
                if h_i == 0:
                    r, g, b = 0.9, t, 0.1
                elif h_i == 1:
                    r, g, b = q, 0.9, 0.1
                elif h_i == 2:
                    r, g, b = 0.1, 0.9, t
                elif h_i == 3:
                    r, g, b = 0.1, q, 0.9
                elif h_i == 4:
                    r, g, b = t, 0.1, 0.9
                else:
                    r, g, b = 0.9, 0.1, q
                
                # Convert to 0-255 and add alpha
                dot_color = (int(r * 255), int(g * 255), int(b * 255), 180)
                
                # Draw 4x4 pixel block for excellent visibility
                if x + 3 < scope_size and y + 3 < scope_size:
                    for dx in range(4):
                        for dy in range(4):
                            scope_surface.set_at((x + dx, y + dy), dot_color)
            
        except Exception as e:
            # If calculation fails, just show graticule
            pass
        
        # Draw label
        font_key = "font_14"
        if font_key not in self._ui_cache:
            self._ui_cache[font_key] = pygame.font.SysFont(None, 14)
        small_font = self._ui_cache[font_key]
        
        label_text = small_font.render("VECTORSCOPE", True, (180, 180, 180, 255))
        scope_surface.blit(label_text, (10, 10))
        
        # Blit scope to screen
        screen.blit(scope_surface, (scope_x, scope_y))

    def _draw_waveform_rgb(self, screen, frame_surface):
        """
        Draw RGB waveform scope overlay
        - Analyzes ORIGINAL cached frames (before zoom/pan/fit transformations)
        - 80% opacity (see-through like Vision bar)
        - Full height deployment (like Histogram)
        - Shows vertical luminance distribution for RGB channels
        - Downsampled for speed (every 4th pixel)
        """
        # Check if we have original frames to analyze
        if self.cached_original_frames is None or self.current_frame >= len(self.cached_original_frames):
            return
        
        screen_width, screen_height = screen.get_size()
        
        # Scope dimensions (covers 1/4 width on right side, full height minus toolbar and Vision bar)
        scope_width = int(screen_width * 0.25)
        scope_height = screen_height - self.toolbar_height - 50  # Minus Vision bar height
        scope_x = screen_width - scope_width
        scope_y = 0
        
        # Create semi-transparent surface (80% opacity)
        scope_surface = pygame.Surface((scope_width, scope_height), pygame.SRCALPHA)
        bg_color = (20, 20, 20, 204)  # Dark background, 80% opaque
        pygame.draw.rect(scope_surface, bg_color, (0, 0, scope_width, scope_height))
        
        # Border
        pygame.draw.rect(scope_surface, (80, 80, 80, 255), (0, 0, scope_width, scope_height), width=2)
        
        # Draw channel divider lines (split into 3 vertical sections)
        line_color = (60, 60, 60, 255)
        channel_height = (scope_height - 20) // 3  # Divide into 3 equal sections
        
        # Top divider (between Red and Green)
        div1_y = 10 + channel_height
        pygame.draw.line(scope_surface, line_color, (10, div1_y), (scope_width - 10, div1_y), 1)
        
        # Bottom divider (between Green and Blue)
        div2_y = 10 + channel_height * 2
        pygame.draw.line(scope_surface, line_color, (10, div2_y), (scope_width - 10, div2_y), 1)
        
        # Calculate waveform from ORIGINAL frame (FAST: downsample by 6x6 for speed)
        try:
            # Get original frame directly from cache
            original_frame = self.cached_original_frames[self.current_frame]
            
            # Downsample: every 6th pixel (was 4th) = faster!
            downsampled = original_frame[::6, ::6, :]
            
            # Get dimensions
            ds_height, ds_width = downsampled.shape[:2]
            
            # Scale X coordinates to fit scope width
            x_scale = (scope_width - 20) / ds_width  # Leave padding
            
            # Plot area parameters - 3 separate channels
            plot_x_start = 10
            
            # Channel sections (each gets 1/3 of height)
            red_channel_top = 10
            red_channel_height = channel_height
            
            green_channel_top = 10 + channel_height
            green_channel_height = channel_height
            
            blue_channel_top = 10 + channel_height * 2
            blue_channel_height = channel_height
            
            # Semi-transparent RGB colors
            color_r = (255, 80, 80, 120)
            color_g = (80, 255, 80, 120)
            color_b = (80, 80, 255, 120)
            
            # Plot each column of pixels
            for x in range(ds_width):
                # Get this column of pixels
                column = downsampled[:, x, :]
                
                # Screen X position for this column
                screen_x = int(plot_x_start + x * x_scale)
                
                if screen_x < 10 or screen_x >= scope_width - 10:
                    continue
                
                # Plot each pixel in the column
                for pixel in column:
                    r, g, b = pixel
                    
                    # Convert 0-255 values to Y position within each channel
                    # RED channel (top third)
                    y_r = int(red_channel_top + (255 - r) / 255.0 * red_channel_height)
                    
                    # GREEN channel (middle third)
                    y_g = int(green_channel_top + (255 - g) / 255.0 * green_channel_height)
                    
                    # BLUE channel (bottom third)
                    y_b = int(blue_channel_top + (255 - b) / 255.0 * blue_channel_height)
                    
                    # Clamp Y values to their respective channels
                    y_r = max(red_channel_top, min(red_channel_top + red_channel_height, y_r))
                    y_g = max(green_channel_top, min(green_channel_top + green_channel_height, y_g))
                    y_b = max(blue_channel_top, min(blue_channel_top + blue_channel_height, y_b))
                    
                    # Draw dots for each channel in its own section
                    if 10 <= screen_x < scope_width - 10:
                        scope_surface.set_at((screen_x, y_r), color_r)
                        scope_surface.set_at((screen_x, y_g), color_g)
                        scope_surface.set_at((screen_x, y_b), color_b)
            
        except Exception as e:
            # If calculation fails, just show reference lines
            pass
        
        # Draw label
        font_key = "font_14"
        if font_key not in self._ui_cache:
            self._ui_cache[font_key] = pygame.font.SysFont(None, 14)
        small_font = self._ui_cache[font_key]
        
        label_text = small_font.render("WAVEFORM RGB", True, (180, 180, 180, 255))
        scope_surface.blit(label_text, (10, scope_height - 25))  # Bottom left
        
        # Draw channel labels on right side
        label_r = small_font.render("R", True, (255, 80, 80, 255))
        label_g = small_font.render("G", True, (80, 255, 80, 255))
        label_b = small_font.render("B", True, (80, 80, 255, 255))
        
        scope_surface.blit(label_r, (scope_width - 20, red_channel_top + red_channel_height // 2 - 7))
        scope_surface.blit(label_g, (scope_width - 20, green_channel_top + green_channel_height // 2 - 7))
        scope_surface.blit(label_b, (scope_width - 20, blue_channel_top + blue_channel_height // 2 - 7))
        
        # Blit scope to screen
        screen.blit(scope_surface, (scope_x, scope_y))

    def _draw_zebra_pattern(self, screen, frame_surface, frame_pos):
        """
        Draw zebra stripe pattern over clipped areas
        - Shows diagonal stripes (45° angle) on overexposed areas (>235)
        - 5 pixel line width
        - Black/white alternating stripes
        """
        if not self.vision_module_open or not self.vision_zebra_active:
            return
        
        if self.cached_original_frames is None or self.current_frame >= len(self.cached_original_frames):
            return
        
        try:
            # Get original frame
            original_frame = self.cached_original_frames[self.current_frame]
            
            # Find overexposed pixels (luminance > 235)
            # Calculate luminance: Y = 0.299*R + 0.587*G + 0.114*B
            luminance = (
                original_frame[:, :, 0] * 0.299 +
                original_frame[:, :, 1] * 0.587 +
                original_frame[:, :, 2] * 0.114
            )
            
            # Mask for clipped areas
            clipped_mask = luminance > 235
            
            if not clipped_mask.any():
                return  # No clipping, nothing to draw
            
            # Get frame position and size
            pos_x, pos_y, draw_w, draw_h = frame_pos
            
            # Create zebra surface matching displayed frame size
            zebra_surface = pygame.Surface((draw_w, draw_h), pygame.SRCALPHA)
            
            # Scale mask to match displayed size
            from scipy.ndimage import zoom
            scale_y = draw_h / original_frame.shape[0]
            scale_x = draw_w / original_frame.shape[1]
            
            # Resize mask to match display size
            if scale_x != 1.0 or scale_y != 1.0:
                clipped_mask_scaled = zoom(clipped_mask.astype(np.float32), (scale_y, scale_x), order=0) > 0.5
            else:
                clipped_mask_scaled = clipped_mask
            
            # Draw diagonal stripes (45° angle)
            # Solid alternating green and magenta lines, touching side-to-side
            line_width = 5  # Width of each stripe
            
            # Colors for alternating stripes (fully opaque)
            color_green = (0, 255, 0, 255)      # Solid bright green
            color_magenta = (255, 0, 255, 255)  # Solid bright magenta
            
            # Create diagonal stripe pattern at 45° (top-left to bottom-right)
            # We'll draw lines perpendicular to the 45° angle to create the stripes
            for offset in range(-draw_h, draw_w, line_width * 2):  # *2 because green+magenta = 2 stripes
                # Draw green stripe at 45° diagonal
                # Line goes from (offset, 0) to (offset + draw_h, draw_h)
                start_x = offset
                start_y = 0
                end_x = offset + draw_h
                end_y = draw_h
                
                pygame.draw.line(zebra_surface, color_green, (start_x, start_y), (end_x, end_y), line_width)
                
                # Draw magenta stripe right next to it
                offset_magenta = offset + line_width
                start_x_m = offset_magenta
                start_y_m = 0
                end_x_m = offset_magenta + draw_h
                end_y_m = draw_h
                
                pygame.draw.line(zebra_surface, color_magenta, (start_x_m, start_y_m), (end_x_m, end_y_m), line_width)
            
            # Apply mask - only show stripes over clipped areas
            zebra_array = pygame.surfarray.pixels_alpha(zebra_surface)
            clipped_mask_scaled = clipped_mask_scaled[:draw_h, :draw_w]  # Crop to exact size
            
            # Zero out alpha where not clipped
            zebra_array[:, :] = (zebra_array[:, :] * clipped_mask_scaled.T).astype(np.uint8)
            del zebra_array
            
            # Blit zebra pattern over video
            screen.blit(zebra_surface, (pos_x, pos_y))
            
        except Exception as e:
            # If zebra fails, just skip it
            pass

    def _draw_aspect_mask(self, screen, frame_pos):
        """
        Draw aspect ratio mask - black bars covering areas outside target ratio
        Adapts to any video aspect/resolution and all transformations
        """
        if self.vision_mask_index == 0:
            return  # No mask active
        
        try:
            # Get current mask ratio
            mask_name, target_ratio = self.vision_mask_ratios[self.vision_mask_index]
            
            if target_ratio is None:
                return
            
            # Get video rectangle in screen space
            pos_x, pos_y, draw_w, draw_h = frame_pos
            
            # Calculate current video aspect ratio
            current_ratio = draw_w / draw_h
            
            # Determine if we need letterbox (horizontal bars) or pillarbox (vertical bars)
            if target_ratio > current_ratio:
                # Target is wider - need letterbox (top/bottom bars)
                # Calculate visible height for target ratio
                visible_h = draw_w / target_ratio
                bar_h = (draw_h - visible_h) / 2
                
                # Draw top bar
                top_bar = pygame.Rect(pos_x, pos_y, draw_w, int(bar_h))
                pygame.draw.rect(screen, (0, 0, 0, 255), top_bar)
                
                # Draw bottom bar
                bottom_bar = pygame.Rect(pos_x, int(pos_y + draw_h - bar_h), draw_w, int(bar_h))
                pygame.draw.rect(screen, (0, 0, 0, 255), bottom_bar)
                
            elif target_ratio < current_ratio:
                # Target is narrower - need pillarbox (left/right bars)
                # Calculate visible width for target ratio
                visible_w = draw_h * target_ratio
                bar_w = (draw_w - visible_w) / 2
                
                # Draw left bar
                left_bar = pygame.Rect(pos_x, pos_y, int(bar_w), draw_h)
                pygame.draw.rect(screen, (0, 0, 0, 255), left_bar)
                
                # Draw right bar
                right_bar = pygame.Rect(int(pos_x + draw_w - bar_w), pos_y, int(bar_w), draw_h)
                pygame.draw.rect(screen, (0, 0, 0, 255), right_bar)
            
            # Display current mask ratio on screen (small text at bottom)
            font_key = "font_14"
            if font_key not in self._ui_cache:
                self._ui_cache[font_key] = pygame.font.SysFont(None, 14)
            small_font = self._ui_cache[font_key]
            
            ratio_text = small_font.render(f"Mask: {mask_name}", True, (200, 200, 200, 255))
            text_x = pos_x + 10
            text_y = pos_y + draw_h - 25
            screen.blit(ratio_text, (text_x, text_y))
            
        except Exception as e:
            # If mask fails, just skip
            pass

    def _draw_guides_overlay(self, screen, frame_pos):
        """
        Draw broadcast guides overlay:
        - Rule of thirds (white)
        - Action safe 5% (green) 
        - Title safe 10% (blue)
        - Format center cross (red)
        All adapt to video aspect/resolution/zoom/pan
        Can persist when Vision module is closed
        """
        # Check if guides should be drawn (active OR persisting when Vision closed)
        if not self.vision_guides_active and not (self.vision_guides_persist and not self.vision_module_open):
            return
        
        try:
            # Get video rectangle in screen space
            pos_x, pos_y, draw_w, draw_h = frame_pos
            
            # Rule of Thirds - White lines
            thirds_color = (255, 255, 255, 180)  # Semi-transparent white
            
            # Vertical thirds
            third_w = draw_w / 3
            v1_x = int(pos_x + third_w)
            v2_x = int(pos_x + third_w * 2)
            pygame.draw.line(screen, thirds_color, (v1_x, pos_y), (v1_x, pos_y + draw_h), 1)
            pygame.draw.line(screen, thirds_color, (v2_x, pos_y), (v2_x, pos_y + draw_h), 1)
            
            # Horizontal thirds
            third_h = draw_h / 3
            h1_y = int(pos_y + third_h)
            h2_y = int(pos_y + third_h * 2)
            pygame.draw.line(screen, thirds_color, (pos_x, h1_y), (pos_x + draw_w, h1_y), 1)
            pygame.draw.line(screen, thirds_color, (pos_x, h2_y), (pos_x + draw_w, h2_y), 1)
            
            # Action Safe - Green rectangle (5% margin = 90% of frame)
            action_margin = 0.05
            action_x = int(pos_x + draw_w * action_margin)
            action_y = int(pos_y + draw_h * action_margin)
            action_w = int(draw_w * (1 - 2 * action_margin))
            action_h = int(draw_h * (1 - 2 * action_margin))
            action_color = (80, 255, 80, 180)  # Semi-transparent green
            pygame.draw.rect(screen, action_color, (action_x, action_y, action_w, action_h), 1)
            
            # Title Safe - Blue rectangle (10% margin = 80% of frame)
            title_margin = 0.10
            title_x = int(pos_x + draw_w * title_margin)
            title_y = int(pos_y + draw_h * title_margin)
            title_w = int(draw_w * (1 - 2 * title_margin))
            title_h = int(draw_h * (1 - 2 * title_margin))
            title_color = (80, 80, 255, 180)  # Semi-transparent blue
            pygame.draw.rect(screen, title_color, (title_x, title_y, title_w, title_h), 1)
            
            # Format Center - Red cross (10px lines)
            center_x = int(pos_x + draw_w / 2)
            center_y = int(pos_y + draw_h / 2)
            cross_size = 10
            center_color = (255, 80, 80, 255)  # Bright red
            
            # Horizontal line
            pygame.draw.line(screen, center_color,
                           (center_x - cross_size, center_y),
                           (center_x + cross_size, center_y), 2)
            # Vertical line
            pygame.draw.line(screen, center_color,
                           (center_x, center_y - cross_size),
                           (center_x, center_y + cross_size), 2)
            
        except Exception as e:
            # If guides fail, just skip
            pass

    def _draw_color_picker(self, screen, frame_pos):
        """
        Draw color picker tool - 5x5 square that samples pixel values
        Shows R, G, B, A, SAT, LUMA in info panel above Vision bar
        """
        if not self.vision_module_open or not self.vision_picker_active:
            return
        
        if self.cached_original_frames is None or self.current_frame >= len(self.cached_original_frames):
            return
        
        try:
            screen_width, screen_height = screen.get_size()
            
            # Initialize picker position if needed (center of screen)
            if self.vision_picker_pos is None:
                self.vision_picker_pos = [screen_width // 2, screen_height // 2]
            
            picker_x, picker_y = self.vision_picker_pos
            picker_size = 5
            
            # Draw 5x5 picker square (pure RED)
            picker_rect = pygame.Rect(picker_x - picker_size//2, picker_y - picker_size//2, picker_size, picker_size)
            pygame.draw.rect(screen, (255, 0, 0, 255), picker_rect, 2)  # Pure red outline
            
            # Draw crosshair (pure RED)
            pygame.draw.line(screen, (255, 0, 0, 255), 
                           (picker_x - 10, picker_y), (picker_x + 10, picker_y), 1)
            pygame.draw.line(screen, (255, 0, 0, 255),
                           (picker_x, picker_y - 10), (picker_x, picker_y + 10), 1)
            
            # Sample pixel from original frame
            original_frame = self.cached_original_frames[self.current_frame]
            frame_height, frame_width = original_frame.shape[:2]
            
            # Convert screen coordinates to original frame coordinates
            pos_x, pos_y, draw_w, draw_h = frame_pos
            
            # Check if picker is over video
            if (picker_x < pos_x or picker_x >= pos_x + draw_w or
                picker_y < pos_y or picker_y >= pos_y + draw_h):
                # Picker outside video area
                sampled_r, sampled_g, sampled_b = 0, 0, 0
                sampled_a = None
                sat = 0.0
                luma = 0
            else:
                # Map screen position to original frame position
                rel_x = (picker_x - pos_x) / draw_w  # 0.0 to 1.0
                rel_y = (picker_y - pos_y) / draw_h  # 0.0 to 1.0
                
                orig_x = int(rel_x * frame_width)
                orig_y = int(rel_y * frame_height)
                
                # Clamp to frame bounds
                orig_x = max(0, min(frame_width - 1, orig_x))
                orig_y = max(0, min(frame_height - 1, orig_y))
                
                # Sample pixel
                pixel = original_frame[orig_y, orig_x]
                sampled_r = int(pixel[0])
                sampled_g = int(pixel[1])
                sampled_b = int(pixel[2])
                sampled_a = None  # No alpha in our frames
                
                # Calculate saturation (0.0 = grey, 1.0 = full color)
                max_rgb = max(sampled_r, sampled_g, sampled_b)
                min_rgb = min(sampled_r, sampled_g, sampled_b)
                if max_rgb == 0:
                    sat = 0.0
                else:
                    sat = (max_rgb - min_rgb) / max_rgb
                
                # Calculate luminance
                luma = int(0.299 * sampled_r + 0.587 * sampled_g + 0.114 * sampled_b)
            
            # Draw info panel above Vision bar
            info_panel_height = 30
            info_panel_y = screen_height - self.toolbar_height - 50 - info_panel_height
            info_panel_width = int(screen_width * 0.25)  # 1/4 width (right side)
            info_panel_x = screen_width - info_panel_width
            
            # Create semi-transparent info panel surface
            info_surface = pygame.Surface((info_panel_width, info_panel_height), pygame.SRCALPHA)
            bg_color = (20, 20, 20, 204)  # 80% opaque like Vision bar
            info_surface.fill(bg_color)
            
            # Border
            pygame.draw.rect(info_surface, (80, 80, 80, 255), (0, 0, info_panel_width, info_panel_height), width=1)
            
            # Draw values with bigger font
            font_key = "font_18"
            if font_key not in self._ui_cache:
                self._ui_cache[font_key] = pygame.font.SysFont(None, 18)
            picker_font = self._ui_cache[font_key]
            
            # Single row: R G B A SAT LUMA (all on one line)
            r_text = picker_font.render(f"{sampled_r}", True, (255, 80, 80, 255))  # Red
            g_text = picker_font.render(f"{sampled_g}", True, (80, 255, 80, 255))  # Green
            b_text = picker_font.render(f"{sampled_b}", True, (80, 80, 255, 255))  # Blue
            a_text = picker_font.render("none" if sampled_a is None else f"{sampled_a:.2f}", 
                                     True, (255, 255, 255, 255))  # White
            sat_text = picker_font.render(f"SAT:{sat:.2f}", True, (180, 180, 180, 255))
            luma_text = picker_font.render(f"LUMA:{luma}", True, (180, 180, 180, 255))
            
            # All on one row, centered vertically
            y_center = (info_panel_height - 18) // 2
            info_surface.blit(r_text, (5, y_center))
            info_surface.blit(g_text, (50, y_center))
            info_surface.blit(b_text, (95, y_center))
            info_surface.blit(a_text, (140, y_center))
            info_surface.blit(sat_text, (200, y_center))
            info_surface.blit(luma_text, (285, y_center))
            
            # Blit info panel to screen
            screen.blit(info_surface, (info_panel_x, info_panel_y))
            
        except Exception as e:
            # If picker fails, just skip it
            pass

    def _open_explorer_at_cache(self):
        """Open Windows Explorer at the runs_cache directory"""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(RUNS_CACHE_DIR)
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', RUNS_CACHE_DIR])
            else:  # Linux and other Unix-like
                subprocess.Popen(['xdg-open', RUNS_CACHE_DIR])
            
            self.log(f"Opened explorer at: {RUNS_CACHE_DIR}")
            return True
        except Exception as e:
            self.log(f"Error opening explorer: {e}")
            return False

    def _open_explorer_at_snapshots(self):
        """Open Windows Explorer at the current snapshots directory (smart or custom)"""
        try:
            snapshot_dir = self.current_snapshot_path  # Use resolved path
            
            if os.name == 'nt':  # Windows
                os.startfile(snapshot_dir)
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['open', snapshot_dir])
            else:  # Linux and other Unix-like
                subprocess.Popen(['xdg-open', snapshot_dir])
            
            self.log(f"Opened explorer at: {snapshot_dir}")
            return True
        except Exception as e:
            self.log(f"Error opening explorer: {e}")
            return False

    def _resolve_snapshot_path(self, user_input):
        """
        Resolve snapshot path from user input.
        - "smart" → default snapshots directory
        - Custom path → path/snapshots/ (auto-created)
        
        Security: Uses path containment check to ensure paths are within allowed directories.
        """
        try:
            # Check if user wants default "smart" path
            if user_input.lower().strip() == "smart":
                self.log("Using default snapshot path (smart mode)")
                return SNAPSHOTS_DIR
            
            # User provided custom path
            # Normalize path separators for current OS
            normalized_path = user_input.replace('\\\\', os.sep).replace('\\', os.sep).replace('/', os.sep)
            
            # Remove trailing separators
            normalized_path = normalized_path.rstrip(os.sep)
            
            # Convert to absolute path and resolve any .. or . components
            normalized_path = os.path.abspath(os.path.realpath(normalized_path))
            
            # Add "snapshots" subdirectory
            snapshots_path = os.path.join(normalized_path, "snapshots")
            
            # Security check: Ensure the path is within allowed locations
            # Allowed: user-specified directory or the default BASE_DIR
            # This prevents path traversal attacks (e.g., ../../etc/passwd)
            allowed_roots = [
                os.path.abspath(os.path.realpath(normalized_path)),
                os.path.abspath(os.path.realpath(BASE_DIR))
            ]
            
            resolved_snapshots = os.path.abspath(os.path.realpath(snapshots_path))
            
            # Check path containment using os.path.commonpath
            is_safe = False
            for allowed_root in allowed_roots:
                try:
                    common = os.path.commonpath([resolved_snapshots, allowed_root])
                    if common == allowed_root:
                        is_safe = True
                        break
                except ValueError:
                    # commonpath raises ValueError if paths are on different drives (Windows)
                    continue
            
            if not is_safe:
                raise Exception(f"Path '{snapshots_path}' is outside allowed directories")
            
            # Try to create the directory
            os.makedirs(snapshots_path, exist_ok=True)
            
            # Verify it's writable with path containment check
            test_file = os.path.join(snapshots_path, ".write_test")
            test_file_resolved = os.path.abspath(os.path.realpath(test_file))
            
            # Ensure test file is within the snapshots directory
            try:
                common = os.path.commonpath([test_file_resolved, resolved_snapshots])
                if common != resolved_snapshots:
                    raise Exception("Test file path outside snapshots directory")
            except ValueError:
                raise Exception("Test file path on different drive")
            
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except:
                raise Exception("Directory is not writable")
            
            self.log(f"Using custom snapshot path: {snapshots_path}")
            return snapshots_path
            
        except Exception as e:
            self.log(f"Error resolving custom snapshot path '{user_input}': {e}")
            self.log("Falling back to default snapshot path")
            return SNAPSHOTS_DIR

    def _draw_snapshot_dropup(self, screen, mouse_pos, snapshot_button_rect=None):
        """Draw the snapshot dropUP menu positioned above the Snapshot button - OPTIMIZED"""
        if not self.snapshot_dropup_open:
            return None

        # Menu dimensions - NOW 3 buttons: Take, Clear, Explore
        menu_width = 150
        item_height = 35
        menu_height = 3 * item_height + 10  # CHANGED: 3 buttons now

        # Position above Snapshot button if provided, otherwise default position
        if snapshot_button_rect:
            # Center menu above the Snapshot button
            menu_x = snapshot_button_rect.centerx - menu_width // 2
            menu_y = snapshot_button_rect.top - menu_height - 5
        else:
            # Fallback position
            vw, vh = screen.get_size()
            menu_x = vw - menu_width - 10
            menu_y = vh - self.toolbar_height - menu_height - 5

        # Ensure menu stays on screen
        menu_x = max(10, min(menu_x, screen.get_width() - menu_width - 10))
        menu_y = max(10, menu_y)

        # Cache font
        font_key = f"font_18"
        if font_key not in self._ui_cache:
            self._ui_cache[font_key] = pygame.font.SysFont(None, 18)
        font = self._ui_cache[font_key]

        # Background
        menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
        pygame.draw.rect(screen, (40, 40, 40), menu_rect, border_radius=4)
        pygame.draw.rect(screen, (80, 80, 80), menu_rect,
                         width=2, border_radius=4)

        # Draw the THREE buttons
        take_rect = pygame.Rect(
            menu_x + 10, menu_y + 5, menu_width - 20, item_height - 5)
        clear_rect = pygame.Rect(
            menu_x + 10, menu_y + 5 + item_height, menu_width - 20, item_height - 5)
        explore_rect = pygame.Rect(
            menu_x + 10, menu_y + 5 + 2 * item_height, menu_width - 20, item_height - 5)

        # Check hover state
        mouse_x, mouse_y = mouse_pos
        take_hover = take_rect.collidepoint(mouse_x, mouse_y)
        clear_hover = clear_rect.collidepoint(mouse_x, mouse_y)
        explore_hover = explore_rect.collidepoint(mouse_x, mouse_y)

        # Visual feedback for recent snapshot or saving in progress
        current_time = pygame.time.get_ticks()
        take_active = (self.snapshot_saving or 
                      (current_time - self.last_snapshot_time < 1000))  # 1 second feedback

        # Draw Take button with DARK GREEN background (matching Snapshot button dark green outline)
        # Same luminance as Clear button for good contrast
        take_bg = (30, 80, 40) if take_hover else (25, 70, 35)  # Dark green background
        
        pygame.draw.rect(screen, take_bg, take_rect, border_radius=3)
        pygame.draw.rect(screen, (120, 120, 120), take_rect, width=2, border_radius=3)  # Light grey outline (always)

        # Draw Clear button with RED background and outline (like ClearCache)
        clear_bg = (100, 40, 40) if clear_hover else (80, 30, 30)
        pygame.draw.rect(screen, clear_bg, clear_rect, border_radius=3)
        pygame.draw.rect(screen, (220, 80, 80), clear_rect, width=2, border_radius=3)

        # Draw Explore button
        explore_bg = (60, 60, 60)
        if explore_hover:
            explore_bg = (70, 70, 70)
        
        pygame.draw.rect(screen, explore_bg, explore_rect, border_radius=3)
        pygame.draw.rect(screen, (120, 120, 120), explore_rect, width=2, border_radius=3)

        # Draw text - change to confirmation if needed
        take_text = font.render("Take", True, (220, 220, 220))
        if self.snapshot_clear_confirm:
            clear_text = font.render("¿ClearForever?", True, (255, 180, 180))  # Confirmation text
        else:
            clear_text = font.render("Clear Snaps", True, (255, 180, 180))  # Normal text
        explore_text = font.render("Explore", True, (220, 220, 220))

        take_text_rect = take_text.get_rect(center=take_rect.center)
        clear_text_rect = clear_text.get_rect(center=clear_rect.center)
        explore_text_rect = explore_text.get_rect(center=explore_rect.center)

        screen.blit(take_text, take_text_rect)
        screen.blit(clear_text, clear_text_rect)
        screen.blit(explore_text, explore_text_rect)

        # Store rects for click detection
        item_rects = [
            ("take", take_rect),
            ("clear", clear_rect),
            ("explore", explore_rect)
        ]

        return (menu_rect, item_rects)

    # =========== NEW FPS DROPUP SYSTEM ===========
    def _draw_fps_dropup(self, screen, mouse_pos, fps_button_rect=None):
        """Draw the FPS dropUP menu with presets, FPS field (yellow), and First Frame field (blue)"""
        if not self.fps_dropup_open:
            return None

        # Menu dimensions
        menu_width = 120
        item_height = 28
        num_presets = len(self.fps_presets)
        fps_field_height = 35  # Yellow FPS field
        first_frame_field_height = 35  # Blue first frame field
        menu_height = (num_presets * item_height) + fps_field_height + first_frame_field_height + 20  # Presets + FPS + First Frame + padding

        # Position above FPS button if provided
        if fps_button_rect:
            menu_x = fps_button_rect.centerx - menu_width // 2
            menu_y = fps_button_rect.top - menu_height - 5
        else:
            # Fallback position
            vw, vh = screen.get_size()
            menu_x = 150
            menu_y = vh - self.toolbar_height - menu_height - 5

        # Ensure menu stays on screen
        menu_x = max(10, min(menu_x, screen.get_width() - menu_width - 10))
        menu_y = max(10, menu_y)

        # Cache font
        font_key = "font_18"
        if font_key not in self._ui_cache:
            self._ui_cache[font_key] = pygame.font.SysFont(None, 18)
        font = self._ui_cache[font_key]

        # Background
        menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
        pygame.draw.rect(screen, (40, 40, 40), menu_rect, border_radius=4)
        pygame.draw.rect(screen, (160, 160, 160), menu_rect, width=2, border_radius=4)  # Grey border

        mouse_x, mouse_y = mouse_pos
        item_rects = []
        current_y = menu_y + 5

        # Draw preset buttons (high to low)
        for i, fps_value in enumerate(self.fps_presets):
            preset_rect = pygame.Rect(menu_x + 5, current_y, menu_width - 10, item_height - 3)
            
            # Check hover
            is_hover = preset_rect.collidepoint(mouse_x, mouse_y)
            is_current = (abs(self.user_fps - fps_value) < 0.01)  # Current FPS
            
            # Background color
            if is_current:
                bg_color = (70, 100, 140)  # Blue highlight for current
            elif is_hover:
                bg_color = (70, 70, 70)  # Grey hover
            else:
                bg_color = (50, 50, 50)  # Default dark
            
            pygame.draw.rect(screen, bg_color, preset_rect, border_radius=2)
            
            # Format FPS value (handle special cases like 23.976, 29.97, 59.94)
            if fps_value == int(fps_value):
                fps_text = f"{int(fps_value)}"
            else:
                fps_text = f"{fps_value:.3f}".rstrip('0').rstrip('.')
            
            text_surf = font.render(fps_text, True, (220, 220, 220))
            text_rect = text_surf.get_rect(center=preset_rect.center)
            screen.blit(text_surf, text_rect)
            
            item_rects.append((f"preset_{i}", preset_rect, fps_value))
            current_y += item_height

        # Draw separator
        separator_y = current_y
        pygame.draw.line(screen, (100, 100, 100), 
                        (menu_x + 5, separator_y), 
                        (menu_x + menu_width - 5, separator_y), 2)
        current_y += 5

        # Draw FPS entry field (YELLOW outline)
        fps_rect = pygame.Rect(menu_x + 5, current_y, menu_width - 10, fps_field_height - 5)
        
        # Background
        fps_field_bg = (60, 60, 80) if self.fps_custom_editing else (50, 50, 50)
        pygame.draw.rect(screen, fps_field_bg, fps_rect, border_radius=2)
        # YELLOW outline for FPS field
        pygame.draw.rect(screen, (220, 180, 60), fps_rect, width=2, border_radius=2)
        
        # Draw FPS text
        fps_text = self.fps_custom_text if self.fps_custom_editing else f"{self.user_fps:.3f}"
        text_surf = font.render(fps_text, True, (220, 220, 220))
        text_rect = text_surf.get_rect(center=fps_rect.center)
        screen.blit(text_surf, text_rect)
        
        # Draw cursor if editing FPS
        if self.fps_custom_editing:
            cursor_x = text_rect.right + 2
            cursor_y_top = text_rect.top
            cursor_y_bottom = text_rect.bottom
            pygame.draw.line(screen, (220, 220, 220), 
                           (cursor_x, cursor_y_top), 
                           (cursor_x, cursor_y_bottom), 2)
        
        item_rects.append(("fps_field", fps_rect, None))
        current_y += fps_field_height
        
        # Draw First Frame entry field (BLUE outline) - BELOW FPS field
        first_frame_rect = pygame.Rect(menu_x + 5, current_y, menu_width - 10, first_frame_field_height - 5)
        
        # Background
        first_frame_bg = (60, 60, 80) if self.first_frame_editing else (50, 50, 50)
        pygame.draw.rect(screen, first_frame_bg, first_frame_rect, border_radius=2)
        # BLUE outline for first frame field
        pygame.draw.rect(screen, (60, 140, 220), first_frame_rect, width=2, border_radius=2)
        
        # Draw first frame text
        first_frame_text = self.first_frame_text if self.first_frame_editing else self.first_frame_text
        text_surf = font.render(first_frame_text, True, (220, 220, 220))
        text_rect = text_surf.get_rect(center=first_frame_rect.center)
        screen.blit(text_surf, text_rect)
        
        # Draw cursor if editing first frame
        if self.first_frame_editing:
            cursor_x = text_rect.right + 2
            cursor_y_top = text_rect.top
            cursor_y_bottom = text_rect.bottom
            pygame.draw.line(screen, (220, 220, 220), 
                           (cursor_x, cursor_y_top), 
                           (cursor_x, cursor_y_bottom), 2)
        
        item_rects.append(("first_frame_field", first_frame_rect, None))

        return (menu_rect, item_rects)

    # =========== NEW CLEARCACHE DROPUP SYSTEM ===========
    def _draw_clearcache_dropup(self, screen, mouse_pos, clearcache_button_rect=None):
        """Draw the ClearCache dropUP menu positioned above the ClearCache button"""
        if not self.clearcache_dropup_open:
            return None

        # Menu dimensions - smaller for just 2 buttons
        menu_width = 150
        item_height = 35
        menu_height = 2 * item_height + 10

        # Position above ClearCache button if provided, otherwise default position
        if clearcache_button_rect:
            # Center menu above the ClearCache button
            menu_x = clearcache_button_rect.centerx - menu_width // 2
            menu_y = clearcache_button_rect.top - menu_height - 5
        else:
            # Fallback position
            vw, vh = screen.get_size()
            menu_x = vw - menu_width - 10
            menu_y = vh - self.toolbar_height - menu_height - 5

        # Ensure menu stays on screen
        menu_x = max(10, min(menu_x, screen.get_width() - menu_width - 10))
        menu_y = max(10, menu_y)

        # Cache font
        font_key = f"font_18"
        if font_key not in self._ui_cache:
            self._ui_cache[font_key] = pygame.font.SysFont(None, 18)
        font = self._ui_cache[font_key]

        # Background
        menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
        pygame.draw.rect(screen, (40, 40, 40), menu_rect, border_radius=4)
        pygame.draw.rect(screen, (80, 80, 80), menu_rect,
                         width=2, border_radius=4)

        # Draw the two buttons
        clear_rect = pygame.Rect(
            menu_x + 10, menu_y + 5, menu_width - 20, item_height - 5)
        explore_rect = pygame.Rect(
            menu_x + 10, menu_y + 5 + item_height, menu_width - 20, item_height - 5)

        # Check hover state
        mouse_x, mouse_y = mouse_pos
        clear_hover = clear_rect.collidepoint(mouse_x, mouse_y)
        explore_hover = explore_rect.collidepoint(mouse_x, mouse_y)

        # Draw Clear button with RED background and outline
        clear_bg = (100, 40, 40) if clear_hover else (80, 30, 30)
        pygame.draw.rect(screen, clear_bg, clear_rect, border_radius=3)
        pygame.draw.rect(screen, (220, 80, 80), clear_rect, width=2, border_radius=3)

        # Draw Explore button
        explore_bg = (60, 60, 60)
        if explore_hover:
            explore_bg = (70, 70, 70)
        
        pygame.draw.rect(screen, explore_bg, explore_rect, border_radius=3)
        pygame.draw.rect(screen, (120, 120, 120), explore_rect, width=2, border_radius=3)

        # Draw text - change to confirmation if needed
        if self.clearcache_clear_confirm:
            clear_text = font.render("¿ClearForever?", True, (255, 180, 180))  # Confirmation text
        else:
            clear_text = font.render("Clear Gens", True, (255, 180, 180))  # Normal text
        explore_text = font.render("Explore", True, (220, 220, 220))

        clear_text_rect = clear_text.get_rect(center=clear_rect.center)
        explore_text_rect = explore_text.get_rect(center=explore_rect.center)

        screen.blit(clear_text, clear_text_rect)
        screen.blit(explore_text, explore_text_rect)

        # Store rects for click detection
        item_rects = [
            ("clear", clear_rect),
            ("explore", explore_rect)
        ]

        return (menu_rect, item_rects)

    def _clear_generations_cache(self):
        """Clear all generation cache folders and files - does NOT delete snapshots"""
        try:
            import shutil
            
            # Get all folders and legacy files in runs_cache directory
            all_items = []
            for item in os.listdir(RUNS_CACHE_DIR):
                item_path = os.path.join(RUNS_CACHE_DIR, item)
                
                # Skip temp_workflow directory
                if item == "temp_workflow":
                    continue
                
                # Add folders (new format) and legacy files
                if os.path.isdir(item_path):
                    all_items.append((item_path, 'folder'))
                elif item.endswith(('.mp4', '.png')):
                    all_items.append((item_path, 'file'))
            
            if not all_items:
                self.log("Generations cache is already empty")
                return False
            
            # Delete all items
            deleted_count = 0
            for item_path, item_type in all_items:
                try:
                    if item_type == 'folder':
                        shutil.rmtree(item_path)
                        deleted_count += 1
                    else:
                        os.remove(item_path)
                        deleted_count += 1
                except Exception as e:
                    self.log(f"Error deleting {item_path}: {e}")
            
            # Clear all generations metadata
            self.generations_metadata.clear()
            self._save_generations_metadata()
            
            # Clear global cache
            with self._global_cache_lock:
                self._global_cache.clear()
            
            # Clear instance caches
            with self.cache_lock:
                self.cached_original_frames = None
                self.cached_scaled_frames = None
                self.cached_surfaces = None
                self.original_resolution = (0, 0)
                self.current_frame = 0
                self.counter_update_frame = 0
                self.current_generation_id = None
            
            self.log(f"Cleared {deleted_count} generation items from cache")
            return True
            
        except Exception as e:
            self.log(f"Error clearing cache: {e}")
            traceback.print_exc()
            return False

    def _clear_snapshots_cache(self):
        """Clear all snapshot files (WebP and PNG) from current snapshot path - does NOT delete generation cache"""
        try:
            # Get all snapshot files in current snapshots directory (smart or custom)
            # Support both .webp (new format) and .png (legacy) files
            all_files = []
            snapshot_dir = self.current_snapshot_path
            for root, dirs, files in os.walk(snapshot_dir):
                for file in files:
                    if file.endswith('.webp') or file.endswith('.png'):  # Delete both WebP and PNG snapshot files
                        file_path = os.path.join(root, file)
                        all_files.append(file_path)
            
            if not all_files:
                self.log(f"Snapshots directory is already empty: {snapshot_dir}")
                return False
            
            # Delete all snapshot files
            deleted_count = 0
            for file_path in all_files:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    self.log(f"Error deleting {file_path}: {e}")
            
            self.log(f"Cleared {deleted_count} snapshot files from: {snapshot_dir}")
            return True
            
        except Exception as e:
            self.log(f"Error clearing snapshots: {e}")
            traceback.print_exc()
            return False

    def _mark_ui_dirty(self):
        """Mark UI cache as dirty (needs re-rendering) - DOES NOT AFFECT FRAME CACHE"""
        self._ui_cache_dirty = True
        self._cached_toolbar_surface = None
        self._cached_timeline_surface = None
        self._cached_counter_surfaces = None

    def _mark_frame_cache_dirty(self):
        """Mark frame cache as dirty (needs re-rendering) - CLEARS MULTI-FRAME CACHE"""
        self._frame_cache_dirty = True
        # Clear multi-frame surface cache
        with self._frame_surface_lock:
            self._frame_surface_cache.clear()

    def _calculate_ui_state_hash(self):
        """Calculate a hash of the current UI state for caching"""
        try:
            state_values = [
                self.viewer_fit_mode,
                self.fullscreen_mode,
                self.user_marks_active,
                self.dropup_open,
                self.color_space_dropup_open,
                self.snapshot_dropup_open,
                self.clearcache_dropup_open,
                self.button_hover,
                self.dropup_hover_index,
                self.dropup_selected_index,
                self.dropup_editing_index,
                self.color_space_hover_index,
                self.color_space_selected_index,
                self.scrubbing,
                self.viewer_paused,
                self.viewer_playback_mode,
                self.counter_update_frame,
                int(self.viewer_zoom * 100),
                int(self.viewer_offset[0]),
                int(self.viewer_offset[1]),
                self.user_in_point,
                self.user_out_point,
                self.selected_color_space,
                int(self.last_snapshot_time > 0),
                int(self.snapshot_saving),
                # NEW: Include wipe state in UI hash
                int(self.wipe_active),
                self.wipe_comparison_gen_id or "",
                int(self.wipe_position * 1000),
            ]
            return hash(tuple(state_values))
        except:
            return 0

    def _calculate_frame_state_hash(self, current_frame, screen_size=None):
        """Calculate a hash of the current frame rendering state"""
        try:
            if screen_size is None:
                screen_size = (1920, 1080)

            state_values = [
                current_frame,
                self.viewer_fit_mode,
                self.fullscreen_mode,
                int(self.viewer_zoom * 1000),  # More precision for zoom
                int(self.viewer_offset[0]),
                int(self.viewer_offset[1]),
                screen_size[0],
                screen_size[1],
                self.toolbar_height if not self.fullscreen_mode else 0,
                self.selected_color_space,
                # NEW: Include wipe state in frame hash
                int(self.wipe_active),
                self.wipe_comparison_gen_id or "",
                int(self.wipe_position * 1000),
            ]
            return hash(tuple(state_values))
        except:
            return 0

    # -----------------------
    # NEW: In-Place Renaming Methods with Disk File Renaming
    # -----------------------
    def _start_rename_generation(self, index):
        """Start renaming a generation in the dropup menu"""
        sorted_generations = self._get_sorted_generations()
        if 0 <= index < len(sorted_generations):
            gen_id, metadata = sorted_generations[index]
            display_name = metadata.get('display_name', '')

            # Parse display name to extract editable part
            # Format: prefix_runsName_timestamp
            parts = display_name.split('_')
            if len(parts) >= 3:
                # Get the editable middle part (runs_name)
                editable_part = '_'.join(parts[1:-1])
                self.dropup_edit_text = editable_part
            else:
                # Fallback: use runs_name from metadata
                self.dropup_edit_text = metadata.get('runs_name', '')

            self.dropup_editing_index = index
            self.dropup_edit_start_time = pygame.time.get_ticks()
            self.dropup_edit_cursor_visible = True
            self.dropup_edit_cursor_pos = len(self.dropup_edit_text)  # Start at end
            self.dropup_edit_initiated_by_keyboard = False  # Mouse initiated
            self._mark_ui_dirty()

            self.log(f"Starting rename for generation at index {index}")

    def _start_rename_generation_keyboard(self):
        """Start renaming the currently selected generation using keyboard"""
        if self.dropup_selected_index >= 0:
            self._start_rename_generation(self.dropup_selected_index)
            self.dropup_edit_initiated_by_keyboard = True
            return True
        return False

    def _rename_file_on_disk(self, old_path, new_name, source_type):
        """Rename the folder/file on disk while preserving the unique identifier part"""
        try:
            if not os.path.exists(old_path):
                self.log(f"Path not found for renaming: {old_path}")
                return None

            # Get the original name and parse its parts
            old_name = os.path.basename(old_path)
            old_dir = os.path.dirname(old_path)

            # Parse the old name: prefix_name_timestamp_uuid (folder) or prefix_name_timestamp_uuid.ext (file)
            # Example folder: v_MyGeneration_12-34-56_abc123def
            # Example file: v_MyGeneration_12-34-56_abc123def.mp4
            
            # Check if it's a folder or file
            is_folder = os.path.isdir(old_path)
            
            if is_folder:
                # Folder format (new): prefix_name_timestamp_uuid
                filename_parts = old_name.split('_')
                
                if len(filename_parts) >= 4:
                    prefix = filename_parts[0]  # v or i
                    timestamp_part = filename_parts[2]
                    uuid_part = filename_parts[3]
                    
                    # Create new folder name
                    new_folder_name = f"{prefix}_{new_name}_{timestamp_part}_{uuid_part}"
                    new_path = os.path.join(old_dir, new_folder_name)
                    
                    # Rename the folder
                    os.rename(old_path, new_path)
                    self.log(f"Renamed folder on disk: {old_name} -> {new_folder_name}")
                    
                    return new_path
                else:
                    self.log(f"Could not parse folder structure: {old_name}")
                    return None
            
            else:
                # File format (legacy): prefix_name_timestamp_uuid.ext
                filename_parts = old_name.split('_')
                
                if len(filename_parts) >= 4:
                    prefix = filename_parts[0]  # v or i
                    timestamp_part = filename_parts[2]
                    
                    # Preserve extension
                    extension = '.mp4'  # Default
                    if old_name.endswith('.png'):
                        extension = '.png'
                        uuid_part = filename_parts[3].replace('.png', '')
                    else:
                        uuid_part = filename_parts[3].replace('.mp4', '')
                    
                    # Create new filename
                    new_filename = f"{prefix}_{new_name}_{timestamp_part}_{uuid_part}{extension}"
                    new_path = os.path.join(old_dir, new_filename)
                    
                    # Rename the file
                    os.rename(old_path, new_path)
                    self.log(f"Renamed file on disk: {old_name} -> {new_filename}")
                    
                    return new_path
                else:
                    self.log(f"Could not parse filename structure: {old_name}")
                    return None

        except Exception as e:
            self.log(f"Error renaming on disk: {e}")
            traceback.print_exc()
            return None

    def _finish_rename_generation(self):
        """Finish renaming a generation and save changes - INCLUDES DISK RENAME"""
        try:
            if self.dropup_editing_index == -1:
                return

            sorted_generations = self._get_sorted_generations()
            if 0 <= self.dropup_editing_index < len(sorted_generations):
                gen_id, metadata = sorted_generations[self.dropup_editing_index]
                # NEW: Support both folder_path and file_path
                old_path = metadata.get('folder_path', metadata.get('file_path', ''))

                if not old_path or not os.path.exists(old_path):
                    self.log("Rename cancelled: path not found or doesn't exist")
                    self._cancel_rename()
                    return

            # Validate new name (no special characters, not empty)
            new_name = self.dropup_edit_text.strip()
            if not new_name:
                self.log("Rename cancelled: empty name")
                self._cancel_rename()
                return

            # Check for invalid characters
            invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '/', '\\']
            for char in invalid_chars:
                if char in new_name:
                    self.log(
                        f"Rename cancelled: invalid character '{char}' in name")
                    self._cancel_rename()
                    return

            # Get original display name parts
            old_display_name = metadata.get('display_name', '')
            parts = old_display_name.split('_')

            if len(parts) >= 3:
                # Get source type from prefix
                prefix = parts[0]  # v or i
                source_type = "video" if prefix == "v" else "images"
                timestamp = parts[-1]  # HH:MM:SS

                # Create new display name
                new_display_name = f"{prefix}_{new_name}_{timestamp}"

                # RENAME THE FOLDER/FILE ON DISK FIRST
                new_path = self._rename_file_on_disk(
                    old_path, new_name, source_type)
                if not new_path:
                    self.log("Rename cancelled: failed to rename on disk")
                    self._cancel_rename()
                    return

                # Update metadata with new values
                self.generations_metadata[gen_id]['display_name'] = new_display_name
                self.generations_metadata[gen_id]['runs_name'] = new_name
                self.generations_metadata[gen_id]['folder_path'] = new_path

                # If this generation is currently loaded, update the cached path
                if gen_id == self.current_generation_id and hasattr(self, 'pending_content'):
                    # Check if pending content references this path
                    if self.pending_content and self.pending_content.get('video_path') == old_path:
                        self.pending_content['video_path'] = new_path

                # Save to disk
                self._save_generations_metadata()

                # Update global cache key if this path was cached
                with self._global_cache_lock:
                    for key, cache_entry in list(self._global_cache.items()):
                        # Check if this cache entry references the old file
                        # We need to check the cache key generation logic
                        if old_path in str(key):
                            # Remove old cache entry
                            del self._global_cache[key]
                            self.log(
                                f"Removed old cache entry for renamed file")

                # If this is the current generation, update its display
                if gen_id == self.current_generation_id:
                    self.log(
                        f"Renamed current generation to: {new_display_name}")
                    self.log(
                        f"Folder/file renamed on disk: {os.path.basename(new_path)}")
                else:
                    self.log(f"Renamed generation to: {new_display_name}")
                    self.log(
                        f"Folder/file renamed on disk: {os.path.basename(new_path)}")

                # Clear editing state
                self.dropup_editing_index = -1
                self.dropup_edit_text = ""
                self.dropup_edit_cursor_pos = 0
                self.dropup_edit_initiated_by_keyboard = False
                self._mark_ui_dirty()
            else:
                self.log(f"Could not parse display name: {old_display_name}")
                self._cancel_rename()

            # Clear editing state
            self.dropup_editing_index = -1
            self.dropup_edit_text = ""
            self.dropup_edit_cursor_pos = 0
            self.dropup_edit_initiated_by_keyboard = False

        except Exception as e:
            self.log(f"ERROR in rename: {e}")
            import traceback
            traceback.print_exc()
            self._cancel_rename()

    def _cancel_rename(self):
        """Cancel the current rename operation"""
        self.dropup_editing_index = -1
        self.dropup_edit_text = ""
        self.dropup_edit_cursor_pos = 0
        self.dropup_edit_initiated_by_keyboard = False
        self._mark_ui_dirty()

    # -----------------------
    # Logo Screen Methods
    # -----------------------
    def _load_logo_screen(self, logo_filename, target_res):
        """Load and resize a logo screen from the logos directory"""
        try:
            logo_path = os.path.join(LOGOS_DIR, logo_filename)
            if not os.path.exists(logo_path):
                self.log(f"Logo file not found: {logo_path}")
                return None

            # Load image
            img = Image.open(logo_path)
            img = img.convert("RGB")

            # Resize to target resolution
            target_w, target_h = target_res
            img_resized = img.resize(
                (target_w, target_h), Image.Resampling.LANCZOS)

            # Convert to numpy array
            logo_array = np.array(img_resized, dtype=np.uint8)
            return logo_array

        except Exception as e:
            self.log(f"Error loading logo {logo_filename}:", e)
            return None

    def _display_logo_screen(self, screen, logo_filename):
        """Display a logo screen centered on the pygame screen"""
        if screen is None:
            return

        try:
            # Get screen dimensions
            screen_w, screen_h = screen.get_size()

            # Load logo
            logo_array = self._load_logo_screen(
                logo_filename, (screen_w, screen_h))
            if logo_array is None:
                # Fallback: black screen
                screen.fill((0, 0, 0))
                pygame.display.flip()
                return

            # Convert to pygame surface and display
            logo_surface = pygame.surfarray.make_surface(
                logo_array.swapaxes(0, 1))
            screen.blit(logo_surface, (0, 0))
            pygame.display.flip()

        except Exception as e:
            self.log(f"Error displaying logo {logo_filename}:", e)
            # Fallback: black screen
            screen.fill((0, 0, 0))
            pygame.display.flip()

    # -----------------------
    # Fullscreen Methods
    # -----------------------
    def _enter_fullscreen(self, screen):
        """Enter fullscreen mode"""
        if not self.fullscreen_mode:
            self.pre_fullscreen_fit_mode = self.viewer_fit_mode
            self.pre_fullscreen_window_size = screen.get_size()
            self.fullscreen_mode = True

            # Clear UI cache when entering fullscreen
            self._mark_ui_dirty()
            self._mark_frame_cache_dirty()  # Frame cache needs update too

            # Switch to fullscreen
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            pygame.display.set_caption(
                "Preview Video Monitor Pro - Fullscreen [ESC] Exit | [SPACE] Play/Pause")
            return screen

    def _exit_fullscreen(self, screen):
        """Exit fullscreen mode"""
        if self.fullscreen_mode:
            self.fullscreen_mode = False
            self.viewer_fit_mode = self.pre_fullscreen_fit_mode

            # Clear UI cache when exiting fullscreen
            self._mark_ui_dirty()
            self._mark_frame_cache_dirty()  # Frame cache needs update too

            # Restore windowed mode
            win_w, win_h = self.pre_fullscreen_window_size
            screen = pygame.display.set_mode(
                (win_w, win_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame.display.set_caption(
                "Preview Video Monitor Pro v5.2 - [q] Close | [SPACE] Play/Pause | [←→] Step | [ENTER] Edit/Confirm | [↑↓] Navigate | [END] Load | [1-5] Fit Modes | [i/o/p] In/Out Marks | Click timeline to scrub | Right-click generation to rename")
            return screen

    # -----------------------
    # Generation Tracking System
    # -----------------------
    def _load_generations_metadata(self):
        """Load generations metadata from JSON file"""
        try:
            if os.path.exists(RUNS_METADATA_FILE):
                with open(RUNS_METADATA_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.log("Error loading generations metadata:", e)
        return {}

    def _save_generations_metadata(self):
        """Save generations metadata to JSON file"""
        try:
            with open(RUNS_METADATA_FILE, 'w') as f:
                json.dump(self.generations_metadata, f, indent=2)
        except Exception as e:
            self.log("Error saving generations metadata:", e)

    def _check_and_migrate_cache(self):
        """Check for old MP4/PNG cache format and clear it"""
        try:
            import shutil
            
            # Look for old .mp4 or .png files (not in folders)
            old_files = []
            for item in os.listdir(RUNS_CACHE_DIR):
                if item == "temp_workflow":
                    continue
                item_path = os.path.join(RUNS_CACHE_DIR, item)
                if os.path.isfile(item_path) and item.endswith(('.mp4', '.png')):
                    old_files.append(item)
            
            if old_files:
                self.log("="*70)
                self.log("CACHE FORMAT UPGRADE DETECTED!")
                self.log(f"Found {len(old_files)} old MP4/PNG files")
                self.log("Old cache will be cleared - new format uses PNG image sequences")
                self.log("Please re-run workflows to rebuild cache with new format")
                self.log("="*70)
                
                # Clear old files
                for file in old_files:
                    try:
                        file_path = os.path.join(RUNS_CACHE_DIR, file)
                        os.remove(file_path)
                        self.log(f"Removed old file: {file}")
                    except Exception as e:
                        self.log(f"Failed to remove {file}: {e}")
                
                # Clear old metadata (will be rebuilt with new generations)
                self.generations_metadata.clear()
                self._save_generations_metadata()
                
                self.log("Old cache cleared successfully")
                self.log("="*70)
        
        except Exception as e:
            self.log(f"Error during cache migration: {e}")

    def _register_new_generation(self, file_path, runs_name, source_type, frame_count=0):
        """Register a new generation in the metadata system with source prefix"""
        try:
            generation_id = str(uuid.uuid4())
            timestamp = time.time()
            time_str = time.strftime("%H:%M:%S")  # Added seconds
            time_str_safe = time_str.replace(":", "-")  # Safe for filenames

            # Add prefix based on source type
            prefix = "v" if source_type == "video" else "i"
            display_name = f"{prefix}_{runs_name}_{time_str}"
            
            # Get folder size for duplicate detection (NEW: folder instead of file)
            total_size = self._get_folder_size(file_path) if os.path.exists(file_path) else 0

            self.generations_metadata[generation_id] = {
                'folder_path': file_path,  # NEW: folder_path instead of file_path
                'display_name': display_name,
                'timestamp': timestamp,
                'runs_name': runs_name,
                'source_type': source_type,
                'total_size': total_size,  # NEW: total_size instead of file_size
                'frame_count': frame_count,
                'format': 'jpg'  # NEW: JPG for speed
            }

            # Storage management is user's responsibility - no auto-cleanup

            # Save metadata
            self._save_generations_metadata()

            self.log(f"Registered new generation: {display_name}")
            self.log(f"Folder saved as: {os.path.basename(file_path)}")
            return generation_id

        except Exception as e:
            self.log("Error registering new generation:", e)
            return None

    def _find_duplicate_generation(self, file_size, frame_count):
        """
        Check if a generation with identical total_size and frame_count already exists.
        Returns (gen_id, metadata) if duplicate found, None otherwise.
        """
        try:
            for gen_id, metadata in self.generations_metadata.items():
                # NEW: Support both old (file_size) and new (total_size) formats
                existing_size = metadata.get('total_size', metadata.get('file_size', -1))
                existing_frames = metadata.get('frame_count', -1)
                
                if existing_size == file_size and existing_frames == frame_count:
                    # Found a duplicate!
                    return gen_id, metadata
            
            # No duplicate found
            return None
            
        except Exception as e:
            self.log(f"Error checking for duplicates: {e}")
            return None

    def _delete_generation(self, gen_id, metadata):
        """Delete a specific generation (folder + metadata)"""
        try:
            display_name = metadata.get('display_name', 'Unknown')
            # NEW: Support both old (file_path) and new (folder_path) formats
            folder_path = metadata.get('folder_path', metadata.get('file_path'))
            
            # Delete the folder or file
            if folder_path and os.path.exists(folder_path):
                if os.path.isdir(folder_path):
                    # New format: Delete entire folder
                    import shutil
                    shutil.rmtree(folder_path)
                    self.log(f"Deleted folder: {os.path.basename(folder_path)}")
                else:
                    # Legacy format: Delete single file
                    os.remove(folder_path)
                    self.log(f"Deleted legacy file: {os.path.basename(folder_path)}")
            
            # Remove from metadata
            if gen_id in self.generations_metadata:
                del self.generations_metadata[gen_id]
                self._save_generations_metadata()
                self.log(f"Deleted generation: {display_name}")
            
            # If this was the current generation, switch to another one
            if self.current_generation_id == gen_id:
                sorted_generations = self._get_sorted_generations()
                
                if sorted_generations:
                    # Find the previous generation (chronologically older)
                    current_idx = -1
                    for i, (check_id, _) in enumerate(sorted_generations):
                        if check_id == gen_id:
                            current_idx = i
                            break
                    
                    # Get the next generation in the list (which is older in time)
                    if current_idx >= 0 and current_idx + 1 < len(sorted_generations):
                        # Switch to the next (older) generation
                        next_gen_id, next_metadata = sorted_generations[current_idx + 1]
                        next_path = next_metadata.get('folder_path', next_metadata.get('file_path'))
                        if next_path and os.path.exists(next_path):
                            self.current_generation_id = next_gen_id
                            self.log(f"Switched to: {next_metadata.get('display_name')}")
                            # Load in background
                            Thread(target=self._load_new_video, args=(next_path,), daemon=True).start()
                    elif len(sorted_generations) > 0:
                        # Was the last in list, switch to first (newest)
                        next_gen_id, next_metadata = sorted_generations[0]
                        next_path = next_metadata.get('folder_path', next_metadata.get('file_path'))
                        if next_path and os.path.exists(next_path):
                            self.current_generation_id = next_gen_id
                            self.log(f"Switched to: {next_metadata.get('display_name')}")
                            Thread(target=self._load_new_video, args=(next_path,), daemon=True).start()
                else:
                    # No more generations - clear display
                    self.current_generation_id = None
                    self.cached_scaled_frames = None
                    self.cached_original_frames = None
                    self.cached_surfaces = None
                    self.log("No more generations - display cleared")
                    self._mark_frame_cache_dirty()
            
            self._mark_ui_dirty()
            return True
            
        except Exception as e:
            self.log(f"Error deleting generation: {e}")
            traceback.print_exc()
            return False

    def _get_sorted_generations(self):
        """Get generations sorted by timestamp (newest first)"""
        try:
            sorted_gens = sorted(
                self.generations_metadata.items(),
                key=lambda x: x[1].get('timestamp', 0),
                reverse=True
            )
            return sorted_gens
        except Exception as e:
            self.log("Error sorting generations:", e)
            return []

    def _clear_all_generations(self):
        """Clear all generations and their files"""
        try:
            # Remove all generation files
            for gen_id, metadata in list(self.generations_metadata.items()):
                file_path = metadata.get('file_path')
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)

            # Clear metadata
            self.generations_metadata.clear()
            self._save_generations_metadata()

            self.log("Cleared all generations")
            return True
        except Exception as e:
            self.log("Error clearing generations:", e)
            return False

    @classmethod
    def get_monitors(cls):
        monitors = []
        if SCREENINFO_AVAILABLE:
            try:
                ms = get_monitors()
                for i, m in enumerate(ms):
                    monitors.append(f"Monitor {i} ({m.width}x{m.height})")
            except Exception:
                pass
        if PYGAME_AVAILABLE and not monitors:
            try:
                d = pygame.display.Info()
                monitors.append(f"Monitor 0 ({d.current_w}x{d.current_h})")
            except Exception:
                pass
        for i in range(len(monitors), 6):
            monitors.append(f"Monitor {i} (unknown resolution)")
        return monitors

    # -----------------------
    # GPU-accelerated helpers
    # -----------------------
    def _gpu_resize_and_center(self, frame_rgb_uint8, target_res, fit_mode):
        """
        Try to use GPU to resize and center frame to target_res.
        Returns uint8 numpy array (target_h, target_w, 3).
        Falls back to CPU implementation if GPU backends are unavailable or an error occurs.
        """
        # Quick sanity checks
        if frame_rgb_uint8 is None:
            return None
        try:
            target_w, target_h = target_res
            frame_h, frame_w = frame_rgb_uint8.shape[:2]

            # Fast path - if size matches
            if frame_w == target_w and frame_h == target_h and fit_mode != "1:1":
                # Already correct size
                canvas = frame_rgb_uint8.copy()
                # If target canvas expected orientation: ensure (target_h, target_w, 3)
                if canvas.shape[0] != target_h or canvas.shape[1] != target_w:
                    canvas = np.ascontiguousarray(cv2.resize(canvas, (target_w, target_h), interpolation=cv2.INTER_LINEAR)) if CV2_AVAILABLE else np.array(
                        Image.fromarray(canvas).resize((target_w, target_h), resample=Image.BILINEAR))
                return canvas

            if self.cv2_cuda:
                # Use OpenCV CUDA
                try:
                    # Create GpuMat and upload
                    gpu_src = cv2.cuda_GpuMat()
                    # OpenCV expects BGR in many CUDA ops — but our frame is RGB.
                    # Convert to BGR on CPU, then upload — cv2.cvtColor has GPU version too,
                    # but converting on CPU is cheap relative to resize for moderate sizes.
                    bgr = cv2.cvtColor(frame_rgb_uint8, cv2.COLOR_RGB2BGR)
                    gpu_src.upload(bgr)

                    # Compute scaling based on fit_mode
                    if fit_mode == "1:1":
                        new_w, new_h = frame_w, frame_h
                    elif fit_mode == "width":
                        scale = target_w / frame_w
                        new_w = max(1, int(frame_w * scale))
                        new_h = max(1, int(frame_h * scale))
                    elif fit_mode == "height":
                        scale = target_h / frame_h
                        new_w = max(1, int(frame_w * scale))
                        new_h = max(1, int(frame_h * scale))
                    else:
                        scale = min(target_w / frame_w, target_h / frame_h)
                        new_w = max(1, int(frame_w * scale))
                        new_h = max(1, int(frame_h * scale))

                    # GPU resize
                    gpu_resized = cv2.cuda.resize(
                        gpu_src, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

                    # Download back to CPU
                    resized_bgr = gpu_resized.download()
                    # Convert back to RGB
                    resized = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

                    # Compose onto canvas
                    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    y_offset = max(0, (target_h - new_h) // 2)
                    x_offset = max(0, (target_w - new_w) // 2)
                    copy_h = min(new_h, target_h - y_offset)
                    copy_w = min(new_w, target_w - x_offset)
                    if copy_h > 0 and copy_w > 0:
                        canvas[y_offset:y_offset+copy_h, x_offset:x_offset +
                               copy_w] = resized[:copy_h, :copy_w]
                    return canvas
                except Exception as e:
                    self.log(
                        "cv2.cuda path failed, falling back to torch/cpu:", e)
                    # fall through to other options

            if self.torch_cuda and TORCH_AVAILABLE:
                # Use PyTorch on GPU to resize with bilinear interpolation
                try:
                    # Convert numpy uint8 [H,W,3] -> float32 [1,3,H,W] scaled 0..1
                    t = torch.from_numpy(frame_rgb_uint8).permute(
                        2, 0, 1).unsqueeze(0).float() / 255.0  # [1,3,H,W]
                    t = t.to('cuda', non_blocking=True)
                    # Compute base dims based on fit_mode similar to CPU path
                    if fit_mode == "1:1":
                        base_w, base_h = frame_w, frame_h
                    elif fit_mode == "width":
                        scale = target_w / frame_w
                        base_w = max(1, int(frame_w * scale))
                        base_h = max(1, int(frame_h * scale))
                    elif fit_mode == "height":
                        scale = target_h / frame_h
                        base_w = max(1, int(frame_w * scale))
                        base_h = max(1, int(frame_h * scale))
                    else:
                        scale = min(target_w / frame_w, target_h / frame_h)
                        base_w = max(1, int(frame_w * scale))
                        base_h = max(1, int(frame_h * scale))

                    # interpolate expects [N,C,H,W], size=(H_out, W_out)
                    t_resized = torch.nn.functional.interpolate(
                        t, size=(base_h, base_w), mode='bilinear', align_corners=False)
                    # back to cpu uint8
                    t_cpu = (t_resized.clamp(0, 1).mul(
                        255.0)).to('cpu', torch.uint8)
                    arr = t_cpu.squeeze(0).permute(1, 2, 0).numpy()  # H,W,3
                    # Compose onto canvas
                    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    y_offset = max(0, (target_h - base_h) // 2)
                    x_offset = max(0, (target_w - base_w) // 2)
                    copy_h = min(base_h, target_h - y_offset)
                    copy_w = min(base_w, target_w - x_offset)
                    if copy_h > 0 and copy_w > 0:
                        canvas[y_offset:y_offset+copy_h,
                               x_offset:x_offset+copy_w] = arr[:copy_h, :copy_w]
                    return canvas
                except Exception as e:
                    self.log(
                        "torch.cuda resize path failed, falling back to CPU:", e)
                    # fall through to CPU

            # If no GPU backend worked or not available: fallback to CPU implementation below
            return self._scale_and_center_for_target_cpu(frame_rgb_uint8, target_res, fit_mode)
        except Exception as e:
            self.log("Error in _gpu_resize_and_center:", e)
            return self._scale_and_center_for_target_cpu(frame_rgb_uint8, target_res, fit_mode)

    def _scale_and_center_for_target_cpu(self, frame_rgb_uint8, target_res, fit_mode):
        """
        Original CPU resizing + centering. Kept as separate function so GPU branch can call it as fallback.
        """
        target_w, target_h = target_res
        frame_h, frame_w = frame_rgb_uint8.shape[:2]

        if fit_mode == "1:1":
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            new_w, new_h = frame_w, frame_h
            copy_w = min(new_w, target_w)
            copy_h = min(new_h, target_h)
            x_offset = max(0, (target_w - copy_w) // 2)
            y_offset = max(0, (target_h - copy_h) // 2)
            canvas[y_offset:y_offset+copy_h, x_offset:x_offset +
                   copy_w] = frame_rgb_uint8[:copy_h, :copy_w]
            return canvas

        if fit_mode == "width":
            scale = target_w / frame_w
        elif fit_mode == "height":
            scale = target_h / frame_h
        else:
            scale = min(target_w / frame_w, target_h / frame_h)

        new_w = max(1, int(frame_w * scale))
        new_h = max(1, int(frame_h * scale))
        # Use cv2 if available for faster resize; fallback to PIL if not
        if CV2_AVAILABLE:
            resized = cv2.resize(
                frame_rgb_uint8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            # PIL fallback
            img = Image.fromarray(frame_rgb_uint8)
            resized = np.array(img.resize(
                (new_w, new_h), resample=Image.BILINEAR))
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        y_offset = max(0, y_offset)
        x_offset = max(0, x_offset)
        copy_h = min(new_h, target_h - y_offset)
        copy_w = min(new_w, target_w - x_offset)
        if copy_h > 0 and copy_w > 0:
            canvas[y_offset:y_offset+copy_h, x_offset:x_offset +
                   copy_w] = resized[:copy_h, :copy_w]
        return canvas

    # -----------------------
    # Helpers: reading frames
    # -----------------------
    def _read_video_frames_from_path(self, video_path):
        """Reads frames from disk; returns numpy array [T,H,W,3] uint8 and fps."""
        if not CV2_AVAILABLE:
            self.log("OpenCV not available")
            return None, 24.0, (0, 0)
        
        # NEW: Detect PNG files and use PNG reader
        if video_path.endswith('.png'):
            return self._read_png_frame(video_path)
        
        # Original MP4/video reading logic
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, 24.0, (0, 0)
        frames = []
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        # Get original resolution from first frame
        ret, first_frame = cap.read()
        if ret:
            # (width, height)
            original_res = (first_frame.shape[1], first_frame.shape[0])
            rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        else:
            original_res = (0, 0)

        try:
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
        except Exception as e:
            self.log("Error reading frames:", e)
        finally:
            cap.release()
        if len(frames) == 0:
            return None, float(video_fps), original_res
        return np.stack(frames, axis=0), float(video_fps), original_res

    def _read_video_frames_original(self, video_input):
        """
        Read original-resolution frames for dynamic path from VideoFromFile-like, path or tensor.
        Returns (frames_numpy, fps, original_resolution)
        """
        if video_input is None:
            return None, 24.0, (0, 0)
        try:
            # VideoFromFile-like
            if hasattr(video_input, '__class__') and 'VideoFromFile' in str(type(video_input)):
                try:
                    s = video_input.get_stream_source()
                    if isinstance(s, str):
                        return self._read_video_frames_from_path(s)
                    else:
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                            tmp.write(s.getvalue())
                            tmp_path = tmp.name
                        try:
                            return self._read_video_frames_from_path(tmp_path)
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass
                except Exception:
                    pass

            # direct file path
            if isinstance(video_input, str) and os.path.exists(video_input):
                return self._read_video_frames_from_path(video_input)

            # torch Tensor
            if TORCH_AVAILABLE and isinstance(video_input, torch.Tensor):
                arr = video_input.detach().cpu().numpy()
                # allow [T,H,W,3] or [B,T,H,W,3]
                if arr.ndim == 5:
                    arr = arr[0]
                if arr.ndim == 4 and arr.shape[-1] == 3:
                    if np.issubdtype(arr.dtype, np.floating):
                        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                    # Get resolution from first frame
                    if len(arr) > 0:
                        original_res = (arr[0].shape[1], arr[0].shape[0])
                    else:
                        original_res = (0, 0)
                    return arr, 24.0, original_res
        except Exception as e:
            self.log("Error reading original frames:", e)
        return None, 24.0, (0, 0)

    def _read_video_frames(self, video_input, target_res):
        """
        Universal reader that returns scaled frames (to target_res) and fps.
        Accepts: path str, VideoFromFile-like, VIDEO object (Comfy), torch tensor, numpy array.
        Implementation: read original frames, then scale each to target_res using GPU if available.
        """
        # First get original frames
        orig, fps, original_res = self._read_video_frames_original(video_input)
        if orig is None:
            return None, fps, original_res
        try:
            # scale each frame to target_res using current fit mode (GPU if available)
            scaled_list = []
            for f in orig:
                # GPU-accelerated path inside _gpu_resize_and_center will fallback to CPU if needed
                scaled = self._gpu_resize_and_center(f, target_res, self.viewer_fit_mode) if self.gpu_available else self._scale_and_center_for_target_cpu(
                    f, target_res, self.viewer_fit_mode)
                scaled_list.append(scaled)
            return np.stack(scaled_list, axis=0), float(fps), original_res
        except Exception as e:
            self.log("Error scaling frames in _read_video_frames:", e)
            return None, float(fps), original_res

    # -----------------------
    # Helpers: images -> mp4 (lightweight cache)
    # -----------------------
    def _normalize_frames_from_images(self, images_input):
        """Normalize various image inputs into list of uint8 RGB numpy arrays."""

        frames = []
        original_res = (0, 0)
        try:
            if TORCH_AVAILABLE and isinstance(images_input, torch.Tensor):
                arr = images_input.detach().cpu().numpy()
                if arr.ndim == 3:
                    arr = arr[None, ...]
                if arr.ndim != 4:
                    return None, original_res
                if np.issubdtype(arr.dtype, np.floating):
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                for f in arr:
                    if f.shape[-1] != 3:
                        return None, original_res
                    frames.append(f.copy())
                if len(frames) > 0:
                    original_res = (frames[0].shape[1], frames[0].shape[0])
                return frames, original_res

            if isinstance(images_input, np.ndarray):
                arr = images_input
                if arr.ndim == 3:
                    arr = arr[None, ...]
                if arr.ndim != 4:
                    return None, original_res
                if np.issubdtype(arr.dtype, np.floating):
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                for f in arr:
                    frames.append(f.copy())
                if len(frames) > 0:
                    original_res = (frames[0].shape[1], frames[0].shape[0])
                return frames, original_res

            if isinstance(images_input, Image.Image):
                frame = np.array(images_input.convert("RGB"), dtype=np.uint8)
                frames.append(frame)
                original_res = (frame.shape[1], frame.shape[0])
                return frames, original_res

            if isinstance(images_input, (list, tuple)):
                for item in images_input:
                    if isinstance(item, Image.Image):
                        frame = np.array(item.convert("RGB"), dtype=np.uint8)
                        frames.append(frame)
                        if original_res == (0, 0):
                            original_res = (frame.shape[1], frame.shape[0])
                    elif isinstance(item, np.ndarray):
                        a = item
                        if a.ndim == 3 and a.shape[-1] == 3:
                            if np.issubdtype(a.dtype, np.floating):
                                a = (a * 255.0).clip(0, 255).astype(np.uint8)
                            frames.append(a.copy())
                            if original_res == (0, 0):
                                original_res = (a.shape[1], a.shape[0])
                        else:
                            return None, original_res
                    elif TORCH_AVAILABLE and isinstance(item, torch.Tensor):
                        a = item.detach().cpu().numpy()
                        if a.ndim == 3 and a.shape[-1] == 3:
                            if np.issubdtype(a.dtype, np.floating):
                                a = (a * 255.0).clip(0, 255).astype(np.uint8)
                            frames.append(a.copy())
                            if original_res == (0, 0):
                                original_res = (a.shape[1], a.shape[0])
                        else:
                            return None, original_res
                    else:
                        return None, original_res
                return frames, original_res
        except Exception as e:
            self.log("Exception normalizing frames:", e)
            return None, original_res
        return None, original_res

    def _write_mp4_from_frames(self, frames_uint8, fps, out_path):
        """Write frames list (RGB uint8) to mp4 on disk using cv2 (mp4v)."""
        # FIXED: Proper numpy array check
        if frames_uint8 is None or (hasattr(frames_uint8, '__len__') and len(frames_uint8) == 0):
            self.log("No frames to write")
            return False
        if not CV2_AVAILABLE:
            self.log("OpenCV not available; cannot write mp4")
            return False

        parent = os.path.dirname(out_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        h, w = frames_uint8[0].shape[0:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp_path = tmp.name

            writer = cv2.VideoWriter(
                tmp_path, fourcc, float(fps), (w, h), True)
            if not writer.isOpened():
                self.log("VideoWriter failed to open", tmp_path)
                return False

            for f in frames_uint8:
                if f.shape[0] != h or f.shape[1] != w:
                    rf = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    rf = f
                bgr = cv2.cvtColor(rf, cv2.COLOR_RGB2BGR)
                writer.write(bgr)

            writer.release()
            writer = None

            # Check if file was written successfully
            # Minimum size threshold
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                shutil.move(tmp_path, out_path)
                self.log(
                    f"Successfully wrote MP4: {out_path} ({os.path.getsize(out_path) / 1024:.2f} KB)")
                return True
            else:
                self.log(
                    f"MP4 write failed: empty or too small file {tmp_path}")
                return False

        except Exception as e:
            self.log("Error writing mp4:", e)
            traceback.print_exc()
            return False
        finally:
            if writer is not None:
                writer.release()
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # -----------------------
    # PNG single-frame support (NEW)
    # -----------------------
    def _write_png_from_frame(self, frame_uint8, out_path):
        """Write single frame to PNG file for single-frame inputs (better quality than 1-frame MP4)"""
        if frame_uint8 is None:
            self.log("No frame to write to PNG")
            return False
        if not CV2_AVAILABLE:
            self.log("OpenCV not available; cannot write PNG")
            return False

        parent = os.path.dirname(out_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        try:
            # Convert RGB to BGR for cv2
            bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            
            # Write PNG with compression level 6 (balanced quality/speed)
            success = cv2.imwrite(out_path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            
            if success and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                self.log(f"Successfully wrote PNG: {out_path} ({os.path.getsize(out_path) / 1024:.2f} KB)")
                return True
            else:
                self.log(f"PNG write failed: {out_path}")
                return False

        except Exception as e:
            self.log("Error writing PNG:", e)
            traceback.print_exc()
            return False

    def _read_png_frame(self, file_path):
        """Read PNG file and return as single-frame array (for single-frame generations)"""
        if not CV2_AVAILABLE:
            self.log("OpenCV not available; cannot read PNG")
            return None, 24.0, (0, 0)

        try:
            # Read PNG
            bgr = cv2.imread(file_path, cv2.IMREAD_COLOR)
            if bgr is None:
                self.log(f"Failed to read PNG: {file_path}")
                return None, 24.0, (0, 0)
            
            # Convert BGR to RGB
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            
            # Get resolution
            h, w = rgb.shape[:2]
            original_res = (w, h)
            
            # Return as single-frame array, fps=24.0 (convention for stills)
            frames = [rgb]
            
            self.log(f"Successfully read PNG: {file_path} ({w}x{h})")
            return frames, 24.0, original_res

        except Exception as e:
            self.log("Error reading PNG:", e)
            traceback.print_exc()
            return None, 24.0, (0, 0)

    def _write_image_sequence(self, frames, folder_path, format='png'):
        """Write frames as PNG/JPG image sequence to folder
        
        Args:
            frames: List of RGB frames (numpy arrays)
            folder_path: Folder to write sequence to
            format: 'png' or 'jpg'
        
        Returns:
            bool: Success status
        """
        if not CV2_AVAILABLE:
            self.log("OpenCV not available; cannot write image sequence")
            return False
        
        try:
            # Ensure folder exists
            os.makedirs(folder_path, exist_ok=True)
            
            # Determine format settings
            if format == 'png':
                params = [cv2.IMWRITE_PNG_COMPRESSION, 6]  # 0-9, 6=balanced
                ext = '.png'
            else:  # jpg
                params = [cv2.IMWRITE_JPEG_QUALITY, 95]  # 0-100, 95=near-lossless
                ext = '.jpg'
            
            # Write each frame
            total_size = 0
            failed_frames = 0
            
            for i, frame in enumerate(frames):
                # Frame numbering: frame_0001.png, frame_0002.png, etc.
                filename = f"frame_{i+1:04d}{ext}"
                filepath = os.path.join(folder_path, filename)
                
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Write frame
                success = cv2.imwrite(filepath, frame_bgr, params)
                if not success:
                    self.log(f"Failed to write frame {i+1}")
                    failed_frames += 1
                else:
                    # Track size
                    total_size += os.path.getsize(filepath)
            
            if failed_frames > 0:
                self.log(f"Warning: {failed_frames} frames failed to write")
            
            size_mb = total_size / (1024 * 1024)
            self.log(f"Wrote {len(frames)} frames to {os.path.basename(folder_path)}")
            self.log(f"Total size: {size_mb:.2f} MB ({format.upper()} format)")
            
            return failed_frames == 0
        
        except Exception as e:
            self.log(f"Error writing image sequence: {e}")
            traceback.print_exc()
            return False

    def _read_image_sequence_original(self, folder_path):
        """Read original frames from PNG/JPG image sequence folder
        
        Args:
            folder_path: Folder containing frame_XXXX.png/jpg files
        
        Returns:
            tuple: (frames_list, fps, resolution) or (None, None, None)
        """
        if not CV2_AVAILABLE:
            self.log("OpenCV not available; cannot read image sequence")
            return None, None, None
        
        try:
            import glob
            
            # Find PNG frames
            pattern = os.path.join(folder_path, "frame_*.png")
            files = sorted(glob.glob(pattern))
            
            if not files:
                # Try JPG frames
                pattern = os.path.join(folder_path, "frame_*.jpg")
                files = sorted(glob.glob(pattern))
            
            if not files:
                self.log(f"No image sequence found in {folder_path}")
                return None, None, None
            
            # Read first frame for resolution
            first_frame = cv2.imread(files[0], cv2.IMREAD_COLOR)
            if first_frame is None:
                self.log(f"Failed to read first frame: {files[0]}")
                return None, None, None
            
            height, width = first_frame.shape[:2]
            resolution = (width, height)
            
            # Read all frames to RAM (like we do with MP4)
            frames = []
            failed_frames = 0
            
            for i, file in enumerate(files):
                frame = cv2.imread(file, cv2.IMREAD_COLOR)
                if frame is None:
                    self.log(f"Failed to read frame {i+1}: {file}")
                    failed_frames += 1
                    continue
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            
            if not frames:
                self.log(f"No frames could be read from {folder_path}")
                return None, None, None
            
            if failed_frames > 0:
                self.log(f"Warning: {failed_frames} frames failed to read")
            
            # NO FPS metadata in image sequences - return None
            # User controls playback FPS via UI
            fps = None
            
            self.log(f"Loaded {len(frames)} frames from {os.path.basename(folder_path)} ({width}x{height})")
            return frames, fps, resolution
        
        except Exception as e:
            self.log(f"Error reading image sequence: {e}")
            traceback.print_exc()
            return None, None, None

    def _read_image_sequence_scaled(self, folder_path, target_res):
        """Read and scale frames from image sequence
        
        Args:
            folder_path: Folder containing image sequence
            target_res: Target resolution tuple (width, height)
        
        Returns:
            tuple: (scaled_frames, fps, original_resolution)
        """
        # Read original frames
        orig_frames, fps, orig_res = self._read_image_sequence_original(folder_path)
        
        if orig_frames is None:
            return None, None, None
        
        # Scale frames
        scaled_frames = []
        for frame in orig_frames:
            scaled = self._scale_and_center_for_target(frame, target_res, "fit")
            scaled_frames.append(scaled)
        
        return scaled_frames, fps, orig_res

    def _get_folder_size(self, folder_path):
        """Calculate total size of all files in folder
        
        Args:
            folder_path: Folder to measure
        
        Returns:
            int: Total size in bytes
        """
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            self.log(f"Error calculating folder size: {e}")
        
        return total_size

    # -----------------------
    # Scale helper (reused)
    # -----------------------
    def _scale_and_center_for_target(self, frame_rgb_uint8, target_res, fit_mode):
        """
        Resize and center frame_rgb_uint8 (H,W,3 uint8) to target_res according to fit_mode.
        This wrapper will prefer a GPU-accelerated implementation when a GPU is present
        and the appropriate backend is available. Otherwise falls back to CPU function.
        Returns uint8 (target_h, target_w, 3)
        """
        if self.gpu_available:
            try:
                return self._gpu_resize_and_center(frame_rgb_uint8, target_res, fit_mode)
            except Exception as e:
                # If GPU fails for any reason, log and fallback to CPU
                self.log("GPU scaling failed, falling back to CPU:", e)
                return self._scale_and_center_for_target_cpu(frame_rgb_uint8, target_res, fit_mode)
        else:
            return self._scale_and_center_for_target_cpu(frame_rgb_uint8, target_res, fit_mode)

    # -----------------------
    # Fast surface creation (precompute for fast blit)
    # -----------------------
    def _create_surfaces_from_frames(self):
        """Create pygame.Surface list from cached_scaled_frames for the fast path."""
        if self.cached_scaled_frames is None:
            return
        if self.cached_surfaces is not None:
            return
        if not PYGAME_AVAILABLE:
            return
        surfaces = []
        try:
            for f in self.cached_scaled_frames:
                # Add shape validation
                if f.ndim != 3 or f.shape[2] != 3:
                    continue
                surf = pygame.surfarray.make_surface(f.swapaxes(0, 1))
                surfaces.append(surf)
            self.cached_surfaces = surfaces
        except Exception as e:
            self.log("Error creating surfaces:", e)
            self.cached_surfaces = None

    def _load_new_video(self, path):
        self.log("Loading generation:", path)
        with self.cache_lock:
            # NEW: Detect folder (image sequence) vs file (legacy)
            if os.path.isdir(path):
                # Image sequence folder
                self.log("Loading image sequence from folder...")
                orig_frames, fps, original_res = self._read_image_sequence_original(path)
                
                if orig_frames is None:
                    self.log("Failed to load image sequence from", path)
                    return
                
                # Scale frames for display
                frames_scaled = []
                for frame in orig_frames:
                    scaled = self._scale_and_center_for_target(frame, self.target_res, "fit")
                    frames_scaled.append(scaled)
            
            else:
                # Legacy file (MP4 or PNG) - for backward compatibility
                self.log("Loading legacy file format...")
                if path.endswith('.png'):
                    # Load PNG as single-frame
                    orig_frames, fps, original_res = self._read_png_frame(path)
                else:
                    # Load MP4 (video)
                    orig_frames, fps, original_res = self._read_video_frames_original(path)
                
                if orig_frames is None:
                    self.log("Failed to load original frames from", path)
                    return
                
                # Scale frames for display
                if path.endswith('.png'):
                    # PNG: scale single frame
                    frames_scaled = []
                    for frame in orig_frames:
                        scaled = self._scale_and_center_for_target(frame, self.target_res, "fit")
                        frames_scaled.append(scaled)
                else:
                    # MP4: use existing video reader
                    frames_scaled, _, _ = self._read_video_frames(path, self.target_res)
                    if frames_scaled is None:
                        frames_scaled = orig_frames

            self.cached_original_frames = orig_frames
            self.cached_scaled_frames = frames_scaled
            self.cached_surfaces = None
            self.current_video_fps = fps
            self.original_resolution = original_res
            
            # Set user FPS from video FPS (with sanity check)
            # For image sequences, fps will be None, so default to 24.0
            # User can still override via FPS dropup
            if fps and 5.0 <= fps <= 200.0:
                self.user_fps = fps
                self.log(f"FPS set to {fps} from video metadata")
            else:
                self.user_fps = 24.0
                if fps is None:
                    self.log(f"No FPS metadata (image sequence), defaulting to 24fps")
                else:
                    self.log(f"Video FPS unreliable ({fps}), defaulting to 24fps")
            
            self.viewer_zoom = 1.0
            self.viewer_offset = [0, 0]
            self.viewer_paused = True
            self.force_counter_update = True
            # Initialize in/out points for the new media
            total_frames = len(orig_frames)
            self.original_in_point = 0
            self.original_out_point = total_frames - 1 if total_frames > 0 else 0
            self.user_in_point = 0
            self.user_out_point = total_frames - 1 if total_frames > 0 else 0
            self.user_marks_active = False
            # FIX: Reset current_frame to 0 and ensure it's within bounds
            self.current_frame = 0
            self.direction = 1
            # Reset showing_cache_clear_screen flag when loading new content
            self.showing_cache_clear_screen = False
            self.cache_clear_start_time = 0  # NEW: Reset timer
            self.showing_wait_screen = False  # CRITICAL FIX: Reset wait screen flag
            # Reset counter frame
            self.counter_update_frame = 0

            # Clear ALL caches for new content
            self._mark_ui_dirty()
            self._mark_frame_cache_dirty()
            # Frame surface cache cleared in _mark_frame_cache_dirty()
            self._ui_cache.clear()

            # Clear wipe comparison cache when loading new video
            self.wipe_comparison_frames = None
            self.wipe_comparison_frame_idx = -1
            # NEW: Reset video display rectangle
            self.current_video_rect = None
            # NEW: Reset comparison video size
            self.comparison_video_size = (0, 0)

            self.log("Loaded new generation with", len(orig_frames), "frames")

    # -----------------------
    # NEW: FRAME SURFACE CACHING SYSTEM WITH COLOR SPACE SUPPORT - UPDATED WITH VIDEO RECT TRACKING
    # -----------------------
    def _get_or_create_frame_surface(self, screen, current_frame, total_frames):
        """Get cached frame surface or create new one with thread-safe locking and color space transform"""
        if screen is None or total_frames <= 0 or self.cached_original_frames is None:
            return None, (0, 0, 0, 0)

        # Ensure frame is within bounds
        if current_frame >= total_frames:
            current_frame = 0
        if current_frame < 0:
            current_frame = 0

        try:
            vw, vh = screen.get_size()
            th = self.toolbar_height if not self.fullscreen_mode else 0
            
            # =========== USE PRE-TRANSFORMED FRAMES FOR SPEED! ===========
            # Instead of transforming every frame during playback,
            # use the pre-transformed cache for instant access
            # =============================================================
            if (self.cached_color_transformed_frames is not None and
                self.cached_color_space == self.selected_color_space and
                current_frame < len(self.cached_color_transformed_frames)):
                # Use pre-transformed frame (INSTANT!) ✅
                f_transformed = self.cached_color_transformed_frames[current_frame]
            else:
                # Fallback: transform on-the-fly (SLOW but works)
                f = self.cached_original_frames[current_frame]
                f_transformed = self._apply_color_space_transform(f)
                if f_transformed is None:
                    f_transformed = f
            # =============================================================

            fh, fw = f_transformed.shape[0:2]  # Note: shape is (height, width, channels)

            # Calculate frame state hash (NOW INCLUDES COLOR SPACE)
            current_state_hash = self._calculate_frame_state_hash(
                current_frame, (vw, vh))

            # Check if we can use cached surface (MULTI-FRAME CACHE!)
            with self._frame_surface_lock:
                if not self._frame_cache_dirty and current_state_hash in self._frame_surface_cache:
                    # Cache hit! Return cached surface (no pygame.surfarray.make_surface!)
                    cached_surf, cached_pos = self._frame_surface_cache[current_state_hash]
                    self.current_video_rect = cached_pos
                    return (cached_surf, cached_pos)

            # Cache miss - need to create new surface
            surf = pygame.surfarray.make_surface(f_transformed.swapaxes(0, 1))

            # Calculate position and size based on fit mode
            current_fit_mode = self.viewer_fit_mode
            if self.fullscreen_mode:
                screen_w, screen_h = screen.get_size()
                scale = min(screen_w / fw, screen_h / fh)
                base_w = int(fw * scale)
                base_h = int(fh * scale)
                draw_w = base_w
                draw_h = base_h
                pos_x = (screen_w - draw_w) // 2
                pos_y = (screen_h - draw_h) // 2
            elif current_fit_mode == "1:1":
                base_w, base_h = fw, fh
                draw_w = max(1, int(base_w * self.viewer_zoom))
                draw_h = max(1, int(base_h * self.viewer_zoom))
                pos_x = (vw - draw_w) // 2 + self.viewer_offset[0]
                pos_y = (vh - draw_h - th) // 2 + self.viewer_offset[1]
            elif current_fit_mode == "width":
                base_w = vw
                base_h = int(fh * (vw / fw))
                draw_w = max(1, int(base_w * self.viewer_zoom))
                draw_h = max(1, int(base_h * self.viewer_zoom))
                pos_x = (vw - draw_w) // 2 + self.viewer_offset[0]
                pos_y = (vh - draw_h - th) // 2 + self.viewer_offset[1]
            elif current_fit_mode == "height":
                base_h = vh - th
                base_w = int(fw * (base_h / fh))
                draw_w = max(1, int(base_w * self.viewer_zoom))
                draw_h = max(1, int(base_h * self.viewer_zoom))
                pos_x = (vw - draw_w) // 2 + self.viewer_offset[0]
                pos_y = (vh - draw_h - th) // 2 + self.viewer_offset[1]
            else:  # fit
                scale = min(vw / fw, (vh - th) / fh)
                base_w = int(fw * scale)
                base_h = int(fh * scale)
                draw_w = max(1, int(base_w * self.viewer_zoom))
                draw_h = max(1, int(base_h * self.viewer_zoom))
                pos_x = (vw - draw_w) // 2 + self.viewer_offset[0]
                pos_y = (vh - draw_h - th) // 2 + self.viewer_offset[1]

            # Scale the surface - fast algorithm for consistent performance
            # OPTIMIZATION: Skip scaling entirely if sizes already match (20-30% GPU savings!)
            if draw_w == fw and draw_h == fh:
                # No scaling needed - use original surface
                scaled_surf = surf
            else:
                # Fast scaling for responsive pan/zoom (scale vs smoothscale)
                scaled_surf = pygame.transform.scale(surf, (draw_w, draw_h))

            # Store the video rectangle for wipe calculations
            self.current_video_rect = (pos_x, pos_y, draw_w, draw_h)

            # Cache the surface in multi-frame cache
            with self._frame_surface_lock:
                self._frame_surface_cache[current_state_hash] = (scaled_surf, (pos_x, pos_y, draw_w, draw_h))
                # Clear dirty flag after successful cache
                self._frame_cache_dirty = False

            return (scaled_surf, (pos_x, pos_y, draw_w, draw_h))

            return scaled_surf, (pos_x, pos_y, draw_w, draw_h)

        except Exception as e:
            self.log("Error creating frame surface:", e)
            return None, (0, 0, 0, 0)

    # -----------------------
    # In/Out Marking System Methods - OPTIMIZED VERSION
    # -----------------------
    def _draw_in_button(self, screen, position, is_active=False, is_hover=False):
        """Draw circular IN button with yellow 'i' - solid fill when active"""
        btn_radius = 15
        btn_rect = pygame.Rect(
            position[0], position[1], btn_radius * 2, btn_radius * 2)

        # Background and border colors
        if is_active or self.user_marks_active:
            # SOLID YELLOW fill when active
            bg_color = (220, 180, 60)  # Full yellow fill
            border_color = (220, 180, 60)  # Yellow border
            text_color = (220, 180, 60)  # Same as background (letter disappears)
        elif is_hover:
            bg_color = (70, 70, 70)  # Medium gray for hover
            border_color = (220, 180, 60)  # Yellow border
            text_color = (160, 130, 50)  # Darker yellow text
        else:
            bg_color = (60, 60, 60)  # Default dark gray
            border_color = (220, 180, 60)  # Yellow border
            text_color = (160, 130, 50)  # Darker yellow text

        # Draw circular button
        pygame.draw.circle(
            screen, bg_color, (position[0] + btn_radius, position[1] + btn_radius), btn_radius)
        pygame.draw.circle(
            screen, border_color, (position[0] + btn_radius, position[1] + btn_radius), btn_radius, width=3)

        # Draw 'i' text
        font_key = f"font_20"
        if font_key not in self._ui_cache:
            self._ui_cache[font_key] = pygame.font.SysFont(None, 20)
        font = self._ui_cache[font_key]

        text_key = f"in_text_yellow_{is_active or self.user_marks_active}_{is_hover}"
        if text_key not in self._ui_cache:
            text_surf = font.render("i", True, text_color)
            self._ui_cache[text_key] = text_surf
        else:
            text_surf = self._ui_cache[text_key]

        text_rect = text_surf.get_rect(
            center=(position[0] + btn_radius, position[1] + btn_radius))
        screen.blit(text_surf, text_rect)

        return btn_rect

    def _draw_out_button(self, screen, position, is_active=False, is_hover=False):
        """Draw circular OUT button with red 'o' - solid fill when active"""
        btn_radius = 15
        btn_rect = pygame.Rect(
            position[0], position[1], btn_radius * 2, btn_radius * 2)

        # Background and border colors
        if is_active or self.user_marks_active:
            # SOLID RED fill when active
            bg_color = (220, 60, 60)  # Full red fill
            border_color = (220, 60, 60)  # Red border
            text_color = (220, 60, 60)  # Same as background (letter disappears)
        elif is_hover:
            bg_color = (70, 70, 70)  # Medium gray for hover
            border_color = (220, 80, 80)  # Red border
            text_color = (180, 60, 60)  # Darker red text
        else:
            bg_color = (60, 60, 60)  # Default dark gray
            border_color = (220, 80, 80)  # Red border
            text_color = (180, 60, 60)  # Darker red text

        # Draw circular button
        pygame.draw.circle(
            screen, bg_color, (position[0] + btn_radius, position[1] + btn_radius), btn_radius)
        pygame.draw.circle(
            screen, border_color, (position[0] + btn_radius, position[1] + btn_radius), btn_radius, width=3)

        # Draw 'o' text
        font_key = f"font_20"
        if font_key not in self._ui_cache:
            self._ui_cache[font_key] = pygame.font.SysFont(None, 20)
        font = self._ui_cache[font_key]

        text_key = f"out_text_red_{is_active or self.user_marks_active}_{is_hover}"
        if text_key not in self._ui_cache:
            text_surf = font.render("o", True, text_color)
            self._ui_cache[text_key] = text_surf
        else:
            text_surf = self._ui_cache[text_key]

        text_rect = text_surf.get_rect(
            center=(position[0] + btn_radius, position[1] + btn_radius))
        screen.blit(text_surf, text_rect)

        return btn_rect

    def _draw_fps_button(self, screen, position, is_active=False, is_hover=False):
        """Draw circular FPS button with grey outline and grey dot inside"""
        btn_radius = 15
        btn_rect = pygame.Rect(
            position[0], position[1], btn_radius * 2, btn_radius * 2)

        # Background color based on state
        if is_active or self.fps_dropup_open:
            bg_color = (70, 70, 90)  # Slightly blue-grey when dropup open
        elif is_hover:
            bg_color = (70, 70, 70)  # Medium gray for hover
        else:
            bg_color = (60, 60, 60)  # Default dark gray

        # Border color - grey for FPS button
        border_color = (160, 160, 160)  # Grey outline

        # Draw circular button background
        pygame.draw.circle(
            screen, bg_color, (position[0] + btn_radius, position[1] + btn_radius), btn_radius)
        
        # Draw circular border
        pygame.draw.circle(
            screen, border_color, (position[0] + btn_radius, position[1] + btn_radius), btn_radius, width=3)

        # Draw small solid grey dot inside (centered)
        dot_radius = 5  # Small dot
        dot_color = (180, 180, 180)  # Light grey dot
        pygame.draw.circle(
            screen, dot_color, (position[0] + btn_radius, position[1] + btn_radius), dot_radius)

        return btn_rect

    # -----------------------
    # Core: display_video node entry - UPDATED WITH WORKFLOW SUPPORT
    # -----------------------
    def display_video(self, video=None, images=None, source="video",
                      # REMOVED: workflow_fps and preview_fps (replaced by UI control)
                      monitor="Monitor 0", power_state=True,
                      target_resolution="1920x1080",
                      generations_name="Generation",  # Displays in UI, value used for cache/file naming
                      snapshot_workflow=True,
                      snapshot_path="smart",  # NEW: "smart" for default, or custom directory path
                      prompt=None, extra_pnginfo=None, unique_id=None):  # NEW: workflow parameters from ComfyUI
        """
        Main entrypoint for the node.
        
        NOTE: 'generations_name' is the UI parameter name. The string value provided
        by the user is used throughout the code for cache keys, file naming, and 
        generation IDs. Changing this parameter name only affects the UI label.
        
        NOTE: 'snapshot_path' accepts "smart" for default behavior, or a custom
        directory path. The system will auto-create a "snapshots" subdirectory.
        """
        # CRITICAL: Signal that we're processing new content IMMEDIATELY
        # This allows the viewer thread to show the wait screen right away
        self.processing_new_content = True
        
        # Store instance ID for unique workflow storage
        self.instance_id = unique_id or str(uuid.uuid4())
        
        # Resolve and store snapshot path
        self.current_snapshot_path = self._resolve_snapshot_path(snapshot_path)
        
        # Store snapshot workflow setting
        self.snapshot_workflow_enabled = snapshot_workflow
        
        # CRITICAL FIX: Save workflow metadata IMMEDIATELY when node runs
        if self.snapshot_workflow_enabled:
            self.log(f"Saving workflow metadata for instance {self.instance_id}...")
            success = self._save_workflow_data_immediately(prompt, extra_pnginfo)
            if success:
                self.log(f"✓ Workflow metadata saved successfully")
            else:
                self.log("✗ Failed to save workflow metadata")
                # Continue even if saving fails - don't break video playback
        else:
            self.log("Workflow snapshot disabled for this instance")
        
        output_video = video
        try:
            # CRITICAL: Check power state FIRST before creating any generations
            if not power_state:
                self.log("Power state Off - stopping (no generation created)")
                self._stop_thread()
                self.showing_wait_screen = False
                return (output_video,)
            
            # Store current source type for prefix system
            self.current_source_type = source

            # first_frame_offset now controlled by UI (see self.first_frame_text in FPS dropup)
            # Default is already set in __init__: self.first_frame_offset = 0

            # If images -> create cached mp4 file first
            if source == "images":
                if images is None:
                    self.log("Source set to images but no images provided.")
                    return (None,)
                frames, original_res = self._normalize_frames_from_images(
                    images)
                if not frames:
                    self.log("Failed to normalize image frames.")
                    return (None,)
                # Store original resolution
                self.original_resolution = original_res

                # Generate unique folder name with prefix, generations_name and timestamp
                time_str = time.strftime("%H:%M:%S")
                time_str_safe = time_str.replace(":", "-")  # Safe for filenames
                unique_id = uuid.uuid4().hex[:8]
                
                # NEW: Create folder for image sequence (even single frames)
                folder_name = f"i_{generations_name}_{time_str_safe}_{unique_id}"
                folder_path = os.path.join(RUNS_CACHE_DIR, folder_name)
                
                # Write as JPG sequence to folder (fast, high quality)
                ok = self._write_image_sequence(frames, folder_path, format='jpg')
                
                if not ok:
                    self.log(f"Failed to write image sequence.")
                    return (None,)

                # Check for duplicate generation before registering
                total_size = self._get_folder_size(folder_path)
                frame_count = len(frames)
                duplicate = self._find_duplicate_generation(total_size, frame_count)
                
                if duplicate:
                    # Duplicate found - use existing generation instead
                    gen_id, metadata = duplicate
                    existing_folder = metadata.get('folder_path')
                    self.log(f"Duplicate content detected (size: {total_size}, frames: {frame_count})")
                    self.log(f"Loading existing generation: {metadata.get('display_name')}")
                    
                    # Remove the duplicate folder we just created
                    if os.path.exists(folder_path) and os.path.isdir(folder_path):
                        import shutil
                        shutil.rmtree(folder_path)
                    
                    # Use existing generation
                    self.current_generation_id = gen_id
                    video = existing_folder
                    output_video = existing_folder
                else:
                    # No duplicate - register this as new generation
                    self.current_generation_id = self._register_new_generation(
                        folder_path, generations_name, "images", frame_count)
                    video = folder_path
                    output_video = video

                # FIX: Reset frame index when switching from video to images
                self.current_frame = 0
                self.counter_update_frame = 0
                self.force_counter_update = True
                
                # Parse monitor parameter
                try:
                    display_idx = int(monitor.split(" ")[1])
                except Exception:
                    display_idx = 0
                
                # Load the new generation
                self._load_new_video(video)
                
                # CRITICAL: Clear processing flag - content is ready!
                self.processing_new_content = False
                
                # Set up pending_content to display it
                self.pending_content = {
                    'frames_scaled': self.cached_scaled_frames,
                    'orig_frames': self.cached_original_frames,
                    'fps': self.user_fps,
                    'original_res': self.original_resolution,
                    'video_path': video,
                    'display_idx': display_idx,  # Use parsed display_idx
                    'target_res': self.target_res,
                    'target_fps': self.user_fps
                }
                
                # Signal new content ready
                self.new_content_ready.set()
                
                # Start viewer thread if not running
                if not self.running:
                    self.running = True
                    self.thread = Thread(target=self._viewer_thread, daemon=True)
                    self.thread.start()
                    self.log("Viewer thread started")
                
                return (output_video,)

            # If source == video and video provided -> cache it like images
            elif source == "video" and video is not None:
                # Read original video and cache it as image sequence
                orig_frames, fps, original_res = self._read_video_frames_original(
                    video)
                if orig_frames is None:
                    self.log("Failed to read video frames for caching.")
                    return (output_video,)
                # Store original resolution
                self.original_resolution = original_res

                # Generate unique folder name with prefix, generations_name and timestamp
                time_str = time.strftime("%H:%M:%S")
                time_str_safe = time_str.replace(":", "-")  # Safe for filenames
                unique_id = uuid.uuid4().hex[:8]
                
                # NEW: Create folder for image sequence (even single frames)
                folder_name = f"v_{generations_name}_{time_str_safe}_{unique_id}"
                folder_path = os.path.join(RUNS_CACHE_DIR, folder_name)
                
                # Write as JPG sequence to folder (fast, high quality)
                ok = self._write_image_sequence(orig_frames, folder_path, format='jpg')
                
                if not ok:
                    self.log(f"Failed to write image sequence.")
                    return (output_video,)

                # Check for duplicate generation before registering
                total_size = self._get_folder_size(folder_path)
                frame_count = len(orig_frames)
                duplicate = self._find_duplicate_generation(total_size, frame_count)
                
                if duplicate:
                    # Duplicate found - use existing generation instead
                    gen_id, metadata = duplicate
                    existing_folder = metadata.get('folder_path')
                    self.log(f"Duplicate content detected (size: {total_size}, frames: {frame_count})")
                    self.log(f"Loading existing generation: {metadata.get('display_name')}")
                    
                    # Remove the duplicate folder we just created
                    if os.path.exists(folder_path) and os.path.isdir(folder_path):
                        import shutil
                        shutil.rmtree(folder_path)
                    
                    # Use existing generation
                    self.current_generation_id = gen_id
                    video = existing_folder
                    output_video = existing_folder
                else:
                    # No duplicate - register this as new generation
                    self.current_generation_id = self._register_new_generation(
                        folder_path, generations_name, "video", frame_count)
                    video = folder_path  # Use cached folder instead of original
                    output_video = video

                # FIX: Reset frame index
                self.current_frame = 0
                self.counter_update_frame = 0
                self.force_counter_update = True
                
                # Parse monitor parameter
                try:
                    display_idx = int(monitor.split(" ")[1])
                except Exception:
                    display_idx = 0
                
                # Load the new generation
                self._load_new_video(video)
                
                # CRITICAL: Clear processing flag - content is ready!
                self.processing_new_content = False
                
                # Set up pending_content to display it
                self.pending_content = {
                    'frames_scaled': self.cached_scaled_frames,
                    'orig_frames': self.cached_original_frames,
                    'fps': self.user_fps,
                    'original_res': self.original_resolution,
                    'video_path': video,
                    'display_idx': display_idx,  # Use parsed display_idx
                    'target_res': self.target_res,
                    'target_fps': self.user_fps
                }
                
                # Signal new content ready
                self.new_content_ready.set()
                
                # Start viewer thread if not running
                if not self.running:
                    self.running = True
                    self.thread = Thread(target=self._viewer_thread, daemon=True)
                    self.thread.start()
                    self.log("Viewer thread started")
                
                return (output_video,)

            # If source == video and no video provided -> graceful stop
            elif source == "video" and video is None:
                self.log("No video provided.")
                self._stop_thread()
                return (output_video,)

            # parse target resolution
            try:
                res_w, res_h = map(int, target_resolution.split("x"))
                target_res = (res_w, res_h)
            except Exception:
                target_res = (1920, 1080)
            self.target_res = target_res

            # monitor selection
            try:
                display_idx = int(monitor.split(" ")[1])
            except Exception:
                display_idx = 0

            # compute cache key (removed fit_mode from cache key since it's now interactive)
            key = self._compute_cache_key(video, target_res)
            self.cache_key = key

            # try global cache
            with self._global_cache_lock:
                cache_entry = self._global_cache.get(key)

            if cache_entry is None:
                # read frames using universal reader and scale to target_res for fast path
                frames_scaled, fps, original_res = self._read_video_frames(
                    video, target_res)
                if frames_scaled is None:
                    self.log("Failed to extract frames from video")
                    return (output_video,)

                # Store original resolution
                self.original_resolution = original_res
                frame_count = len(frames_scaled)
                est_bytes = self._estimate_bytes_for_frames(
                    frame_count, target_res[0], target_res[1])
                if est_bytes <= DEFAULT_MAX_CACHE_BYTES:
                    cache_entry = {"frames": frames_scaled, "fps": fps,
                                   "created_at": time.time(), "size_bytes": est_bytes}
                    with self._global_cache_lock:
                        self._global_cache[key] = cache_entry
                    self.log(
                        f"Cached {frame_count} frames (~{est_bytes/(1024**2):.2f} MB)")
                else:
                    self.log("Too large to cache globally, using local only")
                    cache_entry = {"frames": frames_scaled, "fps": fps, "created_at": time.time(
                    ), "size_bytes": est_bytes, "cached_globally": False}
            else:
                frames_scaled = cache_entry["frames"]
                fps = cache_entry.get("fps", 24.0)
                # For cached entries, we don't have original resolution, so we'll get it from first frame
                if len(frames_scaled) > 0:
                    self.original_resolution = (
                        frames_scaled[0].shape[1], frames_scaled[0].shape[0])
                else:
                    self.original_resolution = (0, 0)

            # Always attempt to obtain original frames for dynamic zoom/pan
            try:
                orig_frames, _, _ = self._read_video_frames_original(video)
            except Exception:
                orig_frames = None
            if orig_frames is None:
                # fallback — use frames_scaled as original if reading originals failed (less ideal)
                try:
                    orig_frames = frames_scaled.copy()
                except Exception:
                    orig_frames = None

            # Store the content to be loaded
            self.pending_content = {
                'frames_scaled': frames_scaled,
                'orig_frames': orig_frames,
                'fps': fps,
                'original_res': self.original_resolution,
                'video_path': video,
                'display_idx': display_idx,
                'target_res': target_res,
                'target_fps': self.user_fps  # Use user-controlled FPS from UI
            }

            # NEW: Force exit from cache clear screen if it's showing
            if self.showing_cache_clear_screen:
                self.showing_cache_clear_screen = False
                self.cache_clear_start_time = 0
                self.log("Exiting cache clear screen to load new content")

            # CRITICAL FIX: Reset wait screen flag when new content is ready
            self.showing_wait_screen = False
            
            # CRITICAL: Clear processing flag - content is ready to load
            self.processing_new_content = False

            # Signal that new content is ready
            self.new_content_ready.set()

            # If thread is not running, start it
            if not self.running:
                self._stop_thread()
                self.running = True
                self.thread = Thread(target=self._viewer_thread, daemon=True)
                self.thread.start()
                self.log("Viewer thread started")
            else:
                self.log("New content queued for display")

        except Exception as e:
            self.log("Error in display_video:", e)
            traceback.print_exc()
        return (output_video,)

    # REMOVED: _calculate_target_fps (replaced by user_fps UI control)

    def _stop_thread(self):
        """Stop the viewer thread with proper thread safety."""
        try:
            with self.thread_lock:
                if self.running:
                    self.running = False
                if self.thread and self.thread.is_alive():
                    self.thread.join(timeout=1.0)
                    if self.thread.is_alive():
                        self.log(
                            "Warning: Viewer thread did not terminate cleanly")
        except Exception as e:
            self.log("Error stopping thread:", e)

    # -----------------------
    # Utility helpers
    # -----------------------
    def _estimate_bytes_for_frames(self, frame_count, width, height):
        return int(frame_count) * int(width) * int(height) * 3

    def _compute_cache_key(self, video_input, target_res):
        try:
            identifier = None
            if hasattr(video_input, '__class__') and 'VideoFromFile' in str(type(video_input)):
                try:
                    s = video_input.get_stream_source()
                    if isinstance(s, str):
                        identifier = f"path:{os.path.abspath(s)}"
                    else:
                        d = s.getvalue()
                        identifier = f"bytes:{hashlib.sha256(d).hexdigest()}"
                except Exception:
                    identifier = f"obj:{str(type(video_input))}"
            elif TORCH_AVAILABLE and isinstance(video_input, torch.Tensor):
                identifier = f"tensor:{tuple(video_input.shape)}:{str(video_input.dtype)}"
            elif isinstance(video_input, str) and os.path.exists(video_input):
                identifier = f"path:{os.path.abspath(video_input)}"
            else:
                identifier = f"objrepr:{str(type(video_input))}"
            key_raw = f"{identifier}|res:{target_res[0]}x{target_res[1]}"
            return hashlib.sha256(key_raw.encode('utf-8')).hexdigest()
        except Exception:
            return str(time.time())

    # -----------------------
    # Enhanced button rendering with color-coded outlines - OPTIMIZED VERSION
    # -----------------------
    def _draw_arrow_button(self, screen, direction, position, is_active=False, is_hover=False):
        """Draw arrow button with triangle pointing left or right. - OPTIMIZED"""
        # Button dimensions
        btn_width, btn_height = 40, 30
        btn_rect = pygame.Rect(position[0], position[1], btn_width, btn_height)

        # Background color based on state
        if is_active:
            bg_color = (60, 70, 90)  # Dark blue-gray for active state
        elif is_hover:
            bg_color = (70, 70, 70)  # Medium gray for hover
        else:
            bg_color = (60, 60, 60)  # Default dark gray

        # Border color - only colored when active, otherwise light grey
        if is_active:
            if direction in ["back", "forward", "Pong"]:  # ADDED "Pong" to this list
                # Green for playback controls (including Pong)
                border_color = (100, 220, 100)
            else:
                border_color = (120, 120, 120)  # Default light grey
        else:
            border_color = (120, 120, 120)  # Light grey outline when inactive

        # Draw button with enhanced outline
        pygame.draw.rect(screen, bg_color, btn_rect, border_radius=4)
        pygame.draw.rect(screen, border_color, btn_rect,
                         width=4, border_radius=4)

        # Add subtle shadow for depth
        shadow_rect = btn_rect.move(2, 2)
        pygame.draw.rect(screen, (30, 30, 30), shadow_rect, border_radius=4)

        # Draw arrow triangle - cached
        triangle_color = (220, 220, 220)
        center_x = position[0] + btn_width // 2
        center_y = position[1] + btn_height // 2
        triangle_size = 8

        # Cache triangle surfaces
        tri_key = f"tri_{direction}_{triangle_size}"
        if tri_key not in self._ui_cache:
            if direction == "back":
                # Left-pointing triangle
                points = [
                    (center_x + triangle_size, center_y - triangle_size),
                    (center_x + triangle_size, center_y + triangle_size),
                    (center_x - triangle_size, center_y)
                ]
            else:  # "forward"
                # Right-pointing triangle
                points = [
                    (center_x - triangle_size, center_y - triangle_size),
                    (center_x - triangle_size, center_y + triangle_size),
                    (center_x + triangle_size, center_y)
                ]
            # Create a small surface for the triangle
            tri_surf = pygame.Surface(
                (triangle_size*2, triangle_size*2), pygame.SRCALPHA)
            pygame.draw.polygon(tri_surf, triangle_color, [
                (triangle_size, 0),
                (triangle_size, triangle_size*2),
                (0 if direction == "back" else triangle_size*2, triangle_size)
            ])
            self._ui_cache[tri_key] = tri_surf
        else:
            tri_surf = self._ui_cache[tri_key]

        # Blit the cached triangle
        screen.blit(tri_surf, (center_x - triangle_size,
                    center_y - triangle_size))

        return btn_rect

    def _draw_button(self, screen, text, position, is_active=False, is_hover=False, color_override=None):
        """Draw a button with color-coded outlines and enhanced styling - OPTIMIZED."""
        # Cache fonts
        font_key = f"font_20"
        if font_key not in self._ui_cache:
            self._ui_cache[font_key] = pygame.font.SysFont(None, 20)
        font = self._ui_cache[font_key]

        # Cache text surfaces for each state
        state_key = f"btn_{text}_{is_active}_{is_hover}_{color_override}"
        if state_key not in self._ui_cache:
            text_surf = font.render(text, True, (220, 220, 220))
            self._ui_cache[state_key] = text_surf
        else:
            text_surf = self._ui_cache[state_key]

        # FIXED: Calculate button rectangle first, then center text in it
        button_height = 30  # Fixed button height for consistent centering
        button_width = text_surf.get_width() + 24  # 12px padding on each side

        # Create button rectangle with top-left at 'position'
        button_rect = pygame.Rect(
            position[0], position[1], button_width, button_height)

        # Center text in the button rectangle
        text_rect = text_surf.get_rect(center=button_rect.center)

        # Background color based on state
        if is_active:
            bg_color = (60, 70, 90)  # Dark blue-gray for active state
        elif is_hover:
            bg_color = (70, 70, 70)  # Medium gray for hover
        else:
            bg_color = (60, 60, 60)  # Default dark gray

        # Border color - only colored when active, otherwise light grey
        if is_active:
            if color_override:
                border_color = color_override
            else:
                # Color scheme based on button type when active
                playback_buttons = ["Pong"]  # Pong is a playback control now
                fit_buttons = ["Fit", "Width", "Height", "1:1", "Fullscreen"]
                reset_button = ["Reset"]
                cache_button = ["ClearCache"]
                generations_button = ["Generations"]
                snapshot_button = ["Snapshot"]  # Snapshot button (cyan outline)

                if text in playback_buttons:
                    # Green outlines for playback controls (including Pong)
                    border_color = (100, 220, 100)
                elif text in fit_buttons:
                    # Violet outlines for fit modes
                    border_color = (180, 100, 220)
                elif text in reset_button:
                    # Blue outline for reset
                    border_color = (100, 140, 220)
                elif text in cache_button:
                    # Orange outline for cache clear
                    border_color = (220, 140, 60)
                elif text in generations_button:
                    # Yellow outline for generations
                    border_color = (220, 220, 60)
                elif text in snapshot_button:
                    # Cyan outline for snapshot
                    border_color = (60, 200, 220)
                else:
                    # Fallback
                    border_color = (120, 120, 120)
        else:
            # Light grey outline when inactive
            border_color = (120, 120, 120)

        # Draw button with enhanced outline (4 pixels thick)
        pygame.draw.rect(screen, bg_color, button_rect, border_radius=4)
        pygame.draw.rect(screen, border_color, button_rect,
                         width=4, border_radius=4)

        # Add subtle shadow for depth
        shadow_rect = button_rect.move(2, 2)
        pygame.draw.rect(screen, (30, 30, 30), shadow_rect, border_radius=4)

        # Draw text (already centered in button_rect)
        screen.blit(text_surf, text_rect)

        return button_rect

    # -----------------------
    # NEW: WIPE COMPARISON METHODS - FIXED VERSION WITH PROPER FIT MODE SUPPORT
    # -----------------------
    def _load_comparison_frame(self, frame_idx):
        """Load a specific frame from the comparison generation (folder or legacy file)"""
        if not self.wipe_comparison_folder or not os.path.exists(self.wipe_comparison_folder):
            return None
        
        try:
            # NEW: Handle folder (image sequence) or file (legacy)
            if os.path.isdir(self.wipe_comparison_folder):
                # Image sequence - load specific frame
                import glob
                pattern = os.path.join(self.wipe_comparison_folder, "frame_*.png")
                files = sorted(glob.glob(pattern))
                
                if not files:
                    pattern = os.path.join(self.wipe_comparison_folder, "frame_*.jpg")
                    files = sorted(glob.glob(pattern))
                
                if not files or frame_idx >= len(files):
                    return None
                
                # Load specific frame
                bgr = cv2.imread(files[frame_idx], cv2.IMREAD_COLOR)
                if bgr is None:
                    return None
                
                rgb_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb_frame = self._apply_color_space_transform(rgb_frame)
                return rgb_frame
            
            else:
                # Legacy file (MP4 or PNG)
                if self.wipe_comparison_folder.endswith('.png'):
                    # Single PNG
                    bgr = cv2.imread(self.wipe_comparison_folder, cv2.IMREAD_COLOR)
                    if bgr is None:
                        return None
                    rgb_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgb_frame = self._apply_color_space_transform(rgb_frame)
                    return rgb_frame
                else:
                    # MP4 - read frame
                    cap = cv2.VideoCapture(self.wipe_comparison_folder)
                    if not cap.isOpened():
                        return None
                    
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames <= 0:
                        cap.release()
                        return None
                    
                    clamped_idx = min(frame_idx, total_frames - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, clamped_idx)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb_frame = self._apply_color_space_transform(rgb_frame)
                        return rgb_frame
                
        except Exception as e:
            self.log(f"Error loading comparison frame: {e}")
            
        return None

    def _get_comparison_video_size(self):
        """Get the dimensions of the comparison generation (folder or legacy file)"""
        if not self.wipe_comparison_folder or not os.path.exists(self.wipe_comparison_folder):
            return (0, 0)
        
        try:
            # NEW: Handle folder or file
            if os.path.isdir(self.wipe_comparison_folder):
                # Image sequence - read first frame
                import glob
                pattern = os.path.join(self.wipe_comparison_folder, "frame_*.png")
                files = sorted(glob.glob(pattern))
                
                if not files:
                    pattern = os.path.join(self.wipe_comparison_folder, "frame_*.jpg")
                    files = sorted(glob.glob(pattern))
                
                if not files:
                    return (0, 0)
                
                bgr = cv2.imread(files[0], cv2.IMREAD_COLOR)
                if bgr is None:
                    return (0, 0)
                
                height, width = bgr.shape[:2]
                return (width, height)
            
            else:
                # Legacy file
                if self.wipe_comparison_folder.endswith('.png'):
                    # Single PNG
                    bgr = cv2.imread(self.wipe_comparison_folder, cv2.IMREAD_COLOR)
                    if bgr is None:
                        return (0, 0)
                    height, width = bgr.shape[:2]
                    return (width, height)
                else:
                    # MP4
                    cap = cv2.VideoCapture(self.wipe_comparison_folder)
                    if not cap.isOpened():
                        return (0, 0)
                    
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    
                    return (width, height)
        except Exception as e:
            self.log(f"Error getting comparison video size: {e}")
            return (0, 0)

    def _load_sbs_comparison_frame(self, frame_idx):
        """Load a specific frame from the SBS comparison generation (same as wipe but separate cache)"""
        if not self.sbs_comparison_folder or not os.path.exists(self.sbs_comparison_folder):
            return None
        
        try:
            # Handle folder (image sequence) or file (legacy)
            if os.path.isdir(self.sbs_comparison_folder):
                # Image sequence - load specific frame
                import glob
                pattern = os.path.join(self.sbs_comparison_folder, "frame_*.png")
                files = sorted(glob.glob(pattern))
                
                if not files:
                    pattern = os.path.join(self.sbs_comparison_folder, "frame_*.jpg")
                    files = sorted(glob.glob(pattern))
                
                if not files or frame_idx >= len(files):
                    return None
                
                # Load specific frame
                bgr = cv2.imread(files[frame_idx], cv2.IMREAD_COLOR)
                if bgr is None:
                    return None
                
                rgb_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb_frame = self._apply_color_space_transform(rgb_frame)
                return rgb_frame
            
            else:
                # Legacy file (MP4 or PNG)
                if self.sbs_comparison_folder.endswith('.png'):
                    # Single PNG
                    bgr = cv2.imread(self.sbs_comparison_folder, cv2.IMREAD_COLOR)
                    if bgr is None:
                        return None
                    rgb_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgb_frame = self._apply_color_space_transform(rgb_frame)
                    return rgb_frame
                else:
                    # MP4 - read frame
                    cap = cv2.VideoCapture(self.sbs_comparison_folder)
                    if not cap.isOpened():
                        return None
                    
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames <= 0:
                        cap.release()
                        return None
                    
                    clamped_idx = min(frame_idx, total_frames - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, clamped_idx)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb_frame = self._apply_color_space_transform(rgb_frame)
                        return rgb_frame
                
        except Exception as e:
            self.log(f"Error loading SBS comparison frame: {e}")
            
        return None

    def _get_sbs_comparison_video_size(self):
        """Get the dimensions of the SBS comparison generation"""
        if not self.sbs_comparison_folder or not os.path.exists(self.sbs_comparison_folder):
            return (0, 0)
        
        try:
            # Handle folder or file
            if os.path.isdir(self.sbs_comparison_folder):
                # Image sequence - read first frame
                import glob
                pattern = os.path.join(self.sbs_comparison_folder, "frame_*.png")
                files = sorted(glob.glob(pattern))
                
                if not files:
                    pattern = os.path.join(self.sbs_comparison_folder, "frame_*.jpg")
                    files = sorted(glob.glob(pattern))
                
                if not files:
                    return (0, 0)
                
                bgr = cv2.imread(files[0], cv2.IMREAD_COLOR)
                if bgr is None:
                    return (0, 0)
                
                height, width = bgr.shape[:2]
                return (width, height)
            
            else:
                # Legacy file
                if self.sbs_comparison_folder.endswith('.png'):
                    # Single PNG
                    bgr = cv2.imread(self.sbs_comparison_folder, cv2.IMREAD_COLOR)
                    if bgr is None:
                        return (0, 0)
                    height, width = bgr.shape[:2]
                    return (width, height)
                else:
                    # MP4
                    cap = cv2.VideoCapture(self.sbs_comparison_folder)
                    if not cap.isOpened():
                        return (0, 0)
                    
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    
                    return (width, height)
        
        except Exception as e:
            self.log(f"Error getting SBS comparison video size: {e}")
            return (0, 0)

    def _calculate_video_rect(self, screen, video_width, video_height):
        """Calculate the video rectangle on screen based on current fit mode, zoom, and pan"""
        if screen is None:
            return (0, 0, 0, 0)
        
        vw, vh = screen.get_size()
        th = self.toolbar_height if not self.fullscreen_mode else 0
        fw, fh = video_width, video_height
        
        if fw <= 0 or fh <= 0:
            return (0, 0, 0, 0)
        
        current_fit_mode = self.viewer_fit_mode
        
        if self.fullscreen_mode:
            screen_w, screen_h = screen.get_size()
            scale = min(screen_w / fw, screen_h / fh)
            base_w = int(fw * scale)
            base_h = int(fh * scale)
            draw_w = base_w
            draw_h = base_h
            pos_x = (screen_w - draw_w) // 2
            pos_y = (screen_h - draw_h) // 2
        elif current_fit_mode == "1:1":
            base_w, base_h = fw, fh
            draw_w = max(1, int(base_w * self.viewer_zoom))
            draw_h = max(1, int(base_h * self.viewer_zoom))
            pos_x = (vw - draw_w) // 2 + self.viewer_offset[0]
            pos_y = (vh - draw_h - th) // 2 + self.viewer_offset[1]
        elif current_fit_mode == "width":
            base_w = vw
            base_h = int(fh * (vw / fw))
            draw_w = max(1, int(base_w * self.viewer_zoom))
            draw_h = max(1, int(base_h * self.viewer_zoom))
            pos_x = (vw - draw_w) // 2 + self.viewer_offset[0]
            pos_y = (vh - draw_h - th) // 2 + self.viewer_offset[1]
        elif current_fit_mode == "height":
            base_h = vh - th
            base_w = int(fw * (base_h / fh))
            draw_w = max(1, int(base_w * self.viewer_zoom))
            draw_h = max(1, int(base_h * self.viewer_zoom))
            pos_x = (vw - draw_w) // 2 + self.viewer_offset[0]
            pos_y = (vh - draw_h - th) // 2 + self.viewer_offset[1]
        else:  # fit
            scale = min(vw / fw, (vh - th) / fh)
            base_w = int(fw * scale)
            base_h = int(fh * scale)
            draw_w = max(1, int(base_w * self.viewer_zoom))
            draw_h = max(1, int(base_h * self.viewer_zoom))
            pos_x = (vw - draw_w) // 2 + self.viewer_offset[0]
            pos_y = (vh - draw_h - th) // 2 + self.viewer_offset[1]
        
        return (pos_x, pos_y, draw_w, draw_h)

    def _draw_wipe_overlay(self, screen, current_frame_surface, current_frame_pos):
        """Draw the wipe comparison overlay - FIXED: Now works with all fit modes, zoom, and pan"""
        if not self.wipe_active or not self.wipe_comparison_folder:
            return
        
        # Get current video rectangle
        video_rect = self.current_video_rect
        if not video_rect:
            return
        
        vid_x, vid_y, vid_w, vid_h = video_rect
        
        # Load comparison frame for current position
        comparison_frame = self._load_comparison_frame(self.current_frame)
        if comparison_frame is None:
            return
        
        # Get comparison video size if not already stored
        if self.comparison_video_size == (0, 0):
            self.comparison_video_size = self._get_comparison_video_size()
        
        comp_width, comp_height = self.comparison_video_size
        if comp_width <= 0 or comp_height <= 0:
            return
        
        # Create surface from comparison frame
        comparison_surface = pygame.surfarray.make_surface(comparison_frame.swapaxes(0, 1))
        
        # Calculate comparison video rectangle using SAME fit mode, zoom, and pan as main video
        comp_rect = self._calculate_video_rect(screen, comp_width, comp_height)
        comp_x, comp_y, comp_w, comp_h = comp_rect
        
        # Scale comparison surface - fast algorithm for responsive wipe
        scaled_comparison = pygame.transform.scale(comparison_surface, (comp_w, comp_h))
        
        # Calculate wipe position in video coordinates
        # Convert video-relative wipe position to screen coordinates
        wipe_x_screen = vid_x + int(self.wipe_position * vid_w)
        
        # Draw split video frames
        # Left side: current frame (up to wipe position)
        left_width = wipe_x_screen - vid_x
        if left_width > 0:
            current_left = current_frame_surface.subsurface((0, 0, left_width, vid_h))
            screen.blit(current_left, (vid_x, vid_y))
        
        # Right side: comparison frame (from wipe position)
        right_width = vid_w - left_width
        if right_width > 0:
            # Calculate corresponding position in comparison video
            comp_left_width = int(self.wipe_position * comp_w)
            comp_right_width = comp_w - comp_left_width
            
            if comp_right_width > 0:
                # Map screen position to comparison video position
                comp_src_x = comp_left_width
                comp_src_width = min(comp_right_width, right_width)
                
                # Create subsurface from comparison frame
                comparison_right = scaled_comparison.subsurface((comp_src_x, 0, comp_src_width, comp_h))
                screen.blit(comparison_right, (wipe_x_screen, vid_y))
        
        # Draw yellow wipe line and square ON TOP
        if not self.wipe_dragging:
            # Draw yellow wipe line on TOP
            pygame.draw.line(screen, (220, 180, 60),
                           (wipe_x_screen, vid_y), 
                           (wipe_x_screen, vid_y + vid_h), 
                           width=3)
            
            # Draw small grab square on TOP (centered on line)
            square_size = 8
            pygame.draw.rect(screen, (255, 220, 80),
                           (wipe_x_screen - square_size//2, 
                            vid_y + vid_h//2 - square_size//2,
                            square_size, square_size))

    def _draw_sbs_overlay(self, screen, current_frame_surface, current_frame_pos):
        """Draw side-by-side comparison - Each side is isolated with clipping boundaries"""
        if not self.sbs_active or not self.sbs_comparison_folder:
            return
        
        # Get screen dimensions
        vw, vh = screen.get_size()
        th = self.toolbar_height if not self.fullscreen_mode else 0
        available_height = vh - th
        
        # FIXED DIVIDER AT SCREEN CENTER
        half_width = vw // 2
        
        # Load comparison frame
        comparison_frame = self._load_sbs_comparison_frame(self.current_frame)
        if comparison_frame is None:
            return
        
        # Get video dimensions
        if self.sbs_comparison_video_size == (0, 0):
            self.sbs_comparison_video_size = self._get_sbs_comparison_video_size()
        
        comp_width, comp_height = self.sbs_comparison_video_size
        if comp_width <= 0 or comp_height <= 0:
            return
        
        # Get original video dimensions from cached frames
        with self.cache_lock:
            if self.cached_original_frames is not None and len(self.cached_original_frames) > 0:
                orig_frame = self.cached_original_frames[self.current_frame]
                orig_height, orig_width = orig_frame.shape[:2]
            else:
                return
        
        # Create surfaces
        original_surface = pygame.surfarray.make_surface(orig_frame.swapaxes(0, 1))
        comparison_surface = pygame.surfarray.make_surface(comparison_frame.swapaxes(0, 1))
        
        # Get current zoom and pan
        zoom = self.viewer_zoom
        pan_x, pan_y = self.viewer_offset
        current_fit_mode = self.viewer_fit_mode
        
        # Fill background with black
        screen.fill((0, 0, 0))
        
        # === LEFT WINDOW: Original frame (with clipping) ===
        # Calculate base fit for original in left half
        if current_fit_mode == "1:1":
            left_base_w, left_base_h = orig_width, orig_height
        elif current_fit_mode == "width":
            left_base_w = half_width
            left_base_h = int(orig_height * (half_width / orig_width))
        elif current_fit_mode == "height":
            left_base_h = available_height
            left_base_w = int(orig_width * (left_base_h / orig_height))
        else:  # fit
            scale = min(half_width / orig_width, available_height / orig_height)
            left_base_w = int(orig_width * scale)
            left_base_h = int(orig_height * scale)
        
        # Apply zoom to left
        left_draw_w = max(1, int(left_base_w * zoom))
        left_draw_h = max(1, int(left_base_h * zoom))
        
        # Calculate position in left window (centered + pan)
        left_pos_x = (half_width - left_draw_w) // 2 + pan_x
        left_pos_y = (available_height - left_draw_h) // 2 + pan_y
        
        # Scale left
        left_scaled = pygame.transform.scale(original_surface, (left_draw_w, left_draw_h))
        
        # CLIP to left window - set clip rect to LEFT half only
        left_clip_rect = pygame.Rect(0, 0, half_width, vh)
        screen.set_clip(left_clip_rect)
        screen.blit(left_scaled, (left_pos_x, left_pos_y))
        
        # === RIGHT WINDOW: Comparison frame (with clipping) ===
        # Calculate base fit for comparison in right half
        if current_fit_mode == "1:1":
            right_base_w, right_base_h = comp_width, comp_height
        elif current_fit_mode == "width":
            right_base_w = half_width
            right_base_h = int(comp_height * (half_width / comp_width))
        elif current_fit_mode == "height":
            right_base_h = available_height
            right_base_w = int(comp_width * (right_base_h / comp_height))
        else:  # fit
            scale = min(half_width / comp_width, available_height / comp_height)
            right_base_w = int(comp_width * scale)
            right_base_h = int(comp_height * scale)
        
        # Apply SAME zoom to right
        right_draw_w = max(1, int(right_base_w * zoom))
        right_draw_h = max(1, int(right_base_h * zoom))
        
        # Calculate position in right window (centered + SAME pan)
        # Position relative to RIGHT window start (half_width)
        right_pos_x = half_width + (half_width - right_draw_w) // 2 + pan_x
        right_pos_y = (available_height - right_draw_h) // 2 + pan_y
        
        # Scale right
        right_scaled = pygame.transform.scale(comparison_surface, (right_draw_w, right_draw_h))
        
        # CLIP to right window - set clip rect to RIGHT half only
        right_clip_rect = pygame.Rect(half_width, 0, half_width, vh)
        screen.set_clip(right_clip_rect)
        screen.blit(right_scaled, (right_pos_x, right_pos_y))
        
        # Clear clipping for divider line
        screen.set_clip(None)
        
        # Draw FIXED vertical dividing line at screen center
        pygame.draw.line(screen, (60, 140, 220),
                       (half_width, 0),
                       (half_width, vh),
                       width=2)

    # -----------------------
    # UPDATED: Dropup Menu with Wipe Buttons
    # -----------------------
    def _draw_dropup_menu(self, screen, mouse_pos, generations_button_rect=None):
        """Draw the generations dropUP menu positioned above the Generations button - OPTIMIZED with editing and wipe buttons"""
        if not self.dropup_open:
            return None

        sorted_generations = self._get_sorted_generations()
        if not sorted_generations:
            return None

        # Menu dimensions
        menu_width = 300
        item_height = 30
        max_visible_items = 10
        menu_height = min(len(sorted_generations),
                          max_visible_items) * item_height + 10

        # Position above Generations button if provided, otherwise default position
        if generations_button_rect:
            # Center menu above the Generations button
            menu_x = generations_button_rect.centerx - menu_width // 2
            menu_y = generations_button_rect.top - menu_height - 5
        else:
            # Fallback position
            vw, vh = screen.get_size()
            menu_x = vw - menu_width - 10
            menu_y = vh - self.toolbar_height - menu_height - 5

        # Ensure menu stays on screen
        menu_x = max(10, min(menu_x, screen.get_width() - menu_width - 10))
        menu_y = max(10, menu_y)

        # Cache font
        font_key = f"font_18"
        if font_key not in self._ui_cache:
            self._ui_cache[font_key] = pygame.font.SysFont(None, 18)
        font = self._ui_cache[font_key]

        # Background
        menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
        pygame.draw.rect(screen, (40, 40, 40), menu_rect, border_radius=4)
        pygame.draw.rect(screen, (80, 80, 80), menu_rect,
                         width=2, border_radius=4)

        # Scroll handling
        total_items = len(sorted_generations)
        visible_items = min(total_items, max_visible_items)

        # Find the index of the currently displayed generation
        current_index = -1
        if self.current_generation_id:
            for i, (gen_id, _) in enumerate(sorted_generations):
                if gen_id == self.current_generation_id:
                    current_index = i
                    break

        # If we have a current generation and it's not already selected, make it the selected one
        if current_index >= 0 and self.dropup_selected_index < 0:
            self.dropup_selected_index = current_index

        # Ensure selected index is valid
        if self.dropup_selected_index >= total_items:
            self.dropup_selected_index = total_items - 1

        # Calculate scroll offset to ensure the selected item is visible
        if self.dropup_selected_index >= 0:
            # Try to center the selected item if possible
            self.dropup_scroll_offset = max(0, min(
                self.dropup_selected_index - visible_items // 2,
                total_items - visible_items
            ))
        else:
            # No selection, start at the top
            self.dropup_scroll_offset = max(
                0, min(self.dropup_scroll_offset, total_items - visible_items))

        start_index = max(0, min(self.dropup_scroll_offset,
                          total_items - visible_items))

        # Draw items with caching
        item_rects = []
        wipe_button_rects = []  # Store wipe button rects for click detection

        for i in range(visible_items):
            idx = start_index + i
            if idx >= len(sorted_generations):
                break

            gen_id, metadata = sorted_generations[idx]
            display_name = metadata.get('display_name', 'Unknown')

            item_y = menu_y + 5 + i * item_height
            item_rect = pygame.Rect(
                menu_x + 5, item_y, menu_width - 10, item_height - 2)
            item_rects.append((gen_id, item_rect, idx))

            # Check if this item is being edited
            is_editing = (idx == self.dropup_editing_index)
            
            # Check if this item has delete confirmation showing
            is_delete_confirm = (idx == self.dropup_delete_confirm_index)

            # Draw wipe button (vertical rectangle, about 1/12th of width)
            wipe_btn_width = max(20, int(item_rect.width * 0.08))  # About 1/12 of width
            wipe_btn_height = item_rect.height - 6  # Slightly shorter than item
            wipe_btn_x = item_rect.right - wipe_btn_width - 2
            wipe_btn_y = item_rect.y + 3
            wipe_btn_rect = pygame.Rect(wipe_btn_x, wipe_btn_y, wipe_btn_width, wipe_btn_height)
            wipe_button_rects.append((gen_id, wipe_btn_rect, idx))

            # Create separate rect for text area (excludes wipe button)
            text_area_rect = pygame.Rect(item_rect.x, item_rect.y, 
                                         item_rect.width - wipe_btn_width - 5, item_rect.height)

            # Check if this generation is selected for wipe OR side-by-side
            is_wipe_selected = (gen_id == self.wipe_comparison_gen_id)
            is_sbs_selected = (gen_id == self.sbs_comparison_gen_id)

            # Draw wipe button background (always dark grey, not affected by hover)
            pygame.draw.rect(screen, (60, 60, 60), wipe_btn_rect, border_radius=2)
            
            # Draw wipe button square with different colors for each mode
            if is_sbs_selected:
                # BLUE fill for side-by-side mode (Shift+Click)
                pygame.draw.rect(screen, (60, 140, 220), wipe_btn_rect, border_radius=2)
                pygame.draw.rect(screen, (80, 180, 255), wipe_btn_rect, width=1, border_radius=2)
            elif is_wipe_selected:
                # YELLOW fill for wipe mode (normal click)
                pygame.draw.rect(screen, (220, 180, 60), wipe_btn_rect, border_radius=2)
                pygame.draw.rect(screen, (255, 220, 80), wipe_btn_rect, width=1, border_radius=2)
            else:
                # Light grey square for unselected
                pygame.draw.rect(screen, (120, 120, 120), wipe_btn_rect, border_radius=2)
                pygame.draw.rect(screen, (90, 90, 90), wipe_btn_rect, width=1, border_radius=2)

            # If delete confirmation is showing for this item, render it
            if is_delete_confirm:
                # Red confirmation overlay spanning full width
                confirm_rect = pygame.Rect(item_rect.x, item_rect.y, item_rect.width, item_rect.height)
                
                # Check if mouse is over delete button
                mouse_x, mouse_y = mouse_pos
                delete_hover = confirm_rect.collidepoint(mouse_x, mouse_y)
                
                # Red background with hover effect
                delete_bg = (100, 40, 40) if delete_hover else (80, 30, 30)
                pygame.draw.rect(screen, delete_bg, confirm_rect, border_radius=3)
                pygame.draw.rect(screen, (220, 80, 80), confirm_rect, width=2, border_radius=3)
                
                # Draw "¿delete?" text centered
                delete_text = font.render("¿delete?", True, (255, 180, 180))
                delete_text_rect = delete_text.get_rect(center=confirm_rect.center)
                screen.blit(delete_text, delete_text_rect)
                
            elif is_editing:
                # Special background for editing item
                pygame.draw.rect(screen, (80, 100, 120),
                                 item_rect, border_radius=3)
                pygame.draw.rect(screen, (100, 150, 200),
                                 item_rect, width=2, border_radius=3)

                # Draw edit text with cursor - UPDATED WITH CURSOR POSITION
                current_time = pygame.time.get_ticks()
                if current_time - self.dropup_edit_start_time > 500:  # Blink every 500ms
                    self.dropup_edit_start_time = current_time
                    self.dropup_edit_cursor_visible = not self.dropup_edit_cursor_visible

                # Render edit text up to cursor position
                text_before_cursor = self.dropup_edit_text[:self.dropup_edit_cursor_pos]
                text_after_cursor = self.dropup_edit_text[self.dropup_edit_cursor_pos:]
                
                # Render the parts separately
                edit_surf_before = font.render(
                    text_before_cursor, True, (220, 240, 255))
                edit_surf_after = font.render(
                    text_after_cursor, True, (220, 240, 255))
                
                # Calculate positions
                before_x = item_rect.x + 5
                before_y = item_rect.y + 6
                cursor_x = before_x + edit_surf_before.get_width()
                
                # Draw selection highlight if active
                if self.dropup_edit_selection_start != -1:
                    start = min(self.dropup_edit_selection_start, self.dropup_edit_selection_end)
                    end = max(self.dropup_edit_selection_start, self.dropup_edit_selection_end)
                    
                    # Calculate pixel positions for selection
                    text_before_sel = self.dropup_edit_text[:start]
                    text_selected = self.dropup_edit_text[start:end]
                    
                    sel_surf_before = font.render(text_before_sel, True, (220, 240, 255))
                    sel_surf = font.render(text_selected, True, (220, 240, 255))
                    
                    sel_x = before_x + sel_surf_before.get_width()
                    sel_width = sel_surf.get_width()
                    
                    # Draw selection background (blue highlight)
                    sel_rect = pygame.Rect(sel_x, before_y - 2, sel_width, item_rect.height - 12)
                    pygame.draw.rect(screen, (60, 120, 200), sel_rect)
                
                # Draw text parts
                screen.blit(edit_surf_before, (before_x, before_y))
                screen.blit(edit_surf_after, (cursor_x, before_y))

                # Draw cursor if visible
                if self.dropup_edit_cursor_visible:
                    cursor_rect = pygame.Rect(
                        cursor_x, item_rect.y + 6, 1, item_rect.height - 12)
                    pygame.draw.rect(screen, (220, 240, 255), cursor_rect)
            else:
                # Normal rendering for non-editing items
                is_current = (gen_id == self.current_generation_id)
                is_keyboard_selected = (idx == self.dropup_selected_index)
                
                # Check hover only for TEXT AREA (not wipe button)
                mouse_x, mouse_y = mouse_pos
                is_text_hover = text_area_rect.collidepoint(mouse_x, mouse_y) and (i == self.dropup_hover_index - start_index)

                if is_editing:
                    # Special background for editing item (ONLY TEXT AREA)
                    pygame.draw.rect(screen, (80, 100, 120),
                                     text_area_rect, border_radius=3)
                    pygame.draw.rect(screen, (100, 150, 200),
                                     text_area_rect, width=2, border_radius=3)
                else:
                    # Draw background ONLY for text area (not including wipe button)
                    if is_current and is_keyboard_selected:
                        # Blue-green for both current and selected
                        pygame.draw.rect(screen, (60, 120, 120),
                                         text_area_rect, border_radius=3)
                    elif is_current:
                        # Green background for current generation
                        pygame.draw.rect(screen, (60, 100, 60),
                                         text_area_rect, border_radius=3)
                    elif is_keyboard_selected:
                        # Blue background for keyboard selection
                        pygame.draw.rect(screen, (60, 80, 120),
                                         text_area_rect, border_radius=3)
                    elif is_text_hover:
                        # Gray background for hover (ONLY text area)
                        pygame.draw.rect(screen, (70, 70, 70),
                                         text_area_rect, border_radius=3)

                if not is_editing:
                    # Draw text - cached (leave room for wipe button)
                    text_key = f"dropup_{display_name}_{is_keyboard_selected}_{is_current}"
                    if text_key not in self._ui_cache:
                        if is_keyboard_selected:
                            # Light blue for keyboard selection
                            text_color = (180, 220, 255)
                        elif is_current:
                            text_color = (180, 255, 180)  # Light green for current
                        else:
                            text_color = (220, 220, 220)  # Normal white

                        text_surf = font.render(display_name, True, text_color)
                        self._ui_cache[text_key] = text_surf
                    else:
                        text_surf = self._ui_cache[text_key]

                    # Position text, leaving space for wipe button
                    text_x = item_rect.x + 5
                    text_width = min(text_surf.get_width(), item_rect.width - wipe_btn_width - 10)
                    screen.blit(text_surf, (text_x, item_rect.y + 6), (0, 0, text_width, text_surf.get_height()))

        return (menu_rect, item_rects, wipe_button_rects)

    # -----------------------
    # OPTIMIZED UI ELEMENT RENDERING
    # -----------------------
    def _render_counter_surfaces(self, current_frame, total_frames, fps, screen_width, screen_height):
        """Pre-render counter surfaces for fast blitting"""
        if total_frames == 0:
            return None

        # Calculate display frame number with custom offset
        display_frame = self.counter_update_frame + self.first_frame_offset + 1
        total_display_frames = total_frames + self.first_frame_offset

        # Create the three text components
        font_key = "font_20"
        if font_key not in self._ui_cache:
            self._ui_cache[font_key] = pygame.font.SysFont(None, 20)
        font = self._ui_cache[font_key]

        # 1. Resolution text
        res_w, res_h = self.original_resolution
        if res_w > 0 and res_h > 0:
            resolution_text = f"Res: {res_w}x{res_h}"
        else:
            resolution_text = "Res: Unknown"

        # 2. In/Out markers text (only when active)
        if self.user_marks_active:
            in_point_display = self.user_in_point + self.first_frame_offset + 1
            out_point_display = self.user_out_point + self.first_frame_offset + 1
            markers_text = f"IN:{in_point_display} OUT:{out_point_display}"
        else:
            markers_text = ""

        # 3. Frame counter with FPS, COLOR SPACE, AND CHANNEL MODE
        # Show channel mode only if not in RGB mode
        if self.channel_mode != "RGB":
            counter_text = f"Frame: {display_frame}/{total_display_frames} | {fps:.1f} fps | {self.selected_color_space} | {self.channel_mode}"
        else:
            counter_text = f"Frame: {display_frame}/{total_display_frames} | {fps:.1f} fps | {self.selected_color_space}"

        # Render all three text surfaces
        resolution_surf = font.render(resolution_text, True, (180, 180, 180))
        counter_surf = font.render(counter_text, True, (220, 220, 220))

        if markers_text:
            markers_surf = font.render(markers_text, True, (220, 180, 180))
        else:
            markers_surf = None

        return {
            'resolution': (resolution_surf, resolution_surf.get_rect()),
            'counter': (counter_surf, counter_surf.get_rect()),
            'markers': (markers_surf, markers_surf.get_rect() if markers_surf else None)
        }

    def _draw_frame_counter_fast(self, screen, counter_surfaces):
        """Fast version of frame counter using pre-rendered surfaces"""
        if not counter_surfaces:
            return

        screen_width, screen_height = screen.get_size()
        padding = 20
        spacing = 15

        # Unpack surfaces
        resolution_surf, resolution_rect = counter_surfaces['resolution']
        counter_surf, counter_rect = counter_surfaces['counter']
        markers_surf, markers_rect = counter_surfaces['markers']

        # Start from the right edge and work left
        # Reserve space for Gamuts button: 3x button width (3 × 12px = 36px)
        gamuts_reserved_space = 36  # 3 times the gamut button width
        right_edge = screen_width - padding - gamuts_reserved_space

        # 1. Frame counter box (rightmost)
        counter_bg_width = counter_rect.width + 20
        counter_bg_height = counter_rect.height + 10

        counter_bg_rect = pygame.Rect(
            right_edge - counter_bg_width,
            screen_height - self.buttons_height +
            (self.buttons_height - counter_bg_height) // 2,
            counter_bg_width,
            counter_bg_height
        )

        # 2. Markers box (middle, only if markers exist)
        if markers_surf and markers_rect:
            markers_bg_width = markers_rect.width + 20
            markers_bg_height = markers_rect.height + 10

            markers_bg_rect = pygame.Rect(
                counter_bg_rect.left - markers_bg_width - spacing,
                screen_height - self.buttons_height +
                (self.buttons_height - markers_bg_height) // 2,
                markers_bg_width,
                markers_bg_height
            )
        else:
            markers_bg_rect = None

        # 3. Resolution box (leftmost)
        resolution_bg_width = resolution_rect.width + 20
        resolution_bg_height = resolution_rect.height + 10

        if markers_bg_rect:
            resolution_bg_rect = pygame.Rect(
                markers_bg_rect.left - resolution_bg_width - spacing,
                screen_height - self.buttons_height +
                (self.buttons_height - resolution_bg_height) // 2,
                resolution_bg_width,
                resolution_bg_height
            )
        else:
            resolution_bg_rect = pygame.Rect(
                counter_bg_rect.left - resolution_bg_width - spacing,
                screen_height - self.buttons_height +
                (self.buttons_height - resolution_bg_height) // 2,
                resolution_bg_width,
                resolution_bg_height
            )

        # Draw all three boxes with backgrounds
        boxes = [
            (resolution_bg_rect, (40, 40, 40), resolution_surf, resolution_rect),
            (counter_bg_rect, (40, 40, 40), counter_surf, counter_rect)
        ]

        if markers_surf and markers_rect and markers_bg_rect:
            boxes.insert(1, (markers_bg_rect, (40, 30, 30),
                         markers_surf, markers_rect))

        # Draw each box
        for bg_rect, bg_color, text_surf, text_rect in boxes:
            # Draw background
            pygame.draw.rect(screen, bg_color, bg_rect, border_radius=3)
            pygame.draw.rect(screen, (80, 80, 80), bg_rect,
                             width=1, border_radius=3)

            # Center text in box
            text_rect.center = bg_rect.center
            screen.blit(text_surf, text_rect)

    def _render_timeline_surface(self, screen, current_frame, total_frames):
        """Pre-render timeline surface for fast blitting"""
        if total_frames <= 1:
            return None

        vw, vh = screen.get_size()
        timeline_height = self.timeline_height
        timeline_margin = 10

        # Timeline position (TOP of toolbar)
        timeline_top = vh - self.toolbar_height + 2
        timeline_rect = pygame.Rect(
            timeline_margin,
            timeline_top,
            vw - 2 * timeline_margin,
            timeline_height
        )

        # Create surface for timeline
        timeline_surface = pygame.Surface(
            (timeline_rect.width, timeline_rect.height), pygame.SRCALPHA)

        # Draw timeline background
        pygame.draw.rect(timeline_surface, (80, 80, 80), (0, 0,
                         timeline_rect.width, timeline_rect.height), border_radius=4)
        pygame.draw.rect(timeline_surface, (120, 120, 120), (0, 0,
                         timeline_rect.width, timeline_rect.height), width=2, border_radius=4)

        # Draw user marked area in yellow/amber gradient if active
        if self.user_marks_active and total_frames > 0:
            in_progress = self.user_in_point / \
                (total_frames - 1) if total_frames > 1 else 0
            out_progress = self.user_out_point / \
                (total_frames - 1) if total_frames > 1 else 1

            marked_width = int(timeline_rect.width *
                               (out_progress - in_progress))
            if marked_width > 0:
                marked_rect = pygame.Rect(
                    int(timeline_rect.width * in_progress),
                    0,
                    marked_width,
                    timeline_rect.height
                )
                # Yellow/amber marking area
                pygame.draw.rect(timeline_surface, (180, 140, 40),
                                 marked_rect, border_radius=4)
                
                # RED PILLARS at IN/OUT extremes
                in_x = int(timeline_rect.width * in_progress)
                out_x = int(timeline_rect.width * out_progress)
                
                # IN pillar (red, left side)
                pygame.draw.rect(timeline_surface, (220, 60, 60),
                                pygame.Rect(in_x - 2, 0, 4, timeline_rect.height),
                                border_radius=1)
                
                # OUT pillar (red, right side)
                pygame.draw.rect(timeline_surface, (220, 60, 60),
                                pygame.Rect(out_x - 2, 0, 4, timeline_rect.height),
                                border_radius=1)

        # Store the rect for hit testing
        self.timeline_extended_rect = timeline_rect

        return (timeline_surface, timeline_rect)

    def _update_timeline_indicator(self, timeline_surface, current_frame, total_frames):
        """Update just the indicator on the timeline surface"""
        if not timeline_surface or total_frames <= 0:
            return timeline_surface

        surf, rect = timeline_surface
        width, height = surf.get_size()

        # Create a copy to avoid modifying the cached surface
        updated_surf = surf.copy()

        # Draw in/out markers if user marks are active (keep this part)
        if self.user_marks_active and total_frames > 0:
            # In marker - yellow
            in_x = int(width * (self.user_in_point / (total_frames - 1))
                       ) if total_frames > 1 else 0
            in_marker = pygame.Rect(
                in_x - 2,
                0,
                4,
                height
            )
            pygame.draw.rect(updated_surf, (220, 180, 60),
                             in_marker, border_radius=1)  # Yellow

            # Out marker - red
            out_x = int(width * (self.user_out_point /
                        (total_frames - 1))) if total_frames > 1 else 0
            out_marker = pygame.Rect(
                out_x - 2,
                0,
                4,
                height
            )
            pygame.draw.rect(updated_surf, (220, 80, 80),
                             out_marker, border_radius=1)  # Red

        return (updated_surf, rect)

    def _render_toolbar_surface(self, screen):
        """Pre-render the static parts of the toolbar"""
        vw, vh = screen.get_size()
        th = self.toolbar_height

        # Create toolbar surface
        toolbar_surface = pygame.Surface((vw, th), pygame.SRCALPHA)

        # Draw toolbar background (fully opaque grey)
        toolbar_rect = pygame.Rect(0, 0, vw, th)
        pygame.draw.rect(toolbar_surface, (36, 36, 36, 255), toolbar_rect)  # Explicit alpha=255

        # Draw separator between timeline and buttons (fully opaque)
        separator_y = th - self.buttons_height
        pygame.draw.line(toolbar_surface, (60, 60, 60, 255),
                         (0, separator_y), (vw, separator_y), 1)

        return toolbar_surface

    def _draw_timeline_scrubber(self, screen, current_frame, total_frames, mouse_pos):
        """Draw timeline scrubber - FIXED VERSION with red indicator on top"""
        if total_frames <= 1:
            return None

        # Check if we need to update the timeline surface
        current_state_hash = self._calculate_ui_state_hash()
        needs_timeline_update = (
            self._ui_cache_dirty or
            self._cached_timeline_surface is None or
            self._last_ui_state_hash != current_state_hash
        )

        if needs_timeline_update or self._last_frame_rendered != current_frame:
            # Render or update timeline surface
            if self._cached_timeline_surface is None:
                self._cached_timeline_surface = self._render_timeline_surface(
                    screen, current_frame, total_frames)
            else:
                # Just update the indicator on the existing surface
                self._cached_timeline_surface = self._update_timeline_indicator(
                    self._cached_timeline_surface, current_frame, total_frames
                )

            self._last_ui_state_hash = current_state_hash

        # Blit the cached timeline surface
        if self._cached_timeline_surface:
            timeline_surface, timeline_rect = self._cached_timeline_surface
            screen.blit(timeline_surface, timeline_rect)

        # FIX: DRAW RED INDICATOR SEPARATELY ON TOP - THIS IS THE KEY FIX
        # Only draw when paused and we have frames
        if self.viewer_paused and total_frames > 0 and self.timeline_extended_rect:
            progress = current_frame / (total_frames - 1) if total_frames > 1 else 0
            indicator_x = timeline_rect.left + int(timeline_rect.width * progress)
            
            # Draw a thicker, more visible red line ON TOP of everything
            pygame.draw.line(screen, (255, 80, 80), 
                           (indicator_x, timeline_rect.top + 2), 
                           (indicator_x, timeline_rect.bottom - 2), 
                           width=3)
            
            # Draw triangle at top for better visibility (on top)
            triangle_points = [
                (indicator_x - 6, timeline_rect.top),
                (indicator_x + 6, timeline_rect.top),
                (indicator_x, timeline_rect.top + 8)
            ]
            pygame.draw.polygon(screen, (255, 80, 80), triangle_points)
            
            # Draw triangle at bottom too
            triangle_points_bottom = [
                (indicator_x - 6, timeline_rect.bottom),
                (indicator_x + 6, timeline_rect.bottom),
                (indicator_x, timeline_rect.bottom - 8)
            ]
            pygame.draw.polygon(screen, (255, 80, 80), triangle_points_bottom)

        # Check if mouse is over timeline area for scrubbing
        mouse_over_timeline = (
            self.timeline_extended_rect and
            self.timeline_extended_rect.collidepoint(mouse_pos)
        )

        return self.timeline_extended_rect if mouse_over_timeline else None

    # -----------------------
    # NEW: Content loading helper - FIXED VERSION
    # -----------------------
    def _load_pending_content(self):
        """Load pending content without restarting the window - FIXED VERSION"""
        if not self.pending_content:
            return False

        content = self.pending_content
        self.pending_content = None

        with self.cache_lock:
            self.cached_scaled_frames = content['frames_scaled']
            self.cached_original_frames = content['orig_frames']
            self.cached_surfaces = None
            self.current_video_fps = content['fps']
            self.original_resolution = content['original_res']
            self.target_res = content['target_res']

            # Initialize in/out points for the new media
            if self.cached_original_frames is not None:
                total_frames = len(self.cached_original_frames)
                self.original_in_point = 0
                self.original_out_point = total_frames - 1 if total_frames > 0 else 0
                self.user_in_point = 0
                self.user_out_point = total_frames - 1 if total_frames > 0 else 0
                self.user_marks_active = False
                self.current_frame = 0
                self.counter_update_frame = 0

            # Reset viewer state
            self.viewer_zoom = 1.0
            self.viewer_offset = [0, 0]
            self.viewer_fit_mode = "fit"
            self.viewer_paused = True
            self.force_counter_update = True
            self.showing_wait_screen = False  # CRITICAL FIX: Reset wait screen flag

            # Clear ALL caches for new content
            self._mark_ui_dirty()
            self._mark_frame_cache_dirty()
            # Frame surface cache cleared in _mark_frame_cache_dirty()
            self._ui_cache.clear()

            # Clear wipe comparison cache
            self.wipe_comparison_frames = None
            self.wipe_comparison_frame_idx = -1
            # NEW: Reset video display rectangle and comparison video size
            self.current_video_rect = None
            self.comparison_video_size = (0, 0)

        # FIX: Use explicit None check instead of boolean evaluation
        frame_count = len(
            self.cached_original_frames) if self.cached_original_frames is not None else 0
        self.log(f"Loaded new content with {frame_count} frames")
        
        # NOTE: Pre-transform removed - GPU acceleration handles it on-the-fly!
        # Transformations happen during playback at 1-2ms per frame (realtime)
        
        return True

    # -----------------------
    # OPTIMIZED VIEWER THREAD - FIXED VERSION WITH UPDATED WIPE HANDLING
    # -----------------------
    def _viewer_thread(self):
        if not PYGAME_AVAILABLE:
            self.log("Pygame not available")
            return

        screen = None
        display_initialized = False
        current_display_idx = 0
        current_target_res = (1920, 1080)
        current_target_fps = 24.0

        # Performance monitoring
        last_perf_log = time.time()
        frames_since_last_log = 0

        try:
            # Initialize pygame once
            if not pygame.get_init():
                pygame.init()
                # Enable key repeat for smooth frame stepping when holding arrow keys
                # (delay=300ms before repeat starts, interval=30ms between repeats)
                pygame.key.set_repeat(300, 30)

            while self.running:
                # Check if we need to initialize or reinitialize the display
                content_just_loaded = False  # Track if we just loaded content
                
                if not display_initialized or self.new_content_ready.is_set():
                    if self.new_content_ready.is_set():
                        # Clear the event
                        self.new_content_ready.clear()

                        # If we have pending content, load it first
                        if self.pending_content:
                            content = self.pending_content
                            current_display_idx = content['display_idx']
                            current_target_res = content['target_res']
                            current_target_fps = content['target_fps']

                            # Show wait screen ONLY if no content currently exists
                            # (Don't flash wait screen when switching between gens)
                            if screen is not None and self.cached_original_frames is None:
                                self._display_logo_screen(
                                    screen, "pvm_wait.jpg")
                                pygame.display.flip()
                                self.showing_wait_screen = True

                            # Load the content - FIX: Use explicit None check
                            self._load_pending_content()
                            content_just_loaded = True  # Mark that we just loaded content
                            self.showing_wait_screen = False  # Clear wait screen immediately

                    # Initialize or reinitialize display if needed
                    if screen is None or not display_initialized:
                        # Try to position on chosen monitor (best-effort)
                        if SCREENINFO_AVAILABLE:
                            try:
                                mons = get_monitors()
                                if current_display_idx < len(mons):
                                    mon = mons[current_display_idx]
                                    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{mon.x},{mon.y}"
                            except Exception:
                                pass

                        # Create or recreate window
                        win_w, win_h = current_target_res
                        if screen is None:
                            screen = pygame.display.set_mode(
                                (win_w, win_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
                        else:
                            # Try to resize existing window
                            try:
                                screen = pygame.display.set_mode(
                                    (win_w, win_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
                            except:
                                # If resize fails, create new window
                                screen = pygame.display.set_mode(
                                    (win_w, win_h), pygame.HWSURFACE | pygame.DOUBLEBUF)

                        pygame.display.set_caption(
                            "Preview Video Monitor Pro v5.2 - [q] Close | [SPACE] Play/Pause | [←→] Step | [ENTER] Edit/Confirm | [↑↓] Navigate | [END] Load | [1-5] Fit Modes | [i/o/p] In/Out Marks | [W] Wipe Toggle | Click timeline to scrub | Right-click generation to rename")
                        display_initialized = True

                        # Clear ALL caches for new window
                        self._mark_ui_dirty()
                        self._mark_frame_cache_dirty()
                        # Frame surface cache cleared in _mark_frame_cache_dirty()
                        self._ui_cache.clear()

                        # Display appropriate logo screen based on session state
                        self.log(f"DEBUG: Display init - first_run={self.first_run_of_session}, showing_wait={self.showing_wait_screen}, content_just_loaded={content_just_loaded}, has_frames={self.cached_original_frames is not None}")
                        
                        if self.first_run_of_session:
                            # First run: show RANDOM welcome screen
                            self.log("DEBUG: Showing welcome screen (first run)")
                            random_hello = self._get_random_hello_image()
                            if random_hello:
                                # Extract just filename for _display_logo_screen method
                                filename = os.path.basename(random_hello)
                                self._display_logo_screen(screen, filename)
                            else:
                                # Fallback to original if no image found
                                self._display_logo_screen(screen, "pvm_hello.jpg")
                            self.first_run_of_session = False
                            # Small delay to show the welcome screen
                            pygame.time.wait(500)
                        elif self.showing_wait_screen:
                            # We're already showing wait screen from above
                            self.log("DEBUG: Keeping existing wait screen")
                            pass
                        elif not content_just_loaded and self.cached_original_frames is None:
                            # Only show wait screen if we didn't just load content AND no content exists
                            self.log("DEBUG: Showing wait screen (no content loaded)")
                            self._display_logo_screen(screen, "pvm_wait.jpg")
                            self.showing_wait_screen = True
                        else:
                            self.log("DEBUG: Skipping wait screen - content is ready")

                # Main loop
                clock = pygame.time.Clock()
                dropup_rect = None
                dropup_item_rects = None
                wipe_button_rects = None
                generations_button_rect = None
                color_space_dropup_rect = None
                color_space_item_rects = None
                gamuts_button_rect = None
                snapshot_dropup_rect = None
                snapshot_item_rects = None
                snapshot_button_rect = None
                clearcache_dropup_rect = None
                clearcache_item_rects = None
                clearcache_button_rect = None
                last_ms = pygame.time.get_ticks()
                # REMOVED: Static delay_ms calculation
                # Now calculated per-frame from self.user_fps

                # Prepare fast surfaces if scaled frames exist
                with self.cache_lock:
                    if self.cached_scaled_frames is not None and self.cached_surfaces is None:
                        self._create_surfaces_from_frames()

                # Pre-render static UI elements if needed
                if not self.fullscreen_mode and screen is not None and self._cached_toolbar_surface is None:
                    self._cached_toolbar_surface = self._render_toolbar_surface(
                        screen)

                self.log(f"DEBUG: Entering main rendering loop - showing_wait={self.showing_wait_screen}, has_frames={self.cached_original_frames is not None}")

                # Main event and rendering loop - OPTIMIZED
                while self.running and not self.new_content_ready.is_set():
                    # Get current time for cache clear timer and editing
                    current_time = pygame.time.get_ticks()

                    # Update snapshot visual feedback
                    if self.last_snapshot_time > 0:
                        if current_time - self.last_snapshot_time > 1000:  # 1 second feedback
                            self.last_snapshot_time = 0
                            self._mark_ui_dirty()

                    with self.cache_lock:
                        # FIX: Use explicit None check instead of boolean evaluation
                        total_orig = 0 if self.cached_original_frames is None else len(
                            self.cached_original_frames)
                        total = total_orig or 0

                    mouse_pos = pygame.mouse.get_pos()
                    mouse_moved = (mouse_pos != self._last_mouse_pos)
                    self._last_mouse_pos = mouse_pos

                    # Handle events - OPTIMIZED: process all events at once
                    events = pygame.event.get()
                    for event in events:
                        if event.type == pygame.QUIT:
                            self.running = False
                            break

                        # Handle keyboard events
                        if event.type == pygame.KEYDOWN:
                            # Handle FPS custom field editing
                            if self.fps_custom_editing:
                                if event.key == pygame.K_BACKSPACE:
                                    # Delete last character
                                    self.fps_custom_text = self.fps_custom_text[:-1]
                                    self._mark_ui_dirty()
                                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                                    # Apply custom FPS
                                    try:
                                        new_fps = float(self.fps_custom_text)
                                        if 0.1 <= new_fps <= 240.0:  # Reasonable FPS range
                                            self.user_fps = new_fps
                                            self.log(f"Custom FPS set to {new_fps}")
                                        else:
                                            self.log(f"FPS out of range (0.1-240): {new_fps}")
                                    except ValueError:
                                        self.log(f"Invalid FPS value: {self.fps_custom_text}")
                                    self.fps_custom_editing = False
                                    self.fps_dropup_open = False
                                    self._mark_ui_dirty()
                                    continue  # Don't process Enter key further (prevent Generations toggle)
                                elif event.key == pygame.K_ESCAPE:
                                    # Cancel editing
                                    self.fps_custom_editing = False
                                    self.fps_dropup_open = False
                                    self._mark_ui_dirty()
                                    continue  # Don't process Escape key further
                                elif event.unicode and (event.unicode.isdigit() or event.unicode == '.'):
                                    # Add digit or decimal point
                                    # Prevent multiple decimal points
                                    if event.unicode == '.' and '.' in self.fps_custom_text:
                                        pass  # Ignore
                                    else:
                                        self.fps_custom_text += event.unicode
                                        self._mark_ui_dirty()
                            
                            # Handle first_frame editing (blue field)
                            elif self.first_frame_editing:
                                if event.key == pygame.K_BACKSPACE:
                                    # Delete last character
                                    self.first_frame_text = self.first_frame_text[:-1]
                                    if not self.first_frame_text:
                                        self.first_frame_text = "1"  # Default to 1 if empty
                                    self._mark_ui_dirty()
                                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                                    # Apply first frame offset
                                    try:
                                        new_first_frame = int(self.first_frame_text)
                                        if new_first_frame >= 0:
                                            self.first_frame_offset = new_first_frame - 1  # Convert to zero-based
                                            self.log(f"First frame set to {new_first_frame}")
                                        else:
                                            self.log(f"First frame must be >= 0")
                                            self.first_frame_text = "1"
                                            self.first_frame_offset = 0
                                    except ValueError:
                                        self.log(f"Invalid first frame value: {self.first_frame_text}")
                                        self.first_frame_text = "1"
                                        self.first_frame_offset = 0
                                    self.first_frame_editing = False
                                    self.fps_dropup_open = False
                                    self._mark_ui_dirty()
                                    continue  # Don't process Enter key further
                                elif event.key == pygame.K_ESCAPE:
                                    # Cancel editing
                                    self.first_frame_editing = False
                                    self.fps_dropup_open = False
                                    self._mark_ui_dirty()
                                    continue  # Don't process Escape key further
                                elif event.unicode and event.unicode.isdigit():
                                    # Add digit only (no decimals for integers)
                                    self.first_frame_text += event.unicode
                                    self._mark_ui_dirty()
                            
                            # NEW: Handle W key for wipe toggle
                            # FIX: Skip wipe toggle when renaming a generation (W/w needed for text input)
                            elif event.key == pygame.K_w and self.dropup_editing_index < 0:
                                if not self.wipe_comparison_gen_id and not self.sbs_comparison_gen_id:
                                    # No generation selected → select CURRENT generation for WIPE
                                    if self.current_generation_id:
                                        # Use the currently loaded generation
                                        self.wipe_comparison_gen_id = self.current_generation_id
                                        # Find its path from metadata
                                        metadata = self.generations_metadata.get(self.current_generation_id)
                                        if metadata:
                                            self.wipe_comparison_folder = metadata.get('folder_path', metadata.get('file_path'))
                                            self.wipe_active = True
                                            self.wipe_position = 0.5  # Center
                                            # Get comparison video size
                                            self.comparison_video_size = self._get_comparison_video_size()
                                            # Deactivate side-by-side (exclusive modes)
                                            self.sbs_active = False
                                            self.sbs_comparison_gen_id = None
                                            self.sbs_comparison_folder = None
                                            self.log(f"Wipe enabled with current generation")
                                            self._mark_ui_dirty()
                                            self._mark_frame_cache_dirty()
                                else:
                                    # Deactivate BOTH wipe and side-by-side completely
                                    self.wipe_comparison_gen_id = None
                                    self.wipe_comparison_folder = None
                                    self.wipe_active = False
                                    self.wipe_dragging = False
                                    self.comparison_video_size = (0, 0)
                                    self.sbs_comparison_gen_id = None
                                    self.sbs_comparison_folder = None
                                    self.sbs_active = False
                                    self.sbs_comparison_frames = None
                                    self.sbs_comparison_frame_idx = -1
                                    self.sbs_comparison_video_size = (0, 0)
                                    self.log("Wipe/Side-by-side deactivated")
                                    self._mark_ui_dirty()
                                    self._mark_frame_cache_dirty()

                            # NEW: Handle text input during rename with improved cursor control
                            if self.dropup_editing_index >= 0:
                                # Handle backspace with repeat support
                                if event.key == pygame.K_BACKSPACE:
                                    if self.dropup_edit_selection_start != -1:
                                        # Delete selected text
                                        start = min(self.dropup_edit_selection_start, self.dropup_edit_selection_end)
                                        end = max(self.dropup_edit_selection_start, self.dropup_edit_selection_end)
                                        self.dropup_edit_text = (
                                            self.dropup_edit_text[:start] +
                                            self.dropup_edit_text[end:]
                                        )
                                        self.dropup_edit_cursor_pos = start
                                        self.dropup_edit_selection_start = -1
                                        self.dropup_edit_selection_end = -1
                                        self._mark_ui_dirty()
                                    elif self.dropup_edit_cursor_pos > 0:
                                        # Delete character before cursor
                                        self.dropup_edit_text = (
                                            self.dropup_edit_text[:self.dropup_edit_cursor_pos - 1] +
                                            self.dropup_edit_text[self.dropup_edit_cursor_pos:]
                                        )
                                        self.dropup_edit_cursor_pos -= 1
                                        self.dropup_backspace_held = True
                                        self.dropup_backspace_start_time = current_time
                                    self._mark_ui_dirty()
                                
                                # Handle arrow keys for cursor movement
                                elif event.key == pygame.K_LEFT:
                                    if self.dropup_edit_cursor_pos > 0:
                                        self.dropup_edit_cursor_pos -= 1
                                        self.dropup_edit_selection_start = -1  # Clear selection
                                        self.dropup_edit_selection_end = -1
                                        self._mark_ui_dirty()
                                elif event.key == pygame.K_RIGHT:
                                    if self.dropup_edit_cursor_pos < len(self.dropup_edit_text):
                                        self.dropup_edit_cursor_pos += 1
                                        self.dropup_edit_selection_start = -1  # Clear selection
                                        self.dropup_edit_selection_end = -1
                                        self._mark_ui_dirty()
                                
                                # Handle Home key
                                elif event.key == pygame.K_HOME:
                                    self.dropup_edit_cursor_pos = 0
                                    self._mark_ui_dirty()
                                
                                # Handle End key
                                elif event.key == pygame.K_END:
                                    self.dropup_edit_cursor_pos = len(self.dropup_edit_text)
                                    self._mark_ui_dirty()
                                
                                # Handle Enter to finish
                                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                                    # Finish rename on Enter
                                    self._finish_rename_generation()
                                
                                # Handle Escape to cancel
                                elif event.key == pygame.K_ESCAPE:
                                    # Cancel rename on Escape
                                    self._cancel_rename()
                                
                                # Handle Delete key
                                elif event.key == pygame.K_DELETE:
                                    if self.dropup_edit_selection_start != -1:
                                        # Delete selected text
                                        start = min(self.dropup_edit_selection_start, self.dropup_edit_selection_end)
                                        end = max(self.dropup_edit_selection_start, self.dropup_edit_selection_end)
                                        self.dropup_edit_text = (
                                            self.dropup_edit_text[:start] +
                                            self.dropup_edit_text[end:]
                                        )
                                        self.dropup_edit_cursor_pos = start
                                        self.dropup_edit_selection_start = -1
                                        self.dropup_edit_selection_end = -1
                                        self._mark_ui_dirty()
                                    elif self.dropup_edit_cursor_pos < len(self.dropup_edit_text):
                                        # Delete character at cursor
                                        self.dropup_edit_text = (
                                            self.dropup_edit_text[:self.dropup_edit_cursor_pos] +
                                            self.dropup_edit_text[self.dropup_edit_cursor_pos + 1:]
                                        )
                                        self._mark_ui_dirty()
                                
                                # Handle Ctrl+A (select all)
                                elif event.key == pygame.K_a and pygame.key.get_mods() & pygame.KMOD_CTRL:
                                    # Select all text
                                    if len(self.dropup_edit_text) > 0:
                                        self.dropup_edit_selection_start = 0
                                        self.dropup_edit_selection_end = len(self.dropup_edit_text)
                                        self.dropup_edit_cursor_pos = len(self.dropup_edit_text)
                                        self._mark_ui_dirty()
                                
                                # Handle regular text input
                                elif event.unicode:
                                    # Add character to edit text at cursor position
                                    if len(event.unicode) == 1 and ord(event.unicode) >= 32:
                                        # If text is selected, replace it
                                        if self.dropup_edit_selection_start != -1:
                                            start = min(self.dropup_edit_selection_start, self.dropup_edit_selection_end)
                                            end = max(self.dropup_edit_selection_start, self.dropup_edit_selection_end)
                                            self.dropup_edit_text = (
                                                self.dropup_edit_text[:start] +
                                                event.unicode +
                                                self.dropup_edit_text[end:]
                                            )
                                            self.dropup_edit_cursor_pos = start + 1
                                            self.dropup_edit_selection_start = -1
                                            self.dropup_edit_selection_end = -1
                                        else:
                                            # No selection - insert at cursor
                                            self.dropup_edit_text = (
                                                self.dropup_edit_text[:self.dropup_edit_cursor_pos] +
                                                event.unicode +
                                                self.dropup_edit_text[self.dropup_edit_cursor_pos:]
                                            )
                                            self.dropup_edit_cursor_pos += 1
                                        self._mark_ui_dirty()
                                continue  # Skip other key handling when editing

                        if event.type == pygame.KEYUP:
                            # Handle backspace key release
                            if event.key == pygame.K_BACKSPACE:
                                self.dropup_backspace_held = False
                                self.dropup_backspace_start_time = 0

                        # Process regular key events (when not editing)
                        if event.type == pygame.KEYDOWN and self.dropup_editing_index == -1:
                            # ESC key - ONLY exits fullscreen mode (not the entire window)
                            if event.key == pygame.K_ESCAPE:
                                if self.fullscreen_mode:
                                    if screen is not None:
                                        screen = self._exit_fullscreen(screen)
                                        self._mark_ui_dirty()
                                        self._mark_frame_cache_dirty()
                                # If not in fullscreen, ESC does nothing (user expects it only for fullscreen)

                            # Shift+Q to quit window (prevent accidental exits)
                            if event.key == pygame.K_q:
                                mods = pygame.key.get_mods()
                                if mods & pygame.KMOD_SHIFT:
                                    self.running = False
                                    break

                            # SPACE BAR for play/pause
                            if event.key == pygame.K_SPACE:
                                self.viewer_paused = not self.viewer_paused
                                if not self.viewer_paused:
                                    self.viewer_playback_mode = "Forward"
                                    # Freeze counter when starting playback
                                    self.counter_update_frame = self.current_frame
                                    self._mark_ui_dirty()
                                else:
                                    # Update counter when pausing
                                    self.counter_update_frame = self.current_frame
                                    self._mark_ui_dirty()

                            # In/Out marking shortcuts
                            if event.key == pygame.K_i:
                                if total > 0:
                                    # TOGGLE: If marks active and at same frame, disable
                                    if self.user_marks_active and self.current_frame == self.user_in_point:
                                        self.user_marks_active = False
                                        self.log("Mark IN/OUT disabled")
                                    else:
                                        # Set new IN point
                                        self.user_in_point = self.current_frame
                                        if self.user_in_point > self.user_out_point:
                                            self.user_out_point = self.user_in_point
                                        self.user_marks_active = True
                                        self.log(
                                            f"Mark IN set to frame {self.user_in_point}")
                                    self._mark_ui_dirty()

                            if event.key == pygame.K_o:
                                if total > 0:
                                    # TOGGLE: If marks active and at same frame, disable
                                    if self.user_marks_active and self.current_frame == self.user_out_point:
                                        self.user_marks_active = False
                                        self.log("Mark IN/OUT disabled")
                                    else:
                                        # Set new OUT point
                                        self.user_out_point = self.current_frame
                                        if self.user_out_point < self.user_in_point:
                                            self.user_in_point = self.user_out_point
                                        self.user_marks_active = True
                                        self.log(
                                            f"Mark OUT set to frame {self.user_out_point}")
                                    self._mark_ui_dirty()

                            if event.key == pygame.K_p:
                                self.user_marks_active = False
                                self.user_in_point = self.original_in_point
                                self.user_out_point = self.original_out_point
                                self.log("Reset to original in/out points")
                                self._mark_ui_dirty()


                            # RGB Channel cycling shortcuts (R, G, B keys)
                            if event.key == pygame.K_r:
                                # Cycle: RGB -> Red -> RGB
                                if self.channel_mode == "RGB":
                                    self.channel_mode = "Red"
                                    self.log("Channel: Red only (grayscale)")
                                else:
                                    self.channel_mode = "RGB"
                                    self.log("Channel: RGB (full color)")
                                self._mark_ui_dirty()
                                self._mark_frame_cache_dirty()
                            
                            if event.key == pygame.K_g:
                                # Cycle: RGB -> Green -> RGB
                                if self.channel_mode == "RGB":
                                    self.channel_mode = "Green"
                                    self.log("Channel: Green only (grayscale)")
                                else:
                                    self.channel_mode = "RGB"
                                    self.log("Channel: RGB (full color)")
                                self._mark_ui_dirty()
                                self._mark_frame_cache_dirty()
                            
                            if event.key == pygame.K_b:
                                # Cycle: RGB -> Blue -> RGB
                                if self.channel_mode == "RGB":
                                    self.channel_mode = "Blue"
                                    self.log("Channel: Blue only (grayscale)")
                                else:
                                    self.channel_mode = "RGB"
                                    self.log("Channel: RGB (full color)")
                                self._mark_ui_dirty()
                                self._mark_frame_cache_dirty()

                            # Fit mode numeric shortcuts - DISABLED when editing FPS or First Frame fields
                            if not self.fps_custom_editing and not self.first_frame_editing:
                                if event.key == pygame.K_5 or event.key == pygame.K_KP5:
                                    if screen is not None:
                                        if self.fullscreen_mode:
                                            screen = self._exit_fullscreen(screen)
                                            self._mark_ui_dirty()
                                            self._mark_frame_cache_dirty()
                                        else:
                                            screen = self._enter_fullscreen(screen)
                                            self._mark_ui_dirty()
                                            self._mark_frame_cache_dirty()
                            
                            # Number key shortcuts (1-4) - DISABLED when editing FPS custom field
                                if event.key == pygame.K_1 or event.key == pygame.K_KP1:
                                    old_fit_mode = self.viewer_fit_mode
                                    if self.fullscreen_mode or self.viewer_fit_mode == "1:1":
                                        if self.fullscreen_mode and screen is not None:
                                            screen = self._exit_fullscreen(screen)
                                            self._mark_ui_dirty()
                                            self._mark_frame_cache_dirty()
                                        self.viewer_fit_mode = "fit"
                                    else:
                                        self.viewer_fit_mode = "1:1"
                                    if old_fit_mode != self.viewer_fit_mode:
                                        self.viewer_zoom = 1.0
                                        self.viewer_offset = [0, 0]
                                        self._mark_frame_cache_dirty()  # Frame cache needs update for fit mode change
                                    self._mark_ui_dirty()
                                elif event.key == pygame.K_2 or event.key == pygame.K_KP2:
                                    old_fit_mode = self.viewer_fit_mode
                                    if self.fullscreen_mode or self.viewer_fit_mode == "fit":
                                        if self.fullscreen_mode and screen is not None:
                                            screen = self._exit_fullscreen(screen)
                                            self._mark_ui_dirty()
                                            self._mark_frame_cache_dirty()
                                        self.viewer_fit_mode = "width"
                                    else:
                                        self.viewer_fit_mode = "fit"
                                    if old_fit_mode != self.viewer_fit_mode:
                                        self.viewer_zoom = 1.0
                                        self.viewer_offset = [0, 0]
                                        self._mark_frame_cache_dirty()  # Frame cache needs update for fit mode change
                                    self._mark_ui_dirty()
                                elif event.key == pygame.K_3 or event.key == pygame.K_KP3:
                                    old_fit_mode = self.viewer_fit_mode
                                    if self.fullscreen_mode or self.viewer_fit_mode == "width":
                                        if self.fullscreen_mode and screen is not None:
                                            screen = self._exit_fullscreen(screen)
                                            self._mark_ui_dirty()
                                            self._mark_frame_cache_dirty()
                                        self.viewer_fit_mode = "height"
                                    else:
                                        self.viewer_fit_mode = "width"
                                    if old_fit_mode != self.viewer_fit_mode:
                                        self.viewer_zoom = 1.0
                                        self.viewer_offset = [0, 0]
                                        self._mark_frame_cache_dirty()  # Frame cache needs update for fit mode change
                                    self._mark_ui_dirty()
                                elif event.key == pygame.K_4 or event.key == pygame.K_KP4:
                                    old_fit_mode = self.viewer_fit_mode
                                    if self.fullscreen_mode or self.viewer_fit_mode == "height":
                                        if self.fullscreen_mode and screen is not None:
                                            screen = self._exit_fullscreen(screen)
                                            self._mark_ui_dirty()
                                            self._mark_frame_cache_dirty()
                                        self.viewer_fit_mode = "1:1"
                                    else:
                                        self.viewer_fit_mode = "height"
                                    if old_fit_mode != self.viewer_fit_mode:
                                        self.viewer_zoom = 1.0
                                        self.viewer_offset = [0, 0]
                                        self._mark_frame_cache_dirty()  # Frame cache needs update for fit mode change
                                    self._mark_ui_dirty()

                            # ENTER key toggles edit mode or confirms edit
                            if event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
                                if self.dropup_editing_index >= 0:
                                    # Already editing - finish the rename
                                    self._finish_rename_generation()
                                elif self.dropup_open and self.dropup_selected_index >= 0:
                                    # Not editing, dropup is open, and something is selected - start editing
                                    self._start_rename_generation_keyboard()
                                elif not (self.fps_dropup_open or self.snapshot_dropup_open or self.clearcache_dropup_open):
                                    # No other dropup open - toggle Generations dropup
                                    self.dropup_open = not self.dropup_open
                                    self.dropup_scroll_offset = 0
                                    if self.dropup_open:
                                        sorted_generations = self._get_sorted_generations()
                                        if sorted_generations and self.current_generation_id:
                                            for i, (gen_id, _) in enumerate(sorted_generations):
                                                if gen_id == self.current_generation_id:
                                                    self.dropup_selected_index = i
                                                    break
                                            else:
                                                self.dropup_selected_index = 0
                                        elif sorted_generations:
                                            self.dropup_selected_index = 0
                                        else:
                                            self.dropup_selected_index = -1
                                    else:
                                        self.dropup_selected_index = -1
                                    self._mark_ui_dirty()

                            # Keyboard navigation in dropup menu
                            if self.dropup_open and self.dropup_editing_index == -1:
                                sorted_generations = self._get_sorted_generations()
                                if sorted_generations:
                                    if event.key == pygame.K_UP:
                                        self.dropup_selected_index = max(
                                            0, self.dropup_selected_index - 1)
                                        if self.dropup_selected_index < self.dropup_scroll_offset:
                                            self.dropup_scroll_offset = self.dropup_selected_index
                                        self._mark_ui_dirty()
                                    elif event.key == pygame.K_DOWN:
                                        self.dropup_selected_index = min(
                                            len(sorted_generations) - 1, self.dropup_selected_index + 1)
                                        max_visible = min(
                                            len(sorted_generations), 10)
                                        if self.dropup_selected_index >= self.dropup_scroll_offset + max_visible:
                                            self.dropup_scroll_offset = self.dropup_selected_index - max_visible + 1
                                        self._mark_ui_dirty()
                                    elif event.key == pygame.K_END:
                                        if 0 <= self.dropup_selected_index < len(sorted_generations):
                                            gen_id, metadata = sorted_generations[self.dropup_selected_index]
                                            gen_path = metadata.get('folder_path', metadata.get('file_path'))

                                            if gen_path and os.path.exists(gen_path):
                                                self.log(
                                                    "Keyboard load selected generation:", gen_id)
                                                self.current_generation_id = gen_id
                                                # Load in background
                                                Thread(target=self._load_new_video, args=(
                                                    gen_path,), daemon=True).start()
                                                self.dropup_open = False
                                                self.dropup_selected_index = -1
                                                self._mark_ui_dirty()
                                                self._mark_frame_cache_dirty()

                            # Step navigation (only when dropup is closed and not editing)
                            if not self.dropup_open and self.dropup_editing_index == -1:
                                if event.key == pygame.K_RIGHT:
                                    if total:
                                        self.current_frame = (
                                            self.current_frame + 1) % total
                                        self.viewer_paused = True
                                        self.counter_update_frame = self.current_frame
                                        self.force_counter_update = True
                                        self._mark_ui_dirty()
                                        # REMOVED: Don't invalidate surface cache on frame change!
                                        # Cache will handle different frames automatically
                                if event.key == pygame.K_LEFT:
                                    if total:
                                        self.current_frame = (
                                            self.current_frame - 1) % total
                                        self.viewer_paused = True
                                        self.counter_update_frame = self.current_frame
                                        self.force_counter_update = True
                                        self._mark_ui_dirty()
                                        # REMOVED: Don't invalidate surface cache on frame change!
                                        # Cache will handle different frames automatically

                            # Close dropup with ESC (if not editing)
                            if event.key == pygame.K_ESCAPE and self.dropup_open and self.dropup_editing_index == -1:
                                self.dropup_open = False
                                self.dropup_selected_index = -1
                                # Reset delete confirmation state
                                self.dropup_delete_confirm_index = -1
                                self.dropup_delete_confirm_gen_id = None
                                self._mark_ui_dirty()

                        # mouse wheel -> zoom anchored on mouse cursor with proper image coordinates
                        if event.type == pygame.MOUSEWHEEL:
                            # Check if we should handle dropup scrolling vs zooming
                            if (self.dropup_open or self.color_space_dropup_open or 
                                self.snapshot_dropup_open or self.clearcache_dropup_open):
                                # Handle dropup scrolling
                                if self.color_space_dropup_open:
                                    # Color space dropup scroll
                                    self.dropup_scroll_offset = max(0, self.dropup_scroll_offset - event.y)
                                elif self.dropup_open:
                                    # Generations dropup scroll  
                                    self.dropup_scroll_offset = max(0, self.dropup_scroll_offset - event.y)
                                else:
                                    # Other dropups - close on scroll
                                    if self.snapshot_dropup_open:
                                        self.snapshot_dropup_open = False
                                        self.snapshot_clear_confirm = False  # Reset Snapshot confirmation
                                    if self.clearcache_dropup_open:
                                        self.clearcache_dropup_open = False
                                        self.clearcache_clear_confirm = False  # Reset confirmation
                                self._mark_ui_dirty()
                            elif self.dropup_editing_index == -1:  # Don't zoom while editing
                                # SPECIAL HANDLING FOR SBS MODE
                                if self.sbs_active:
                                    # SBS mode: zoom within each frame's space independently
                                    mx, my = pygame.mouse.get_pos()
                                    if event.y != 0:
                                        factor = 1.15 if event.y > 0 else (1.0 / 1.15)
                                        prev_zoom = self.viewer_zoom
                                        new_zoom = max(0.1, min(16.0, self.viewer_zoom * factor))
                                        
                                        # Determine which side the mouse is on
                                        vw, vh = screen.get_size()
                                        half_width = vw // 2
                                        th = self.toolbar_height
                                        available_height = vh - th
                                        
                                        # Calculate which frame to use for anchor calculation
                                        is_left_side = (mx < half_width)
                                        
                                        with self.cache_lock:
                                            if total > 0 and self.cached_original_frames is not None:
                                                try:
                                                    # Get frame dimensions
                                                    orig_frame = self.cached_original_frames[self.current_frame]
                                                    orig_height, orig_width = orig_frame.shape[:2]
                                                    
                                                    # Get comparison dimensions
                                                    if self.sbs_comparison_video_size == (0, 0):
                                                        self.sbs_comparison_video_size = self._get_sbs_comparison_video_size()
                                                    comp_width, comp_height = self.sbs_comparison_video_size
                                                    
                                                    # Choose dimensions based on which side mouse is on
                                                    if is_left_side:
                                                        fw, fh = orig_width, orig_height
                                                        window_x = 0
                                                    else:
                                                        fw, fh = comp_width, comp_height
                                                        window_x = half_width
                                                    
                                                    # Calculate base fit for this frame in its half
                                                    current_fit_mode = self.viewer_fit_mode
                                                    if current_fit_mode == "1:1":
                                                        base_w, base_h = fw, fh
                                                    elif current_fit_mode == "width":
                                                        base_w = half_width
                                                        base_h = int(fh * (half_width / fw))
                                                    elif current_fit_mode == "height":
                                                        base_h = available_height
                                                        base_w = int(fw * (base_h / fh))
                                                    else:  # fit
                                                        scale = min(half_width / fw, available_height / fh)
                                                        base_w = int(fw * scale)
                                                        base_h = int(fh * scale)
                                                    
                                                    # Calculate current frame position with previous zoom
                                                    draw_w = max(1, int(base_w * prev_zoom))
                                                    draw_h = max(1, int(base_h * prev_zoom))
                                                    pos_x = window_x + (half_width - draw_w) // 2 + self.viewer_offset[0]
                                                    pos_y = (available_height - draw_h) // 2 + self.viewer_offset[1]
                                                    
                                                    # Calculate mouse position relative to the frame
                                                    # Mouse position in window-relative coordinates
                                                    rel_x = (mx - pos_x) / draw_w if draw_w > 0 else 0.5
                                                    rel_y = (my - pos_y) / draw_h if draw_h > 0 else 0.5
                                                    
                                                    # Calculate new dimensions with new zoom
                                                    new_draw_w = max(1, int(base_w * new_zoom))
                                                    new_draw_h = max(1, int(base_h * new_zoom))
                                                    new_pos_x = window_x + (half_width - new_draw_w) // 2
                                                    new_pos_y = (available_height - new_draw_h) // 2
                                                    
                                                    # Adjust offset to keep the same point under mouse
                                                    self.viewer_offset[0] = int(mx - new_pos_x - (rel_x * new_draw_w))
                                                    self.viewer_offset[1] = int(my - new_pos_y - (rel_y * new_draw_h))
                                                    
                                                    self.viewer_zoom = new_zoom
                                                    
                                                except Exception as e:
                                                    self.log("SBS zoom calculation error:", e)
                                                    self.viewer_zoom = new_zoom
                                            else:
                                                self.viewer_zoom = new_zoom
                                        
                                        self._mark_frame_cache_dirty()
                                        self._mark_ui_dirty()
                                else:
                                    # Normal zoom behavior (NOT SBS)
                                    mx, my = pygame.mouse.get_pos()
                                    if event.y != 0:
                                        factor = 1.15 if event.y > 0 else (1.0 / 1.15)
                                        prev_zoom = self.viewer_zoom
                                        new_zoom = max(0.1, min(16.0, self.viewer_zoom * factor))
                                        
                                        # Calculate where the image is actually positioned on screen
                                        vw, vh = screen.get_size()
                                        th = self.toolbar_height
                                        
                                        with self.cache_lock:
                                            if total > 0 and self.cached_original_frames is not None:
                                                try:
                                                    f = self.cached_original_frames[self.current_frame]
                                                    fh, fw = f.shape[0:2]  # Note: shape is (height, width, channels)
                                                    
                                                    # Use current interactive fit mode to calculate base dimensions
                                                    current_fit_mode = self.viewer_fit_mode
                                                    if self.fullscreen_mode:
                                                        # Fullscreen mode calculation
                                                        screen_w, screen_h = screen.get_size()
                                                        scale = min(screen_w / fw, screen_h / fh)
                                                        base_w = int(fw * scale)
                                                        base_h = int(fh * scale)
                                                        draw_w = base_w
                                                        draw_h = base_h
                                                        pos_x = (screen_w - draw_w) // 2
                                                        pos_y = (screen_h - draw_h) // 2
                                                    elif current_fit_mode == "1:1":
                                                        base_w, base_h = fw, fh
                                                        draw_w = max(1, int(base_w * prev_zoom))
                                                        draw_h = max(1, int(base_h * prev_zoom))
                                                        pos_x = (vw - draw_w) // 2 + self.viewer_offset[0]
                                                        pos_y = (vh - draw_h - th) // 2 + self.viewer_offset[1]
                                                    elif current_fit_mode == "width":
                                                        base_w = vw
                                                        base_h = int(fh * (vw / fw))
                                                        draw_w = max(1, int(base_w * prev_zoom))
                                                        draw_h = max(1, int(base_h * prev_zoom))
                                                        pos_x = (vw - draw_w) // 2 + self.viewer_offset[0]
                                                        pos_y = (vh - draw_h - th) // 2 + self.viewer_offset[1]
                                                    elif current_fit_mode == "height":
                                                        base_h = vh - th
                                                        base_w = int(fw * (base_h / fh))
                                                        draw_w = max(1, int(base_w * prev_zoom))
                                                        draw_h = max(1, int(base_h * prev_zoom))
                                                        pos_x = (vw - draw_w) // 2 + self.viewer_offset[0]
                                                        pos_y = (vh - draw_h - th) // 2 + self.viewer_offset[1]
                                                    else:  # fit
                                                        scale = min(vw / fw, (vh - th) / fh)
                                                        base_w = int(fw * scale)
                                                        base_h = int(fh * scale)
                                                        draw_w = max(1, int(base_w * prev_zoom))
                                                        draw_h = max(1, int(base_h * prev_zoom))
                                                        pos_x = (vw - draw_w) // 2 + self.viewer_offset[0]
                                                        pos_y = (vh - draw_h - th) // 2 + self.viewer_offset[1]
                                                    
                                                    # Calculate mouse position relative to the image
                                                    if pos_x <= mx <= pos_x + draw_w and pos_y <= my <= pos_y + draw_h:
                                                        # Mouse is over the image - zoom around mouse pointer
                                                        rel_x = (mx - pos_x) / draw_w
                                                        rel_y = (my - pos_y) / draw_h
                                                        
                                                        # Calculate new dimensions
                                                        if self.fullscreen_mode:
                                                            new_draw_w = base_w
                                                            new_draw_h = base_h
                                                        else:
                                                            new_draw_w = max(1, int(base_w * new_zoom))
                                                            new_draw_h = max(1, int(base_h * new_zoom))
                                                        
                                                        if self.fullscreen_mode:
                                                            new_pos_x = (screen.get_size()[0] - new_draw_w) // 2
                                                            new_pos_y = (screen.get_size()[1] - new_draw_h) // 2
                                                        else:
                                                            new_pos_x = (vw - new_draw_w) // 2
                                                            new_pos_y = (vh - new_draw_h - th) // 2
                                                        
                                                        # Adjust offset to keep the same point under mouse
                                                        if not self.fullscreen_mode:
                                                            self.viewer_offset[0] = int(mx - new_pos_x - (rel_x * new_draw_w))
                                                            self.viewer_offset[1] = int(my - new_pos_y - (rel_y * new_draw_h))
                                                    
                                                    self.viewer_zoom = new_zoom
                                                    
                                                except Exception as e:
                                                    self.log("Zoom calculation error:", e)
                                                    self.viewer_zoom = new_zoom
                                            else:
                                                self.viewer_zoom = new_zoom
                                        
                                        self._mark_frame_cache_dirty()  # Frame cache needs update for zoom change
                                        self._mark_ui_dirty()  # UI also needs update for zoom display

                        # mouse buttons
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            # Flag to prevent Vision from processing clicks handled by dropups
                            event_handled_by_dropup = False
                            
                            if event.button == 1:  # Left click
                                # Check if clicking on ClearCache dropup menu
                                if self.clearcache_dropup_open and clearcache_dropup_rect and clearcache_dropup_rect.collidepoint(event.pos):
                                    # Find which item was clicked
                                    item_clicked = False
                                    for action, item_rect in clearcache_item_rects:
                                        if item_rect.collidepoint(event.pos):
                                            item_clicked = True
                                            
                                            if action == "clear":
                                                # Two-click confirmation for clearing generations
                                                if not self.clearcache_clear_confirm:
                                                    # First click - show confirmation
                                                    self.clearcache_clear_confirm = True
                                                    self._mark_ui_dirty()
                                                else:
                                                    # Second click - actually clear
                                                    if self._clear_generations_cache():
                                                        self.log("Generations cache cleared successfully")
                                                        self.showing_cache_clear_screen = True
                                                        self.cache_clear_start_time = pygame.time.get_ticks()
                                                        if screen is not None:
                                                            self._display_logo_screen(
                                                                screen, "pvm_cacheclear.jpg")
                                                        self._mark_ui_dirty()
                                                        self._mark_frame_cache_dirty()
                                                    self.clearcache_dropup_open = False
                                                    self.clearcache_clear_confirm = False
                                                    self._mark_ui_dirty()
                                            elif action == "explore":
                                                # Open explorer
                                                self._open_explorer_at_cache()
                                                self.clearcache_dropup_open = False
                                                self._mark_ui_dirty()
                                            break

                                    if not item_clicked:
                                        # Clicked in empty area of dropup
                                        self.clearcache_dropup_open = False
                                        self.clearcache_clear_confirm = False  # Reset confirmation
                                        self._mark_ui_dirty()

                                # Check if clicking on snapshot dropup menu
                                elif self.snapshot_dropup_open and snapshot_dropup_rect and snapshot_dropup_rect.collidepoint(event.pos):
                                    # Find which item was clicked
                                    item_clicked = False
                                    for action, item_rect in snapshot_item_rects:
                                        if item_rect.collidepoint(event.pos):
                                            item_clicked = True
                                            
                                            if action == "take":
                                                # Take snapshot
                                                if self._take_snapshot():
                                                    self.log("Snapshot taken successfully")
                                                self.snapshot_dropup_open = False
                                                self._mark_ui_dirty()
                                            elif action == "clear":
                                                # Two-click confirmation for clearing snapshots
                                                if not self.snapshot_clear_confirm:
                                                    # First click - show confirmation
                                                    self.snapshot_clear_confirm = True
                                                    self._mark_ui_dirty()
                                                else:
                                                    # Second click - actually clear
                                                    if self._clear_snapshots_cache():
                                                        self.log("Snapshots cleared successfully")
                                                    self.snapshot_dropup_open = False
                                                    self.snapshot_clear_confirm = False
                                                    self._mark_ui_dirty()
                                            elif action == "explore":
                                                # Open explorer at snapshots directory
                                                self._open_explorer_at_snapshots()
                                                self.snapshot_dropup_open = False
                                                self._mark_ui_dirty()
                                            break

                                    if not item_clicked:
                                        # Clicked in empty area of dropup
                                        self.snapshot_dropup_open = False
                                        self.snapshot_clear_confirm = False  # Reset confirmation
                                        self._mark_ui_dirty()

                                # Check if clicking on FPS dropup menu
                                elif self.fps_dropup_open and fps_dropup_rect and fps_dropup_rect.collidepoint(event.pos):
                                    # Find which item was clicked
                                    item_clicked = False
                                    if fps_item_rects:
                                        for item_data in fps_item_rects:
                                            if len(item_data) == 3:
                                                item_name, item_rect, fps_value = item_data
                                            else:
                                                continue
                                            
                                            if item_rect.collidepoint(event.pos):
                                                item_clicked = True
                                                
                                                if item_name.startswith("preset_"):
                                                    # Clicked a preset - set FPS
                                                    self.user_fps = fps_value
                                                    self.fps_custom_text = f"{fps_value:.3f}"
                                                    self.fps_dropup_open = False
                                                    self.log(f"FPS set to {fps_value}")
                                                    self._mark_ui_dirty()
                                                    break
                                                elif item_name == "fps_field":
                                                    # Clicked FPS field (yellow) - enable editing
                                                    self.fps_custom_editing = True
                                                    self.fps_custom_text = ""  # Clear for new input
                                                    self._mark_ui_dirty()
                                                    break
                                                elif item_name == "first_frame_field":
                                                    # Clicked First Frame field (blue) - enable editing
                                                    self.first_frame_editing = True
                                                    self.first_frame_text = ""  # Clear for new input
                                                    self._mark_ui_dirty()
                                                    break

                                    if not item_clicked:
                                        # Clicked in empty area of dropup
                                        self.fps_dropup_open = False
                                        self.fps_custom_editing = False
                                        self.first_frame_editing = False
                                        self._mark_ui_dirty()

                                # Check if clicking on color space dropup menu
                                # Check if clicking on generations dropup menu (including wipe buttons)
                                elif self.dropup_open and dropup_rect and dropup_rect.collidepoint(event.pos):
                                    # FIRST: Check if delete confirmation is showing and user clicked it
                                    if self.dropup_delete_confirm_index >= 0:
                                        sorted_generations = self._get_sorted_generations()
                                        if 0 <= self.dropup_delete_confirm_index < len(sorted_generations):
                                            gen_id, metadata = sorted_generations[self.dropup_delete_confirm_index]
                                            # Check if click is on the delete confirmation button
                                            for check_gen_id, item_rect, idx in dropup_item_rects:
                                                if idx == self.dropup_delete_confirm_index and item_rect.collidepoint(event.pos):
                                                    # User confirmed deletion - delete this generation
                                                    self._delete_generation(gen_id, metadata)
                                                    # Clear delete confirmation
                                                    self.dropup_delete_confirm_index = -1
                                                    self.dropup_delete_confirm_gen_id = None
                                                    self._mark_ui_dirty()
                                                    break
                                        # Reset confirmation if clicked anywhere else
                                        self.dropup_delete_confirm_index = -1
                                        self.dropup_delete_confirm_gen_id = None
                                        self._mark_ui_dirty()
                                    
                                    # SECOND: Check if clicking on wipe button
                                    else:
                                        wipe_button_clicked = False
                                        if wipe_button_rects:
                                            for gen_id, wipe_btn_rect, idx in wipe_button_rects:
                                                if wipe_btn_rect.collidepoint(event.pos):
                                                    wipe_button_clicked = True
                                                    sorted_generations = self._get_sorted_generations()
                                                    if 0 <= idx < len(sorted_generations):
                                                        gen_id, metadata = sorted_generations[idx]
                                                        
                                                        # 3-STATE CYCLING: Off → Wipe → SBS → Off
                                                        # Check current state for THIS generation
                                                        is_wipe = (gen_id == self.wipe_comparison_gen_id and self.wipe_active)
                                                        is_sbs = (gen_id == self.sbs_comparison_gen_id and self.sbs_active)
                                                        
                                                        if is_wipe:
                                                            # STATE: Wipe → Switch to SBS
                                                            self.wipe_comparison_gen_id = None
                                                            self.wipe_comparison_folder = None
                                                            self.wipe_active = False
                                                            self.wipe_dragging = False
                                                            self.comparison_video_size = (0, 0)
                                                            
                                                            self.sbs_comparison_gen_id = gen_id
                                                            self.sbs_comparison_folder = metadata.get('folder_path', metadata.get('file_path'))
                                                            self.sbs_active = True
                                                            self.sbs_comparison_frames = None
                                                            self.sbs_comparison_frame_idx = -1
                                                            self.log(f"Switched to side-by-side: {metadata.get('display_name', gen_id)}")
                                                        
                                                        elif is_sbs:
                                                            # STATE: SBS → Turn Off
                                                            self.sbs_comparison_gen_id = None
                                                            self.sbs_comparison_folder = None
                                                            self.sbs_active = False
                                                            self.sbs_comparison_video_size = (0, 0)
                                                            self.sbs_comparison_frames = None
                                                            self.sbs_comparison_frame_idx = -1
                                                            self.log("Comparison deactivated")
                                                        
                                                        else:
                                                            # STATE: Off → Turn on Wipe
                                                            # Clear any other active comparisons first
                                                            self.wipe_comparison_gen_id = gen_id
                                                            self.wipe_comparison_folder = metadata.get('folder_path', metadata.get('file_path'))
                                                            self.wipe_active = True
                                                            self.wipe_position = 0.5  # Reset to center
                                                            self.wipe_dragging = False
                                                            self.comparison_video_size = self._get_comparison_video_size()
                                                            
                                                            # Deactivate SBS if another gen was active
                                                            if self.sbs_active:
                                                                self.sbs_active = False
                                                                self.sbs_comparison_gen_id = None
                                                                self.sbs_comparison_folder = None
                                                            
                                                            self.log(f"Wipe enabled: {metadata.get('display_name', gen_id)}")
                                                        
                                                        self._mark_ui_dirty()
                                                        self._mark_frame_cache_dirty()
                                                    break

                                        # THIRD: If not a wipe button, check if clicking on generation item
                                        if not wipe_button_clicked:
                                            item_clicked = False
                                            for gen_id, item_rect, idx in dropup_item_rects:
                                                if item_rect.collidepoint(event.pos):
                                                    item_clicked = True
                                                    
                                                    # CHECK: If this item is being edited, handle text selection
                                                    if idx == self.dropup_editing_index:
                                                        # Calculate character position from mouse X
                                                        mouse_x = event.pos[0]
                                                        text_x = item_rect.x + 5  # Text starts 5px from left
                                                        
                                                        # Find closest character position
                                                        font = pygame.font.SysFont('Arial', 15)
                                                        click_pos = 0
                                                        for i in range(len(self.dropup_edit_text) + 1):
                                                            text_before = self.dropup_edit_text[:i]
                                                            width = font.render(text_before, True, (255, 255, 255)).get_width()
                                                            if text_x + width > mouse_x:
                                                                break
                                                            click_pos = i
                                                        
                                                        # Start selection
                                                        self.dropup_edit_selection_start = click_pos
                                                        self.dropup_edit_selection_end = click_pos
                                                        self.dropup_edit_cursor_pos = click_pos
                                                        self.dropup_edit_mouse_dragging = True
                                                        self._mark_ui_dirty()
                                                        break

                                                    # Check if this is a right-click (we'll simulate with modifier key check)
                                                    # For now, handle as left click - load generation
                                                    sorted_generations = self._get_sorted_generations()
                                                    if 0 <= idx < len(sorted_generations):
                                                        gen_id, metadata = sorted_generations[idx]
                                                        gen_path = metadata.get('folder_path', metadata.get('file_path'))

                                                        if gen_path and os.path.exists(gen_path):
                                                            self.log(
                                                                "Mouse load selected generation:", gen_id)
                                                            self.current_generation_id = gen_id
                                                            # Load in background
                                                            Thread(target=self._load_new_video, args=(
                                                                gen_path,), daemon=True).start()
                                                            self.dropup_open = False
                                                            self.dropup_selected_index = -1
                                                            self.dropup_editing_index = -1  # NEW: Cancel any editing
                                                            self._mark_ui_dirty()
                                                            self._mark_frame_cache_dirty()
                                                            # Mark event as handled by dropup
                                                            event_handled_by_dropup = True
                                                    break

                                            if not item_clicked:
                                                # Clicked in empty area of dropup
                                                self.dropup_open = False
                                                self.dropup_selected_index = -1
                                                self.dropup_editing_index = -1  # NEW: Cancel any editing
                                                self._mark_ui_dirty()

                                # Check if clicking on extended timeline area for scrubbing
                                elif self.timeline_extended_rect and self.timeline_extended_rect.collidepoint(event.pos):
                                    self.scrubbing = True
                                    rel_x = event.pos[0] - \
                                        self.timeline_extended_rect.left
                                    progress = max(
                                        0, min(1, rel_x / self.timeline_extended_rect.width))
                                    new_frame = int(
                                        progress * (total - 1)) if total > 1 else 0
                                    self.current_frame = max(
                                        0, min(new_frame, total - 1))
                                    self.counter_update_frame = self.current_frame
                                    self.viewer_paused = True
                                    self.force_counter_update = True
                                    self._mark_ui_dirty()
                                    # REMOVED: Don't invalidate surface cache on frame change!
                                
                                # FIXED: Wipe line click handling with proper video-relative position calculation
                                elif self.wipe_active and self.current_video_rect:
                                    # Get video rectangle
                                    vid_x, vid_y, vid_w, vid_h = self.current_video_rect
                                    
                                    # Check if clicking within video area
                                    if vid_x <= event.pos[0] <= vid_x + vid_w and vid_y <= event.pos[1] <= vid_y + vid_h:
                                        # Calculate current wipe line position in screen coordinates
                                        wipe_x_screen = vid_x + int(self.wipe_position * vid_w)
                                        line_threshold = 10  # pixels around line to grab
                                        
                                        # Check if clicking near the wipe line
                                        if abs(event.pos[0] - wipe_x_screen) < line_threshold:
                                            self.wipe_dragging = True
                                            # CRITICAL FIX: Close all dropups when clicking wipe line
                                            self.dropup_open = False
                                            self.dropup_selected_index = -1
                                            self.dropup_editing_index = -1
                                            # Reset confirmation states
                                            self.dropup_delete_confirm_index = -1
                                            self.dropup_delete_confirm_gen_id = None
                                            self.color_space_dropup_open = False
                                            self.snapshot_dropup_open = False
                                            self.snapshot_clear_confirm = False  # Reset Snapshot confirmation
                                            self.clearcache_dropup_open = False
                                            self.clearcache_clear_confirm = False  # Reset ClearCache confirmation
                                            self._mark_ui_dirty()
                                
                                else:
                                    # FIXED: Close dropups if clicking in empty toolbar area
                                    vw, vh = screen.get_size()
                                    th = self.toolbar_height
                                    # REMOVED: Don't close dropups on toolbar click - buttons need to handle their own toggles!
                                    # ONLY trigger Play if clicking in the main video area (not toolbar) AND Vision module is closed
                                    if event.pos[1] < (vh - th) and not self.vision_module_open:
                                        self.viewer_paused = False
                                        self.viewer_playback_mode = "Forward"
                                        self.counter_update_frame = self.current_frame
                                        self.dropup_open = False
                                        self.dropup_selected_index = -1
                                        self.dropup_editing_index = -1  # NEW: Cancel any editing
                                        self.color_space_dropup_open = False
                                        self.snapshot_dropup_open = False
                                        self.clearcache_dropup_open = False
                                        self._mark_ui_dirty()
                                
                                # Check Vision module (AFTER all dropups have been checked)
                                # IMPORTANT: Only if NO dropups are open AND event wasn't already handled by dropup
                                if (self.vision_module_open and event.button == 1 and 
                                    not event_handled_by_dropup and
                                    not self.dropup_open and 
                                    not self.color_space_dropup_open and 
                                    not self.snapshot_dropup_open and 
                                    not self.clearcache_dropup_open):
                                    vw, vh = screen.get_size()
                                    module_height = 50
                                    module_y = vh - self.toolbar_height - module_height
                                    
                                    # PRIORITY: Check if clicking on color picker (if active)
                                    if self.vision_picker_active and self.vision_picker_pos:
                                        picker_x, picker_y = self.vision_picker_pos
                                        picker_hit_size = 20  # Larger hit area for easier clicking
                                        picker_hit_rect = pygame.Rect(
                                            picker_x - picker_hit_size//2,
                                            picker_y - picker_hit_size//2,
                                            picker_hit_size,
                                            picker_hit_size
                                        )
                                        
                                        if picker_hit_rect.collidepoint(event.pos):
                                            # Start dragging picker
                                            self.vision_picker_dragging = True
                                            # Don't process other Vision interactions
                                            continue
                                    
                                    # Check if clicking in Vision module area
                                    if event.pos[1] >= module_y and event.pos[1] <= module_y + module_height:
                                        # Calculate slider section dimensions
                                        slider_section_width = int(vw * 0.75)
                                        slider_width = slider_section_width // 3
                                        
                                        # Check if clicking in Gain slider area
                                        padding = 10
                                        bar_y = module_y + 15 + 15  # Adjusted for label position
                                        bar_x_start = 10 + padding
                                        bar_x_end = 10 + slider_width - padding
                                        bar_width = bar_x_end - bar_x_start
                                        
                                        # Slider interaction rect
                                        slider_rect = pygame.Rect(bar_x_start, bar_y - 15, bar_width, 30)
                                        
                                        if slider_rect.collidepoint(event.pos):
                                            # Left click - start dragging
                                            self.vision_dragging_slider = "gain"
                                            
                                            # Immediately set value at click position
                                            mouse_x = event.pos[0]
                                            rel_x = mouse_x - bar_x_start
                                            normalized = max(0.0, min(1.0, rel_x / bar_width))
                                            min_val = -6.0
                                            max_val = 6.0
                                            new_value = min_val + normalized * (max_val - min_val)
                                            new_value = round(new_value * 10) / 10  # Snap to 0.1
                                            self.vision_gain = max(min_val, min(max_val, new_value))
                                            self._mark_ui_dirty()
                                            self._mark_frame_cache_dirty()
                                        
                                        # Check GAMMA slider (second third)
                                        gamma_bar_x_start = 10 + slider_width + padding
                                        gamma_bar_x_end = 10 + slider_width * 2 - padding
                                        gamma_bar_width = gamma_bar_x_end - gamma_bar_x_start
                                        gamma_slider_rect = pygame.Rect(gamma_bar_x_start, bar_y - 15, gamma_bar_width, 30)
                                        
                                        if gamma_slider_rect.collidepoint(event.pos):
                                            # Left click
                                            self.vision_dragging_slider = "gamma"
                                            mouse_x = event.pos[0]
                                            rel_x = mouse_x - gamma_bar_x_start
                                            normalized = max(0.0, min(1.0, rel_x / gamma_bar_width))
                                            new_value = 0.0 + normalized * 4.0  # 0 to 4
                                            new_value = round(new_value * 10) / 10
                                            self.vision_gamma = max(0.0, min(4.0, new_value))
                                            self._mark_ui_dirty()
                                            self._mark_frame_cache_dirty()
                                        
                                        # Check SATURATION slider (third third)
                                        sat_bar_x_start = 10 + slider_width * 2 + padding
                                        sat_bar_x_end = 10 + slider_width * 3 - padding
                                        sat_bar_width = sat_bar_x_end - sat_bar_x_start
                                        sat_slider_rect = pygame.Rect(sat_bar_x_start, bar_y - 15, sat_bar_width, 30)
                                        
                                        if sat_slider_rect.collidepoint(event.pos):
                                            # Left click
                                            self.vision_dragging_slider = "saturation"
                                            mouse_x = event.pos[0]
                                            rel_x = mouse_x - sat_bar_x_start
                                            normalized = max(0.0, min(1.0, rel_x / sat_bar_width))
                                            new_value = 0.0 + normalized * 4.0  # 0 to 4
                                            new_value = round(new_value * 10) / 10
                                            self.vision_saturation = max(0.0, min(4.0, new_value))
                                            self._mark_ui_dirty()
                                            self._mark_frame_cache_dirty()
                                        
                                        # Check SCOPE BUTTONS (H, V, W) in Analysis Tools section
                                        button_section_start_x = slider_section_width + 10
                                        button_size = 30
                                        button_spacing = 10
                                        button_y_center = module_y + (module_height - button_size) // 2
                                        
                                        scope_buttons_map = [
                                            ("H", "histogram", 0),
                                            ("V", "vector", 1),
                                            ("W", "waveform", 2)
                                        ]
                                        
                                        for label, scope_type, index in scope_buttons_map:
                                            button_x = button_section_start_x + index * (button_size + button_spacing)
                                            button_rect = pygame.Rect(button_x, button_y_center, button_size, button_size)
                                            
                                            if button_rect.collidepoint(event.pos):
                                                # Toggle scope: if clicking active scope, turn it off; otherwise activate
                                                if self.vision_active_scope == scope_type:
                                                    self.vision_active_scope = None  # Turn off
                                                else:
                                                    self.vision_active_scope = scope_type  # Turn on this scope
                                                self._mark_ui_dirty()
                                        
                                        # Check ZEBRA button (Z) - after scope buttons
                                        zebra_button_x = button_section_start_x + 3 * (button_size + button_spacing)
                                        zebra_button_rect = pygame.Rect(zebra_button_x, button_y_center, button_size, button_size)
                                        
                                        if zebra_button_rect.collidepoint(event.pos):
                                            # Toggle zebra
                                            self.vision_zebra_active = not self.vision_zebra_active
                                            self._mark_ui_dirty()
                                        
                                        # Check PICKER button (P) - after Zebra
                                        picker_button_x = button_section_start_x + 4 * (button_size + button_spacing)
                                        picker_button_rect = pygame.Rect(picker_button_x, button_y_center, button_size, button_size)
                                        
                                        if picker_button_rect.collidepoint(event.pos):
                                            # Toggle picker
                                            self.vision_picker_active = not self.vision_picker_active
                                            if self.vision_picker_active and self.vision_picker_pos is None:
                                                # Initialize picker at screen center
                                                vw_init, vh_init = screen.get_size()
                                                self.vision_picker_pos = [vw_init // 2, vh_init // 2]
                                            self._mark_ui_dirty()
                                        
                                        # Check RGB channel buttons (R, G, B) - after Picker
                                        rgb_buttons = [
                                            ("R", "Red", 5),
                                            ("G", "Green", 6),
                                            ("B", "Blue", 7)
                                        ]
                                        
                                        for label, mode, offset in rgb_buttons:
                                            button_x = button_section_start_x + offset * (button_size + button_spacing)
                                            button_rect = pygame.Rect(button_x, button_y_center, button_size, button_size)
                                            
                                            if button_rect.collidepoint(event.pos):
                                                # Toggle channel mode
                                                if self.channel_mode == mode:
                                                    # If already active, go back to RGB
                                                    self.channel_mode = "RGB"
                                                else:
                                                    # Activate this channel
                                                    self.channel_mode = mode
                                                self._mark_ui_dirty()
                                                self._mark_frame_cache_dirty()
                                        
                                        # Check GUIDES button - after RGB buttons
                                        guides_button_x = button_section_start_x + 8 * (button_size + button_spacing)
                                        guides_button_rect = pygame.Rect(guides_button_x, button_y_center, button_size, button_size)
                                        
                                        if guides_button_rect.collidepoint(event.pos):
                                            # Toggle guides
                                            self.vision_guides_active = not self.vision_guides_active
                                            self._mark_ui_dirty()
                                        
                                        # Check MASK button (M) - last slot! Cycles through 10 ratios
                                        mask_button_x = button_section_start_x + 9 * (button_size + button_spacing)
                                        mask_button_rect = pygame.Rect(mask_button_x, button_y_center, button_size, button_size)
                                        
                                        if mask_button_rect.collidepoint(event.pos):
                                            # Cycle to next mask ratio
                                            self.vision_mask_index = (self.vision_mask_index + 1) % len(self.vision_mask_ratios)
                                            self._mark_ui_dirty()
                                        
                                        # Check FLIP/FLOP button - slot 10! Cycles: Normal → H-Flip → V-Flip → Both → Normal
                                        flip_button_x = button_section_start_x + 10 * (button_size + button_spacing)
                                        flip_button_rect = pygame.Rect(flip_button_x, button_y_center, button_size, button_size)
                                        
                                        if flip_button_rect.collidepoint(event.pos):
                                            # Cycle through flip modes: 0=normal, 1=h-flip, 2=v-flip, 3=both
                                            self.vision_flip_mode = (self.vision_flip_mode + 1) % 4
                                            self._mark_ui_dirty()
                                            self._mark_frame_cache_dirty()

                            elif event.button == 2:  # Middle click
                                # Check if middle-clicking on generations dropup menu
                                if self.dropup_open and dropup_rect and dropup_rect.collidepoint(event.pos):
                                    # Find which generation was middle-clicked
                                    if dropup_item_rects:
                                        for gen_id, item_rect, idx in dropup_item_rects:
                                            if item_rect.collidepoint(event.pos):
                                                # Show delete confirmation for this generation
                                                self.dropup_delete_confirm_index = idx
                                                self.dropup_delete_confirm_gen_id = gen_id
                                                self.log(f"Delete confirmation for generation {idx}")
                                                self._mark_ui_dirty()
                                                break
                                else:
                                    # Middle click -> reset view (original behavior)
                                    # OR reset Vision sliders if clicking on them
                                    should_reset_view = True
                                    
                                    if self.vision_module_open:
                                        vw_check, vh_check = screen.get_size()
                                        module_height = 50
                                        module_y = vh_check - self.toolbar_height - module_height
                                        
                                        # Check if mouse is over Vision module
                                        if event.pos[1] >= module_y and event.pos[1] <= module_y + module_height:
                                            # Mouse is over Vision module - check for slider resets
                                            should_reset_view = False
                                            
                                            # Calculate slider dimensions
                                            slider_section_width = int(vw_check * 0.75)
                                            slider_width = slider_section_width // 3
                                            padding = 10
                                            bar_y = module_y + 15 + 15
                                            
                                            # Check GAIN slider
                                            gain_bar_x_start = 10 + padding
                                            gain_bar_x_end = 10 + slider_width - padding
                                            gain_bar_width = gain_bar_x_end - gain_bar_x_start
                                            gain_slider_rect = pygame.Rect(gain_bar_x_start, bar_y - 15, gain_bar_width, 30)
                                            
                                            if gain_slider_rect.collidepoint(event.pos):
                                                self.vision_gain = 0.0  # Reset to 0 stops
                                                self._mark_ui_dirty()
                                                self._mark_frame_cache_dirty()
                                            
                                            # Check GAMMA slider
                                            gamma_bar_x_start = 10 + slider_width + padding
                                            gamma_bar_x_end = 10 + slider_width * 2 - padding
                                            gamma_bar_width = gamma_bar_x_end - gamma_bar_x_start
                                            gamma_slider_rect = pygame.Rect(gamma_bar_x_start, bar_y - 15, gamma_bar_width, 30)
                                            
                                            if gamma_slider_rect.collidepoint(event.pos):
                                                self.vision_gamma = 1.0  # Reset to 1.0
                                                self._mark_ui_dirty()
                                                self._mark_frame_cache_dirty()
                                                # Clear cached LUT
                                                self.vision_gamma_lut = None
                                                self.vision_gamma_lut_value = None
                                            
                                            # Check SATURATION slider
                                            sat_bar_x_start = 10 + slider_width * 2 + padding
                                            sat_bar_x_end = 10 + slider_width * 3 - padding
                                            sat_bar_width = sat_bar_x_end - sat_bar_x_start
                                            sat_slider_rect = pygame.Rect(sat_bar_x_start, bar_y - 15, sat_bar_width, 30)
                                            
                                            if sat_slider_rect.collidepoint(event.pos):
                                                self.vision_saturation = 1.0  # Reset to 1.0
                                                self._mark_ui_dirty()
                                                self._mark_frame_cache_dirty()
                                            
                                            # Check MASK button (M) - middle-click to reset to OFF
                                            button_section_width = vw_check - slider_section_width
                                            button_section_start_x = slider_section_width
                                            button_size = 30
                                            button_spacing = 10
                                            button_y_center = module_y + (module_height - button_size) // 2
                                            
                                            mask_button_x = button_section_start_x + 9 * (button_size + button_spacing)
                                            mask_button_rect = pygame.Rect(mask_button_x, button_y_center, button_size, button_size)
                                            
                                            if mask_button_rect.collidepoint(event.pos):
                                                # Reset mask to OFF
                                                self.vision_mask_index = 0
                                                self._mark_ui_dirty()
                                    
                                    if should_reset_view:
                                        self.viewer_zoom = 1.0
                                        self.viewer_offset = [0, 0]
                                        self.viewer_fit_mode = "fit"
                                        self.dropup_open = False
                                        self.dropup_selected_index = -1
                                        self.dropup_editing_index = -1  # NEW: Cancel any editing
                                        self.dropup_delete_confirm_index = -1  # NEW: Cancel any delete confirmation
                                        self.color_space_dropup_open = False
                                        self.snapshot_dropup_open = False
                                        self.clearcache_dropup_open = False
                                        self.fps_dropup_open = False  # Close FPS dropup
                                        self.fps_custom_editing = False  # Cancel FPS editing
                                        self._mark_ui_dirty()
                                        self._mark_frame_cache_dirty()  # Frame cache needs update for view reset

                            elif event.button == 3:  # Right click
                                # FIXED: Right click should NOT close dropup if clicking inside it
                                if self.dropup_open and dropup_rect and dropup_rect.collidepoint(event.pos):
                                    # Right-click inside dropup - keep it open for rename
                                    self.dragging = False
                                    self.drag_button = None
                                    self._mark_ui_dirty()
                                elif self.color_space_dropup_open and color_space_dropup_rect and color_space_dropup_rect.collidepoint(event.pos):
                                    # Right-click inside color space dropup - ignore
                                    self.dragging = False
                                    self.drag_button = None
                                    self._mark_ui_dirty()
                                elif self.snapshot_dropup_open and snapshot_dropup_rect and snapshot_dropup_rect.collidepoint(event.pos):
                                    # Right-click inside snapshot dropup - ignore
                                    self.dragging = False
                                    self.drag_button = None
                                    self._mark_ui_dirty()
                                elif self.clearcache_dropup_open and clearcache_dropup_rect and clearcache_dropup_rect.collidepoint(event.pos):
                                    # Right-click inside ClearCache dropup - ignore
                                    self.dragging = False
                                    self.drag_button = None
                                    self._mark_ui_dirty()
                                elif self.fps_dropup_open and fps_dropup_rect and fps_dropup_rect.collidepoint(event.pos):
                                    # Right-click inside FPS dropup - ignore
                                    self.dragging = False
                                    self.drag_button = None
                                    self._mark_ui_dirty()
                                else:
                                    # Right-click outside dropup - enable dragging for panning
                                    self.dragging = True
                                    self.drag_button = 3
                                    self.last_mouse_pos = event.pos
                                    self.dropup_open = False
                                    self.dropup_selected_index = -1
                                    self.dropup_editing_index = -1  # NEW: Cancel any editing
                                    # Reset confirmation states
                                    self.dropup_delete_confirm_index = -1
                                    self.dropup_delete_confirm_gen_id = None
                                    self.color_space_dropup_open = False
                                    self.snapshot_dropup_open = False
                                    self.snapshot_clear_confirm = False  # Reset Snapshot confirmation
                                    self.clearcache_dropup_open = False
                                    self.clearcache_clear_confirm = False  # Reset ClearCache confirmation
                                    self.fps_dropup_open = False  # Close FPS dropup
                                    self.fps_custom_editing = False  # Cancel FPS editing
                                    self._mark_ui_dirty()

                        if event.type == pygame.MOUSEBUTTONUP:
                            if event.button == 1:  # Left button up
                                # Stop text selection dragging
                                self.dropup_edit_mouse_dragging = False
                                
                                # Stop Vision picker dragging
                                self.vision_picker_dragging = False
                                
                                # Stop Vision slider dragging
                                self.vision_dragging_slider = None
                                
                                self.scrubbing = False
                                self.wipe_dragging = False
                                self._mark_ui_dirty()

                            elif event.button == 3:  # Right button up - FIXED: Only handle rename if inside dropup
                                # Check if this was a right-click on a dropup item
                                if self.dropup_open and dropup_rect and dropup_rect.collidepoint(event.pos):
                                    # Find which item was right-clicked
                                    for gen_id, item_rect, idx in dropup_item_rects:
                                        if item_rect.collidepoint(event.pos):
                                            # Start rename on this item
                                            self._start_rename_generation(idx)
                                            break

                                self.dragging = False
                                self.drag_button = None
                                self._mark_ui_dirty()

                        if event.type == pygame.MOUSEMOTION:
                            # PRIORITY 0: Text selection dragging in edit mode
                            if self.dropup_edit_mouse_dragging and self.dropup_editing_index >= 0:
                                # Find the edit item rect
                                if dropup_item_rects:
                                    for gen_id, item_rect, idx in dropup_item_rects:
                                        if idx == self.dropup_editing_index and item_rect.collidepoint(event.pos):
                                            # Calculate character position from mouse X
                                            mouse_x = event.pos[0]
                                            text_x = item_rect.x + 5
                                            
                                            # Find closest character position
                                            font = pygame.font.SysFont('Arial', 15)
                                            drag_pos = 0
                                            for i in range(len(self.dropup_edit_text) + 1):
                                                text_before = self.dropup_edit_text[:i]
                                                width = font.render(text_before, True, (255, 255, 255)).get_width()
                                                if text_x + width > mouse_x:
                                                    break
                                                drag_pos = i
                                            
                                            # Update selection end
                                            self.dropup_edit_selection_end = drag_pos
                                            self.dropup_edit_cursor_pos = drag_pos
                                            self._mark_ui_dirty()
                                            break
                            
                            # PRIORITY 1: Color picker dragging (if active and dragging)
                            elif self.vision_picker_dragging and self.vision_picker_active:
                                # Update picker position
                                self.vision_picker_pos = list(event.pos)
                                self._mark_ui_dirty()
                            
                            # PRIORITY 1: Vision module slider dragging (only if NO dropups open!)
                            elif (self.vision_module_open and self.vision_dragging_slider and
                                not self.dropup_open and 
                                not self.color_space_dropup_open and 
                                not self.snapshot_dropup_open and 
                                not self.clearcache_dropup_open):
                                # Handle Vision slider dragging
                                # Get screen dimensions
                                vw, vh = screen.get_size()
                                slider_section_width = int(vw * 0.75)
                                slider_width = slider_section_width // 3
                                padding = 10
                                
                                # Get mouse X position
                                mouse_x = event.pos[0]
                                
                                if self.vision_dragging_slider == "gain":
                                    # GAIN slider (first third)
                                    bar_x_start = 10 + padding
                                    bar_x_end = 10 + slider_width - padding
                                    bar_width = bar_x_end - bar_x_start
                                    
                                    rel_x = mouse_x - bar_x_start
                                    normalized = max(0.0, min(1.0, rel_x / bar_width))
                                    
                                    min_val = -6.0
                                    max_val = 6.0
                                    new_value = min_val + normalized * (max_val - min_val)
                                    new_value = round(new_value * 10) / 10
                                    
                                    self.vision_gain = max(min_val, min(max_val, new_value))
                                    
                                elif self.vision_dragging_slider == "gamma":
                                    # GAMMA slider (second third)
                                    bar_x_start = 10 + slider_width + padding
                                    bar_x_end = 10 + slider_width * 2 - padding
                                    bar_width = bar_x_end - bar_x_start
                                    
                                    rel_x = mouse_x - bar_x_start
                                    normalized = max(0.0, min(1.0, rel_x / bar_width))
                                    
                                    new_value = 0.0 + normalized * 4.0  # 0 to 4
                                    new_value = round(new_value * 10) / 10
                                    
                                    self.vision_gamma = max(0.0, min(4.0, new_value))
                                    
                                elif self.vision_dragging_slider == "saturation":
                                    # SATURATION slider (third third)
                                    bar_x_start = 10 + slider_width * 2 + padding
                                    bar_x_end = 10 + slider_width * 3 - padding
                                    bar_width = bar_x_end - bar_x_start
                                    
                                    rel_x = mouse_x - bar_x_start
                                    normalized = max(0.0, min(1.0, rel_x / bar_width))
                                    
                                    new_value = 0.0 + normalized * 4.0  # 0 to 4
                                    new_value = round(new_value * 10) / 10
                                    
                                    self.vision_saturation = max(0.0, min(4.0, new_value))
                                
                                self._mark_ui_dirty()
                                self._mark_frame_cache_dirty()  # Need to reapply effects to frame
                                
                            elif self.dragging and self.drag_button == 3:
                                dx, dy = event.rel
                                self.viewer_offset[0] += dx
                                self.viewer_offset[1] += dy
                                self._mark_frame_cache_dirty()  # Frame cache needs update for offset change
                                self._mark_ui_dirty()
                            elif self.scrubbing and self.timeline_extended_rect:
                                rel_x = event.pos[0] - \
                                    self.timeline_extended_rect.left
                                progress = max(
                                    0, min(1, rel_x / self.timeline_extended_rect.width))
                                new_frame = int(
                                    progress * (total - 1)) if total > 1 else 0
                                self.current_frame = max(
                                    0, min(new_frame, total - 1))
                                self.counter_update_frame = self.current_frame
                                self.viewer_paused = True
                                self.force_counter_update = True
                                self._mark_ui_dirty()
                                # REMOVED: Don't invalidate surface cache on frame change!
                            elif self.wipe_dragging and self.current_video_rect:
                                # Get video rectangle
                                vid_x, vid_y, vid_w, vid_h = self.current_video_rect
                                
                                # Calculate new wipe position in video-relative coordinates
                                if vid_w > 0:
                                    # Clamp mouse position to video bounds
                                    mouse_x_clamped = max(vid_x, min(event.pos[0], vid_x + vid_w))
                                    new_position = (mouse_x_clamped - vid_x) / vid_w
                                    new_position = max(0.0, min(1.0, new_position))
                                    
                                    if abs(new_position - self.wipe_position) > 0.001:  # Only update if changed
                                        self.wipe_position = new_position
                                        self._mark_frame_cache_dirty()  # Frame cache needs update for wipe position change
                                        self._mark_ui_dirty()
                            elif self.color_space_dropup_open and color_space_dropup_rect and color_space_dropup_rect.collidepoint(event.pos):
                                menu_x, menu_y = color_space_dropup_rect.topleft
                                rel_y = event.pos[1] - menu_y - 5
                                item_index = rel_y // 28  # 28px item height

                                total_items = len(self.color_spaces)
                                visible_items = total_items  # Show all items
                                start_index = 0  # No scrolling
                                new_hover_index = start_index + item_index
                                if 0 <= new_hover_index < total_items and new_hover_index != self.color_space_hover_index:
                                    self.color_space_hover_index = new_hover_index
                                    self._mark_ui_dirty()
                            elif self.dropup_open and dropup_rect and dropup_rect.collidepoint(event.pos):
                                menu_x, menu_y = dropup_rect.topleft
                                rel_y = event.pos[1] - menu_y - 5
                                item_index = rel_y // 30

                                total_items = len(
                                    self._get_sorted_generations())
                                visible_items = min(total_items, 10)
                                start_index = max(
                                    0, min(self.dropup_scroll_offset, total_items - visible_items))
                                new_hover_index = start_index + item_index
                                if new_hover_index != self.dropup_hover_index:
                                    self.dropup_hover_index = new_hover_index
                                    self._mark_ui_dirty()
                            elif self.snapshot_dropup_open and snapshot_dropup_rect and snapshot_dropup_rect.collidepoint(event.pos):
                                # Snapshot dropup hover handled in _draw_snapshot_dropup
                                pass
                            elif self.clearcache_dropup_open and clearcache_dropup_rect and clearcache_dropup_rect.collidepoint(event.pos):
                                # ClearCache dropup hover handled in _draw_clearcache_dropup
                                pass
                            else:
                                if self.color_space_hover_index != -1:
                                    self.color_space_hover_index = -1
                                    self._mark_ui_dirty()
                                if self.dropup_hover_index != -1:
                                    self.dropup_hover_index = -1
                                    self._mark_ui_dirty()

                    # Handle continuous backspace when held down during editing
                    if self.dropup_editing_index >= 0 and self.dropup_backspace_held:
                        if current_time - self.dropup_backspace_start_time > 500:  # Initial delay
                            # Repeat backspace every 50ms after initial delay
                            if (current_time - self.dropup_backspace_start_time - 500) % 50 < 10:  # Rough timing
                                if self.dropup_edit_cursor_pos > 0:
                                    # Delete character before cursor
                                    self.dropup_edit_text = (
                                        self.dropup_edit_text[:self.dropup_edit_cursor_pos - 1] +
                                        self.dropup_edit_text[self.dropup_edit_cursor_pos:]
                                    )
                                    self.dropup_edit_cursor_pos -= 1
                                    self._mark_ui_dirty()

                    # Update hover state for toolbar buttons - OPTIMIZED: only when mouse moves
                    if mouse_moved and not self.fullscreen_mode and screen is not None:
                        mx, my = mouse_pos
                        old_button_hover = self.button_hover
                        self.button_hover = None

                        # Separate Y positions for different button groups
                        # For circular/arrow buttons (Group 1: Playback controls)
                        btn_y_circular = screen.get_height() - 35
                        # 5px LOWER for text buttons (Group 2: Fit/UI controls)
                        btn_y_text = screen.get_height() - 30

                        # Text buttons group (Group 2)
                        text_buttons = ["1:1", "Fit", "Width", "Height", "Fullscreen", "Reset",
                                        "Generations", "Snapshot", "ClearCache"]  # UPDATED: ClearCache is now a dropup

                        # Button position (centered in button area)
                        btn_x = 10

                        # UPDATED BUTTON ORDER: FPS replaces reset_markers
                        btns = ["in", "back", "fps_control", "forward", "out", "Pong", "1:1", "Fit", "Width", "Height",
                                "Fullscreen", "Reset", "Generations", "Snapshot", "ClearCache"]

                        for b in btns:
                            # Determine Y position based on button type
                            if b in text_buttons:
                                # Use lower position for text buttons (Group 2)
                                current_y = btn_y_text
                            else:
                                # Use original position for circular/arrow/Pong buttons (Group 1)
                                current_y = btn_y_circular

                            if b in ["in", "out", "fps_control"]:
                                btn_rect = pygame.Rect(
                                    btn_x, current_y, 30, 30)
                                if btn_rect.collidepoint((mx, my)) and not self.dropup_open and not self.snapshot_dropup_open and not self.clearcache_dropup_open and not self.fps_dropup_open:
                                    self.button_hover = b
                                btn_x += 40
                            elif b in ["back", "forward"]:
                                btn_rect = pygame.Rect(
                                    btn_x, current_y, 40, 30)
                                if btn_rect.collidepoint((mx, my)) and not self.dropup_open and not self.snapshot_dropup_open and not self.clearcache_dropup_open:
                                    self.button_hover = b
                                btn_x += 50
                            elif b == "Pong":
                                # Pong is now with playback controls, so it should be at circular height
                                # Same dimensions as arrow buttons
                                btn_rect = pygame.Rect(
                                    btn_x, current_y, 40, 30)
                                if btn_rect.collidepoint((mx, my)) and not self.dropup_open and not self.snapshot_dropup_open and not self.clearcache_dropup_open:
                                    self.button_hover = b
                                btn_x += 50  # Same spacing as arrow buttons
                            else:
                                # Use cached font
                                font_key = f"font_20"
                                if font_key not in self._ui_cache:
                                    self._ui_cache[font_key] = pygame.font.SysFont(
                                        None, 20)
                                font = self._ui_cache[font_key]

                                label = font.render(b, True, (220, 220, 220))
                                button_width = label.get_width() + 24
                                button_height = 30
                                full = pygame.Rect(
                                    btn_x, current_y, button_width, button_height)
                                if full.collidepoint((mx, my)) and not self.dropup_open and not self.snapshot_dropup_open and not self.clearcache_dropup_open:
                                    self.button_hover = b

                                # Add 10px spacing between all buttons in Group 2
                                btn_x += full.width + 10

                        # Mark UI dirty if hover state changed
                        if old_button_hover != self.button_hover:
                            self._mark_ui_dirty()

                    # ============================================================
                    # CRITICAL FPS TIMING: Calculate frame delay from user-controlled FPS
                    # This is the ONLY timing mechanism - no dual timing conflicts!
                    # delay_ms controls when frames advance (pure frame cycling)
                    # ============================================================
                    delay_ms = 1000.0 / max(0.1, self.user_fps)  # Prevent division by zero

                    # advance frame based on user FPS & pause with in/out marking support
                    now = pygame.time.get_ticks()
                    if now - last_ms >= delay_ms and not self.scrubbing:
                        last_ms = now
                        frames_since_last_log += 1

                        if not self.viewer_paused and total > 0:
                            # Get current playback bounds
                            if self.user_marks_active:
                                start_frame = self.user_in_point
                                end_frame = self.user_out_point
                            else:
                                start_frame = self.original_in_point
                                end_frame = self.original_out_point

                            if self.viewer_playback_mode == "Pause":
                                nxt = self.current_frame
                            elif self.viewer_playback_mode == "Forward":
                                nxt = self.current_frame + 1
                                if nxt > end_frame:
                                    nxt = start_frame
                                self.direction = 1
                            elif self.viewer_playback_mode == "Backward":
                                nxt = self.current_frame - 1
                                if nxt < start_frame:
                                    nxt = end_frame
                                self.direction = -1
                            elif self.viewer_playback_mode == "Ping-Pong":
                                nxt = self.current_frame + self.direction
                                if nxt > end_frame:
                                    nxt = end_frame - 1
                                    self.direction = -1
                                if nxt < start_frame:
                                    nxt = start_frame + 1
                                    self.direction = 1
                            else:
                                nxt = self.current_frame + 1
                                if nxt > end_frame:
                                    nxt = start_frame

                            # clamp with additional bounds checking
                            old_frame = self.current_frame
                            self.current_frame = max(
                                start_frame, min(nxt, end_frame))

                            # Mark frame cache dirty only if frame actually changed
                            if old_frame != self.current_frame:
                                self._mark_frame_cache_dirty()

                            # Note: counter_update_frame stays frozen during playback

                    # Handle showing_cache_clear_screen flag with automatic timeout
                    if self.showing_cache_clear_screen:
                        # Check if we need to start the timer
                        if self.cache_clear_start_time == 0:
                            self.cache_clear_start_time = current_time
                            self.log("Cache clear screen started")

                        # Check if 3 seconds have passed
                        elif current_time - self.cache_clear_start_time > 3000:  # 3000ms = 3 seconds
                            self.showing_cache_clear_screen = False
                            self.cache_clear_start_time = 0
                            self.log(
                                "Cache clear screen timeout - returning to normal")
                            self._mark_ui_dirty()
                            self._mark_frame_cache_dirty()
                            continue

                        # Display cache clear screen
                        self._display_logo_screen(screen, "pvm_cacheclear.jpg")
                        pygame.display.flip()
                        clock.tick(current_target_fps)
                        continue

                    # Handle processing_new_content flag - show wait screen during MP4 creation
                    if self.processing_new_content:
                        # We're processing new content (writing MP4), show wait screen
                        self._display_logo_screen(screen, "pvm_wait.jpg")
                        pygame.display.flip()
                        clock.tick(current_target_fps)
                        continue

                    # Handle showing_wait_screen flag
                    if self.showing_wait_screen:
                        # Check if we should exit wait screen
                        if not self.pending_content and self.cached_original_frames is not None:
                            # Content is loaded, exit wait screen immediately
                            self.showing_wait_screen = False
                            self._mark_ui_dirty()
                            self._mark_frame_cache_dirty()
                            # Don't show wait screen, skip to next iteration to render content
                            continue
                        
                        # Still waiting - show wait screen
                        self._display_logo_screen(screen, "pvm_wait.jpg")
                        pygame.display.flip()
                        clock.tick(60)
                        continue

                    # Performance logging - DISABLED (was too spammy)
                    # if time.time() - last_perf_log > 5.0 and frames_since_last_log > 0:
                    #     avg_fps = frames_since_last_log / 5.0
                    #     if avg_fps < current_target_fps * 0.8:
                    #         self.log(f"Performance: {avg_fps:.1f} fps (target: {current_target_fps:.1f})")
                    #     frames_since_last_log = 0
                    #     last_perf_log = time.time()

                    # drawing - OPTIMIZED RENDERING PIPELINE WITH FIXED WIPE SYSTEM
                    if screen is not None:
                        screen.fill((0, 0, 0))
                        vw, vh = screen.get_size()

                        # 1. ALWAYS RENDER THE FRAME FIRST (never skip this)
                        if total > 0 and self.cached_original_frames is not None:
                            # Get or create frame surface with thread-safe locking
                            frame_surface, frame_pos = self._get_or_create_frame_surface(
                                screen, self.current_frame, total)
                            if frame_surface:
                                # Apply Vision module effects if open (gain/gamma/saturation)
                                if self.vision_module_open and (self.vision_gain != 0.0 or self.vision_gamma != 1.0 or self.vision_saturation != 1.0):
                                    # Create a copy to apply effects without modifying cached surface
                                    modified_surface = frame_surface.copy()
                                    
                                    # Get pixel array
                                    pixels = pygame.surfarray.pixels3d(modified_surface)
                                    
                                    # 1. Apply GAIN (exposure stops formula)
                                    if self.vision_gain != 0.0:
                                        # Convert stops to multiplier: 2^stops
                                        # -6 stops = 2^-6 = 0.015625 (very dark)
                                        #  0 stops = 2^0 = 1.0 (neutral)
                                        # +6 stops = 2^6 = 64.0 (very bright)
                                        actual_gain = np.power(2.0, self.vision_gain)
                                        pixels[:] = np.clip(pixels * actual_gain, 0, 255)
                                    
                                    # 2. Apply GAMMA (power curve) - OPTIMIZED WITH LUT
                                    if self.vision_gamma != 1.0:
                                        # Clamp gamma to avoid division by zero
                                        safe_gamma = max(0.01, self.vision_gamma)
                                        
                                        # Build or use cached LUT (Lookup Table)
                                        # Only rebuild if gamma value changed significantly (tolerance for speed)
                                        if (self.vision_gamma_lut is None or 
                                            self.vision_gamma_lut_value is None or
                                            abs(safe_gamma - self.vision_gamma_lut_value) > 0.01):  # Tolerance for speed
                                            # Build LUT: pre-calculate all 256 possible gamma corrections
                                            # This happens only when gamma changes significantly
                                            input_values = np.arange(256, dtype=np.float32) / 255.0
                                            corrected_values = np.power(input_values, 1.0 / safe_gamma)
                                            self.vision_gamma_lut = (corrected_values * 255).astype(np.uint8)
                                            self.vision_gamma_lut_value = safe_gamma
                                        
                                        # Apply LUT: super fast array indexing (no power operations!)
                                        # Instead of: pixels^(1/gamma) for each pixel
                                        # We do: lookup[pixel_value] for each pixel
                                        pixels[:] = self.vision_gamma_lut[pixels]
                                    
                                    # 3. Apply SATURATION (lerp between grayscale and color) - ULTRA OPTIMIZED
                                    if self.vision_saturation != 1.0:
                                        # ULTRA FAST: Use integer weights (avoid float multiplication)
                                        # Y = 0.299*R + 0.587*G + 0.114*B
                                        # Scaled to integers: Y = (76*R + 150*G + 29*B) >> 8
                                        gray = ((pixels[:,:,0].astype(np.uint16) * 76 + 
                                                pixels[:,:,1].astype(np.uint16) * 150 + 
                                                pixels[:,:,2].astype(np.uint16) * 29) >> 8).astype(np.uint8)
                                        
                                        # OPTIMIZED: Direct broadcasting without np.newaxis
                                        gray_3d = gray[:, :, np.newaxis]
                                        
                                        # Lerp with saturation
                                        pixels[:] = np.clip(
                                            gray_3d + ((pixels.astype(np.int16) - gray_3d) * self.vision_saturation),
                                            0, 255
                                        ).astype(np.uint8)
                                    
                                    # Convert to uint8
                                    pixels[:] = pixels.astype(np.uint8)
                                    
                                    # Delete reference to unlock surface
                                    del pixels
                                    
                                    frame_surface = modified_surface
                                
                                # Apply flip/flop transform if active
                                if self.vision_flip_mode > 0:
                                    flip_h = (self.vision_flip_mode == 1 or self.vision_flip_mode == 3)
                                    flip_v = (self.vision_flip_mode == 2 or self.vision_flip_mode == 3)
                                    frame_surface = pygame.transform.flip(frame_surface, flip_h, flip_v)
                                
                                # Blit the frame surface (with or without gain/flip applied)
                                pos_x, pos_y, draw_w, draw_h = frame_pos
                                screen.blit(frame_surface, (pos_x, pos_y))
                                
                                # Draw wipe overlay if active (FIXED: Now works with all fit modes)
                                if self.wipe_active and self.current_video_rect:
                                    self._draw_wipe_overlay(screen, frame_surface, frame_pos)
                                
                                # Draw side-by-side overlay if active (exclusive with wipe)
                                if self.sbs_active and self.current_video_rect:
                                    self._draw_sbs_overlay(screen, frame_surface, frame_pos)
                                
                                # Draw zebra pattern if active (clipping indicator)
                                if self.vision_zebra_active:
                                    self._draw_zebra_pattern(screen, frame_surface, frame_pos)
                                
                                # Draw broadcast guides if active (OR persisting when Vision closed)
                                if self.vision_guides_active or (self.vision_guides_persist and not self.vision_module_open):
                                    self._draw_guides_overlay(screen, frame_pos)
                                
                                # Draw aspect mask if active (OR persisting when Vision closed)
                                if self.vision_mask_index > 0 or (self.vision_mask_persist and not self.vision_module_open):
                                    self._draw_aspect_mask(screen, frame_pos)
                                
                                # Draw color picker if active
                                if self.vision_picker_active:
                                    self._draw_color_picker(screen, frame_pos)
                                
                                # Draw Vision scopes if active (histogram, vector, waveform)
                                if self.vision_module_open and self.vision_active_scope == "histogram":
                                    self._draw_histogram_scope(screen, frame_surface)
                                elif self.vision_module_open and self.vision_active_scope == "vector":
                                    self._draw_vectorscope(screen, frame_surface)
                                elif self.vision_module_open and self.vision_active_scope == "waveform":
                                    self._draw_waveform_rgb(screen, frame_surface)

                        # 2. Only draw UI if not in fullscreen mode
                        if not self.fullscreen_mode and screen is not None:
                            # Recreate toolbar surface if cache was cleared (e.g., by hover state change)
                            if self._cached_toolbar_surface is None:
                                self._cached_toolbar_surface = self._render_toolbar_surface(screen)
                            
                            # Blit pre-rendered toolbar background
                            if self._cached_toolbar_surface:
                                screen.blit(self._cached_toolbar_surface,
                                            (0, vh - self.toolbar_height))

                            # Draw timeline scrubber (TOP of toolbar) - FIXED VERSION WITH RED INDICATOR ON TOP
                            self._draw_timeline_scrubber(
                                screen, self.current_frame, total, mouse_pos)
                            
                            # Draw Vision module (covers timeline when active)
                            if self.vision_module_open:
                                self._draw_vision_module(screen, mouse_pos)

                            # Draw buttons with SEPARATE Y positions for different button groups
                            btn_x = 10
                            # Original position for circular/arrow buttons (Group 1)
                            btn_y_circular = vh - 35
                            # 5px LOWER for text buttons (Group 2: Fit/UI controls)
                            btn_y_text = vh - 30

                            btn_rects = []
                            generations_button_rect = None
                            gamuts_button_rect = None  # Renamed from gamuts_button_rect
                            snapshot_button_rect = None
                            clearcache_button_rect = None
                            fps_button_rect = None  # NEW: FPS button rect

                            # Text buttons group (Group 2)
                            text_buttons = ["1:1", "Fit", "Width", "Height", "Fullscreen", "Reset",
                                            "Generations", "Snapshot", "ClearCache", "Vision"]  # Vision module toggle

                            # Draw all buttons - FPS REPLACES RESET_MARKERS
                            btns_draw_order = ["in", "back", "fps_control", "forward", "out", "Pong", "1:1", "Fit", "Width", "Height", 
                                               "Fullscreen", "Reset", "Generations", "Snapshot", "ClearCache", "Vision"]

                            # Check if snapshot button should be active (recent snapshot or saving in progress)
                            is_snapshot_active = (self.snapshot_saving or 
                                                 (self.last_snapshot_time > 0 and
                                                  current_time - self.last_snapshot_time < 1000))

                            for b in btns_draw_order:
                                # Determine Y position based on button type
                                if b in text_buttons:
                                    # Use lower position for text buttons and gamuts (Group 2)
                                    current_y = btn_y_text
                                else:
                                    # Use original position for circular/arrow/Pong buttons (Group 1)
                                    current_y = btn_y_circular

                                if b == "in":
                                    is_hover = (b == self.button_hover)
                                    btn_rect = self._draw_in_button(
                                        screen, (btn_x, current_y), self.user_marks_active, is_hover)
                                    btn_rects.append((b, btn_rect))
                                    btn_x += 40
                                elif b == "out":
                                    is_hover = (b == self.button_hover)
                                    btn_rect = self._draw_out_button(
                                        screen, (btn_x, current_y), self.user_marks_active, is_hover)
                                    btn_rects.append((b, btn_rect))
                                    btn_x += 40
                                elif b == "fps_control":
                                    is_hover = (b == self.button_hover)
                                    btn_rect = self._draw_fps_button(
                                        screen, (btn_x, current_y), self.fps_dropup_open, is_hover)
                                    btn_rects.append((b, btn_rect))
                                    fps_button_rect = btn_rect  # Store for dropup positioning
                                    btn_x += 40
                                elif b in ["back", "forward"]:
                                    is_active = (
                                        (b == "back" and self.viewer_playback_mode == "Backward") or
                                        (b == "forward" and self.viewer_playback_mode == "Forward")
                                    )
                                    is_hover = (b == self.button_hover)

                                    btn_rect = self._draw_arrow_button(
                                        screen, b, (btn_x, current_y), is_active, is_hover)
                                    btn_rects.append((b, btn_rect))
                                    btn_x += 50
                                elif b == "Pong":
                                    # Pong is now with playback controls, so it should be at circular height
                                    is_active = (
                                        self.viewer_playback_mode == "Ping-Pong")
                                    is_hover = (b == self.button_hover)

                                    # Draw Pong button with green outline when active
                                    btn_rect = self._draw_button(
                                        screen, b, (btn_x, current_y), is_active, is_hover, (100, 220, 100))
                                    btn_rects.append((b, btn_rect))
                                    btn_x += btn_rect.width + 10  # Standard spacing for Group 1
                                else:
                                    # For Group 2 buttons (Fit/UI controls)
                                    is_active = (
                                        (b.lower() == self.viewer_fit_mode) or
                                        (b == self.viewer_playback_mode) or
                                        (b == "Generations" and self.dropup_open) or
                                        (b == "Fullscreen" and self.fullscreen_mode) or
                                        (b == "Snapshot" and (self.snapshot_dropup_open or is_snapshot_active)) or  # Active when dropup open OR snapshot taken
                                        (b == "ClearCache" and self.clearcache_dropup_open) or
                                        (b == "Vision" and self.vision_module_open)  # Active when Vision module is open
                                    )
                                    is_hover = (b == self.button_hover)

                                    color_override = None
                                    if b == "ClearCache":
                                        color_override = (220, 140, 60)  # Orange outline
                                    elif b == "Generations":
                                        color_override = (220, 220, 60)  # Yellow outline
                                    elif b == "Snapshot":
                                        # Dark green when dropup open, cyan when snapshot taken, grey otherwise
                                        if self.snapshot_dropup_open:
                                            color_override = (60, 140, 80)  # Dark green when dropup open
                                        elif is_snapshot_active:
                                            color_override = (60, 200, 220)  # Cyan flash when snapshot taken
                                        # else: None = grey outline (default)
                                    elif b == "Vision":
                                        color_override = (60, 140, 220)  # Blue outline

                                    btn_rect = self._draw_button(
                                        screen, b, (btn_x, current_y), is_active, is_hover, color_override)
                                    btn_rects.append((b, btn_rect))

                                    if b == "Generations":
                                        generations_button_rect = btn_rect
                                    elif b == "Snapshot":
                                        snapshot_button_rect = btn_rect
                                    elif b == "ClearCache":
                                        clearcache_button_rect = btn_rect

                                    # Add 10px spacing between all buttons in Group 2
                                    btn_x += btn_rect.width + 10

                            # Draw ClearCache dropup menu
                            clearcache_dropup_result = self._draw_clearcache_dropup(
                                screen, mouse_pos, clearcache_button_rect)
                            if clearcache_dropup_result:
                                clearcache_dropup_rect, clearcache_item_rects = clearcache_dropup_result
                            else:
                                clearcache_dropup_rect = None
                                clearcache_item_rects = None

                            # Draw snapshot dropup menu
                            snapshot_dropup_result = self._draw_snapshot_dropup(
                                screen, mouse_pos, snapshot_button_rect)
                            if snapshot_dropup_result:
                                snapshot_dropup_rect, snapshot_item_rects = snapshot_dropup_result
                            else:
                                snapshot_dropup_rect = None
                                snapshot_item_rects = None

                            # Draw FPS dropup menu  
                            fps_dropup_result = self._draw_fps_dropup(
                                screen, mouse_pos, fps_button_rect)
                            if fps_dropup_result:
                                fps_dropup_rect, fps_item_rects = fps_dropup_result
                            else:
                                fps_dropup_rect = None
                                fps_item_rects = None

                            # Color space dropup removed (locked to sRGB)
                            color_space_dropup_rect = None
                            color_space_item_rects = None

                            # Draw generations dropup menu
                            dropup_result = self._draw_dropup_menu(
                                screen, mouse_pos, generations_button_rect)
                            if dropup_result:
                                dropup_rect, dropup_item_rects, wipe_button_rects = dropup_result
                            else:
                                dropup_rect = None
                                dropup_item_rects = None
                                wipe_button_rects = None

                            # Draw frame counter, markers, and resolution boxes - OPTIMIZED
                            # Check if we need to update counter surfaces
                            current_state_hash = self._calculate_ui_state_hash()
                            needs_counter_update = (
                                self._ui_cache_dirty or
                                self._cached_counter_surfaces is None or
                                self._last_ui_state_hash != current_state_hash or
                                self.force_counter_update
                            )

                            if needs_counter_update:
                                self._cached_counter_surfaces = self._render_counter_surfaces(
                                    self.current_frame, total, self.user_fps, vw, vh
                                )
                                self.force_counter_update = False

                            if self._cached_counter_surfaces:
                                self._draw_frame_counter_fast(
                                    screen, self._cached_counter_surfaces)
                            
                            # Draw camera iris animation (on top of everything)
                            if self.iris_animation_active:
                                self._draw_camera_iris_animation(screen)

                        # ============================================================
                        # TOOLBAR BUTTON CLICK HANDLING WITH DEBOUNCING
                        # Uses get_pressed() to check current state, but debounces
                        # by only triggering on the transition from not-pressed to pressed
                        # This prevents multiple triggers when button is held down
                        # ============================================================
                        mouse_pressed = pygame.mouse.get_pressed()[0]
                        
                        # Only trigger on button DOWN (transition from False to True)
                        mouse_just_pressed = mouse_pressed and not self._prev_mouse_pressed
                        self._prev_mouse_pressed = mouse_pressed  # Update for next frame
                        
                        if mouse_just_pressed and not self.scrubbing and not self.fullscreen_mode:
                            mx, my = mouse_pos
                            button_clicked = False
                            for (b, rect) in btn_rects:
                                # Special handling for circular FPS button
                                if b == "fps_control":
                                    # Check if click is inside the circle (radius 15)
                                    center_x = rect.centerx
                                    center_y = rect.centery
                                    dist = ((mx - center_x)**2 + (my - center_y)**2)**0.5
                                    if dist <= 15:  # Radius of FPS button
                                        button_clicked = True
                                    else:
                                        continue  # Not inside circle, skip
                                elif rect.collidepoint((mx, my)):
                                    button_clicked = True
                                else:
                                    continue  # Not in rect, skip
                                
                                # Process button click
                                if button_clicked:
                                    if b == "in":
                                        if total > 0:
                                            # TOGGLE: If marks already active and user clicks IN again, disable
                                            if self.user_marks_active and self.current_frame == self.user_in_point:
                                                self.user_marks_active = False
                                                self.log("Mark IN/OUT disabled")
                                            else:
                                                # Set new IN point
                                                self.user_in_point = self.current_frame
                                                if self.user_in_point > self.user_out_point:
                                                    self.user_out_point = self.user_in_point
                                                self.user_marks_active = True
                                                self.log(
                                                    f"Mark IN set to frame {self.user_in_point}")
                                            self.active_button = "in"
                                            self._mark_ui_dirty()
                                    elif b == "out":
                                        if total > 0:
                                            # TOGGLE: If marks already active and user clicks OUT again, disable
                                            if self.user_marks_active and self.current_frame == self.user_out_point:
                                                self.user_marks_active = False
                                                self.log("Mark IN/OUT disabled")
                                            else:
                                                # Set new OUT point
                                                self.user_out_point = self.current_frame
                                                if self.user_out_point < self.user_in_point:
                                                    self.user_in_point = self.user_out_point
                                                self.user_marks_active = True
                                                self.log(
                                                    f"Mark OUT set to frame {self.user_out_point}")
                                            self.active_button = "out"
                                            self._mark_ui_dirty()
                                    elif b == "fps_control":
                                        # Toggle FPS dropup
                                        self.fps_dropup_open = not self.fps_dropup_open
                                        if self.fps_dropup_open:
                                            # Close other dropups
                                            self.dropup_open = False
                                            # Reset confirmation states when closing dropups
                                            self.dropup_delete_confirm_index = -1
                                            self.dropup_delete_confirm_gen_id = None
                                            self.color_space_dropup_open = False
                                            self.snapshot_dropup_open = False
                                            self.snapshot_clear_confirm = False  # Reset Snapshot confirmation
                                            self.clearcache_dropup_open = False
                                            self.clearcache_clear_confirm = False  # Reset ClearCache confirmation
                                        self.log(f"FPS dropup {'opened' if self.fps_dropup_open else 'closed'}")
                                        self._mark_ui_dirty()
                                    elif b == "back":
                                        self.viewer_playback_mode = "Backward"
                                        self.viewer_paused = False
                                        self.counter_update_frame = self.current_frame
                                        self.active_button = "back"
                                        self._mark_ui_dirty()
                                    elif b == "forward":
                                        self.viewer_playback_mode = "Forward"
                                        self.viewer_paused = False
                                        self.counter_update_frame = self.current_frame
                                        self.active_button = "forward"
                                        self._mark_ui_dirty()
                                    elif b == "Pong":
                                        self.viewer_playback_mode = "Ping-Pong"
                                        self.viewer_paused = False
                                        self.counter_update_frame = self.current_frame
                                        self.active_button = "Pong"
                                        self._mark_ui_dirty()
                                    elif b == "Fit":
                                        self.viewer_fit_mode = "fit"
                                        self.viewer_zoom = 1.0
                                        self.viewer_offset = [0, 0]
                                        self.active_button = "Fit"
                                        self._mark_ui_dirty()
                                        self._mark_frame_cache_dirty()
                                    elif b == "Width":
                                        self.viewer_fit_mode = "width"
                                        self.viewer_zoom = 1.0
                                        self.viewer_offset = [0, 0]
                                        self.active_button = "Width"
                                        self._mark_ui_dirty()
                                        self._mark_frame_cache_dirty()
                                    elif b == "Height":
                                        self.viewer_fit_mode = "height"
                                        self.viewer_zoom = 1.0
                                        self.viewer_offset = [0, 0]
                                        self.active_button = "Height"
                                        self._mark_ui_dirty()
                                        self._mark_frame_cache_dirty()
                                    elif b == "1:1":
                                        self.viewer_fit_mode = "1:1"
                                        self.viewer_zoom = 1.0
                                        self.viewer_offset = [0, 0]
                                        self.active_button = "1:1"
                                        self._mark_ui_dirty()
                                        self._mark_frame_cache_dirty()
                                    elif b == "Fullscreen":
                                        if screen is not None:
                                            screen = self._enter_fullscreen(
                                                screen)
                                            self._mark_ui_dirty()
                                            self._mark_frame_cache_dirty()
                                        self.active_button = "Fullscreen"
                                    elif b == "Reset":
                                        self.viewer_zoom = 1.0
                                        self.viewer_offset = [0, 0]
                                        self.active_button = "Reset"
                                        self._mark_ui_dirty()
                                        self._mark_frame_cache_dirty()
                                    elif b == "Generations":
                                        self.dropup_open = not self.dropup_open
                                        self.dropup_scroll_offset = 0
                                        if self.dropup_open:
                                            # Close other dropups if open
                                            self.color_space_dropup_open = False
                                            self.snapshot_dropup_open = False
                                            self.snapshot_clear_confirm = False  # Reset Snapshot confirmation
                                            self.clearcache_dropup_open = False
                                            self.clearcache_clear_confirm = False  # Reset ClearCache confirmation
                                            # Stop any Vision slider dragging
                                            self.vision_dragging_slider = None

                                            sorted_generations = self._get_sorted_generations()
                                            if sorted_generations and self.current_generation_id:
                                                for i, (gen_id, _) in enumerate(sorted_generations):
                                                    if gen_id == self.current_generation_id:
                                                        self.dropup_selected_index = i
                                                        break
                                                else:
                                                    self.dropup_selected_index = 0
                                            elif sorted_generations:
                                                self.dropup_selected_index = 0
                                            else:
                                                self.dropup_selected_index = -1
                                        else:
                                            self.dropup_selected_index = -1
                                            self.dropup_delete_confirm_index = -1  # Cancel delete confirmation when closing
                                        self.active_button = "Generations"
                                        self._mark_ui_dirty()
                                    elif b == "Snapshot":
                                        self.snapshot_dropup_open = not self.snapshot_dropup_open
                                        if self.snapshot_dropup_open:
                                            # Close other dropups if open
                                            self.dropup_open = False
                                            self.dropup_selected_index = -1
                                            # Reset confirmation states
                                            self.dropup_delete_confirm_index = -1
                                            self.dropup_delete_confirm_gen_id = None
                                            self.color_space_dropup_open = False
                                            self.clearcache_dropup_open = False
                                            self.clearcache_clear_confirm = False  # Reset ClearCache confirmation
                                            # Stop any Vision slider dragging
                                            self.vision_dragging_slider = None
                                        self.active_button = "Snapshot"
                                        self._mark_ui_dirty()
                                    elif b == "ClearCache":
                                        # ClearCache is now a dropup
                                        self.clearcache_dropup_open = not self.clearcache_dropup_open
                                        if self.clearcache_dropup_open:
                                            # Close other dropups if open
                                            self.dropup_open = False
                                            self.dropup_selected_index = -1
                                            # Reset Generations delete confirmation
                                            self.dropup_delete_confirm_index = -1
                                            self.dropup_delete_confirm_gen_id = None
                                            self.color_space_dropup_open = False
                                            self.snapshot_dropup_open = False
                                            self.snapshot_clear_confirm = False  # Reset Snapshot confirmation
                                            # Stop any Vision slider dragging
                                            self.vision_dragging_slider = None
                                        self.active_button = "ClearCache"
                                        self._mark_ui_dirty()
                                    elif b == "Vision":
                                        # Toggle Vision module
                                        self.vision_module_open = not self.vision_module_open
                                        if self.vision_module_open:
                                            # Close all dropups when Vision opens
                                            self.dropup_open = False
                                            self.dropup_selected_index = -1
                                            self.color_space_dropup_open = False
                                            self.snapshot_dropup_open = False
                                            self.clearcache_dropup_open = False
                                        else:
                                            # When closing Vision, set persistence flags for guides/masks
                                            self.vision_guides_persist = self.vision_guides_active
                                            self.vision_mask_persist = (self.vision_mask_index > 0)
                                        self.active_button = "Vision"
                                        self._mark_ui_dirty()
                                    # Small delay to prevent rapid clicking
                                    pygame.time.wait(120)

                        # Update display and clear dirty flags
                        if screen is not None:
                            pygame.display.flip()

                        # Clear dirty flags after rendering
                        self._ui_cache_dirty = False
                        # Note: frame_cache_dirty is cleared inside _get_or_create_frame_surface

                # ============================================================
                # CRITICAL: Clock tick MUST be at while loop level
                # If inside 'if screen' block, it gets skipped on continues!
                # This caused 2x slower playback bug
                # ============================================================
                clock.tick(60)

                # If we get here and new content is ready, break to outer loop to handle it
                if self.new_content_ready.is_set():
                    continue

        except Exception as e:
            self.log("Viewer thread error:", e)
            traceback.print_exc()
        finally:
            try:
                if pygame.get_init():
                    pygame.quit()
            except Exception:
                pass
            self.running = False

    # =========== RANDOM WELCOME SCREEN SELECTION ===========
    def _get_random_hello_image(self):
        """
        Randomly select a welcome screen image from pvm_hello_01.jpg to pvm_hello_99.jpg
        Falls back to pvm_hello.jpg if numbered files don't exist
        Returns full path to selected image or None if no images found
        """
        try:
            import glob
            # Get all numbered hello images (01-99)
            pattern = os.path.join(LOGOS_DIR, "pvm_hello_*.jpg")
            files = glob.glob(pattern)
            
            if not files:
                # Fallback to single pvm_hello.jpg if it exists
                fallback = os.path.join(LOGOS_DIR, "pvm_hello.jpg")
                if os.path.exists(fallback):
                    self.log("No numbered hello images found, using fallback")
                    return fallback
                return None
            
            # Use system time as seed for randomness
            random.seed()  # Re-seed with system time
            selected = random.choice(files)
            
            # Log which image was selected (for debugging)
            filename = os.path.basename(selected)
            self.log(f"Random welcome screen selected: {filename}")
            
            return selected
            
        except Exception as e:
            self.log(f"Error selecting random welcome image: {e}")
            # Fallback to default if error
            fallback = os.path.join(LOGOS_DIR, "pvm_hello.jpg")
            if os.path.exists(fallback):
                return fallback
            return None

    def __del__(self):
        """Cleanup when instance is destroyed"""
        self._stop_thread()


# Node registration
NODE_CLASS_MAPPINGS = {
    "PreviewVideoMonitorPro": PreviewVideoMonitorPro
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewVideoMonitorPro": "🖥️ Preview Video Monitor Pro"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
