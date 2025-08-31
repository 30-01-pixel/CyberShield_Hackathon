import os
from pathlib import Path
import multiprocessing

# Directory paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
SPLITS_DIR = DATA_DIR / "splits"
ANALYZED_DIR = DATA_DIR / "analyzed"
LOGS_DIR = BASE_DIR / "logs"

# Model directories
MODELS_DIR = BASE_DIR / "models"

# Hashtag tracking directories
HASHTAGS_DIR = DATA_DIR / "hashtags"

# Processing settings
DEFAULT_NUM_SPLITS = 5
MIN_ROWS_PER_SPLIT = 10

# API settings
API_HOST = "127.0.0.1"
API_PORT = 8000

# Analysis fields
IMAGE_URL_FIELD = "media_url"
DESCRIPTION_FIELD = "content_text"

# Analysis control
SKIP_IMAGE_ANALYSIS_DEFAULT = False  # Set to True to skip image analysis by default

# Output fields for analysis results
IMG_ANALYSIS_FIELDS = ["img_is_anti_india", "img_confidence", "img_reasoning"]
TEXT_ANALYSIS_FIELDS = ["text_is_anti_india", "text_confidence", "text_reasoning"]

# Parallel processing settings
MAX_WORKERS_MULTIPROCESSING = min(8, multiprocessing.cpu_count())  # Cap at 8 to avoid overwhelming system
MAX_WORKERS_THREADING = min(4, multiprocessing.cpu_count())  # Threading for I/O bound tasks
CHUNK_SIZE_FOR_PARALLEL = 100  # Process data in chunks for better memory management

# MLX Model settings
MLX_TEXT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
MLX_VISION_MODEL = "mlx-community/Qwen2.5-VL-72B-Instruct-4bit"

# MLX Inference parameters
MLX_TEMPERATURE = 0.1
MLX_MAX_TOKENS = 512
MLX_TEXT_MAX_TOKENS = 200  # Shorter for analysis tasks

# Model caching and memory settings
MLX_CACHE_DIR = MODELS_DIR / "cache"
MLX_WARMUP_ENABLED = True  # Warm up models on startup
MLX_AUTO_CLEANUP = True    # Automatically cleanup models when needed

# Sparrow VLM settings (when integrated)
SPARROW_MODEL_PATH = MODELS_DIR / "sparrow"
SPARROW_BACKEND = "mlx"  # Use MLX backend for Sparrow

# Image processing settings
IMAGE_MAX_SIZE = (1024, 1024)  # Max image dimensions for processing
IMAGE_DOWNLOAD_TIMEOUT = 30    # Timeout for downloading remote images
IMAGE_TEMP_CLEANUP = True      # Clean up temporary downloaded images

# Hashtag tracking settings
TRACK_HASHTAG_USAGE = True     # Enable hashtag extraction and logging
HASHTAG_FIELD = "hashtags"     # Column name containing hashtags
HASHTAG_CSV_COLUMNS = [
    "timestamp", "dataset_id", "split_number", "hashtag", 
    "row_id", "username", "platform", "language", 
    "text_is_anti_india", "text_confidence"
]

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"