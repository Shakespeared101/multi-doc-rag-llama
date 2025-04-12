# fix_llama.py
import sys

try:
    # Import the module normally.
    from llama_index.readers.file.video_audio import VideoAudioReader
except ImportError:
    # If the module isn't available, do nothing.
    VideoAudioReader = None

# If the VideoAudioReader is imported and VideoAudioParser is missing, add an alias.
module_name = "llama_index.readers.file.video_audio"
if module_name in sys.modules and VideoAudioReader is not None:
    mod = sys.modules[module_name]
    if not hasattr(mod, "VideoAudioParser"):
        setattr(mod, "VideoAudioParser", VideoAudioReader)
