import shutil
import os

def clear_index(index_dir="index"):
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)

clear_index()