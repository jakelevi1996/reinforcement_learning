import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(SOURCE_DIR)

np.set_printoptions(
    precision=3,
    linewidth=10000,
    suppress=True,
    threshold=10000,
)
