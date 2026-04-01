"""pytest configuration — adds project root to sys.path so tests can import src.*"""

import sys
import os

# Allow "from src.xxx import ..." when pytest runs from openmath-agents/
sys.path.insert(0, os.path.dirname(__file__))
