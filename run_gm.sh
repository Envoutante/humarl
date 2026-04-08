#!/usr/bin/env python3
"""Wrapper to run GPU manager with proper output logging."""
import subprocess
import sys
import os

# Create a custom stdout that flushes immediately
class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)

sys.stdout = Unbuffered(sys.stdout)
sys.stderr = Unbuffered(sys.stderr)

# Now run the actual GPU manager
os.execv(sys.executable, [sys.executable, '-u', '-c', '''
import sys
sys.path.insert(0, "src")
from gpu_manager import main
main()
'''])
