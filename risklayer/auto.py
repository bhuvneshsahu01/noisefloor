"""
Drop-in auto-instrumentation module.
Just `import risklayer.auto` at the top of your script to automatically protect all LLM calls.
"""
from risklayer.instrumentation import patch_all

# Automatically patch upon import
patch_all()
