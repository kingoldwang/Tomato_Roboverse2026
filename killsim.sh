#!/bin/bash
echo "Killing sim processes..."

pkill -9 -f "gz sim" 2>/dev/null
pkill -9 -f "px4" 2>/dev/null
pkill -9 -f "save_photo" 2>/dev/null
pkill -9 -f "keyboardcontrol" 2>/dev/null
pkill -9 -f "MicroXRCEAgent" 2>/dev/null

sleep 1

remaining=$(ps aux | grep -E "gz|px4" | grep -v grep)
if [ -z "$remaining" ]; then
    echo "Done. All clear."
else
    echo "Warning: some processes still running:"
    echo "$remaining"
fi
