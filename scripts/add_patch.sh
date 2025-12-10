#!/bin/bash

# Fix OmniAvatar repository bug - Replace files
echo "=========================================="
echo "Starting to update code for original repository..."
echo "=========================================="

echo "Replacing model_config.py ..."
cp -f utils/Omniavatarpatch/model_config.py OmniAvatar/OmniAvatar/configs/model_config.py
echo "✓ model_config.py replaced successfully"

echo "Replacing wan_video_dit.py ..."
cp -f utils/Omniavatarpatch/wan_video_dit.py OmniAvatar/OmniAvatar/models/wan_video_dit.py
echo "✓ wan_video_dit.py replaced successfully"

echo "Replacing wan_video.py ..."
cp -f utils/Omniavatarpatch/wan_video.py OmniAvatar/OmniAvatar/wan_video.py
echo "✓ wan_video.py replaced successfully"

echo "Replacing wav2vec.py ..."
cp -f utils/Omniavatarpatch/wav2vec.py OmniAvatar/OmniAvatar/models/wav2vec.py
echo "✓ wav2vec.py replaced successfully"

echo "Replacing args_config.py ..."
cp -f utils/Omniavatarpatch/args_config.py OmniAvatar/OmniAvatar/utils/args_config.py
echo "✓ args_config.py replaced successfully"

echo "Replacing flow_match.py ..."
cp -f utils/Omniavatarpatch/flow_match.py OmniAvatar/OmniAvatar/schedulers/flow_match.py
echo "✓ flow_match.py replaced successfully"

echo "=========================================="
echo "All files replaced successfully!"
echo "=========================================="
