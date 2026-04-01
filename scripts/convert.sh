#!/bin/bash

# 1. Check if the user provided exactly 3 arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_video.mp4> <subtitle_file.srt> <output_video.mp4>"
    echo "Example: $0 5_emo_S.mp4 5_emo_S_smiles.srt output_5_emo_S_smiles.mp4"
    exit 1
fi

# Assign arguments to readable variables
INPUT_VIDEO="$1"
SUBTITLE_FILE="$2"
OUTPUT_VIDEO="$3"

# 2. Verify that the input video exists
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Cannot find the video file '$INPUT_VIDEO'."
    exit 1
fi

# 3. Verify that the subtitle file exists
if [ ! -f "$SUBTITLE_FILE" ]; then
    echo "Error: Cannot find the subtitle file '$SUBTITLE_FILE'."
    exit 1
fi

# 4. Execute the FFmpeg command
echo "Burning subtitles into $INPUT_VIDEO..."
ffmpeg -i "$INPUT_VIDEO" -vf subtitles="$SUBTITLE_FILE" "$OUTPUT_VIDEO"

echo "Success! Your new video is saved as: $OUTPUT_VIDEO"