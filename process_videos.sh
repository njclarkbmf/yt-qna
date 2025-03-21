#!/bin/bash

# Script to process multiple YouTube videos from a text file
# Usage: ./process_videos.sh videos.txt [--tags tag1,tag2] [--force] [--whisper-model base]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <video_list_file> [additional options]"
    echo "Example: $0 videos.txt --tags machine-learning,tutorial"
    exit 1
fi

VIDEO_FILE=$1
shift  # Remove the first argument (the file name)

# Check if the file exists
if [ ! -f "$VIDEO_FILE" ]; then
    echo "Error: File '$VIDEO_FILE' not found."
    exit 1
fi

# Additional options to pass to the YouTube QA Bot
ADDITIONAL_OPTS="$@"

# Count the total number of videos
TOTAL=$(grep -v '^\s*#' "$VIDEO_FILE" | grep -v '^\s*$' | wc -l)
echo "Processing $TOTAL videos from $VIDEO_FILE"
echo "Additional options: $ADDITIONAL_OPTS"
echo

# Process each video URL in the file
COUNT=0
while IFS= read -r line || [[ -n "$line" ]]; do
    # Skip empty lines and comments
    if [[ -z "$line" || "$line" == \#* ]]; then
        continue
    fi
    
    COUNT=$((COUNT + 1))
    echo "[$COUNT/$TOTAL] Processing: $line"
    
    # Run the YouTube QA Bot command
    python youtube_qa_app.py add "$line" $ADDITIONAL_OPTS
    
    echo "-------------------------------------------"
    echo
done < "$VIDEO_FILE"

echo "Completed processing $COUNT videos."
