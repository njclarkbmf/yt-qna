#!/bin/bash
# fast_start.sh - A more efficient way to use the YouTube QA Bot

# Display help message
show_help() {
    echo "YouTube QA Bot - Fast Start Helper"
    echo ""
    echo "Usage:"
    echo "  ./fast_start.sh [command] [arguments]"
    echo ""
    echo "Commands:"
    echo "  download URL [--tags tag1,tag2]     Download a video for later processing"
    echo "  batch FILE [--tags tag1,tag2]       Download videos from a file containing URLs"
    echo "  process [--all | --video-id ID]     Process downloaded videos"
    echo "  list                               List videos in the database"
    echo "  delete VIDEO_ID                    Delete a video from the database"
    echo "  query \"Your question\"             Ask a question about your videos"
    echo "  help                               Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./fast_start.sh download https://www.youtube.com/watch?v=VIDEO_ID --tags ml,tutorial"
    echo "  ./fast_start.sh batch videos.txt"
    echo "  ./fast_start.sh process --all"
    echo "  ./fast_start.sh query \"What is explained in these videos?\""
}

# Check if at least one argument was provided
if [ $# -lt 1 ]; then
    show_help
    exit 1
fi

# Process command
COMMAND=$1
shift

case $COMMAND in
    download)
        if [ $# -lt 1 ]; then
            echo "Error: URL required"
            exit 1
        fi
        URL=$1
        shift
        python fast_download.py "$URL" $@
        ;;
        
    batch)
        if [ $# -lt 1 ]; then
            echo "Error: File required"
            exit 1
        fi
        FILE=$1
        shift
        python fast_download.py "$FILE" --batch $@
        ;;
        
    process)
        python process_later.py $@
        ;;
        
    list)
        python youtube_qa_app.py list --fast
        ;;
        
    delete)
        if [ $# -lt 1 ]; then
            echo "Error: VIDEO_ID required"
            exit 1
        fi
        VIDEO_ID=$1
        python youtube_qa_app.py delete --video-id "$VIDEO_ID" --fast
        ;;
        
    query)
        if [ $# -lt 1 ]; then
            echo "Error: Query string required"
            exit 1
        fi
        QUERY=$1
        shift
        python youtube_qa_app.py query "$QUERY" $@
        ;;
        
    help|*)
        show_help
        ;;
esac
