#!/bin/bash

# Check if text argument is provided
if [ $# -eq 0 ]; then
  echo "Usage: ./search_pretty.sh \"your search query here\""
  exit 1
fi

# Use all arguments as the search text
QUERY="$*"

echo "🔍 Searching for: \"$QUERY\""
echo "------------------------------"

# Send request to API and process with jq for pretty output
# If jq is not installed, fall back to basic curl
if command -v jq &> /dev/null; then
  RESULTS=$(curl -s -X POST http://localhost:8001/search/input \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"$QUERY\"}")

  # Count results
  COUNT=$(echo $RESULTS | jq length)
  echo "Found $COUNT similar threads"
  echo "------------------------------"

  # Process each result
  echo "$RESULTS" | jq -c '.[]' | while read -r thread; do
    TITLE=$(echo "$thread" | jq -r '.title')
    CATEGORY=$(echo "$thread" | jq -r '.category')
    ID=$(echo "$thread" | jq -r '.ed_thread_id')
    SIMILARITY=$(echo "$thread" | jq -r '.similarity')
    SIMILARITY_PCT=$(echo "$SIMILARITY * 100" | bc -l | xargs printf "%.2f")

    echo "📄 $TITLE"
    echo "📁 Category: $CATEGORY"
    echo "🆔 Thread ID: $ID"
    echo "✓ Similarity: ${SIMILARITY_PCT}%"
    echo "------------------------------"
  done
else
  # Fallback to basic curl if jq is not available
  curl -X POST http://localhost:8001/search/input \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"$QUERY\"}"
fi
