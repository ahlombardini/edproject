#!/bin/bash

# Check if text argument is provided
if [ $# -eq 0 ]; then
  echo "Usage: ./search_api.sh \"your search query here\""
  exit 1
fi

# Use all arguments as the search text
QUERY="$*"

# Send request to API
curl -X POST http://localhost:8001/search/input \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$QUERY\"}"
