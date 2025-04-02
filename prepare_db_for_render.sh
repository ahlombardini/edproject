#!/bin/bash

# Create a backup directory if it doesn't exist
mkdir -p backup

# Get current timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Copy and compress the database
echo "Creating backup of edapi.db..."
cp edapi.db backup/edapi_${TIMESTAMP}.db
echo "Compressing database for upload..."
tar -czf backup/edapi_${TIMESTAMP}.tar.gz edapi.db

echo "Database backup created at: backup/edapi_${TIMESTAMP}.db"
echo "Compressed database created at: backup/edapi_${TIMESTAMP}.tar.gz"
echo ""
echo "Next steps:"
echo "1. Download the compressed file: backup/edapi_${TIMESTAMP}.tar.gz"
echo "2. Go to your Render dashboard"
echo "3. Open the shell for your service"
echo "4. Upload the compressed file using: curl -o edapi.tar.gz <your-file-url>"
echo "5. Extract it using: tar -xzf edapi.tar.gz"
echo "6. Move it to the data directory: mv edapi.db /data/"
echo ""
echo "Or you can run these commands directly in Render shell:"
echo "curl -o edapi.tar.gz <your-file-url>"
echo "tar -xzf edapi.tar.gz"
echo "mv edapi.db /data/"
