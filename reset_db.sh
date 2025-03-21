#!/bin/bash

# Extract LANCEDB_PATH from .env file
LANCEDB_PATH=$(grep LANCEDB_PATH .env | cut -d '=' -f2)

# If not found, use default
if [ -z "$LANCEDB_PATH" ]; then
    LANCEDB_PATH="lancedb"
fi

# Create backup with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="${LANCEDB_PATH}_backup_${TIMESTAMP}"

echo "Creating backup at $BACKUP_PATH"
cp -r "$LANCEDB_PATH" "$BACKUP_PATH"

echo "Removing $LANCEDB_PATH"
rm -rf "$LANCEDB_PATH"

echo "Database reset complete. You can now reprocess your videos."
