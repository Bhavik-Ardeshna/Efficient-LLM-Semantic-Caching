#!/bin/bash

# Reset Qdrant Vector Database
# This script completely resets the Qdrant container and data

set -e

echo "🚀 Qdrant Reset Script"
echo "====================="

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found in current directory"
    exit 1
fi

# Check if Qdrant service is defined
if ! grep -q "qdrant" docker-compose.yml; then
    echo "❌ Qdrant service not found in docker-compose.yml"
    exit 1
fi

echo "📊 Current Qdrant status:"
docker-compose ps qdrant || echo "Qdrant container not running"

echo ""
echo "⚠️  WARNING: This will completely reset Qdrant!"
echo "   - All vector data will be lost"
echo "   - All collections will be deleted"
echo "   - Container will be recreated"
echo ""

read -p "Are you sure you want to continue? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy]es?$ ]]; then
    echo "❌ Operation cancelled"
    exit 1
fi

echo ""
echo "🛑 Stopping Qdrant container..."
docker-compose stop qdrant

echo "🗑️  Removing Qdrant container..."
docker-compose rm -f qdrant

echo "🧹 Removing Qdrant volumes (if any)..."
docker volume ls -q | grep qdrant | xargs -r docker volume rm || echo "No Qdrant volumes found"

echo "🔄 Recreating Qdrant container..."
docker-compose up -d qdrant

echo "⏳ Waiting for Qdrant to be ready..."
sleep 5

# Check if Qdrant is running
if docker-compose ps qdrant | grep -q "Up"; then
    echo "✅ Qdrant reset completed successfully!"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Run: python reset_vector_store.py --confirm"
    echo "   2. Restart your application"
    echo "   3. Vector store will be repopulated automatically"
else
    echo "❌ Qdrant failed to start properly"
    echo "Check logs with: docker-compose logs qdrant"
    exit 1
fi

echo ""
echo "🎉 Qdrant is ready for use!" 