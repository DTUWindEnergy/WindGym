#!/bin/bash
# Build script for generating API documentation using Sphinx
# This script generates Markdown files from Python docstrings that can be
# integrated with the Docusaurus documentation site.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Building WindGym API documentation..."
echo "Project root: $PROJECT_ROOT"

# Change to the Docs directory
cd "$SCRIPT_DIR"

# Install Sphinx dependencies if needed
if ! python -c "import sphinx_markdown_builder" 2>/dev/null; then
    echo "Installing Sphinx dependencies..."
    pip install -r requirements-docs.txt
fi

# Clean previous builds
rm -rf sphinx-source/_build
rm -rf api-docs/*

# Run Sphinx to generate Markdown files
echo "Running Sphinx build..."
sphinx-build -b markdown sphinx-source api-docs

# Copy the generated API docs to the Docusaurus docs folder
echo "Copying API docs to Docusaurus..."
mkdir -p docs/api

# Copy all generated markdown files
if [ -d "api-docs" ] && [ "$(ls -A api-docs/*.md 2>/dev/null)" ]; then
    cp api-docs/*.md docs/api/
fi

# Copy module docs if they exist
if [ -d "api-docs/modules" ]; then
    mkdir -p docs/api/modules
    cp api-docs/modules/*.md docs/api/modules/
fi

echo "API documentation build complete!"
echo "Generated files are in: $SCRIPT_DIR/docs/api/"
