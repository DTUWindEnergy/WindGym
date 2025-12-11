#!/bin/bash
set -e

echo "[✓] Converting simulations.ipynb to Markdown..."
# Ensure the .ipynb file exists in docs/
if [ -f "docs/simulations.ipynb" ]; then
  # --to markdown: converts to Markdown
  # --output-dir docs/: saves the output .md file back into the docs/ directory
  # --output simulations: names the output file simulations.md
  # Note: This will also create a simulations_files/ folder with images
  pixi run jupyter nbconvert --to markdown --execute docs/simulations.ipynb --output-dir docs/ --output simulations --allow-errors

  # Ensure the images folder exists (created by nbconvert)
  if [ -d "docs/simulations_files" ]; then
    echo "[✓] Image files found in docs/simulations_files/"
  else
    echo "[!] Warning: No image files generated from notebook"
  fi

  echo "[✓] Successfully converted simulations.ipynb to docs/simulations.md"
else
  echo "[!] Warning: docs/simulations.ipynb not found. Skipping notebook conversion."
fi

#echo "[✓] Copying plots to static/evals/..."
#mkdir -p static/evals
#cp ../WindGym-Zoo/results/*.png static/evals/ || echo "No .png plots to copy"
#cp ../WindGym-Zoo/results/*.csv static/evals/ || echo "No .csv files to copy"
