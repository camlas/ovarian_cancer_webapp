#!/bin/bash
echo "🎨 Building Tailwind CSS..."
npx tailwindcss -i ./static/css/src/input.css -o ./static/css/output.css --watch
