#!/bin/bash

# Script to push to GitHub
# Usage: ./push_to_github.sh <github-username> <repo-name>

GITHUB_USERNAME=${1:-"hardikarora"}
REPO_NAME=${2:-"agent-environment-hackathon"}

echo "🚀 Pushing to GitHub: $GITHUB_USERNAME/$REPO_NAME"
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "❌ Not a git repository. Initializing..."
    git init
    git branch -m main
fi

# Add all changes
echo "📦 Adding files..."
git add .

# Commit if there are changes
if git diff-index --quiet HEAD --; then
    echo "✅ No changes to commit"
else
    echo "💾 Committing changes..."
    git commit -m "Fix: JSON serialization error and add custom model support

- Fixed ArgSpec JSON serialization error in LLM judge
- Added support for custom OpenRouter models
- Updated model dropdown with more options
- Improved error handling and feedback
- Added Hugging Face deployment configuration"
fi

# Check if remote exists
if git remote | grep -q "^origin$"; then
    echo "✅ Remote 'origin' already exists"
    git remote set-url origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
else
    echo "🔗 Adding GitHub remote..."
    git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
fi

echo ""
echo "📤 Pushing to GitHub..."
echo ""
echo "⚠️  You may need to authenticate with GitHub."
echo "   If the repository doesn't exist, create it first at:"
echo "   https://github.com/new"
echo ""

# Try to push
if git push -u origin main 2>&1; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo "🌐 Repository: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
else
    echo ""
    echo "⚠️  Push failed. This might be because:"
    echo "   1. The repository doesn't exist yet"
    echo "   2. You need to authenticate"
    echo ""
    echo "📝 Manual steps:"
    echo "   1. Create a new repository at: https://github.com/new"
    echo "   2. Name it: $REPO_NAME"
    echo "   3. Don't initialize with README (we already have files)"
    echo "   4. Then run:"
    echo "      git remote set-url origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    echo "      git push -u origin main"
fi
