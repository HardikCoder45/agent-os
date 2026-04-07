#!/usr/bin/env python3
"""
Create a GitHub repository and push code
"""
import subprocess
import sys

def run_command(cmd, check=True):
    """Run a shell command and return output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {result.stderr}")
        return None
    return result.stdout.strip()

def main():
    repo_name = "agent-environment-hackathon"
    
    print("🚀 Setting up GitHub repository...")
    print()
    
    # Get GitHub username (you'll need to provide this)
    print("📝 Please provide your GitHub username:")
    github_username = input("Username: ").strip()
    
    if not github_username:
        print("❌ Username is required")
        sys.exit(1)
    
    repo_url = f"https://github.com/{github_username}/{repo_name}.git"
    
    print(f"\n📦 Repository will be: {github_username}/{repo_name}")
    print()
    
    # Check if remote exists
    current_remote = run_command("git remote get-url origin 2>/dev/null", check=False)
    
    if current_remote:
        print(f"✅ Remote 'origin' exists: {current_remote}")
        print("🔄 Updating remote URL...")
        run_command(f"git remote set-url origin {repo_url}")
    else:
        print("🔗 Adding remote 'origin'...")
        run_command(f"git remote add origin {repo_url}")
    
    print()
    print("⚠️  IMPORTANT: Before pushing, you need to:")
    print(f"   1. Go to: https://github.com/new")
    print(f"   2. Create a repository named: {repo_name}")
    print(f"   3. Make it Public or Private (your choice)")
    print(f"   4. DON'T initialize with README, .gitignore, or license")
    print()
    
    response = input("Have you created the repository? (y/n): ").strip().lower()
    
    if response != 'y':
        print()
        print("📝 When ready, run:")
        print(f"   git push -u origin main")
        print()
        print("Or run this script again.")
        sys.exit(0)
    
    print()
    print("📤 Pushing to GitHub...")
    
    # Push to GitHub
    result = subprocess.run(
        ["git", "push", "-u", "origin", "main"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print()
        print("✅ Successfully pushed to GitHub!")
        print(f"🌐 Repository: https://github.com/{github_username}/{repo_name}")
        print()
        print("🎉 Next steps:")
        print("   - View your code online")
        print("   - Set up GitHub Actions for CI/CD")
        print("   - Add collaborators if needed")
        print()
    else:
        print()
        print("❌ Push failed!")
        print(f"Error: {result.stderr}")
        print()
        print("💡 Common issues:")
        print("   - Repository doesn't exist yet")
        print("   - Authentication required (use GitHub token or SSH)")
        print("   - Branch protection rules")
        print()
        print("📝 Try manually:")
        print(f"   git push -u origin main")

if __name__ == "__main__":
    main()
