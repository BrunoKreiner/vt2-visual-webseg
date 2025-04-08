import os
import mimetypes
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def should_ignore_file(file_path):
    """Check if file should be ignored completely."""
    if file_path.name.startswith('.'):
        return True
    if file_path.name.endswith('.zone.identifier'):
        return True
    if file_path.name.upper() == "LICENSE" or file_path.name.upper().startswith("LICENSE."):
        return True
    # Skip binary and large files
    ext = file_path.suffix.lower()
    if ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg', 
               '.mp3', '.mp4', '.avi', '.mov', '.wav', '.flac',
               '.zip', '.tar', '.gz', '.rar', '.7z',
               '.dll', '.exe', '.so', '.dylib',
               '.pyc', '.pyd', '.pyo', '.pyx',
               '.pt', '.pth', '.pkl', '.pickle', '.npy',
               '.db', '.sqlite', '.sqlite3']:
        return True
    return False

def should_include_content(file_path, max_size_kb=100):
    """Check if the file's content should be included."""
    # Only include python and markdown files under the size limit
    ext = file_path.suffix.lower()
    
    # First check extension
    if ext not in ['.py', '.md']:
        return False
    
    # Then check size
    try:
        size_kb = os.path.getsize(file_path) / 1024
        if size_kb > max_size_kb:
            return False
    except:
        return False
        
    return True

def is_important_file(file_path):
    """Check if file is particularly important to understand the repository."""
    name = file_path.name.lower()
    if name in ['readme.md', 'requirements.txt', 'setup.py', 'main.py', 'app.py', 
                'index.py', 'config.py', 'settings.py', '__init__.py',
                'pyproject.toml', 'environment.yml']:
        return True
    return False

def truncate_content(content, max_lines=100):
    """Truncate content to a maximum number of lines."""
    lines = content.split('\n')
    if len(lines) <= max_lines:
        return content
    
    # Take first and last lines to give context
    first_half = lines[:max_lines//2]
    last_half = lines[-(max_lines//2):]
    
    return '\n'.join(first_half + ['[...truncated...]'] + last_half)

def generate_repo_summary(root_dir, max_examples=3, max_file_size_kb=100, max_content_lines=100):
    """Generate a concise summary of the repository structure and key files."""
    summary = []
    
    # Add header
    summary.append("# Repository Summary")
    summary.append(f"Root Directory: {os.path.basename(root_dir)}")
    summary.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append("")
    
    # Count files by type
    file_counts = defaultdict(int)
    important_files = []
    sample_files = []
    total_files = 0
    
    # Walk directory tree
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            file_path = Path(os.path.join(root, file))
            
            # Skip files that should be ignored
            if should_ignore_file(file_path):
                continue
            
            # Count by extension
            ext = file_path.suffix.lower() or 'no_extension'
            file_counts[ext] += 1
            total_files += 1
            
            # Track important files
            if is_important_file(file_path):
                important_files.append(file_path)
            
            # Collect some sample files for each extension (for diversity)
            if len(sample_files) < 20 and should_include_content(file_path, max_file_size_kb):
                # Don't add too many of the same type
                if sum(1 for f in sample_files if f.suffix == ext) < max_examples:
                    sample_files.append(file_path)
    
    # Add file statistics
    summary.append("## Repository Statistics")
    summary.append(f"- Total files: {total_files}")
    summary.append("- Files by type:")
    for ext, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True):
        summary.append(f"  - {ext}: {count}")
    summary.append("")
    
    # Add important files section
    if important_files:
        summary.append("## Key Files")
        for file_path in important_files:
            rel_path = os.path.relpath(file_path, root_dir)
            summary.append(f"### {rel_path}")
            
            # Include content for important files
            if should_include_content(file_path, max_file_size_kb * 2):  # Allow larger size for important files
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        content = truncate_content(content, max_content_lines)
                        summary.append("```")
                        summary.append(content)
                        summary.append("```")
                except:
                    summary.append("[Could not read file content]")
            else:
                summary.append("[Content not included due to size]")
            summary.append("")
    
    # Add directory structure (simplified)
    summary.append("## Directory Structure")
    
    # Get top-level directories and files
    top_entries = []
    try:
        with os.scandir(root_dir) as entries:
            for entry in entries:
                if not entry.name.startswith('.'):
                    top_entries.append(entry)
    except:
        pass
    
    # Sort: directories first, then files
    top_dirs = sorted([e for e in top_entries if e.is_dir()], key=lambda e: e.name)
    top_files = sorted([e for e in top_entries if e.is_file()], key=lambda e: e.name)
    
    summary.append("```")
    summary.append(f"./{os.path.basename(root_dir)}/")
    
    # Add top-level directories with counts
    for entry in top_dirs:
        dir_path = entry.path
        file_count = 0
        
        # Count files in this directory (recursively)
        for root, _, files in os.walk(dir_path):
            file_count += len([f for f in files if not should_ignore_file(Path(os.path.join(root, f)))])
        
        summary.append(f"├── {entry.name}/ ({file_count} files)")
    
    # Add top-level files
    for i, entry in enumerate(top_files):
        if i == len(top_files) - 1:
            summary.append(f"└── {entry.name}")
        else:
            summary.append(f"├── {entry.name}")
    summary.append("```")
    summary.append("")
    
    # Add sample code files
    if sample_files:
        summary.append("## Sample Code")
        for file_path in sample_files:
            rel_path = os.path.relpath(file_path, root_dir)
            summary.append(f"### {rel_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    content = truncate_content(content, max_content_lines)
                    summary.append("```")
                    summary.append(content)
                    summary.append("```")
            except:
                summary.append("[Could not read file content]")
            summary.append("")
    
    return '\n'.join(summary)

def main():
    # Get the root directory (where the script is run)
    root_dir = os.getcwd()
    output_file_path = os.path.join(root_dir, "repo_summary.md")
    
    # Generate the repository summary
    summary = generate_repo_summary(
        root_dir, 
        max_examples=3,          # Max examples of each file type
        max_file_size_kb=5000,     # Max file size for inclusion (50KB)
        max_content_lines=10000    # Max lines of content per file
    )
    
    # Write to file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Repository summary complete. Output saved to: {output_file_path}")

if __name__ == "__main__":
    main()