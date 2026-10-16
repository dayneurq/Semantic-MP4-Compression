#!/usr/bin/env python3
"""
Dependency Checker for Semantic Audio Compressor
Verifies all required packages are installed before running expensive API calls
"""

import sys

def check_package(package_name, import_name=None, min_version=None):
    """Check if a package is installed and optionally verify version"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        
        if min_version and version != 'unknown':
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                return False, f"❌ {package_name} {version} (needs >= {min_version})"
        
        return True, f"✅ {package_name} {version}"
    except ImportError:
        return False, f"❌ {package_name} - NOT INSTALLED"

def main():
    print("=" * 60)
    print("Semantic Audio Compressor - Dependency Check")
    print("=" * 60)
    print()
    
    # Core dependencies
    dependencies = [
        ("numpy", "numpy", None),
        ("librosa", "librosa", "0.9.0"),
        ("soundfile", "soundfile", None),
        ("requests", "requests", None),
    ]
    
    # Optional but recommended
    optional_deps = [
        ("pyannote.audio", "pyannote.audio", None),
        ("torch", "torch", None),
        ("demucs", "demucs", None),
    ]
    
    print("REQUIRED DEPENDENCIES:")
    print("-" * 60)
    all_required = True
    for pkg, import_name, min_ver in dependencies:
        success, message = check_package(pkg, import_name, min_ver)
        print(f"  {message}")
        if not success:
            all_required = False
    
    print()
    print("OPTIONAL DEPENDENCIES (for advanced features):")
    print("-" * 60)
    for pkg, import_name, min_ver in optional_deps:
        success, message = check_package(pkg, import_name, min_ver)
        print(f"  {message}")
    
    print()
    print("=" * 60)
    
    if not all_required:
        print("❌ MISSING REQUIRED DEPENDENCIES!")
        print()
        print("Install missing packages with:")
        print("  pip install numpy librosa soundfile requests")
        print()
        return False
    else:
        print("✅ ALL REQUIRED DEPENDENCIES INSTALLED!")
        print()
        print("Optional features available:")
        print("  - Speaker diarization: ", end="")
        try:
            __import__("pyannote.audio")
            print("✅ Available")
        except:
            print("❌ Not available (install: pip install pyannote.audio)")
        
        print("  - Stem separation: ", end="")
        try:
            __import__("demucs")
            print("✅ Available")
        except:
            print("❌ Not available (install: pip install demucs)")
        
        print()
        print("You're ready to run the audio compressor!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)