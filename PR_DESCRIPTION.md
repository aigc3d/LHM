# ComfyUI Node for Large Animatable Human Model (LHM)

## Overview
This pull request adds a ComfyUI node implementation for the Large Animatable Human Model (LHM), enabling users to integrate LHM's 3D human reconstruction and animation capabilities directly into ComfyUI workflows.

## Features
- **Modular architecture** with both full implementation and simplified fallback mode
- **Automatic dependency detection** with graceful degradation when optional dependencies are missing
- **Client-side UI enhancements** including progress bars and real-time status updates
- **Comprehensive documentation** including installation guides and troubleshooting
- **Multiple node types** for different use cases (reconstruction, testing, etc.)
- **Installation scripts** for different platforms (bash and Python versions)

## Implementation Details
- `full_implementation.py`: Complete implementation with all LHM features
- `__init__.py`: Entry point with automatic fallback to simplified mode
- `lhm_import_fix.py`: Robust Python path handling for dependency resolution
- `install_dependencies.py/sh`: Cross-platform installation scripts
- `routes.py`: API endpoints for progress updates and resource management
- `web/js/lhm.js`: Client-side UI enhancements
- `TROUBLESHOOTING.md`: Detailed guide for resolving common issues

## Quality Assurance
All code has undergone comprehensive error checking with:
- Validated error handling in both full and simplified implementations
- Confirmed proper import paths and dependency management
- Verified no TODOs or outstanding issues remain
- Tested module interaction and fallback mechanisms

## Changelog
See the included `CHANGELOG.md` for a detailed history of changes.

## Testing
The implementation has been tested with:
- ComfyUI latest version
- Various dependency configurations
- Multiple platforms (macOS, Windows, Linux)
- Different input image types and sizes

## Notes
- The modular architecture ensures compatibility with environments that may not have all optional dependencies installed
- Users can start with the simplified implementation and gradually install components for full functionality
- All JavaScript modules use modern ES6 module syntax for better compatibility with ComfyUI 