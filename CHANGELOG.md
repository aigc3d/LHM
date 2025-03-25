# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2024-03-26

### Added
- Created modular architecture for ComfyUI LHM node:
  - Implemented a flexible system with standalone components
  - Added fallback implementations for environments with missing dependencies
  - Created `full_implementation.py` with complete functionality
  - Added robust import path resolution via `lhm_import_fix.py`
- Enhanced installation and dependency management:
  - Created `install_dependencies.sh` bash script for automatic installation
  - Added `install_dependencies.py` Python script for cross-platform support
  - Implemented progressive loading of features based on available dependencies
- Improved client-side implementation:
  - Enhanced progress bar with detailed status updates
  - Added custom styling with gradients and visual indicators
  - Implemented responsive text-wrapping for status messages
- New troubleshooting resources:
  - Updated troubleshooting guide with common installation issues
  - Added step-by-step solutions for dependency problems
  - Created simplified test node for diagnostics

### Changed
- Refactored code structure for better maintainability:
  - Separated full and simplified implementations
  - Improved module loading with graceful fallbacks
  - Enhanced error handling and user feedback
- Reorganized API routes implementation:
  - Created more robust websocket communication
  - Added dummy server for offline development

### Fixed
- Resolved Pinokio integration issues:
  - Fixed Python path resolution in Pinokio environments
  - Added comprehensive path discovery for LHM codebase
  - Implemented fallback mechanisms for missing dependencies
- Improved cross-platform compatibility:
  - Better handling of file paths on Windows and Unix systems
  - Conditional dependency installation based on platform

## [1.0.0] - 2023-11-29

### Added
- Enhanced ComfyUI node with full standards compliance:
  - Implemented lifecycle hooks (onNodeCreated, onNodeRemoved) for resource management
  - Added lazy evaluation support with IS_CHANGED method
  - Created client-side settings UI with customization options
  - Added progress bars directly on the node UI
  - Implemented memory optimization with configurable resource cleanup
  - Added preview scaling option for better performance with large images
  - Proper error handling and recovery from failures
  - Created server-side API routes for resource management
  - Created package.json following ComfyUI registry standards
- Initial ComfyUI node implementation for LHM project:
  - Created `comfy_lhm_node` directory with core functionality
  - Added requirements.txt for ComfyUI node dependencies
  - Added comprehensive README.md with installation and usage instructions
  - Implemented image preprocessing with background removal and recentering
  - Added model loading and initialization functionality
  - Implemented inference pipeline with motion sequence support
  - Added client-side JavaScript for progress updates and UI enhancements
  - Created example workflow JSON demonstrating usage
  - Added proper logging and error handling

### Changed
- Refactored JavaScript to use modular structure with import statements
- Updated node styling to use user-configurable colors
- Modified tensor handling to support batched inputs

### Deprecated
- None

### Removed
- None

### Fixed
- Fixed memory leaks by properly cleaning up resources on node removal
- Improved tensor shape handling for better compatibility with ComfyUI workflows

### Security
- None

## Additional Notes
- Documentation has been updated to reflect new features and settings
- Example workflow demonstrates all key functionality
- The node is now fully compliant with ComfyUI registry standards and ready for submission 