# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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