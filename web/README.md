# ComfyUI 3D Viewer

Interactive WebGL-based viewer for Gaussian Splats and Point Clouds, designed for ComfyUI workflows.

## Features

- **Interactive 3D Visualization**: View .ply and .splat files in real-time
- **Dual Rendering Modes**:
  - Point Cloud mode for dense 3D points
  - Gaussian Splat mode for volumetric rendering
- **Intuitive Controls**:
  - Left-click + drag to orbit
  - Right-click + drag to pan
  - Scroll wheel to zoom
- **Customizable Display**:
  - Adjustable point size
  - Auto-rotation toggle
  - Grid and axes helpers
- **Zero Dependencies**: Uses CDN-hosted Three.js (no local npm/build required)

## Usage

### From ComfyUI Node

1. Add the **"View 3D in Browser"** node to your workflow
2. Connect a file path from `SavePointCloud` or `Save3DGaussians`
3. Select the rendering mode (auto, pointcloud, or splat)
4. Execute the workflow - the viewer opens automatically in your browser

### Standalone Usage

Run the viewer server directly from command line:

```bash
python viewer_server.py /path/to/file.ply pointcloud
```

Then open `http://localhost:8765/index.html?file=/path/to/file.ply&mode=pointcloud` in your browser.

## Keyboard Shortcuts

- **G**: Toggle grid visibility
- **A**: Toggle axes visibility
- **R**: Reset camera view
- **H**: Toggle UI overlay

## Technical Details

### File Format Support

- **PLY** (Stanford Polygon format): Point clouds and Gaussian splats
- **SPLAT**: Gaussian splatting format (experimental)

### Requirements

- Modern web browser with WebGL support
- Python 3.7+ (for the HTTP server)
- No GPU required for viewing (runs entirely in browser)

### Architecture

```
web/
├── index.html      - Main viewer interface
├── style.css       - UI styling
└── README.md       - This file

viewer_server.py    - Python HTTP server for serving files
```

The viewer uses:
- **Three.js r160**: 3D rendering engine
- **PLYLoader**: Loading .ply files
- **OrbitControls**: Camera interaction

All libraries are loaded from CDN (jsdelivr.net) - no local installation needed.

## Troubleshooting

### Viewer doesn't open
- Check that port 8765 is not in use
- Try a different port using the `port` parameter
- Ensure the file path is valid and the file exists

### File doesn't load
- Verify the .ply file is valid (try opening in another viewer)
- Check browser console for errors (F12)
- Ensure the file path is absolute, not relative

### Performance issues
- Large point clouds (>10M points) may be slow
- Try reducing point size
- Use confidence filtering when exporting from ComfyUI

## Browser Compatibility

- Chrome/Edge: ✅ Fully supported
- Firefox: ✅ Fully supported
- Safari: ✅ Fully supported
- Mobile browsers: ⚠️ Limited (touch controls may vary)

## License

Part of the ComfyUI-HunyuanWorld-Mirror node pack.
Apache-2.0 License
