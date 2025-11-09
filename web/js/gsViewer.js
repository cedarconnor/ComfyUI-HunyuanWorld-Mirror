import * as SPLAT from 'gsplat';

const canvas = document.getElementById("canvas");
const loading = document.getElementById("loading");
const errorDiv = document.getElementById("error");
const infoDiv = document.getElementById("info");
const visualizer = document.getElementById("visualizer");

let renderer, scene, camera, controls;
let currentFilepath = "";
let lastTimestamp = "";
let animationId = null;

// Initialize renderer
function initRenderer() {
    try {
        renderer = new SPLAT.WebGLRenderer(canvas);
        scene = new SPLAT.Scene();
        camera = new SPLAT.Camera();

        // Set initial camera position
        camera.position.set(0, 0, 5);

        // Setup controls
        controls = new SPLAT.OrbitControls(camera, canvas);
        controls.orbitSpeed = 1.5;
        controls.panSpeed = 0.8;
        controls.zoomSpeed = 1.0;
        controls.dampingFactor = 0.05;

        // Handle window resize
        window.addEventListener('resize', onWindowResize);
        onWindowResize();

        console.log("[GSViewer] Renderer initialized");
    } catch (error) {
        console.error("[GSViewer] Failed to initialize renderer:", error);
        showError("Failed to initialize WebGL renderer. Your browser may not support WebGL.");
    }
}

function onWindowResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;

    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
}

async function loadGaussianSplats(filepath) {
    loading.style.display = 'block';
    errorDiv.style.display = 'none';
    infoDiv.style.display = 'none';

    try {
        // Construct URL for file access
        const url = window.location.origin;
        const params = new URLSearchParams({ filepath: filepath });
        const fileURL = `${url}/viewfile?${params}`;

        console.log("[GSViewer] Loading:", fileURL);

        // Clear previous scene
        scene.reset();

        // Load PLY file
        const onProgress = (progress) => {
            loading.textContent = `Loading... ${Math.round(progress * 100)}%`;
        };

        await SPLAT.PLYLoader.LoadAsync(fileURL, scene, onProgress);

        console.log("[GSViewer] Loaded successfully. Splat count:", scene.splatCount);

        loading.style.display = 'none';
        infoDiv.style.display = 'block';

        // Fit camera to scene
        fitCameraToScene();

    } catch (error) {
        console.error("[GSViewer] Failed to load file:", error);
        showError(`Failed to load Gaussian Splatting file:\n${error.message}`);
        loading.style.display = 'none';
    }
}

function fitCameraToScene() {
    if (!scene || scene.splatCount === 0) return;

    // Simple camera positioning - adjust as needed
    camera.position.set(0, 0, 5);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    loading.style.display = 'none';
}

// Rendering loop
function animate() {
    animationId = requestAnimationFrame(animate);

    // Check if filepath has changed
    const filepath = visualizer.getAttribute("filepath");
    const timestamp = visualizer.getAttribute("timestamp");

    if (timestamp && timestamp !== lastTimestamp && filepath) {
        currentFilepath = filepath;
        lastTimestamp = timestamp;
        loadGaussianSplats(filepath);
    }

    // Update controls and render
    if (controls) {
        controls.update();
    }

    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }
}

// Initialize on load
window.addEventListener('DOMContentLoaded', () => {
    initRenderer();
    animate();

    // Initial load check
    const filepath = visualizer.getAttribute("filepath");
    const timestamp = visualizer.getAttribute("timestamp");

    if (filepath && timestamp) {
        currentFilepath = filepath;
        lastTimestamp = timestamp;
        loadGaussianSplats(filepath);
    }
});

// Cleanup on unload
window.addEventListener('beforeunload', () => {
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
    if (scene) {
        scene.reset();
    }
});
