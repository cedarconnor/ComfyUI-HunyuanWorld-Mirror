import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

const container = document.getElementById("canvas");
const loading = document.getElementById("loading");
const errorDiv = document.getElementById("error");
const infoDiv = document.getElementById("info");
const statsDiv = document.getElementById("stats");
const pointCountDiv = document.getElementById("point-count");
const visualizer = document.getElementById("visualizer");

let scene, camera, renderer, controls;
let pointCloud = null;
let currentFilepath = "";
let lastTimestamp = "";
let animationId = null;

// Initialize Three.js scene
function initScene() {
    try {
        // Check if Three.js library loaded
        if (typeof THREE === 'undefined') {
            throw new Error("Three.js library not loaded. Check network connection and CDN availability.");
        }

        // Check WebGL support
        const testCanvas = document.createElement('canvas');
        const gl = testCanvas.getContext('webgl2') || testCanvas.getContext('webgl');
        if (!gl) {
            throw new Error("WebGL not supported in this browser.");
        }

        // Scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a1a);

        // Camera
        camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        camera.position.set(0, 0, 5);

        // Renderer
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        // Controls
        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.rotateSpeed = 1.0;
        controls.zoomSpeed = 1.2;
        controls.panSpeed = 0.8;

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);

        // Grid helper (optional)
        const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x222222);
        gridHelper.visible = false; // Hidden by default
        scene.add(gridHelper);

        // Handle window resize
        window.addEventListener('resize', onWindowResize);

        console.log("[PointCloudViewer] Scene initialized successfully");

        // Hide loading initially
        loading.style.display = 'none';

    } catch (error) {
        console.error("[PointCloudViewer] Failed to initialize scene:", error);
        showError(`Failed to initialize 3D renderer:\n${error.message}`);
    }
}

function onWindowResize() {
    const width = window.innerWidth;
    const height = window.innerHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

async function loadPointCloud(filepath) {
    if (!renderer || !scene) {
        console.warn("[PointCloudViewer] Scene not initialized, skipping load");
        return;
    }

    loading.style.display = 'block';
    errorDiv.style.display = 'none';
    infoDiv.style.display = 'none';
    statsDiv.style.display = 'none';

    try {
        // Construct URL for file access
        const url = window.location.origin;
        const params = new URLSearchParams({ filepath: filepath });
        const fileURL = `${url}/viewfile?${params}`;

        console.log("[PointCloudViewer] Loading:", fileURL);
        console.log("[PointCloudViewer] Full filepath:", filepath);

        // Remove previous point cloud
        if (pointCloud) {
            scene.remove(pointCloud);
            pointCloud.geometry.dispose();
            if (pointCloud.material) {
                pointCloud.material.dispose();
            }
            pointCloud = null;
        }

        // Load PLY file
        const loader = new PLYLoader();

        const timeout = setTimeout(() => {
            console.warn("[PointCloudViewer] Loading timeout - request taking too long");
        }, 30000);  // 30 second timeout warning

        loader.load(
            fileURL,
            (geometry) => {
                clearTimeout(timeout);
                console.log("[PointCloudViewer] Loaded successfully");

                // Create point cloud material
                const material = new THREE.PointsMaterial({
                    size: 0.01,
                    vertexColors: true,
                    sizeAttenuation: true
                });

                // Check if geometry has colors
                if (!geometry.hasAttribute('color')) {
                    // Default to white if no colors
                    const colors = new Float32Array(geometry.attributes.position.count * 3);
                    colors.fill(1.0); // White
                    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                }

                // Create point cloud
                pointCloud = new THREE.Points(geometry, material);
                scene.add(pointCloud);

                // Center and fit to view
                fitCameraToPointCloud(geometry);

                // Update stats
                const pointCount = geometry.attributes.position.count;
                pointCountDiv.textContent = `Points: ${pointCount.toLocaleString()}`;

                loading.style.display = 'none';
                infoDiv.style.display = 'block';
                statsDiv.style.display = 'block';

                console.log("[PointCloudViewer] Point count:", pointCount);
            },
            (progress) => {
                if (progress.lengthComputable) {
                    const percent = (progress.loaded / progress.total) * 100;
                    loading.innerHTML = `<div class="spinner"></div><div>Loading... ${Math.round(percent)}%</div>`;
                    console.log(`[PointCloudViewer] Progress: ${percent}%`);
                }
            },
            (error) => {
                clearTimeout(timeout);
                console.error("[PointCloudViewer] Load error:", error);
                console.error("[PointCloudViewer] Error details:", {
                    message: error.message,
                    filepath: filepath
                });
                showError(`Failed to load point cloud file:\n${error.message}\n\nCheck browser console for details.`);
                loading.style.display = 'none';
            }
        );

    } catch (error) {
        console.error("[PointCloudViewer] Failed to load file:", error);
        console.error("[PointCloudViewer] Error details:", {
            message: error.message,
            stack: error.stack,
            filepath: filepath
        });
        showError(`Failed to load point cloud file:\n${error.message}\n\nCheck browser console for details.`);
        loading.style.display = 'none';
    }
}

function fitCameraToPointCloud(geometry) {
    if (!geometry) return;

    // Compute bounding box
    geometry.computeBoundingBox();
    const boundingBox = geometry.boundingBox;

    if (!boundingBox) return;

    // Get center and size
    const center = new THREE.Vector3();
    boundingBox.getCenter(center);

    const size = new THREE.Vector3();
    boundingBox.getSize(size);

    // Calculate camera distance
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = camera.fov * (Math.PI / 180);
    let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
    cameraZ *= 1.5; // Add some padding

    // Position camera
    camera.position.set(center.x, center.y, center.z + cameraZ);
    camera.lookAt(center);

    // Update controls target
    controls.target.copy(center);
    controls.update();

    // Update camera near/far planes
    camera.near = cameraZ / 100;
    camera.far = cameraZ * 100;
    camera.updateProjectionMatrix();
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    loading.style.display = 'none';
}

// Animation loop
function animate() {
    animationId = requestAnimationFrame(animate);

    // Check if filepath has changed
    const filepath = visualizer.getAttribute("filepath");
    const timestamp = visualizer.getAttribute("timestamp");

    if (timestamp && timestamp !== lastTimestamp && filepath) {
        currentFilepath = filepath;
        lastTimestamp = timestamp;
        loadPointCloud(filepath);
    }

    // Update controls
    if (controls) {
        controls.update();
    }

    // Render scene
    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }
}

// Initialize on load
window.addEventListener('DOMContentLoaded', () => {
    initScene();
    animate();

    // Initial load check
    const filepath = visualizer.getAttribute("filepath");
    const timestamp = visualizer.getAttribute("timestamp");

    if (filepath && timestamp) {
        currentFilepath = filepath;
        lastTimestamp = timestamp;
        loadPointCloud(filepath);
    }
});

// Cleanup on unload
window.addEventListener('beforeunload', () => {
    if (animationId) {
        cancelAnimationFrame(animationId);
    }
    if (renderer) {
        renderer.dispose();
    }
    if (pointCloud) {
        scene.remove(pointCloud);
        pointCloud.geometry.dispose();
        pointCloud.material.dispose();
    }
});
