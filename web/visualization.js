import { app } from "../../scripts/app.js";

/**
 * HunyuanWorld-Mirror 3D Visualization Extension
 * Adds preview capabilities for Gaussian Splatting and Point Clouds
 */

class Visualizer {
    constructor(node, typeName) {
        this.node = node;
        this.typeName = typeName;
        this.container = null;
        this.iframe = null;
        this.filepath = "";
        this.timestamp = Date.now();
        this.iframeLoaded = false;
    }

    createContainer() {
        console.log(`[Visualizer] 🔨 Creating container for ${this.typeName}`);

        // Create container div
        const container = document.createElement("div");
        container.style.position = "absolute";
        container.style.overflow = "hidden";
        container.style.display = "none";  // Hidden until positioned
        container.id = `HunyuanWorld_${this.typeName}_${Date.now()}`;

        // Create iframe inside container
        const iframe = document.createElement("iframe");
        iframe.style.width = "100%";
        iframe.style.height = "100%";
        iframe.style.border = "none";
        iframe.scrolling = "no";
        iframe.src = `/extensions/ComfyUI-HunyuanWorld-Mirror/html/${this.typeName}.html`;

        console.log(`[Visualizer] 📄 iframe src: ${iframe.src}`);

        // Wait for iframe to load before trying to access content
        iframe.addEventListener('load', () => {
            console.log(`[Visualizer] ✅ iframe loaded successfully: ${this.typeName}`);
            this.iframeLoaded = true;
            // If we have a pending filepath, update it now
            if (this.filepath) {
                console.log(`[Visualizer] 📦 Applying pending filepath: ${this.filepath}`);
                const filepath = this.filepath;
                this.filepath = null;  // Reset to trigger update
                this.updateVisual(filepath);
            } else {
                console.log(`[Visualizer] ⏳ No pending filepath yet`);
            }
        });

        iframe.addEventListener('error', (e) => {
            console.error(`[Visualizer] ❌ iframe load error:`, e);
        });

        container.appendChild(iframe);
        this.iframe = iframe;

        console.log(`[Visualizer] ✓ Container created with ID: ${container.id}`);
        return container;
    }

    updateVisual(filepath) {
        console.log(`[Visualizer] 🔄 updateVisual called with filepath: ${filepath}`);
        console.log(`[Visualizer] Current filepath: ${this.filepath}, iframeLoaded: ${this.iframeLoaded}`);

        if (filepath && filepath !== this.filepath) {
            this.filepath = filepath;
            this.timestamp = Date.now();

            console.log(`[Visualizer] 📝 Set new filepath: ${this.filepath}, timestamp: ${this.timestamp}`);

            // If iframe not loaded yet, the load event will handle the update
            if (!this.iframeLoaded) {
                console.log(`[Visualizer] ⏰ iframe not loaded yet, will update after load`);
                return;
            }

            if (this.iframe && this.iframe.contentWindow) {
                try {
                    const iframeDoc = this.iframe.contentDocument || this.iframe.contentWindow.document;
                    console.log(`[Visualizer] 📄 Got iframe document:`, iframeDoc ? '✓' : '✗');

                    if (iframeDoc) {
                        const visualizer = iframeDoc.getElementById("visualizer");
                        console.log(`[Visualizer] 🔍 Looking for visualizer element:`, visualizer ? '✓ Found' : '✗ Not found');

                        if (visualizer) {
                            visualizer.setAttribute("filepath", this.filepath);
                            visualizer.setAttribute("timestamp", this.timestamp.toString());
                            console.log(`[Visualizer] ✅ Updated attributes - filepath: ${this.filepath}, timestamp: ${this.timestamp}`);
                        } else {
                            console.warn("[Visualizer] ❌ Script element #visualizer not found in iframe");
                            console.log("[Visualizer] Available elements:", Array.from(iframeDoc.querySelectorAll('[id]')).map(el => el.id));
                        }
                    }
                } catch (error) {
                    console.error("[Visualizer] ❌ Error accessing iframe:", error);
                    console.error("[Visualizer] Error stack:", error.stack);
                }
            } else {
                console.warn("[Visualizer] ❌ iframe or contentWindow not available");
            }
        } else {
            console.log(`[Visualizer] ⏭️ Skipping update - no change or invalid filepath`);
        }
    }
}

function registerVisualizer(nodeType, nodeData, nodeName, typeName) {
    if (nodeData?.name === nodeName) {
        const onNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const visualizer = new Visualizer(this, typeName);
            const node = this;

            const widget = this.addCustomWidget({
                name: "3d_preview",
                type: "3d_preview",
                value: "",
                draw: function(ctx, node, width, y) {
                    // Position container using proper ComfyUI coordinate conversion
                    if (!visualizer.container || node.flags.collapsed) {
                        if (visualizer.container) {
                            visualizer.container.style.display = "none";
                        }
                        return;
                    }

                    // Ensure container is attached to DOM
                    const canvasEl = node.graph.list_of_graphcanvas[0].canvas;
                    if (visualizer.container.parentElement !== canvasEl.parentElement) {
                        console.log(`[${nodeName}] 🔗 Attaching container to DOM`);
                        canvasEl.parentElement.appendChild(visualizer.container);
                    }

                    // Only show if zoomed in enough
                    const scale = app.canvas.ds.scale;
                    if (scale < 0.5) {
                        visualizer.container.style.display = "none";
                        return;
                    }

                    // Get node position
                    const [x, y] = node.getBounding();
                    const [left, top] = app.canvasPosToClientPos([x, y]);

                    // Calculate offset for node title and margins
                    const topOffset = LiteGraph.NODE_TITLE_HEIGHT + 30;

                    // Calculate dimensions
                    const viewerHeight = 600;
                    const containerWidth = node.size[0] * scale;
                    const containerHeight = viewerHeight * scale;

                    // Position container
                    Object.assign(visualizer.container.style, {
                        display: "block",
                        left: `${left}px`,
                        top: `${top + (topOffset * scale)}px`,
                        width: `${containerWidth}px`,
                        height: `${containerHeight}px`,
                        zIndex: "5",
                        pointerEvents: "auto"
                    });

                    // Log positioning details periodically (every 100 frames)
                    if (!this._drawCallCount) this._drawCallCount = 0;
                    if (++this._drawCallCount % 100 === 0) {
                        console.log(`[${nodeName}] 📐 Positioned container at (${left}, ${top + (topOffset * scale)}) size ${containerWidth}x${containerHeight}`);
                    }
                },
                computeSize: function(width) {
                    return [width, 600];  // Fixed height for viewer
                },
                serialize: false
            });

            widget.visualizer = visualizer;

            // Create container with iframe
            visualizer.container = visualizer.createContainer();
            this.visualizer = visualizer;

            return result;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (this.visualizer?.container) {
                this.visualizer.container.remove();
            }
            return onRemoved?.apply(this, arguments);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            console.log(`[${nodeName}] 📨 onExecuted called with message:`, message);

            const result = onExecuted?.apply(this, arguments);

            if (message?.previews?.[0]?.filepath) {
                const filepath = message.previews[0].filepath;
                console.log(`[${nodeName}] 📁 Got filepath from previews:`, filepath);
                console.log(`[${nodeName}] Visualizer exists:`, !!this.visualizer);

                if (this.visualizer) {
                    this.visualizer.updateVisual(filepath);
                } else {
                    console.error(`[${nodeName}] ❌ Visualizer not initialized!`);
                }
            } else {
                console.warn(`[${nodeName}] ⚠️ No filepath in message.previews`);
                console.log(`[${nodeName}] Message structure:`, JSON.stringify(message, null, 2));
            }

            return result;
        };
    }
}

// Register Gaussian Splatting Viewer
app.registerExtension({
    name: "HunyuanWorld.Visualizer.GaussianSplatting",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        registerVisualizer(nodeType, nodeData, "Preview3DGS", "gsViewer");
    },
});

// Register Point Cloud Viewer
app.registerExtension({
    name: "HunyuanWorld.Visualizer.PointCloud",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        registerVisualizer(nodeType, nodeData, "PreviewPointCloud", "pointcloudViewer");
    },
});
