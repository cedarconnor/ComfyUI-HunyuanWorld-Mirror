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
        // Create container div
        const container = document.createElement("div");
        container.style.position = "absolute";
        container.style.overflow = "hidden";
        container.style.display = "none";  // Hidden until positioned

        // Create iframe inside container
        const iframe = document.createElement("iframe");
        iframe.style.width = "100%";
        iframe.style.height = "100%";
        iframe.style.border = "none";
        iframe.scrolling = "no";
        iframe.src = `/extensions/ComfyUI-HunyuanWorld-Mirror/html/${this.typeName}.html`;

        // Wait for iframe to load before trying to access content
        iframe.addEventListener('load', () => {
            console.log(`[Visualizer] iframe loaded: ${this.typeName}`);
            this.iframeLoaded = true;
            // If we have a pending filepath, update it now
            if (this.filepath) {
                const filepath = this.filepath;
                this.filepath = null;  // Reset to trigger update
                this.updateVisual(filepath);
            }
        });

        container.appendChild(iframe);
        this.iframe = iframe;

        return container;
    }

    updateVisual(filepath) {
        if (filepath && filepath !== this.filepath) {
            this.filepath = filepath;
            this.timestamp = Date.now();

            // If iframe not loaded yet, the load event will handle the update
            if (!this.iframeLoaded) {
                console.log(`[Visualizer] iframe not loaded yet, will update after load`);
                return;
            }

            if (this.iframe && this.iframe.contentWindow) {
                try {
                    const iframeDoc = this.iframe.contentDocument || this.iframe.contentWindow.document;
                    if (iframeDoc) {
                        const visualizer = iframeDoc.getElementById("visualizer");
                        if (visualizer) {
                            visualizer.setAttribute("filepath", this.filepath);
                            visualizer.setAttribute("timestamp", this.timestamp.toString());
                            console.log(`[Visualizer] Updated: ${this.filepath}`);
                        } else {
                            console.warn("[Visualizer] Script element not found in iframe");
                        }
                    }
                } catch (error) {
                    console.error("[Visualizer] Error accessing iframe:", error);
                }
            }
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
            const result = onExecuted?.apply(this, arguments);

            if (message?.previews?.[0]?.filepath) {
                this.visualizer?.updateVisual(message.previews[0].filepath);
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
