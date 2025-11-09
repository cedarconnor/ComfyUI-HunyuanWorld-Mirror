import { app } from "../../scripts/app.js";

/**
 * HunyuanWorld-Mirror 3D Visualization Extension
 * Adds preview capabilities for Gaussian Splatting and Point Clouds
 */

class Visualizer {
    constructor(node, typeName) {
        this.node = node;
        this.typeName = typeName;
        this.iframe = null;
        this.filepath = "";
        this.timestamp = Date.now();
    }

    createIframe() {
        const iframe = document.createElement("iframe");
        iframe.style.width = "100%";
        iframe.style.height = "100%";
        iframe.style.border = "none";
        iframe.src = `/extensions/ComfyUI-HunyuanWorld-Mirror/html/${this.typeName}.html`;
        return iframe;
    }

    updateVisual(filepath) {
        if (filepath && filepath !== this.filepath) {
            this.filepath = filepath;
            this.timestamp = Date.now();

            if (this.iframe && this.iframe.contentWindow) {
                const visualizer = this.iframe.contentDocument.getElementById("visualizer");
                if (visualizer) {
                    visualizer.setAttribute("filepath", this.filepath);
                    visualizer.setAttribute("timestamp", this.timestamp.toString());
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

            const widget = this.addCustomWidget({
                name: "3d_preview",
                type: "3d_preview",
                value: "",
                draw: function(ctx, node, width, y) {
                    // Widget drawing handled by iframe
                },
                computeSize: function(width) {
                    return [width, 600];  // Fixed height for viewer
                },
                serialize: false
            });

            widget.visualizer = visualizer;

            // Create and append iframe
            visualizer.iframe = visualizer.createIframe();
            this.visualizer = visualizer;

            return result;
        };

        const onDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function (ctx) {
            const result = onDrawBackground?.apply(this, arguments);

            if (this.flags.collapsed || !this.visualizer?.iframe) {
                return result;
            }

            // Position iframe over the node canvas
            const parent = this.graph.list_of_graphcanvas[0].canvas.parentElement;
            if (this.visualizer.iframe.parentElement !== parent) {
                parent.appendChild(this.visualizer.iframe);
            }

            // Calculate position
            const transform = this.graph.list_of_graphcanvas[0].ds;
            const x = (this.pos[0] + 15) * transform.scale + transform.offset[0];
            const y = (this.pos[1] + 100) * transform.scale + transform.offset[1];
            const width = (this.size[0] - 30) * transform.scale;
            const height = 600 * transform.scale;

            this.visualizer.iframe.style.position = "absolute";
            this.visualizer.iframe.style.left = x + "px";
            this.visualizer.iframe.style.top = y + "px";
            this.visualizer.iframe.style.width = width + "px";
            this.visualizer.iframe.style.height = height + "px";
            this.visualizer.iframe.style.zIndex = "5";
            this.visualizer.iframe.style.pointerEvents = "auto";

            return result;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (this.visualizer?.iframe) {
                this.visualizer.iframe.remove();
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
