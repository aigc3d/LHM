/**
 * LHM Node Client-Side JavaScript
 * Handles progress updates and custom styling for the LHM node in ComfyUI
 */

import { app } from "/scripts/app.js";

// Track our node instances
const lhmNodes = {};

// Wait for the document to load and ComfyUI to initialize
document.addEventListener("DOMContentLoaded", () => {
    // Register event listeners once app is ready
    setTimeout(() => {
        registerLHMNode();
        setupWebsocketListeners();
    }, 1000);
});

/**
 * Setup websocket listeners for progress updates
 */
function setupWebsocketListeners() {
    // Listen for progress updates
    app.registerExtension({
        name: "LHM.ProgressUpdates",
        init() {
            // Add socket listeners
            const onSocketMessage = function(event) {
                try {
                    const message = JSON.parse(event.data);
                    
                    // Handle LHM progress updates
                    if (message?.type === "lhm.progress") {
                        const data = message.data;
                        const nodeId = data.node_id;
                        const progress = data.value;
                        const text = data.text || "";
                        
                        // Update the node if we have it registered
                        if (nodeId && lhmNodes[nodeId]) {
                            updateNodeProgress(nodeId, progress, text);
                        } else {
                            // Log progress when node ID is not available
                            console.log(`LHM progress (unknown node): ${progress}% - ${text}`);
                        }
                    }
                } catch (error) {
                    console.error("Error processing websocket message:", error);
                }
            };
            
            // Add listener for incoming messages
            if (app.socket) {
                app.socket.addEventListener("message", onSocketMessage);
            }
        }
    });
}

/**
 * Register handlers for the LHM nodes
 */
function registerLHMNode() {
    // Find existing ComfyUI node registration system
    if (!app.registerExtension) {
        console.error("Cannot register LHM node - ComfyUI app.registerExtension not found");
        return;
    }
    
    app.registerExtension({
        name: "LHM.NodeSetup",
        async beforeRegisterNodeDef(nodeType, nodeData) {
            // Check if this is our node type
            if (nodeData.name === "LHMReconstructionNode" || nodeData.name === "LHMTestNode") {
                // Store original methods we'll be enhancing
                const onNodeCreated = nodeType.prototype.onNodeCreated;
                const onRemoved = nodeType.prototype.onRemoved;
                
                // Add our custom progress bar
                addProgressBarWidget(nodeType);
                
                // Add our own widget for displaying progress text
                addProgressTextWidget(nodeType);
                
                // Replace the onNodeCreated method
                nodeType.prototype.onNodeCreated = function() {
                    // Add this node to our tracking
                    lhmNodes[this.id] = {
                        instance: this,
                        progress: 0,
                        text: "Initialized",
                    };
                    
                    // Set initial progress
                    updateNodeProgress(this.id, 0, "Ready");

                    // Add custom styling class to the node
                    const element = document.getElementById(this.id);
                    if (element) {
                        element.classList.add("lhm-node");
                    }
                    
                    // Call the original method if it exists
                    if (onNodeCreated) {
                        onNodeCreated.apply(this, arguments);
                    }
                };
                
                // Replace the onRemoved method
                nodeType.prototype.onRemoved = function() {
                    // Remove this node from our tracking
                    delete lhmNodes[this.id];
                    
                    // Call the original method if it exists
                    if (onRemoved) {
                        onRemoved.apply(this, arguments);
                    }
                };
            }
        },
        async nodeCreated(node) {
            // Additional setup when a node is created in the graph
            if (node.type === "LHMReconstructionNode" || node.type === "LHMTestNode") {
                // Add custom styling
                const element = document.getElementById(node.id);
                if (element) {
                    element.classList.add("lhm-node");
                }
            }
        }
    });
    
    // Add custom CSS for styling the node
    addCustomCSS();
}

/**
 * Add a progress bar widget to the node
 */
function addProgressBarWidget(nodeType) {
    // Get the node's widgets system
    const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
    
    // Add our progress bar widget class
    class ProgressBarWidget {
        constructor(node) {
            this.node = node;
            this.value = 0;
            this.visible = true;
        }

        // Draw the widget in the node
        draw(ctx, node, width, pos, height) {
            if (!this.visible) return;
            
            // Draw progress bar background
            const margin = 10;
            ctx.fillStyle = "#2a2a2a";
            ctx.fillRect(margin, pos[1], width - margin * 2, 10);
            
            // Draw progress bar fill
            const progress = Math.max(0, Math.min(this.value, 100)) / 100;
            ctx.fillStyle = progress > 0 ? "#4CAF50" : "#555";
            ctx.fillRect(margin, pos[1], (width - margin * 2) * progress, 10);
            
            // Add percentage text
            ctx.fillStyle = "#fff";
            ctx.font = "10px Arial";
            ctx.textAlign = "center";
            ctx.fillText(
                Math.round(progress * 100) + "%",
                width / 2,
                pos[1] + 8
            );
            
            return 14; // Height used by the widget
        }
    }
    
    // Create an instance of the widget when the node is created
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() {
        if (!this.widgets) {
            this.widgets = [];
        }
        
        // Add our progress bar widget first
        this.progressBar = new ProgressBarWidget(this);
        this.widgets.push(this.progressBar);
        
        // Call the original method
        if (origOnNodeCreated) {
            origOnNodeCreated.apply(this, arguments);
        }
    };
}

/**
 * Add a progress text widget to the node
 */
function addProgressTextWidget(nodeType) {
    // Define our custom progress text widget
    class ProgressTextWidget {
        constructor(node) {
            this.node = node;
            this.text = "Ready";
            this.visible = true;
        }
        
        // Draw the widget
        draw(ctx, node, width, pos, height) {
            if (!this.visible) return;
            
            // Draw the status text
            const margin = 10;
            ctx.fillStyle = "#ddd";
            ctx.font = "11px Arial";
            ctx.textAlign = "left";
            
            // Split text into multiple lines if needed
            const maxWidth = width - margin * 2;
            const words = this.text.split(' ');
            let line = '';
            let y = pos[1] + 3;
            let lineHeight = 14;
            
            for (let i = 0; i < words.length; i++) {
                const testLine = line + (line ? ' ' : '') + words[i];
                const metrics = ctx.measureText(testLine);
                
                if (metrics.width > maxWidth && i > 0) {
                    // Draw the current line and start a new one
                    ctx.fillText(line, margin, y);
                    line = words[i];
                    y += lineHeight;
                } else {
                    line = testLine;
                }
            }
            
            // Draw the final line
            ctx.fillText(line, margin, y);
            
            return y - pos[1] + lineHeight; // Return height used
        }
    }
    
    // Add our widget when the node is created
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function() {
        if (!this.widgets) {
            this.widgets = [];
        }
        
        // Add our text widget after the progress bar
        this.progressText = new ProgressTextWidget(this);
        this.widgets.push(this.progressText);
        
        // Call the original method
        if (origOnNodeCreated) {
            origOnNodeCreated.apply(this, arguments);
        }
    };
}

/**
 * Update the progress display for a node
 */
function updateNodeProgress(nodeId, progress, text) {
    if (!lhmNodes[nodeId]) return;
    
    const nodeInfo = lhmNodes[nodeId];
    const node = nodeInfo.instance;
    
    if (node && node.progressBar) {
        // Update progress bar
        node.progressBar.value = progress;
        nodeInfo.progress = progress;
        
        // Update text
        if (text && node.progressText) {
            node.progressText.text = text;
            nodeInfo.text = text;
        }
        
        // Force redraw of the node
        app.graph.setDirtyCanvas(true, false);
    }
}

/**
 * Add custom CSS styles for LHM nodes
 */
function addCustomCSS() {
    const style = document.createElement('style');
    style.textContent = `
        .lhm-node {
            --lhm-primary-color: #4CAF50;
            --lhm-secondary-color: #2E7D32;
            --lhm-background: #1E1E1E;
        }
        
        .lhm-node .nodeheader {
            background: linear-gradient(to right, var(--lhm-secondary-color), var(--lhm-primary-color));
            color: white;
        }
        
        .lhm-node .nodeheader .nodeTitle {
            text-shadow: 0px 1px 2px rgba(0,0,0,0.5);
            font-weight: bold;
        }
        
        .lhm-node.LHMReconstructionNode .nodeheader {
            background: linear-gradient(to right, #2E7D32, #4CAF50);
        }
        
        .lhm-node.LHMTestNode .nodeheader {
            background: linear-gradient(to right, #0D47A1, #2196F3);
        }
    `;
    document.head.appendChild(style);
} 