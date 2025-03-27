// LHM ComfyUI Node - Client-side Extensions
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "lhm.humanreconstruction",
    async setup() {
        // Store settings locally
        let settings = {
            progressColor: "#5a8db8",
            fps: 24,
            memoryMode: "balanced",
            autoUnload: true,
            debugMode: false
        };
        
        // Try to load settings (will be available if lhm_settings.js is loaded)
        try {
            const lhmSettings = app.extensions["lhm.settings"];
            if (lhmSettings && lhmSettings.getSettings) {
                settings = await lhmSettings.getSettings();
            }
        } catch (e) {
            console.log("LHM: Settings extension not found, using defaults");
        }
        
        // Listen for settings changes
        document.addEventListener("lhm-settings-changed", (event) => {
            settings = event.detail;
            applySettings();
        });
        
        // Apply current settings
        function applySettings() {
            // Update progress bar color in CSS
            const progressBarStyle = document.getElementById("lhm-progress-bar-style");
            if (progressBarStyle) {
                progressBarStyle.innerHTML = `
                    .lhm-progress-bar .progress {
                        background-color: ${settings.progressColor} !important;
                    }
                `;
            }
            
            // Configure memory usage based on settings
            if (settings.memoryMode === "conservative") {
                // Add code to reduce memory usage
                app.nodeOutputsCacheLimit = Math.min(app.nodeOutputsCacheLimit, 2);
            } else if (settings.memoryMode === "performance") {
                // Add code to prioritize performance
                app.nodeOutputsCacheLimit = Math.max(app.nodeOutputsCacheLimit, 10);
            }
            
            // Apply debug mode
            if (settings.debugMode) {
                // Enable debug logging
                app.ui.settings.showDebugLogs = true;
                console.log("LHM: Debug mode enabled");
            }
        }
        
        // Register event listeners for progress updates
        function progressHandler(event) {
            // Display progress updates in the UI
            const { value, text } = event.detail;
            
            // Update any visible progress bars
            const progressBars = document.querySelectorAll(".lhm-progress-bar .progress");
            progressBars.forEach(bar => {
                bar.style.width = `${value}%`;
            });
            
            const progressTexts = document.querySelectorAll(".lhm-progress-text");
            progressTexts.forEach(textEl => {
                textEl.textContent = text;
            });
            
            // Log to console if in debug mode
            if (settings.debugMode) {
                console.log(`LHM Progress: ${value}% - ${text}`);
            }
        }
        
        // Add a custom CSS class for LHM nodes to style them uniquely
        const style = document.createElement('style');
        style.innerHTML = `
            .lhm-node {
                background: linear-gradient(45deg, rgba(51,51,51,1) 0%, rgba(75,75,75,1) 100%);
                border: 2px solid ${settings.progressColor} !important;
            }
            .lhm-node .title {
                color: #a3cfff !important;
                text-shadow: 0px 0px 3px rgba(0,0,0,0.5);
            }
            
            .lhm-progress-bar {
                width: 100%;
                height: 4px;
                background-color: #333;
                border-radius: 2px;
                overflow: hidden;
                margin-top: 5px;
            }
            
            .lhm-progress-bar .progress {
                height: 100%;
                background-color: ${settings.progressColor};
                width: 0%;
                transition: width 0.3s ease-in-out;
            }
            
            .lhm-progress-text {
                font-size: 11px;
                color: #ccc;
                text-align: center;
                margin-top: 2px;
            }
        `;
        document.head.appendChild(style);
        
        // Add a separate style element for progress bar that can be updated
        const progressBarStyle = document.createElement('style');
        progressBarStyle.id = "lhm-progress-bar-style";
        progressBarStyle.innerHTML = `
            .lhm-progress-bar .progress {
                background-color: ${settings.progressColor} !important;
            }
        `;
        document.head.appendChild(progressBarStyle);
        
        // Register event listeners
        app.api.addEventListener("lhm.progress", progressHandler);
        
        // Apply settings initially
        applySettings();
        
        // Clean up resources when workflow is cleared if auto-unload is enabled
        app.graph.addEventListener("clear", () => {
            if (settings.autoUnload) {
                // Send message to server to clean up resources
                app.api.fetchApi('/extensions/lhm/unload_resources', { 
                    method: 'POST',
                    body: JSON.stringify({ unload: true })
                });
                
                if (settings.debugMode) {
                    console.log("LHM: Resources unloaded");
                }
            }
        });
    },
    
    // Add custom behavior when node is added to graph
    nodeCreated(node) {
        if (node.type === "LHMReconstructionNode") {
            // Add custom class to the node element
            node.element.classList.add("lhm-node");
            
            // Add progress bar to node
            const container = node.domElements.content;
            const progressContainer = document.createElement('div');
            progressContainer.innerHTML = `
                <div class="lhm-progress-bar">
                    <div class="progress"></div>
                </div>
                <div class="lhm-progress-text">Ready</div>
            `;
            container.appendChild(progressContainer);
        }
    },
    
    // Custom widget rendering
    getCustomWidgets() {
        return {
            // Example of customizing a boolean widget
            BOOLEAN: (node, inputName, inputData) => {
                // Only customize for LHM nodes
                if (node.type !== "LHMReconstructionNode") {
                    return null;
                }
                
                // Customize labels for better UX
                if (inputName === "export_mesh") {
                    inputData.label_on = "Export 3D";
                    inputData.label_off = "Skip 3D";
                } else if (inputName === "remove_background") {
                    inputData.label_on = "No BG";  
                    inputData.label_off = "Keep BG";
                } else if (inputName === "recenter") {
                    inputData.label_on = "Center";
                    inputData.label_off = "Original";
                }
                
                return null; // Return null to use default widget with our customizations
            }
        };
    }
}); 