// LHM ComfyUI Node - Client-side Extensions
app.registerExtension({
    name: "lhm.humanreconstruction",
    async setup() {
        // Register event listeners for progress updates
        function progressHandler(event) {
            // Display progress updates in the UI
            const { value, text } = event.detail;
            
            // Log to console
            console.log(`LHM Progress: ${value}% - ${text}`);
            
            // You could update UI elements here if needed
            // This is useful for long-running operations like model loading
        }
        
        // Add a custom CSS class for LHM nodes to style them uniquely
        const style = document.createElement('style');
        style.innerHTML = `
            .lhm-node {
                background: linear-gradient(45deg, rgba(51,51,51,1) 0%, rgba(75,75,75,1) 100%);
                border: 2px solid #5a8db8 !important;
            }
            .lhm-node .title {
                color: #a3cfff !important;
                text-shadow: 0px 0px 3px rgba(0,0,0,0.5);
            }
        `;
        document.head.appendChild(style);
        
        // Register event listeners
        app.api.addEventListener("lhm.progress", progressHandler);
    },
    
    // Add custom behavior when node is added to graph
    nodeCreated(node) {
        if (node.type === "LHMReconstructionNode") {
            // Add custom class to the node element
            node.element.classList.add("lhm-node");
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