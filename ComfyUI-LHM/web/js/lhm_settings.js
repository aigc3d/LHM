// LHM ComfyUI Node - Settings Configuration
import { app } from "../../scripts/app.js";

// Register extension settings
app.registerExtension({
    name: "lhm.settings",
    
    async setup() {
        // Create a settings section for LHM
        const configSection = document.createElement('div');
        configSection.innerHTML = `
            <h3>LHM Human Reconstruction</h3>
            <div class="lhm-settings">
                <label>Progress Bar Color: 
                    <input type="color" id="lhm-progress-color" value="#5a8db8" />
                </label>
                <br />
                <label>Animation Preview FPS: 
                    <input type="number" id="lhm-fps" min="1" max="60" value="24" />
                </label>
                <br />
                <label>Memory Optimization: 
                    <select id="lhm-memory-mode">
                        <option value="balanced">Balanced</option>
                        <option value="performance">Performance (uses more memory)</option>
                        <option value="conservative">Conservative (uses less memory)</option>
                    </select>
                </label>
                <br />
                <label>Auto-unload unused models: 
                    <input type="checkbox" id="lhm-auto-unload" checked />
                </label>
                <br />
                <label>Debug Mode: 
                    <input type="checkbox" id="lhm-debug-mode" />
                </label>
            </div>
            <button id="lhm-reset-settings">Reset to Defaults</button>
        `;
        
        // Get the settings element from ComfyUI
        const settings = document.querySelector(".comfy-settings");
        settings?.appendChild(configSection);
        
        // Save settings to localStorage
        function saveSettings() {
            const settings = {
                progressColor: document.getElementById("lhm-progress-color").value,
                fps: document.getElementById("lhm-fps").value,
                memoryMode: document.getElementById("lhm-memory-mode").value,
                autoUnload: document.getElementById("lhm-auto-unload").checked,
                debugMode: document.getElementById("lhm-debug-mode").checked
            };
            
            localStorage.setItem("lhm_node_settings", JSON.stringify(settings));
            
            // Dispatch event so other scripts can react to settings changes
            document.dispatchEvent(new CustomEvent("lhm-settings-changed", { 
                detail: settings 
            }));
        }
        
        // Load settings from localStorage
        function loadSettings() {
            const savedSettings = localStorage.getItem("lhm_node_settings");
            if (savedSettings) {
                const settings = JSON.parse(savedSettings);
                
                // Apply the settings to the UI
                document.getElementById("lhm-progress-color").value = settings.progressColor || "#5a8db8";
                document.getElementById("lhm-fps").value = settings.fps || 24;
                document.getElementById("lhm-memory-mode").value = settings.memoryMode || "balanced";
                document.getElementById("lhm-auto-unload").checked = settings.autoUnload !== undefined ? settings.autoUnload : true;
                document.getElementById("lhm-debug-mode").checked = settings.debugMode || false;
            }
        }
        
        // Reset settings to defaults
        function resetSettings() {
            document.getElementById("lhm-progress-color").value = "#5a8db8";
            document.getElementById("lhm-fps").value = 24;
            document.getElementById("lhm-memory-mode").value = "balanced";
            document.getElementById("lhm-auto-unload").checked = true;
            document.getElementById("lhm-debug-mode").checked = false;
            
            saveSettings();
        }
        
        // Add event listeners
        document.getElementById("lhm-progress-color")?.addEventListener("change", saveSettings);
        document.getElementById("lhm-fps")?.addEventListener("change", saveSettings);
        document.getElementById("lhm-memory-mode")?.addEventListener("change", saveSettings);
        document.getElementById("lhm-auto-unload")?.addEventListener("change", saveSettings);
        document.getElementById("lhm-debug-mode")?.addEventListener("change", saveSettings);
        document.getElementById("lhm-reset-settings")?.addEventListener("click", resetSettings);
        
        // Load saved settings
        loadSettings();
    },
    
    // Provide helper to access settings from other extensions
    async getSettings() {
        const savedSettings = localStorage.getItem("lhm_node_settings");
        if (savedSettings) {
            return JSON.parse(savedSettings);
        }
        
        // Default settings
        return {
            progressColor: "#5a8db8",
            fps: 24,
            memoryMode: "balanced",
            autoUnload: true,
            debugMode: false
        };
    }
}); 