"""
API Routes for LHM node
Handles registration of node instances and provides API endpoints for the LHM node.
"""

import os
import sys
import json
import time
from collections import defaultdict

# Track node instances by their ID
node_instances = {}

# Try importing the PromptServer for API registration
try:
    from server import PromptServer
    has_prompt_server = True
except ImportError:
    has_prompt_server = False
    print("Warning: PromptServer not found, API routes will not be available")
    
    # Create a dummy PromptServer for compatibility
    class DummyPromptServer:
        instance = None
        
        def __init__(self):
            self.routes = {}
            
        def add_route(self, route_path, handler, **kwargs):
            self.routes[route_path] = handler
            print(f"Registered route {route_path} (dummy)")
            
        @staticmethod
        def send_sync(*args, **kwargs):
            pass
    
    PromptServer = DummyPromptServer
    PromptServer.instance = PromptServer()

def register_node_instance(node_id, instance):
    """Register a node instance for API access."""
    node_instances[node_id] = instance
    print(f"Registered LHM node: {node_id}")
    
def unregister_node_instance(node_id):
    """Unregister a node instance."""
    if node_id in node_instances:
        del node_instances[node_id]
        print(f"Unregistered LHM node: {node_id}")

def cleanup_node_resources(node_id):
    """Clean up resources used by a specific node"""
    instance = node_instances.get(node_id)
    if instance:
        # Set models to None to allow garbage collection
        if hasattr(instance, 'model') and instance.model is not None:
            instance.model = None
        
        if hasattr(instance, 'pose_estimator') and instance.pose_estimator is not None:
            instance.pose_estimator = None
            
        if hasattr(instance, 'face_detector') and instance.face_detector is not None:
            instance.face_detector = None
        
        # Explicitly run garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return True
    return False

def cleanup_all_resources():
    """Clean up resources used by all LHM nodes"""
    for node_id in list(node_instances.keys()):
        cleanup_node_resources(node_id)
    return True

@PromptServer.instance.routes.post("/extensions/lhm/unload_resources")
async def unload_resources(request):
    """API endpoint to unload resources when requested by the client"""
    try:
        json_data = await request.json()
        unload = json_data.get("unload", False)
        node_id = json_data.get("node_id", None)
        
        if unload:
            if node_id:
                # Unload resources for a specific node
                success = cleanup_node_resources(node_id)
            else:
                # Unload all resources
                success = cleanup_all_resources()
                
            return web.json_response({"success": success})
        
        return web.json_response({"success": False, "error": "No action requested"})
        
    except Exception as e:
        print(f"Error in unload_resources: {str(e)}")
        return web.json_response({"success": False, "error": str(e)})

def setup_routes():
    """Set up API routes for the LHM node."""
    if not has_prompt_server:
        print("Skipping LHM API route setup - PromptServer not available")
        return
        
    # API endpoint to get node status
    @PromptServer.instance.routes.get("/lhm/node/status")
    async def api_get_node_status(request):
        """Return status information for all registered LHM nodes."""
        try:
            node_status = {}
            for node_id, instance in node_instances.items():
                node_status[node_id] = {
                    "node_id": node_id,
                    "type": instance.__class__.__name__,
                    "is_running": getattr(instance, "is_running", False)
                }
            
            return {"status": "success", "nodes": node_status}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
    
    # API endpoint to send progress updates to the client
    @PromptServer.instance.routes.post("/lhm/progress/{node_id}")
    async def api_update_progress(request):
        """Update the progress of a specific node."""
        try:
            node_id = request.match_info.get("node_id", None)
            if not node_id or node_id not in node_instances:
                return {"status": "error", "message": f"Node {node_id} not found"}
                
            data = await request.json()
            value = data.get("value", 0)
            text = data.get("text", "")
            
            # Send the progress update to clients
            PromptServer.instance.send_sync("lhm.progress", {
                "node_id": node_id,
                "value": value,
                "text": text
            })
            
            return {"status": "success"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
            
    # API endpoint to check if models are loaded
    @PromptServer.instance.routes.get("/lhm/models/status")
    async def api_get_model_status(request):
        """Return information about loaded LHM models."""
        try:
            model_status = {}
            for node_id, instance in node_instances.items():
                # Get model info if available
                if hasattr(instance, "model") and instance.model is not None:
                    model_status[node_id] = {
                        "loaded": True,
                        "version": getattr(instance, "last_model_version", "unknown"),
                        "device": str(getattr(instance, "device", "unknown"))
                    }
                else:
                    model_status[node_id] = {
                        "loaded": False
                    }
            
            return {"status": "success", "models": model_status}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
            
    print("LHM API routes registered successfully") 