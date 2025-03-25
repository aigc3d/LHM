import json
from aiohttp import web
import torch
import gc
from server import PromptServer

# Store a reference to node instances for resource management
node_instances = {}

def register_node_instance(node_id, instance):
    """Register a node instance for resource management"""
    node_instances[node_id] = instance
    
def unregister_node_instance(node_id):
    """Unregister a node instance when it's deleted"""
    if node_id in node_instances:
        del node_instances[node_id]

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

# Make sure routes are registered
def setup_routes():
    """Ensure routes are properly set up"""
    # This function is called from __init__.py
    pass 