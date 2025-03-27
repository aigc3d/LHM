import os
import imageio
import numpy as np
import torch
from tqdm import tqdm

from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.structures import Meshes

class NormalShader(ShaderBase):
    def __init__(self, device = "cpu", **kwargs):
        super().__init__(device=device, **kwargs)

    def forward(self, fragments, meshes, **kwargs):
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = fragments.bary_coords.clone()
        texels = texels.permute(0, 3, 1, 2, 4)
        texels = texels * 2 - 1  # 将 bary_coords 映射到 [-1, 1]

        # 获取法线
        verts_normals = meshes.verts_normals_packed()
        faces_normals = verts_normals[meshes.faces_packed()]
        bary_coords = fragments.bary_coords

        pixel_normals = (bary_coords[..., None] * faces_normals[fragments.pix_to_face]).sum(dim=-2)
        pixel_normals = pixel_normals / pixel_normals.norm(dim=-1, keepdim=True)

        # 将法线映射到颜色空间
        # colors = (pixel_normals + 1) / 2  # 将法线映射到 [0, 1]
        colors = torch.clamp(pixel_normals, -1, 1)
        print(colors.shape)
        mask = (fragments.pix_to_face > 0).float()
        colors = torch.cat([colors, mask.unsqueeze(-1)], dim=-1)
        # colors[fragments.pix_to_face < 0] = 0

        # 混合颜色
        # images = self.blend(texels, colors, fragments, blend_params)
        return colors

def overlay_image_onto_background(image, mask, bbox, background):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    out_image = background.copy()
    bbox = bbox[0].int().cpu().numpy().copy()
    roi_image = out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    if len(roi_image) < 1 or len(roi_image[1]) < 1:
        return out_image
    try:
        roi_image[mask] = image[mask]
    except Exception as e:
        raise e
    out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi_image

    return out_image


def update_intrinsics_from_bbox(K_org, bbox):
    '''
    update intrinsics for cropped images
    '''
    device, dtype = K_org.device, K_org.dtype
    
    K = torch.zeros((K_org.shape[0], 4, 4)
    ).to(device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1
    
    image_sizes = []
    for idx, bbox in enumerate(bbox):
        left, upper, right, lower = bbox
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))

    return K, image_sizes


def perspective_projection(x3d, K, R=None, T=None):
    if R != None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T != None:
        x3d = x3d + T.transpose(1, 2)

    x2d = torch.div(x3d, x3d[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    return x2d


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    left = torch.clamp(X.min(1)[0][:, 0], min=0, max=img_w)
    right = torch.clamp(X.max(1)[0][:, 0], min=0, max=img_w)
    top = torch.clamp(X.min(1)[0][:, 1], min=0, max=img_h)
    bottom = torch.clamp(X.max(1)[0][:, 1], min=0, max=img_h)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = (right - left)
    height = (bottom - top)

    new_left = torch.clamp(cx - width/2 * scaleFactor, min=0, max=img_w-1)
    new_right = torch.clamp(cx + width/2 * scaleFactor, min=1, max=img_w)
    new_top = torch.clamp(cy - height / 2 * scaleFactor, min=0, max=img_h-1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor, min=1, max=img_h)

    bbox = torch.stack((new_left.detach(), new_top.detach(),
                        new_right.detach(), new_bottom.detach())).int().float().T
    return bbox


class Renderer():
    def __init__(self, width, height, K, device, faces=None):

        self.width = width
        self.height = height
        self.K = K

        self.device = device

        if faces is not None:
            self.faces = torch.from_numpy(
                (faces).astype('int')
            ).unsqueeze(0).to(self.device)

        self.initialize_camera_params()
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])
        self.create_renderer()

    def create_camera(self, R=None, T=None):
        if R is not None:
            self.R = R.clone().view(1, 3, 3).to(self.device)
        if T is not None:
            self.T = T.clone().view(1, 3).to(self.device)

        return PerspectiveCameras(
            device=self.device,
            R=self.R.mT,
            T=self.T,
            K=self.K_full,
            image_size=self.image_sizes,
            in_ndc=False)

    def create_renderer(self):
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=self.image_sizes[0],
                    blur_radius=1e-5,),
            ),
            shader=SoftPhongShader(
                device=self.device,
                lights=self.lights,
            )
        )

    def create_normal_renderer(self):
        normal_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=RasterizationSettings(
                    image_size=self.image_sizes[0],
                ),
            ),
            shader=NormalShader(device=self.device),
        )
        return normal_renderer

    def initialize_camera_params(self):
        """Hard coding for camera parameters
        TODO: Do some soft coding"""

        # Extrinsics
        self.R = torch.diag(
            torch.tensor([1, 1, 1])
        ).float().to(self.device).unsqueeze(0)

        self.T = torch.tensor(
            [0, 0, 0]
        ).unsqueeze(0).float().to(self.device)

        # Intrinsics
        self.K = self.K.unsqueeze(0).float().to(self.device)
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        self.cameras = self.create_camera()

    def render_normal(self, vertices):
        vertices = vertices.unsqueeze(0)

        mesh = Meshes(verts=vertices, faces=self.faces)
        normal_renderer = self.create_normal_renderer()
        results = normal_renderer(mesh)
        results = torch.flip(results, [1, 2])
        return results

    def render_mesh(self, vertices, background, colors=[0.8, 0.8, 0.8]):

        self.update_bbox(vertices[::50], scale=1.2)
        vertices = vertices.unsqueeze(0)

        if colors[0] > 1: colors = [c / 255. for c in colors]
        verts_features = torch.tensor(colors).reshape(1, 1, 3).to(device=vertices.device, dtype=vertices.dtype)
        verts_features = verts_features.repeat(1, vertices.shape[1], 1)
        textures = TexturesVertex(verts_features=verts_features)

        mesh = Meshes(verts=vertices,
                      faces=self.faces,
                      textures=textures,)

        materials = Materials(
            device=self.device,
            specular_color=(colors, ),
            shininess=0
            )

        results = torch.flip(
            self.renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights),
            [1, 2]
        )
        image = results[0, ..., :3] * 255
        mask = results[0, ..., -1] > 1e-3

        image = overlay_image_onto_background(image, mask, self.bboxes, background.copy())
        self.reset_bbox()
        return image

    def update_bbox(self, x3d, scale=2.0, mask=None):
        """ Update bbox of cameras from the given 3d points

        x3d: input 3D keypoints (or vertices), (num_frames, num_points, 3)
        """
        if x3d.size(-1) != 3:
            x2d = x3d.unsqueeze(0)
        else:
            x2d = perspective_projection(x3d.unsqueeze(0), self.K, self.R, self.T.reshape(1, 3, 1))

        if mask is not None:
            x2d = x2d[:, ~mask]
        bbox = compute_bbox_from_points(x2d, self.width, self.height, scale)
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def reset_bbox(self,):
        bbox = torch.zeros((1, 4)).float().to(self.device)
        bbox[0, 2] = self.width
        bbox[0, 3] = self.height
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

class RendererUtil():
    def __init__(self, K, w, h, device, faces, keep_origin=True):
        self.keep_origin = keep_origin
        self.default_R = torch.eye(3)
        self.default_T = torch.zeros(3)
        self.device = device
        self.renderer =  Renderer(w, h, K, device, faces)

    def set_extrinsic(self, R, T):
        self.default_R = R
        self.default_T = T

    def render_normal(self, verts_list):
        if not len(verts_list) == 1:
            return None
        
        self.renderer.create_camera(self.default_R, self.default_T)
        normal_map = self.renderer.render_normal(verts_list[0])
        return normal_map[0, :, :, 0]

    def render_frame(self, humans, pred_rend_array, verts_list=None, color_list=None):
        if not isinstance(pred_rend_array, np.ndarray):
            pred_rend_array = np.asarray(pred_rend_array)
        self.renderer.create_camera(self.default_R, self.default_T)
        _img = pred_rend_array
        if humans is not None:
            for human in humans:
                _img = self.renderer.render_mesh(human['v3d'].to(self.device), _img)
        else:
            for i, verts in enumerate(verts_list):
                if color_list is None:
                    _img = self.renderer.render_mesh(verts.to(self.device), _img)
                else:
                    _img = self.renderer.render_mesh(verts.to(self.device), _img, color_list[i])
        if self.keep_origin:
            _img = np.concatenate([np.asarray(pred_rend_array), _img],1).astype(np.uint8)
        return _img

    def render_video(self, results, pil_bis_frames, fps, out_path):
        writer = imageio.get_writer(
             out_path,
             fps=fps, mode='I', format='FFMPEG', macro_block_size=1
        )
        for i, humans in enumerate(tqdm(results)):
            pred_rend_array = pil_bis_frames[i]
            _img = self.render_frame( humans, pred_rend_array)
            try:
                writer.append_data(_img)
            except:
                print('Error in writing video')
                print(type(_img))
        writer.close()
def render_frame(renderer, humans, pred_rend_array, default_R, default_T, device, keep_origin=True):
    
    if not isinstance(pred_rend_array, np.ndarray):
        pred_rend_array = np.asarray(pred_rend_array)
    renderer.create_camera(default_R, default_T)
    _img = pred_rend_array
    if humans is None:
        humans = []
    if isinstance(humans, dict):
        humans = [humans]
    for human in humans:
        if isinstance(human, dict):
            v3d = human['v3d'].to(device)
        else:
            v3d = human
        _img = renderer.render_mesh(v3d, _img)
        
    if keep_origin:
        _img = np.concatenate([np.asarray(pred_rend_array), _img],1).astype(np.uint8)
    return _img


def render_video(results, faces, K, pil_bis_frames, fps, out_path, device, keep_origin=True):    
    # results [F, N, ...]
    if isinstance(pil_bis_frames[0], np.ndarray):
        height, width, _ = pil_bis_frames[0].shape
    else:
        shape = pil_bis_frames[0].size
        width, height = shape[1], shape[0]
    renderer = Renderer(width, height, K[0], device, faces)
    
    
    # build default camera
    default_R, default_T = torch.eye(3), torch.zeros(3)
    
    writer = imageio.get_writer(
             out_path,
             fps=fps, mode='I', format='FFMPEG', macro_block_size=1
        )
    for i, humans in enumerate(tqdm(results)):
        pred_rend_array = pil_bis_frames[i]
        _img = render_frame(renderer, humans, pred_rend_array, default_R, default_T, device, keep_origin)
        try:
            writer.append_data(_img)
        except:
            print('Error in writing video')
            print(type(_img))
    writer.close()
