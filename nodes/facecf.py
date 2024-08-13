import logging
import os
import sys

from spandrel import ModelLoader


import comfy.utils
import cv2
import folder_paths
import numpy as np
import torch
from basicsr.utils import img2tensor, tensor2img
from comfy import model_management
from torchvision.transforms.functional import normalize

from basicsr.utils.registry import ARCH_REGISTRY
from . import stylegan2_bilinear_arch
import basicsr.archs

basicsr.archs.stylegan2_bilinear_arch = stylegan2_bilinear_arch

from .facexlib.facexlib.utils.face_restoration_helper import FaceRestoreHelper

dir_facerestore_models = os.path.join(folder_paths.models_dir, "facerestore_models")
dir_facedetection_models = os.path.join(folder_paths.models_dir, "facedetection")
os.makedirs(dir_facerestore_models, exist_ok=True)
os.makedirs(dir_facedetection_models, exist_ok=True)
folder_paths.folder_names_and_paths["facerestore_models"] = (
    [dir_facerestore_models],
    folder_paths.supported_pt_extensions,
)
folder_paths.folder_names_and_paths["facedetection_models"] = (
    [dir_facedetection_models],
    folder_paths.supported_pt_extensions,
)


class CropFace:
    face_helper = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "facedetection": (
                    [
                        "retinaface_resnet50",
                        "retinaface_mobile0.25",
                        "YOLOv5l",
                        "YOLOv5n",
                    ],
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "crop_face"

    CATEGORY = "facerestore_cf"

    def crop_face(self, image, facedetection):
        device = model_management.get_torch_device()
        if self.face_helper is None:
            self.face_helper = FaceRestoreHelper(
                1,
                face_size=512,
                crop_ratio=(1, 1),
                det_model=facedetection,
                save_ext="png",
                use_parse=True,
                device=device,
            )

        image_np = 255.0 * image.cpu().numpy()

        total_images = image_np.shape[0]
        out_images = np.ndarray(shape=(total_images, 512, 512, 3))
        next_idx = 0

        for i in range(total_images):
            cur_image_np = image_np[i, :, :, ::-1]

            original_resolution = cur_image_np.shape[0:2]

            if self.face_helper is None:
                return image

            self.face_helper.clean_all()
            self.face_helper.read_image(cur_image_np)
            self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
            self.face_helper.align_warp_face()

            faces_found = len(self.face_helper.cropped_faces)
            if faces_found == 0:
                next_idx += 1  # output black image for no face
            if out_images.shape[0] < next_idx + faces_found:
                print(out_images.shape)
                print((next_idx + faces_found, 512, 512, 3))
                print("aaaaa")
                out_images = np.resize(out_images, (next_idx + faces_found, 512, 512, 3))
                print(out_images.shape)
            for j in range(faces_found):
                cropped_face_1 = self.face_helper.cropped_faces[j]
                cropped_face_2 = img2tensor(cropped_face_1 / 255.0, bgr2rgb=True, float32=True)
                normalize(cropped_face_2, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_3 = cropped_face_2.unsqueeze(0).to(device)
                cropped_face_4 = tensor2img(cropped_face_3, rgb2bgr=True, min_max=(-1, 1)).astype("uint8")
                cropped_face_5 = cv2.cvtColor(cropped_face_4, cv2.COLOR_BGR2RGB)
                out_images[next_idx] = cropped_face_5
                next_idx += 1

        cropped_face_6 = np.array(out_images).astype(np.float32) / 255.0
        cropped_face_7 = torch.from_numpy(cropped_face_6)
        return (cropped_face_7,)


class UpscaleFaceImpletementdByQw2406:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "codeformer_fidelity": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1, "step": 0.05},
                ),
                "modelName": (["codeformer.pth", "GFPGANv1.3.pth"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "faceRestore"

    def __init__(self, model_name="codeformer.pth", upscale=2, **kargs):
        self.upscale = upscale
        self.device = model_management.get_torch_device()

        self.face_helper = FaceRestoreHelper(
            self.upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
        )
        self.modelName = None
        self._loadModel(model_name)

    def _loadModel(self, model_name):
        if model_name == self.modelName:
            print("use cache:%s" % model_name)
            return

        import folder_paths

        self.modelName = model_name

        if "codeformer" in model_name.lower():
            model_path = folder_paths.get_full_path("facerestore_models", model_name)
            codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            ).to(
                self.device,
            )
            checkpoint = torch.load(model_path)["params_ema"]
            codeformer_net.load_state_dict(checkpoint)
            self.model = codeformer_net.eval().to(self.device)
            self.useCodeformer = True
            return

        model_path = folder_paths.get_full_path("facerestore_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        self.model = ModelLoader().load_from_state_dict(sd).eval().to(self.device)
        self.useCodeformer = False

    def run(self, image, **kargs):
        """image, b, h, w, c/255"""
        return self.restore_face(image[0], **kargs)

    def restore_face(self, image, **kargs):
        out_image = self.__call__(image, **kargs)[-1]
        restored_img_np = np.array([out_image]).astype(np.float32) / 255.0
        restored_img_tensor = torch.from_numpy(restored_img_np)
        return (restored_img_tensor,)

    def __call__(self, img, only_center_face=False, paste_back=True, **kargs):
        modelName = kargs.get("modelName")
        if modelName:
            self._loadModel(modelName)
        codeformer_fidelity = kargs.get("codeformer_fidelity")
        img = 255.0 * img.cpu().numpy()
        self.face_helper.clean_all()
        self.face_helper.read_image(img)
        self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5, resize=640)
        self.face_helper.align_warp_face()

        for cropped_face in self.face_helper.cropped_faces:
            cropped_face = cropped_face[:, :, ::-1]
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
            try:
                param = {}
                if self.useCodeformer:
                    """
                    self._call_fn = lambda model, image: model(
                        image, weight=codeformer_fidelity
                    )
                    """
                    param = dict(w=codeformer_fidelity)
                with torch.no_grad():
                    output = self.model(cropped_face_t, **param)[0]
                    restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                restored_face = cropped_face
            restored_face = restored_face[:, :, ::-1]

            restored_face = restored_face.astype("uint8")
            self.face_helper.add_restored_face(restored_face)

        if paste_back:
            self.face_helper.get_inverse_affine(None)
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=None)
            return (
                self.face_helper.cropped_faces,
                self.face_helper.restored_faces,
                restored_img,
            )
        return self.face_helper.cropped_faces, self.face_helper.restored_faces, None


NODE_CLASS_MAPPINGS = {
    "FaceRestoreCFWithModelImplementedByQw2406": UpscaleFaceImpletementdByQw2406,
    "CropFaceWithModelImplementedByQw2406": CropFace,
}
