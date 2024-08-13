import torch
import numpy
import PIL.Image

class DwposeDetector:
    from .openpose import OpenposeDetector
    detector = OpenposeDetector()
    detector.load_dw_model()
    try:
        detector((numpy.asarray(PIL.Image.new(mode="RGB", size=(1024, 1024), color=(255, 255, 255))).astype(numpy.uint8)), use_dw_pose=True)
    except:
        import traceback
        traceback.print_exc()

    @classmethod
    def INPUT_TYPES(s):
        from .utils import create_node_input_types
        input_types = create_node_input_types(
            detect_hand=(["enable", "disable"], {"default": "enable"}),
            detect_body=(["enable", "disable"], {"default": "enable"}),
            detect_face=(["enable", "disable"], {"default": "enable"})
        )
        input_types["optional"] = {
            **input_types["optional"],
            "bbox_detector": (
                ["yolox_l.torchscript.pt", "yolox_l.onnx", "yolo_nas_l_fp16.onnx", "yolo_nas_m_fp16.onnx", "yolo_nas_s_fp16.onnx"],
                {"default": "yolox_l.onnx"}
            ),
            "pose_estimator": (["dw-ll_ucoco_384_bs5.torchscript.pt", "dw-ll_ucoco_384.onnx", "dw-ll_ucoco.onnx"], {"default": "dw-ll_ucoco_384_bs5.torchscript.pt"})
        }
        return input_types

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    FUNCTION = "__call__"


    def __init__(self, *args, **kargs):
        print("use qw dwpose")
        pass

    @classmethod
    def from_pretrained(cls, *args, **kargs):
        return cls()

    def  __call__(self, image,  *args, **kargs):
        return [self.detector(image, *args, use_dw_pose=True, json_pose_callback=None, **kargs), {}]

    def estimate_pose(self, image, *args, **kargs):
        image = image.cpu().numpy()
        if len(image.shape) == 4:
            images = image
        elif len(image.shape) == 3:
            images = [image, ]
        else:
            raise RuntimeError("image shape error, qw dwpose")
        
        out_tensor = None
        for i, image in enumerate(images):
            np_image = numpy.asarray(image * 255., dtype=numpy.uint8)
            np_result = self.detector(image, *args, use_dw_pose=True, json_pose_callback=None, **kargs)
            out = torch.from_numpy(np_result.astype(numpy.float32) / 255.0)
            if out_tensor is None:
                out_tensor = torch.zeros(len(images), *out.shape, dtype=torch.float32)
            out_tensor[i] = out
        return out_tensor, {}

        # need return tensor # 202403之前可用
        # return [resImage, {}]
        # resImage = self.detector(image, *args, use_dw_pose=True, json_pose_callback=None, **kargs)
        # return resImage, {}



NODE_CLASS_MAPPINGS = {
    "qwDWPreprocessor": DwposeDetector
}
