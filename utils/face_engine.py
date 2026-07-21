from insightface.app import FaceAnalysis

from utils.config import onnx_providers


class FaceEngine:
    def __init__(self):
        providers = onnx_providers()
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
        )
        # ctx_id 0 = first GPU; -1 = CPU
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def get_faces(self, image):
        return self.app.get(image)
