import numpy as np
import torch
import onnx

if torch.cuda.is_available():
    try:
        import onnxruntime_gpu as ort
    except ImportError:
        import onnxruntime as ort
else:
    import onnxruntime as ort


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class OnnxConverter:
    # https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

    def to_onnx(self, model, dummy_input:torch.tensor, onnx_path:str, verbose=False, export_params=True, **kwargs):
        # https://pytorch.org/docs/stable/onnx.html#functions
        model.eval()
        return torch.onnx.export(
                model, dummy_input, onnx_path, 
                verbose=verbose, export_params=export_params, 
                **kwargs
            )

    def from_onnx(self, onnx_path:str):
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        return model

    def create_session(self, onnx_path):
        return onnxruntime.InferenceSession(onnx_path)

    def runtime(self, onnx_session, X, onnx_format=False):
        onnx_input = {onnx_session.get_inputs()[0].name: to_numpy(X)}
        onnx_output = onnx_session.run(None, onnx_input)
        return onnx_output if onnx_format else onnx_output[0]

    def compare_with_torch(self, y_torch, y_onnx) -> None:
        np.testing.assert_allclose(to_numpy(y_torch), y_onnx[0], rtol=1e-03, atol=1e-05)

        
""" refenrence
* https://github.com/onnx/tutorials
* https://github.com/onnx/tutorials/blob/master/tutorials/PytorchAddExportSupport.md
* https://github.com/onnx/tutorials/blob/master/tutorials/PytorchOnnxExport.ipynb
* https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
* https://medium.com/@hemanths933/convert-bert-model-from-pytorch-to-onnx-and-run-inference-39150161fb23
"""
