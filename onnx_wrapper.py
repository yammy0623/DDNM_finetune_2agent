import torch

class ONNXUNetWrapper:
    def __init__(self, ort_session):
        self.ort_session = ort_session
        self.onnx_input_names = [input.name for input in ort_session.get_inputs()]
        self.onnx_output_name = ort_session.get_outputs()[0].name

    def __call__(self, input_tensor, timestep_tensor):
        sample = input_tensor.detach().cpu().numpy()
        timestep = timestep_tensor.detach().cpu().numpy()
        
        ort_inputs = {
            self.onnx_input_names[0]: sample,
            self.onnx_input_names[1]: timestep
        }
        ort_outs = self.ort_session.run([self.onnx_output_name], ort_inputs)
        out_sample = torch.from_numpy(ort_outs[0]).to("cuda")
        class OutputWrapper:
            def __init__(self, sample):
                self.sample = sample
        return OutputWrapper(out_sample)
