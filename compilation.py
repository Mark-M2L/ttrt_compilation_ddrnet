from models.ddrnet_23_slim import DualResNet_imagenet

import torch

import tensorrt
import torch_tensorrt
import torch_tensorrt.logging as logging   

# logging.set_reportable_log_level(logging.Level.Graph)    

logging.set_reportable_log_level(logging.Level.Debug)


cuda = torch.device('cuda')


# Load model with correct weights
model = DualResNet_imagenet(None)

# Test
model = model.eval().to(cuda)

batch_size = 2

scripted_model = torch.jit.freeze(torch.jit.script(model))
# traced_model = torch.jit.trace(model, (torch.rand(1,3,600,400).to(cuda)))

inputs = [torch_tensorrt.Input(
            min_shape=[2, 3, 600, 400],
            opt_shape=[2, 3, 600, 400],
            max_shape=[2, 3, 600, 400],
            dtype=torch.float,
        )]
enabled_precisions = {torch.float, torch.half}
# enabled_precisions = {torch.half}
# enabled_precisions = {torch.float}

# trt_ts_module = torch_tensorrt.compile(model, inputs=inputs, enabled_precisions=enabled_precisions, require_full_compilation=True)

with torch_tensorrt.logging.debug():
    # trt_ts_module = torch_tensorrt.compile(model, inputs=inputs, enabled_precisions=enabled_precisions, torch_executed_ops=["prim::ListConstruct"], min_block_size=1)
    # trt_ts_module = torch_tensorrt.compile(scripted_model, inputs=inputs, enabled_precisions=enabled_precisions, torch_executed_ops=["prim::ListConstruct"], min_block_size=1)
    # trt_ts_module = torch_tensorrt.compile(scripted_model, inputs=inputs, enabled_precisions=enabled_precisions, torch_executed_ops=["prim::ListConstruct"])
    # trt_ts_module = torch_tensorrt.compile(scripted_model, inputs=inputs, enabled_precisions=enabled_precisions, min_block_size=1)
    trt_ts_module = torch_tensorrt.compile(scripted_model, inputs=inputs, enabled_precisions=enabled_precisions)

torch.jit.save(trt_ts_module, "tensor_rt/trt_torchscript_module.ts")