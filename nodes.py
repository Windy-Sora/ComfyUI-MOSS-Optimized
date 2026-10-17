import folder_paths
from comfy_api.latest import ComfyExtension, io
from comfy_api.torch_helpers import set_torch_compile_wrapper

from .loaders import DTYPES, load_clip, load_model


class MOSSOptimizedLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MOSSOptimizedLoader", display_name="MOSS Optimized Loader",
            category="MOSS Optimized",
            inputs=[
                io.Combo.Input("unet_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("weight_dtype", options=list(DTYPES), default="default"),
                io.Boolean.Input("compile_situ", default=False, tooltip="Compile SiTU only. Requires a working torch.compile toolchain; first use takes longer."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, unet_name, weight_dtype, compile_situ):
        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = load_model(path, weight_dtype)
        if compile_situ:
            count = len(model.get_model_object("diffusion_model.layers"))
            set_torch_compile_wrapper(
                model=model, backend="inductor", dynamic=True,
                keys=[f"diffusion_model.layers.{index}.situ" for index in range(count)],
            )
        return io.NodeOutput(model)


class MOSSOptimizedTextEncoderLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MOSSOptimizedTextEncoderLoader", display_name="MOSS Qwen3.5 Text Encoder",
            category="MOSS Optimized",
            inputs=[
                io.Combo.Input("clip_name", options=folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("device", options=["default", "cpu"], default="default", advanced=True),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, clip_name, device):
        path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        return io.NodeOutput(load_clip(path, device))


class MOSSOptimizedExtension(ComfyExtension):
    async def get_node_list(self):
        return [MOSSOptimizedLoader, MOSSOptimizedTextEncoderLoader]


async def comfy_entrypoint():
    return MOSSOptimizedExtension()
