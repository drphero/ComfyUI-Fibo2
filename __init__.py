from comfy_api.latest import ComfyExtension

from .nodes import Fibo2DiffusionModelLoader, Fibo2TextEncoderLoader, Fibo2EditConditioning, Fibo2Scheduler


class Fibo2Extension(ComfyExtension):
    async def get_node_list(self):
        return [Fibo2DiffusionModelLoader, Fibo2TextEncoderLoader, Fibo2EditConditioning, Fibo2Scheduler]


async def comfy_entrypoint():
    return Fibo2Extension()
