from torch.distributed.device_mesh import init_device_mesh

from . import distenv


def build_mesh(shape, dim_names, device_type: str | None = None):
    if device_type is None:
        device_type = distenv.device_type()
    return init_device_mesh(device_type, tuple(shape),
                            mesh_dim_names=tuple(dim_names))
