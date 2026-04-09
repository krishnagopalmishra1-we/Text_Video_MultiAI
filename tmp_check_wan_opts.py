from diffusers import WanPipeline
methods = sorted([m for m in dir(WanPipeline) if not m.startswith('_')])
filtered = [m for m in methods if any(k in m for k in ['enable','fuse','set_','cache','tea','compile','slicing','sequential','offload'])]
print('\n'.join(filtered))
