import os
import torch
from torch.utils.ffi import create_extension


sources = ['src/online_triplet_loss_layer.c']
headers = ['src/online_triplet_loss_layer.h']
defines = []
with_cuda = False

if with_cuda:
    assert torch.cuda.is_available(), "cuda support need"
    print('Including CUDA code.')
    defines += [('WITH_CUDA', None)]

extra_objects = ['src/online_triplet_loss.o']

extra_compile_args = ['-fopenmp', '-std=c99']

this_file = os.path.dirname(os.path.realpath(__file__))
print(this_file)
sources = [os.path.join(this_file, fname) for fname in sources]
headers = [os.path.join(this_file, fname) for fname in headers]
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    '_ext.online_triplet_loss',
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
    extra_compile_args=extra_compile_args
)

if __name__ == '__main__':
    ffi.build()
