# source me: isolated ROCm 7.14 + jax-rocm7 0.10.0 venv (lane o5-port; CPU-only deviceless use)
S=<DATA_ROOT>/wrf_gpu2/scratch/o5-port
SP=$S/venv_rocm/lib/python3.13/site-packages
export O5_ROCM_PY=$S/venv_rocm/bin/python
export LD_LIBRARY_PATH=$SP/_rocm_sdk_core/lib:$SP/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SP/_rocm_sdk_libraries/lib:$SP/_rocm_sdk_libraries/lib/rocm_sysdeps/lib
export JAX_ROCM_PLUGIN_INTERNAL_BITCODE_PATH=$SP/_rocm_sdk_core/lib/llvm/amdgcn/bitcode
export HIP_DEVICE_LIB_PATH=$JAX_ROCM_PLUGIN_INTERNAL_BITCODE_PATH
export JAX_ROCM_PLUGIN_INTERNAL_LLD_PATH=$SP/_rocm_sdk_core/lib/llvm/bin
export ROCM_PATH=$SP/_rocm_sdk_core
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES=
