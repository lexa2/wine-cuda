/***
 * This is a wrapper for cudart32_40_17.dll and libcudart.so.4.0.17
 * Original work Copyrighted by Seth Shelnutt under the LGPL v2.1 or later
 * Some minor additions to port it from Cuda 3.x to Cuda 4.x done by
 * Alexey Loukianov <mooroon2@mail.ru> during October 2011.
 */


#include "cuda_runtime_api.h"

void** __cudaRegisterFatBinary(void *fatCubin);

void __cudaUnregisterFatBinary(void **fatCubinHandle);


void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
                            const char *deviceName, int thread_limit, uint3 *tid,
                            uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize);

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                       const char  *deviceName, int ext, int size, int constant,
                       int global);

void __cudaRegisterTexture( void **fatCubinHandle, const struct textureReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext);

void __cudaRegisterShared(void **fatCubinHandle, void **devicePtr);

void __cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr, size_t size,
                             size_t alignment, int storage);
