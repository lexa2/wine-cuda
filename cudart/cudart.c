/***
 * This is a wrapper for cudart32_40_17.dll and libcudart.so.4.0.17
 * Original work Copyrighted by Seth Shelnutt under the LGPL v2.1 or later
 * Some minor additions to port it from Cuda 3.x to Cuda 4.x done by
 * Alexey Loukianov <mooroon2@mail.ru> during October 2011.
 */


#include <windows.h>
#include "crt/host_runtime.h"
//#include "device_functions.h"  //used in the functions cudaSynchronizeThreads
#include "sm_20_atomic_functions.h"
#include "texture_fetch_functions.h"
#include "cuda_runtime_api.h"
#include "cuda_gl_interop.h"
#include "driver_functions.h"
#include "driver_types.h"
//#include "math_functions.h"
#include <stdlib.h>
#include <GL/gl.h>
#include <string.h>
#include "wine/debug.h"
#include "cudart.h"




WINE_DEFAULT_DEBUG_CHANNEL(cuda);

    #define QUEUE_MAX       20
    #define HALF_QUEUE_MAX  ((QUEUE_MAX)/2)

    static unsigned int numQueued = 0;
    static BOOL eventInitialized = FALSE;
    static cudaEvent_t event;
//    static unsigned long sleepAmount = 150;


    static const char* cudaErrorString[] = {
        "cudaSuccess",
        "cudaErrorMissingConfiguration",
        "cudaErrorMemoryAllocation",
        "cudaErrorInitializationError",
        "cudaErrorLaunchFailure",
        "cudaErrorPriorLaunchFailure",
        "cudaErrorLaunchTimeout",
        "cudaErrorLaunchOutOfResources",
        "cudaErrorInvalidDeviceFunction",
        "cudaErrorInvalidConfiguration",
        "cudaErrorInvalidDevice",
        "cudaErrorInvalidValue",
        "cudaErrorInvalidPitchValue",
        "cudaErrorInvalidSymbol",
        "cudaErrorMapBufferObjectFailed",
        "cudaErrorUnmapBufferObjectFailed",
        "cudaErrorInvalidHostPointer",
        "cudaErrorInvalidDevicePointer",
        "cudaErrorInvalidTexture",
        "cudaErrorInvalidTextureBinding",
        "cudaErrorInvalidChannelDescriptor",
        "cudaErrorInvalidMemcpyDirection",
        "cudaErrorAddressOfConstant",
        "cudaErrorTextureFetchFailed",
        "cudaErrorTextureNotBound",
        "cudaErrorSynchronizationError",
        "cudaErrorInvalidFilterSetting",
        "cudaErrorInvalidNormSetting",
        "cudaErrorMixedDeviceExecution",
        "cudaErrorCudartUnloading",
        "cudaErrorUnknown",
        "cudaErrorNotYetImplemented",
        "cudaErrorMemoryValueTooLarge",
        "cudaErrorInvalidResourceHandle",
        "cudaErrorNotReady"
    };

static const char* debug_cudaError(cudaError_t err) {
        if (cudaErrorStartupFailure == err) {
            WINE_TRACE("\n");
	return "cudaErrorStartupFailure";
        }

        if (cudaErrorApiFailureBase == err) {
            WINE_TRACE("\n");
	return "cudaErrorApiFailureBase";
        }

        if (err >= 0 && err < sizeof(cudaErrorString)/sizeof(cudaErrorString[0])) {
            return cudaErrorString[err];
        }

        WINE_TRACE("unknown error %d\n", err);
	return "unknown CUDA error";
    }

BOOL WINAPI DllMain(HINSTANCE hInstDLL, DWORD fdwReason, LPVOID lpv)
    {
        if (DLL_PROCESS_DETACH == fdwReason)
        {
            /* Cleanup */
            if (eventInitialized) {
                WINE_TRACE("releasing event %d\n", event);

                cudaError_t err = cudaEventDestroy(event);

                if (err) {
                    WINE_TRACE("cudaEventDestroy: %s\n", debug_cudaError(err));
                }
            }
        }
	return TRUE;
    }



/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaMalloc3D( struct cudaPitchedPtr* pitchDevPtr, struct cudaExtent extent ){
        WINE_TRACE("\n");
	return cudaMalloc3D( pitchDevPtr, extent );
}


cudaError_t WINAPI wine_cudaMalloc3DArray( struct cudaArray** arrayPtr, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags ){
        WINE_TRACE("\n");
	return cudaMalloc3DArray( arrayPtr, desc, extent, flags);
}

cudaError_t WINAPI wine_cudaMemset3D( struct cudaPitchedPtr pitchedDevPtr, int value, struct cudaExtent extent ){
	WINE_TRACE("\n");
	return cudaMemset3D( pitchedDevPtr, value, extent );
}


cudaError_t WINAPI wine_cudaMemcpy3D( const struct cudaMemcpy3DParms *p ){
        WINE_TRACE("\n");
	return cudaMemcpy3D( p );
}

cudaError_t WINAPI wine_cudaMemcpy3DAsync( const struct cudaMemcpy3DParms *p, cudaStream_t stream ){
	WINE_TRACE("\n");
	return cudaMemcpy3DAsync( p, stream );
}

/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaMalloc(void **devPtr, size_t size) {
        WINE_TRACE("\n");
	return cudaMalloc(devPtr, size);
    }

cudaError_t WINAPI wine_cudaMallocHost( void** hostPtr, size_t size ){
	WINE_TRACE("\n");
	return cudaMallocHost( hostPtr, size);
}

cudaError_t WINAPI wine_cudaMallocPitch( void** devPtr, size_t* pitch, size_t widthInBytes, size_t height ){
	WINE_TRACE("\n");
	return cudaMallocPitch( devPtr, pitch, widthInBytes, height );
}

cudaError_t WINAPI wine_cudaMallocArray( struct cudaArray** array, const struct cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int flags ){
	WINE_TRACE("\n");
	return cudaMallocArray( array, desc, width, height, flags );
}


cudaError_t WINAPI wine_cudaFree(void *devPtr) {
        WINE_TRACE("\n");
	return cudaFree(devPtr);
}

cudaError_t WINAPI wine_cudaFreeHost( void* hostPtr ){
	WINE_TRACE("\n");
	return cudaFreeHost( hostPtr);
}

cudaError_t WINAPI wine_cudaFreeArray( struct cudaArray* array ){
	WINE_TRACE("\n");
	return cudaFreeArray( array );
}


cudaError_t WINAPI wine_cudaHostAlloc(void **pHost, size_t bytes, unsigned int flags){
	WINE_TRACE("\n");
	return cudaHostAlloc( pHost, bytes, flags );
}
cudaError_t WINAPI wine_cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags){
	WINE_TRACE("\n");
	return cudaHostGetDevicePointer( pDevice, pHost, flags );
}
cudaError_t WINAPI wine_cudaHostGetFlags(unsigned int *pFlags, void *pHost){
	WINE_TRACE("\n");
	return cudaHostGetFlags( pFlags, pHost );
}



/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaMemGetInfo(size_t *free, size_t *total){
	WINE_TRACE("\n");
	return cudaMemGetInfo( free, total );
}

cudaError_t WINAPI wine_cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
        WINE_TRACE("\n");
	return cudaMemcpy(dst, src, count, kind);
}

cudaError_t WINAPI wine_cudaMemcpyToArray( struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind ){
	WINE_TRACE("\n");
	return cudaMemcpyToArray( dst, wOffset, hOffset, src, count, kind );
}

cudaError_t WINAPI wine_cudaMemcpyFromArray( void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind ){
	WINE_TRACE("\n");
	return cudaMemcpyFromArray( dst, src, wOffset, hOffset, count, kind );
}

cudaError_t WINAPI wine_cudaMemcpyArrayToArray( struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t count, enum cudaMemcpyKind kind ){
	WINE_TRACE("\n");
	return cudaMemcpyArrayToArray( dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind );
}

cudaError_t WINAPI wine_cudaMemcpy2D( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind ){
	WINE_TRACE("\n");
	return cudaMemcpy2D( dst, dpitch, src, spitch, width, height, kind );
}

cudaError_t WINAPI wine_cudaMemcpy2DToArray( struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind ){
	WINE_TRACE("\n");
	return cudaMemcpy2DToArray( dst, wOffset, hOffset, src, spitch, width, height, kind );
}

cudaError_t WINAPI wine_cudaMemcpy2DFromArray( void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind ){
	WINE_TRACE("\n");
	return cudaMemcpy2DFromArray( dst, dpitch, src, wOffset, hOffset, width, height, kind );
}

cudaError_t WINAPI wine_cudaMemcpy2DArrayToArray( struct cudaArray *dst, size_t wOffsetDst, size_t hOffsetDst, const struct cudaArray *src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, enum cudaMemcpyKind kind ){
	WINE_TRACE("\n");
	return cudaMemcpy2DArrayToArray( dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind );
}

cudaError_t WINAPI wine_cudaMemcpyToSymbol( const char *symbol, const void *src, size_t count, size_t offset , enum cudaMemcpyKind kind ) {
	WINE_TRACE("\n");
	return cudaMemcpyToSymbol( symbol, src, count, offset, kind);
}

cudaError_t WINAPI wine_cudaMemcpyFromSymbol( void *dst, const char *symbol, size_t count, size_t offset , enum cudaMemcpyKind kind ) {
	WINE_TRACE("\n");
	return cudaMemcpyFromSymbol( dst, symbol, count, offset, kind );
}

/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaMemcpyAsync( void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream ){
	WINE_TRACE("\n");
	return cudaMemcpyAsync( dst, src, count, kind, stream );
}

cudaError_t WINAPI wine_cudaMemcpyToArrayAsync( struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream ){
	WINE_TRACE("\n");
	return cudaMemcpyToArrayAsync( dst, wOffset, hOffset, src, count, kind, stream );\
}

cudaError_t WINAPI wine_cudaMemcpyFromArrayAsync( void *dst, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream ){
	WINE_TRACE("\n");
	return cudaMemcpyFromArrayAsync( dst, src, wOffset, hOffset, count, kind, stream );
}

cudaError_t WINAPI wine_cudaMemcpy2DAsync( void *dst, size_t dpitch, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream ){
	WINE_TRACE("\n");
	return cudaMemcpy2DAsync( dst, dpitch, src, spitch, width, height, kind, stream );
}

cudaError_t WINAPI wine_cudaMemcpy2DToArrayAsync( struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream ){
	WINE_TRACE("\n");
	return cudaMemcpy2DToArrayAsync( dst, wOffset, hOffset, src, spitch, width, height, kind, stream );
}

cudaError_t WINAPI wine_cudaMemcpy2DFromArrayAsync( void *dst, size_t dpitch, const struct cudaArray *src, size_t wOffset, size_t hOffset, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream ){
	WINE_TRACE("\n");
	return cudaMemcpy2DFromArrayAsync( dst, dpitch, src, wOffset, hOffset, width, height, kind, stream );
}

cudaError_t WINAPI wine_cudaMemcpyToSymbolAsync( const char *symbol, const void *src, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream ){
	WINE_TRACE("\n");
	return cudaMemcpyToSymbolAsync( symbol, src, count, offset, kind, stream );
}

cudaError_t WINAPI wine_cudaMemcpyFromSymbolAsync( void *dst, const char *symbol, size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream ){
	WINE_TRACE("\n");
	return cudaMemcpyFromSymbolAsync( dst, symbol, count, offset, kind, stream);
}




/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaMemset( void *devPtr, int value, size_t count ){
	WINE_TRACE("\n");
	return cudaMemset( devPtr, value, count );
}

cudaError_t WINAPI wine_cudaMemset2D( void *devPtr, size_t pitch, int value, size_t width, size_t height ){
	WINE_TRACE("\n");
	return cudaMemset2D( devPtr, pitch, value, width, height );
}

/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaGetSymbolAddress( void **devPtr, const char *symbol ){
	WINE_TRACE("\n");
	return cudaGetSymbolAddress( devPtr, symbol );
}

cudaError_t WINAPI wine_cudaGetSymbolSize( size_t *size, const char *symbol ){
	WINE_TRACE("\n");
	return cudaGetSymbolSize( size, symbol );
}

/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaGetDeviceCount( int* count ){
	WINE_TRACE("\n");
	return cudaGetDeviceCount( count );
}

cudaError_t WINAPI wine_cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
	WINE_TRACE("\n");
	return cudaGetDeviceProperties(prop, device);
}

cudaError_t WINAPI wine_cudaChooseDevice( int *device, const struct cudaDeviceProp *prop ){
	WINE_TRACE("\n");
	return cudaChooseDevice( device, prop );
}

cudaError_t WINAPI wine_cudaSetDevice( int device ) {
	WINE_TRACE("\n");
	return cudaSetDevice( device );
}

cudaError_t WINAPI wine_cudaGetDevice( int* device ){
	WINE_TRACE("\n");
	return cudaGetDevice( device );
}


cudaError_t WINAPI wine_cudaSetValidDevices( int *device_arr, int len ){
	WINE_TRACE("\n");
	return cudaSetValidDevices( device_arr, len);
}

cudaError_t WINAPI wine_cudaSetDeviceFlags( int flags ){
	WINE_TRACE("\n");
	return cudaSetDeviceFlags( flags );
}

cudaError_t WINAPI wine_cudaDeviceReset( void ){
	WINE_TRACE("\n");
	cudaDeviceReset();
}

/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaBindTexture( size_t *offset, const struct textureReference *texref, const void *devPtr, const struct cudaChannelFormatDesc *desc, size_t size ){
	WINE_TRACE("\n");
	return cudaBindTexture( offset, texref, devPtr, desc, size );
}

cudaError_t WINAPI wine_cudaBindTexture2D( size_t *offset,const struct textureReference *texref,const void *devPtr, const struct cudaChannelFormatDesc *desc,size_t width, size_t height, size_t pitch ){
	WINE_TRACE("\n");
	return cudaBindTexture2D( offset, texref, devPtr, desc, width, height, pitch );
}

cudaError_t WINAPI wine_cudaBindTextureToArray( const struct textureReference *texref, const struct cudaArray *array, const struct cudaChannelFormatDesc *desc ){
	WINE_TRACE("\n");
	return cudaBindTextureToArray( texref, array, desc );
}

cudaError_t WINAPI wine_cudaUnbindTexture( const struct textureReference *texRef ){
	WINE_TRACE("\n");
	return cudaUnbindTexture( texRef );
}

cudaError_t WINAPI wine_cudaGetTextureAlignmentOffset( size_t *offset, const struct textureReference *texRef ){
	WINE_TRACE("\n");
	return cudaGetTextureAlignmentOffset( offset, texRef );
}

cudaError_t WINAPI wine_cudaGetTextureReference( const struct textureReference **texref, const char *symbol ){
	WINE_TRACE("\n");
	return cudaGetTextureReference( texref, symbol );
}

/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaGetChannelDesc( struct cudaChannelFormatDesc *desc, const struct cudaArray *array ){
	WINE_TRACE("\n");
	return cudaGetChannelDesc( desc, array );
}

struct cudaChannelFormatDesc WINAPI wine_cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f){
	WINE_TRACE("\n");
	return cudaCreateChannelDesc(x, y, z, w, f);
}

/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaGetLastError() {
	cudaError_t err = cudaGetLastError();

        WINE_TRACE("return %s\n", debug_cudaError(err));

        return err;
}


const char* WINAPI wine_cudaGetErrorString(cudaError_t error) {
	WINE_TRACE("\n");
	return cudaGetErrorString(error);
    }

/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaConfigureCall( dim3 gridDim, dim3 blockDim, size_t sharedMem, cudaStream_t stream ) {
	WINE_TRACE("\n");
	return cudaConfigureCall( gridDim, blockDim, sharedMem, stream );
}

cudaError_t WINAPI wine_cudaSetupArgument( const void *arg, size_t size, size_t offset ){
	WINE_TRACE("\n");
	return cudaSetupArgument( arg, size, offset );
}

cudaError_t WINAPI wine_cudaFuncSetCacheConfig( const char *func, enum cudaFuncCache cacheConfig ){
	WINE_TRACE("\n");
	return cudaFuncSetCacheConfig( func, cacheConfig );
}

cudaError_t WINAPI wine_cudaLaunch(const char *entry) {
	 WINE_TRACE("%p\n", entry);

        if (QUEUE_MAX == numQueued) {
            cudaError_t evtErr;

            if (WINE_TRACE_ON(cuda)) {
                /* print out if event was recorded or not */
                WINE_TRACE("check event recorded %s\n", debug_cudaError(cudaEventQuery(event)));
            }

            /* wait for event */
	    unsigned int sleepCount = 0;

	    while ((cudaEventQuery(event) == cudaErrorNotReady) && (sleepCount < 150)) {
                usleep(150, NULL);
                sleepCount++;
            }

/*
	    if ((sleepCount > 10) && (sleepAmount < 300)) {
		sleepAmount += 10;

	    }

	    if ((sleepCount < 1) && (sleepAmount > 20)) {
		sleepAmount -= 10;
            }
*/
            WINE_TRACE("slept %u times\n", sleepCount);
//	    WINE_TRACE("next sleep will be for %uns\n",sleepAmount);

//    #else
            evtErr = cudaEventSynchronize(event);

            if (evtErr) {
                WINE_ERR("cudaEventSynchronize: %s\n", debug_cudaError(evtErr));
            }
//    #endif

            WINE_TRACE("event recorded, continuing\n");

            /* record a new event and subtract HALF_QUEUE_MAX from numQueued */
            numQueued = HALF_QUEUE_MAX;
            evtErr = cudaEventRecord(event, 0);

            if (evtErr) {
                WINE_ERR("cudaEventRecord: %s\n", debug_cudaError(evtErr));
            }
        }

        cudaError_t err = cudaLaunch(entry);

        if (!eventInitialized) {
            /* Create an event on the first cudaLaunch call.  This is done here so the calling program
             * has a chance to select the GPU device with cudaSetDevice if desired. */
            cudaError_t evtErr = cudaEventCreate(&event);

            if (evtErr) {
                WINE_ERR("cudaEventCreate: %s\n", debug_cudaError(evtErr));
            }

            /* cudaEventCreate can WINE_TRACE("\n");
	return errors from previous asynchronous calls, so an error here does
             * not necessarily mean the event wasn't created.  Assume it was created for now. */
            eventInitialized = TRUE;
            WINE_TRACE("created event %d\n", event);
        }

        /* record an event at HALF_QUEUE_MAX */
        if (HALF_QUEUE_MAX == ++numQueued) {
            cudaError_t evtErr = cudaEventRecord(event, 0);  /* Assuming everything using stream 0 */

            if (evtErr) {
                WINE_ERR("cudaEventRecord: %s\n", debug_cudaError(evtErr));
            }
        }

        if (err) {
            WINE_TRACE("return %s\n", debug_cudaError(err));
        }

	return err;
    }


cudaError_t WINAPI wine_cudaFuncGetAttributes( struct cudaFuncAttributes *attr, const char *func ){
	WINE_TRACE("\n");
	return cudaFuncGetAttributes( attr, func );
}

/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaStreamCreate( cudaStream_t *pStream ){
	WINE_TRACE("\n");
	return cudaStreamCreate( pStream );
}

cudaError_t WINAPI wine_cudaStreamDestroy( cudaStream_t stream ){
	WINE_TRACE("\n");
	return cudaStreamDestroy( stream );
}

cudaError_t WINAPI wine_cudaStreamSynchronize( cudaStream_t stream ){
	WINE_TRACE("\n");
	return cudaStreamSynchronize( stream );
}


cudaError_t WINAPI wine_cudaStreamQuery(cudaStream_t stream){
	WINE_TRACE("\n");
	return cudaStreamQuery( stream );
}

/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaEventCreate( cudaEvent_t *event ){
	WINE_TRACE("\n");
	return cudaEventCreate( event );
}

cudaError_t WINAPI wine_cudaEventCreateWithFlags( cudaEvent_t *event, int flags ){
	WINE_TRACE("\n");
	return cudaEventCreateWithFlags( event, flags );
}

cudaError_t WINAPI wine_cudaEventRecord( cudaEvent_t event, cudaStream_t stream ){
	WINE_TRACE("\n");
	return cudaEventRecord( event, stream );
}

cudaError_t WINAPI wine_cudaEventQuery( cudaEvent_t event ){
	WINE_TRACE("\n");
	return cudaEventQuery( event );
}

cudaError_t WINAPI wine_cudaEventSynchronize( cudaEvent_t event ){
	WINE_TRACE("\n");
	return cudaEventSynchronize( event );
}

cudaError_t WINAPI wine_cudaEventDestroy( cudaEvent_t event ){
	WINE_TRACE("\n");
	return cudaEventDestroy( event );
}

cudaError_t WINAPI wine_cudaEventElapsedTime( float *ms, cudaEvent_t start, cudaEvent_t end ){
	WINE_TRACE("\n");
	return cudaEventElapsedTime( ms, start, end );
}

/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaSetDoubleForDevice( double *d ){
	WINE_TRACE("\n");
	return cudaSetDoubleForDevice( d );
}

cudaError_t WINAPI wine_cudaSetDoubleForHost( double *d ){
	WINE_TRACE("\n");
	return cudaSetDoubleForHost( d );
}

/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaThreadExit( void ){
	WINE_TRACE("\n");
	return cudaThreadExit(  );
}

cudaError_t WINAPI wine_cudaThreadSynchronize( void ){
	WINE_TRACE("\n");
	return cudaThreadSynchronize(  );
}

/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaDriverGetVersion( int *driverVersion ){
	WINE_TRACE("\n");
	return cudaDriverGetVersion( driverVersion );
}

cudaError_t WINAPI wine_cudaRuntimeGetVersion( int *runtimeVersion ){
	WINE_TRACE("\n");
	return cudaRuntimeGetVersion( runtimeVersion );
}

/*******************************************************************************
*                                                                              *
*             cuda_runtime_api.h                                               *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaGraphicsUnregisterResource( struct cudaGraphicsResource *resource ){
	WINE_TRACE("\n");
	return cudaGraphicsUnregisterResource( resource );
}

cudaError_t WINAPI wine_cudaGraphicsResourceSetMapFlags( struct cudaGraphicsResource *resource, unsigned int flags ){
	WINE_TRACE("\n");
	return cudaGraphicsResourceSetMapFlags( resource, flags );
}
 
cudaError_t WINAPI wine_cudaGraphicsMapResources( int count, struct cudaGraphicsResource **resources, cudaStream_t stream ){
	WINE_TRACE("\n");
	return cudaGraphicsMapResources( count, resources, stream );
}

cudaError_t WINAPI wine_cudaGraphicsUnmapResources( int count, struct cudaGraphicsResource **resources, cudaStream_t stream ){
	WINE_TRACE("\n");
	return cudaGraphicsUnmapResources( count, resources, stream );
}

cudaError_t WINAPI wine_cudaGraphicsResourceGetMappedPointer( void **devPtr, size_t *size, struct cudaGraphicsResource *resource){
	WINE_TRACE("\n");
	return cudaGraphicsResourceGetMappedPointer( devPtr, size, resource );
}

cudaError_t WINAPI wine_cudaGraphicsSubResourceGetMappedArray( struct cudaArray **arrayPtr, struct cudaGraphicsResource *resource, unsigned int arrayIndex, unsigned int mipLevel){
	WINE_TRACE("\n");
	return cudaGraphicsSubResourceGetMappedArray( arrayPtr, resource, arrayIndex, mipLevel );
}

/*******************************************************************************
*                                                                              *
*                  cuda_gl_interop.h                                           *
*                                                                              *
*******************************************************************************/


cudaError_t WINAPI wine_cudaGLSetGLDevice(int device){
	WINE_TRACE("\n");
	return cudaGLSetGLDevice( device );
}

cudaError_t WINAPI wine_cudaGraphicsGLRegisterImage( struct cudaGraphicsResource **resource, GLuint image, GLenum target, unsigned int Flags ){
	WINE_TRACE("\n");
	return cudaGraphicsGLRegisterImage( resource, image, target, Flags );
}

cudaError_t WINAPI wine_cudaGraphicsGLRegisterBuffer( struct cudaGraphicsResource **resource, GLuint buffer, unsigned int Flags ){
	WINE_TRACE("\n");
	return cudaGraphicsGLRegisterBuffer( resource, buffer, Flags );
}

cudaError_t WINAPI wine_cudaWGLGetDevice( int *device, HGPUNV hGpu ){
	WINE_TRACE("\n");
	return wine_cudaWGLGetDevice( device, hGpu );
}

/*******************************************************************************
*                                                                              *
*                 cuda_gl_interop.h                                            *
*                                                                              *
*******************************************************************************/

cudaError_t WINAPI wine_cudaGLRegisterBufferObject( GLuint bufObj ){
	WINE_TRACE("\n");
	return cudaGLRegisterBufferObject( bufObj );
}

cudaError_t WINAPI wine_cudaGLMapBufferObject( void** devPtr, GLuint bufObj ){
	WINE_TRACE("\n");
	return cudaGLMapBufferObject( devPtr, bufObj );
}

cudaError_t WINAPI wine_cudaGLUnmapBufferObject( GLuint bufObj ){
	WINE_TRACE("\n");
	return cudaGLUnmapBufferObject( bufObj );
}

cudaError_t WINAPI wine_cudaGLUnregisterBufferObject( GLuint bufObj ){
	WINE_TRACE("\n");
	return cudaGLUnregisterBufferObject( bufObj );
}


cudaError_t WINAPI wine_cudaGLSetBufferObjectMapFlags( GLuint bufObj, unsigned int flags ){
	WINE_TRACE("\n");
	return cudaGLSetBufferObjectMapFlags( bufObj, flags ); 
}

cudaError_t WINAPI wine_cudaGLMapBufferObjectAsync( void **devPtr, GLuint bufObj, cudaStream_t stream ){
	WINE_TRACE("\n");
	return cudaGLMapBufferObjectAsync( devPtr, bufObj, stream );
}

cudaError_t WINAPI wine_cudaGLUnmapBufferObjectAsync( GLuint bufObj, cudaStream_t stream){
	WINE_TRACE("\n");
	return cudaGLUnmapBufferObjectAsync( bufObj, stream );
}

/*******************************************************************************
*                                                                              *
*                  host_runtime.h                                              *
*                                                                              *
*******************************************************************************/


void** WINAPI wine_cudaRegisterFatBinary( void* fatCubin ) {
        WINE_TRACE("\n");
	return __cudaRegisterFatBinary( fatCubin );
    }

void WINAPI wine_cudaUnregisterFatBinary( void **fatCubinHandle ) {
        WINE_TRACE("\n");
	return __cudaUnregisterFatBinary( fatCubinHandle );
    }

void WINAPI wine_cudaRegisterVar( void **fatCubinHandle, char *hostVar, char *deviceAddress, const char  *deviceName, int ext, int size, int constant, int global ){
        WINE_TRACE("\n");
	return __cudaRegisterVar( fatCubinHandle, hostVar, deviceAddress, deviceName, ext, size, constant, global );
    }

void WINAPI wine_cudaRegisterTexture( void **fatCubinHandle, const struct textureReference *hostVar, const void **deviceAddress, const char *deviceName, int dim, int norm, int ext ){
	WINE_TRACE("\n");
	return __cudaRegisterTexture( fatCubinHandle, hostVar, deviceAddress, deviceName, dim, norm, ext );
}

void WINAPI wine_cudaRegisterShared( void **fatCubinHandle, void **devicePtr ) {
	WINE_TRACE("\n");
	return __cudaRegisterShared( fatCubinHandle, devicePtr );
    }

void WINAPI wine_cudaRegisterSharedVar(void **fatCubinHandle, void **devicePtr, size_t size, size_t alignment, int storage) {
	WINE_TRACE("\n");
	return __cudaRegisterSharedVar( fatCubinHandle, devicePtr, size, alignment, storage);
    }


void WINAPI wine_cudaRegisterFunction( void **fatCubinHandle, const char *hostFun, char *deviceFun, const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize ) {
	WINE_TRACE("\n");
	return __cudaRegisterFunction( fatCubinHandle, hostFun, deviceFun, deviceName, thread_limit, tid, bid, bDim, gDim, wSize );
    }

/*******************************************************************************
*                                                                              *
*               device_functions.h                                             *
*                                                                              *
*******************************************************************************/

int WINAPI wine_cudaSynchronizeThreads( void **one, void *two  ){
	WINE_TRACE("\n");
	return __cudaSynchronizeThreads( one, two );
}


/*******************************************************************************
*                                                                              *
*                texture_fetch_functions.h                                     *
*                                                                              *
*******************************************************************************/

void WINAPI wine_cudaTextureFetch( const void *tex, void *index, int integer, void *val ){
	WINE_TRACE("\n");
	__cudaTextureFetch( tex, index, integer, val );
}

/*******************************************************************************
*                                                                              *
*                sm_20_atomic_functions.h                                      *
*                                                                              *
*******************************************************************************/

void WINAPI wine_cudaMutexOperation( int lock ){
	WINE_TRACE("\n");
	__cudaMutexOperation( lock );
}

