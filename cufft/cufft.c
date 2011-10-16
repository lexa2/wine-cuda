/*This is a wrapper for cufft32_30_14.dll and libcufft.so.3.0.14
 Copyrighted by Seth Shelnutt under the LGPL v2.1 or later
 */

#include <windows.h>
#include "cufft.h"
#include <stdlib.h>
#include <GL/gl.h>
#include <string.h>
#include "wine/debug.h"

WINE_DEFAULT_DEBUG_CHANNEL(cuda);

/*******************************************************************************
*                                                                              *
*             cufft.h		                                               *
*                                                                              *
*******************************************************************************/

cufftResult WINAPI wine_cufftPlan1d( cufftHandle *plan, int nx, cufftType type, int batch ){
	WINE_TRACE("\n");
	return cufftPlan1d( plan, nx, type, batch );
}

cufftResult WINAPI wine_cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type){
	WINE_TRACE("\n");
	return cufftPlan2d( plan, nx, ny, type );
}

cufftResult WINAPI wine_cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type){
	WINE_TRACE("\n");
	return cufftPlan3d( plan, nx, ny, nz, type );
}

cufftResult WINAPI wine_cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed, int istride, int idist, int *onembed, int ostride, int odist, cufftType type, int batch){
	WINE_TRACE("\n");
	return cufftPlanMany( plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch);
}

cufftResult WINAPI wine_cufftDestroy(cufftHandle plan){
	WINE_TRACE("\n");
	return cufftDestroy( plan );
}

cufftResult WINAPI wine_cufftExecC2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction){
	WINE_TRACE("\n");
	return cufftExecC2C( plan, idata, odata, direction );
}

cufftResult WINAPI wine_cufftExecR2C(cufftHandle plan, cufftReal *idata, cufftComplex *odata){
	WINE_TRACE("\n");
	return cufftExecR2C( plan, idata, odata );
}

cufftResult WINAPI wine_cufftExecC2R(cufftHandle plan, cufftComplex *idata, cufftReal *odata){
	WINE_TRACE("\n");
	return cufftExecC2R( plan, idata, odata );
}

cufftResult WINAPI wine_cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleComplex *odata, int direction){
	WINE_TRACE("\n");
	return cufftExecZ2Z( plan, idata, odata, direction );
}

cufftResult WINAPI wine_cufftExecD2Z( cufftHandle plan, cufftDoubleReal *idata, cufftDoubleComplex *odata ){
	WINE_TRACE("\n");
	return cufftExecD2Z( plan, idata, odata );
}

cufftResult WINAPI wine_cufftExecZ2D( cufftHandle plan, cufftDoubleComplex *idata, cufftDoubleReal *odata ){
	WINE_TRACE("\n");
	return cufftExecZ2D( plan, idata, odata );
}

cufftResult WINAPI wine_cufftSetStream( cufftHandle plan, cudaStream_t stream ){
	WINE_TRACE("\n");
	return cufftSetStream( plan, stream );
}
