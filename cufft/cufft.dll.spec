#
# File: cufft.dll.spec
#
# Copyrighted by Seth Shelnutt under the LGPL v2.1 or later
#
# Wine spec file for the cudart.dll built-in library (a minimal wrapper around the
# linux library libcufft.so)
#
# For further details of wine spec files see the Winelib documentation at
# www.winehq.org


@ stdcall cufftPlan1d( ptr long long long ) wine_cufftPlan1d
@ stdcall cufftPlan2d( ptr long long long ) wine_cufftPlan2d
@ stdcall cufftPlan3d( ptr long long long ) wine_cufftPlan3d
@ stdcall cufftPlanMany( ptr long ptr ptr long long ptr long long long long ) wine_cufftPlanMany
@ stdcall cufftDestroy( long ) wine_cufftDestroy
@ stdcall cufftExecC2C( long ptr ptr long ) wine_cufftExecC2C
@ stdcall cufftExecR2C( long ptr ptr ) wine_cufftExecR2C
@ stdcall cufftExecC2R( long ptr ptr ) wine_cufftExecC2R
@ stdcall cufftExecZ2Z( long ptr ptr long ) wine_cufftExecZ2Z
@ stdcall cufftExecD2Z( long ptr ptr ) wine_cufftExecD2Z
@ stdcall cufftExecZ2D( long ptr ptr ) wine_cufftExecZ2D
@ stdcall cufftSetStream( long long ) wine_cufftSetStream
