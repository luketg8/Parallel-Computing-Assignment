__kernel void avgTemp(__global const int* temperature, __global int* output, __local int* scratch){ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	
	//cache all values from global memory to local memory
	scratch[lid] = temperature[id];

	barrier(CLK_LOCAL_MEM_FENCE); //wait for all local threads to finish copying from global to local

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N))
			if(scratch[lid] + i != 1000) {
			scratch[lid] += scratch[lid + i];
			}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atom_add(&output[0],scratch[lid]);

	}
}

__kernel void maxTemperature(__global const int* temperature, __global int* output, __local int* scratch){ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all values from global memory to local memory
	scratch[lid] = temperature[id];
	barrier(CLK_LOCAL_MEM_FENCE);

		if (!lid){
				atom_max(&output[0],scratch[lid]);
		}
}

//histogram calculation
__kernel void hist_simple(__global const int* temperature, __global int* output, int bincount, int minval, int maxval) { 
	int id = get_global_id(0);
	int n = 0;
	//get range of values it can be
	int range = maxval-minval;
	int i = temperature[id];
	//each increment is range / the number of bins wanted
	int increment = range/bincount;
	//initial increment
	int compareval = minval + increment;
	while (i > compareval)
	{
	//get next increment
		compareval += increment;
		n++;
	}
	atomic_inc(&output[n]);
}

__kernel void minTemperature(__global const int* temperature, __global int* output, __local int* scratch){ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cace all values from global memory to local memory
	scratch[lid] = temperature[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2){
		if (!(scratch[lid] <= scratch[lid + i]))
			scratch[lid] = scratch[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid){
				atom_min(&output[0],scratch[lid]);
		}

}