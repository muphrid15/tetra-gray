#ifndef STEPSIZE_HDR
#define STEPSIZE_HDR
#include "integrator.cuh"
#include "types.cuh"
#include "image.cuh"

namespace ray
{
	struct DynamicStepsizeAdjuster
	{
		template<typename R>
		__host__ __device__ ParticleData<R> operator()(const R& dparam0, const ParticleData<R>& data) const
		{
			const R ratio = abs(data.value.momentum[0]);
			return ParticleData<R>(data.value, data.param, dparam0/ratio);
		}
	};

	struct DynamicStepsizeStopCondition
	{
		template<typename R>
		__host__ __device__ bool operator()(const R& max_step_ratio, const R& dparam0, const ParticleData<R>& data) const
		{
			return (dparam0/data.dparam) >= max_step_ratio;
		}
	};

	struct CombinedDynamicStepsizeStopCondition
	{
		template<typename R>
		__host__ __device__ bool operator()(const R& radius, const R& max_parameter, const R& max_step_ratio, const R& dparam0, const ParticleData<R>& data) const
		{
			return DynamicStepsizeStopCondition()(max_step_ratio, dparam0, data) || CombinedStopCondition()(radius, max_parameter, data);
		}
	};

}

#endif
