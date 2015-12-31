#ifndef RHS_HDR
#define RHS_HDR
#include "particle.cuh"
#include "integrator.cuh"
#include "types.cuh"

namespace ray
{
		struct FlatRHS
		{
			template<typename R>
			__host__ __device__ Particle<R> operator()(const Particle<R>& data, const R& proper_time) const
			{
				const auto momrhs = Vector<R>();

				const auto posrhs = data.momentum;
				const auto rhs = pt::makeParticle(posrhs, momrhs);

				return rhs;
			}
		};
}
#endif
