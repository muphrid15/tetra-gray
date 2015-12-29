#ifndef RHS_HDR
#define RHS_HDR
#include "particle.cuh"
#include "integrator.cuh"

namespace ray
{
	template<uint plus_dim, uint minus_dim = 0, uint zero_dim = 0, typename R = double>
		struct FlatRHS
		{
			using Pt = Particle<plus_dim, minus_dim, zero_dim, R>;
			__host__ __device__ Pt operator()(const Pt& data, const R& proper_time) const
			{
				const auto momrhs = typename Pt::Smv();

				const auto posrhs = data.momentum;
				const auto rhs = ray::makeParticle(posrhs, momrhs);

				return rhs;
			}
		};
}
#endif
