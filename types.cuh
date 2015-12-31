#ifndef TYPES_HDR
#define TYPES_HDR
#include "clifford.cuh"
#include "particle.cuh"
#include "integrator.cuh"

namespace ray
{
	constexpr double PI = 3.14159265358979323846;
	template<typename R>
	using Multivector = mv::Multivector<3, 1, 0, R>;

	template<typename R>
	using Vector = mv::SingleGradedMultivector<1, 3, 1, 0, R>;
	
	template<typename R>
	using Particle = pt::Particle<3, 1, 0, R>;

	template<typename R>
	using ParticleData = ode::ODEData<Particle<R>, R>;
}

#endif
