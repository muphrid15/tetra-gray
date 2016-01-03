#ifndef TYPES_HDR
#define TYPES_HDR
#include "clifford-static.cuh"
#include "particle.cuh"
#include "integrator.cuh"

namespace ray
{
	constexpr double PI = 3.14159265358979323846;
	template<typename R>
	using Versor = multi::Versor<R>;

	template<typename R>
	using Vector = multi::Vector<R>;

	template<typename R>
	using Bivector = multi::Bivector<R>;
	
	template<typename R>
	using Particle = pt::Particle<R>;

	template<typename R>
	using ParticleData = ode::ODEData<Particle<R>, R>;
}

#endif
