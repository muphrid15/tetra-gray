#ifndef COORD_SYSTEM_HDR
#define COORD_SYSTEM_HDR
#include "types.cuh"

namespace ray
{
	template<typename R>
	__host__ __device__ Vector<R> sphericalCoordinatesFromCartesian(const Vector<R>& posvec)
	{
		const R x = posvec.extractComponent(0);
		const R y = posvec.extractComponent(1);
		const R z = posvec.extractComponent(2);
		const R t = posvec.extractComponent(3);

		const R r = sqrt(x*x + y*y + z*z);
		const R theta = (r > 0.) ? acos(z/r) : 0.;
		const R phi = atan2(x, y);
		const R spherical_coordinates[4] = { r, theta, phi, t };
		return Vector<R>(spherical_coordinates);
	}


	template<typename R>
	__host__ __device__ inline Vector<R> spheroidalCoordinatesFromCartesian(const R& scale_factor_a, const Vector<R>& posvec)
	{
		const R x = posvec.extractComponent(0);
		const R y = posvec.extractComponent(1);
		const R z = posvec.extractComponent(2);
		const R t = posvec.extractComponent(3);

		const R phi = atan2(x, y);
		const R rho = x*x + y*y;
		const R d1 = sqrt((rho+scale_factor_a)*(rho+scale_factor_a) + z*z);
		const R d2 = sqrt((rho-scale_factor_a)*(rho-scale_factor_a) + z*z);
		
		const R mu = acosh((d1 + d2)/(2*scale_factor_a));
		const R nu = asin((d1 - d2)/(2*scale_factor_a));

		const R spheroidal_coordinates[4] = { mu, nu, phi, t };
		return Vector<R>(spheroidal_coordinates);
	} 

	template<typename R>
	__host__ __device__ inline Vector<R> spheroidalBasisVectorEmu(const R& sinh_mu, const R& cosh_mu, const R& sin_nu, const R& cos_nu, const R& sin_phi, const R& cos_phi)
	{
		const R components[4] = { sinh_mu*sin_nu*cos_phi, sinh_mu*sin_nu*sin_phi, cosh_mu*cos_nu, R(0.) };
		return Vector<R>(components)*R(1./sqrt(sinh_mu*sinh_mu + cos_nu*cos_nu));
	}

	template<typename R>
	__host__ __device__ inline Vector<R> spheroidalBasisVectorEnu(const R& sinh_mu, const R& cosh_mu, const R& sin_nu, const R& cos_nu, const R& sin_phi, const R& cos_phi)
	{
		const R components[4] = { -cosh_mu*cos_nu*cos_phi, -cosh_mu*cos_nu*sin_phi, sinh_mu*sin_nu, R(0.) };
		return Vector<R>(components)*R(1./sqrt(sinh_mu*sinh_mu + cos_nu*cos_nu));
	}

	template<typename R>
	__host__ __device__ inline Vector<R> spheroidalBasisVectorPhi(const R& sin_phi, const R& cos_phi)
	{
		const R components[4] = { cos_phi, sin_phi, R(0.), R(0.) };
		return Vector<R>(components);
	}
	
	template<typename R>
	__host__ __device__ Vector<R> spheroidalBasisVectorT()
	{
		const R components[4] = { R(0.), R(0.), R(0.), R(1.) };
		return Vector<R>(components);
	}
}
#endif
