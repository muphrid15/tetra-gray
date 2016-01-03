#ifndef COORD_SYSTEM_HDR
#define COORD_SYSTEM_HDR
#include "types.cuh"

namespace ray
{
	template<typename R>
	__host__ __device__ Vector<R> sphericalCoordinatesFromCartesian(const Vector<R>& posvec)
	{
		const R x = posvec[1];
		const R y = posvec[2];
		const R z = posvec[3];
		const R t = posvec[0];

		const R r = sqrt(x*x + y*y + z*z);
		const R theta = (r > 0.) ? acos(z/r) : 0.;
		const R phi = atan2(y, x);
		const R spherical_coordinates[4] = {t, r, theta, phi};
		return Vector<R>(spherical_coordinates);
	}


	//Note that our spheroidal coordinates are different from, e.g. wikipedia's oblate spheroidal
	//In particular, our angle nu is more like spherical coordinates' theta
	//nu runs the range [0,pi] as a result
	//posvec is assumed to be in cartesian coordinates
	template<typename R>
	__host__ __device__ inline Vector<R> spheroidalCoordinatesFromCartesian(const R& scale_factor_a, const Vector<R>& posvec)
	{
		const R x = posvec[1];
		const R y = posvec[2];
		const R z = posvec[3];
		const R t = posvec[0];

		const R phi = atan2(y, x);
		const R rho = sqrt(x*x + y*y);
		const R d1 = sqrt((rho+scale_factor_a)*(rho+scale_factor_a) + z*z);
		const R d2 = sqrt((rho-scale_factor_a)*(rho-scale_factor_a) + z*z);
		
		const R cosh_mu = ((d1 + d2)/(2*scale_factor_a));
		const R mu = acosh(cosh_mu);
		const R cos2_nu = 1. - (d1-d2)*(d1-d2)/(4.*scale_factor_a*scale_factor_a);
		const R cos_nu_sign = copysign(sqrt(fmax(cos2_nu, R(0.))), z);
		const R nu = acos(cos_nu_sign);
//		const R nu = acos(z/(scale_factor_a*cosh_mu));

		const R spheroidal_coordinates[4] = { t, mu, nu, phi};
		return Vector<R>(spheroidal_coordinates);
	} 

	//All of the following assume t, x, y, z ordering for components, and return the spheroidal basis vector in terms of Cartesian components
	//Angles are inputs here to reduce reduplication of computation of trig functions
	template<typename R>
	__host__ __device__ inline Vector<R> spheroidalBasisVectorEmu(const R& sinh_mu, const R& cosh_mu, const R& sin_nu, const R& cos_nu, const R& sin_phi, const R& cos_phi)
	{
		const R components[4] = { R(0.), sinh_mu*sin_nu*cos_phi, sinh_mu*sin_nu*sin_phi, cosh_mu*cos_nu };
		return Vector<R>(components)*R(1./sqrt(sinh_mu*sinh_mu + cos_nu*cos_nu));
	}

	template<typename R>
	__host__ __device__ inline Vector<R> spheroidalBasisVectorEnu(const R& sinh_mu, const R& cosh_mu, const R& sin_nu, const R& cos_nu, const R& sin_phi, const R& cos_phi)
	{
		const R components[4] = { R(0.), cosh_mu*cos_nu*cos_phi, cosh_mu*cos_nu*sin_phi, -sinh_mu*sin_nu};
		return Vector<R>(components)*R(1./sqrt(sinh_mu*sinh_mu + cos_nu*cos_nu));
	}

	template<typename R>
	__host__ __device__ inline Vector<R> spheroidalBasisVectorPhi(const R& sin_phi, const R& cos_phi)
	{
		const R components[4] = { R(0.), -sin_phi, cos_phi, R(0.) };
		return Vector<R>(components);
	}
	
	template<typename R>
	__host__ __device__ Vector<R> spheroidalBasisVectorT()
	{
		const R components[4] = { R(1.), R(0.), R(0.), R(0.) };
		return Vector<R>(components);
	}
}
#endif
