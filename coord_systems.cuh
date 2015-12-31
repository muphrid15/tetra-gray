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

}
#endif
