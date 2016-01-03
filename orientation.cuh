#ifndef ORIENTATION_HDR
#define ORIENTATION_HDR
#include "types.cuh"

namespace ray
{
	//note this is not numerically stable for rotations near 180 degrees
	/*
	template<uint plus_dim, uint minus_dim, uint zero_dim, typename R>
		__host__ __device__ mv::Multivector<plus_dim, minus_dim, zero_dim, R> simpleRotor(const mv::SingleGradedMultivector<1, plus_dim, minus_dim, zero_dim, R>& vstart, const mv::SingleGradedMultivector<1, plus_dim, minus_dim, zero_dim, R>& vend)
		{
			const auto vstart_normed = vstart.normalize();
			const auto vend_normed = vend.normalize();
			const auto bisector = vstart_normed + vend_normed;
			return bisector*vstart;
		}
		*/

	template<typename R>
		__host__ __device__ Versor<R> simpleRotorFromAngle(const Vector<R> v1, const Vector<R>& v2,  const R& angle)
		{
			const auto bivec = v1^v2;
			const R bivec_squared = bivec|bivec;
			const R abs_bivec_norm = sqrt(abs(bivec_squared));
			const R half_angle = angle/2.;
			const auto mvone = Versor<R>(1.);

			//Euclidean case
			if(bivec_squared < R(0.))
			{
				const auto scalar_part = mvone*cos(half_angle);
				const auto bivector_part = Versor<R>(bivec*sin(half_angle)/abs_bivec_norm);
				//the minus sign is correct for clifford, different from quats
				return scalar_part - bivector_part;
			}
			//Hyperbolic case
			else if(bivec_squared > R(0.))
			{
				const auto scalar_part = mvone*cosh(half_angle);
				const auto bivector_part = Versor<R>(bivec*sinh(half_angle)/abs_bivec_norm);
				return scalar_part - bivector_part;
			}
			//Galilean case
			return mvone - Versor<R>(bivec*half_angle);
		}

	/*
	template<typename X>
		__host__ __device__ X bilinearMultiply(const X& versor, const X& input)
		{
			return versor*input*versor.inverse();
		}
		*/
}

#endif
