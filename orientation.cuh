#ifndef ORIENTATION_HDR
#define ORIENTATION_HDR
#include "clifford.cuh"

namespace ray
{

	//note this is not numerically stable for rotations near 180 degrees
	template<uint plus_dim, uint minus_dim, uint zero_dim, typename R>
	__host__ __device__ mv::Multivector<plus_dim, minus_dim, zero_dim, R> simpleRotor(const mv::SingleGradedMultivector<1, plus_dim, minus_dim, zero_dim, R>& vstart, const mv::SingleGradedMultivector<1, plus_dim, minus_dim, zero_dim, R>& vend)
	{
		const auto vstart_normed = vstart.normalize();
		const auto vend_normed = vend.normalize();
		const auto bisector = vstart_normed + vend_normed;
		return bisector*vstart;
	}

	template<uint plus_dim, uint minus_dim, uint zero_dim, typename R>
	__host__ __device__ mv::Multivector<plus_dim, minus_dim, zero_dim, R> simpleRotorFromAngle(const mv::SingleGradedMultivector<1, plus_dim, minus_dim, zero_dim, R>& v1, const mv::SingleGradedMultivector<1, plus_dim, minus_dim, zero_dim, R>& v2, const R& angle)
	{
		const auto bivec = (v1*v2) % 2;
		const auto bivec_squared = (bivec*bivec).scalarPart();
		const auto half_angle = angle/2.;
		const auto mvone = mv::Multivector<plus_dim, minus_dim, zero_dim>(1.);
		return bivec_squared < 0 ? cos(half_angle)*mvone - sin(half_angle)*(bivec.normalize()) : (bivec_squared > 0) ? (cosh(half_angle)*mvone - sinh(half_angle)*(bivec.normalize())) : mvone - half_angle*bivec;
		
	}

	template<typename X>
	__host__ __device__ X bilinearMultiply(const X& versor, const X& input)
	{
		return versor*input*versor.inverse();
	}
}

#endif
