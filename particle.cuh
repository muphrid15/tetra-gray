#ifndef PARTICLE_HDR
#define PARTICLE_HDR
#include "clifford.cuh"

namespace ray
{
	template<uint plus_dim, uint minus_dim, uint zero_dim, typename R>
	struct Particle
	{
		private:
			__host__ __device__ constexpr static uint dimensions()
			{
				return plus_dim + minus_dim + zero_dim;
			}

		public:
			using Smv = mv::SingleGradedMultivector<1U, plus_dim, minus_dim, zero_dim, R>;
			Smv position, momentum;

			__host__ __device__ Particle(const R (&posarr)[dimensions()], const R (&momarr)[dimensions()])
			{
				position = Smv(posarr);
				momentum = Smv(momarr);
			}
			__host__ __device__ Particle(const Smv& pos, const Smv& mom) : position(pos), momentum(mom) {}

			__host__ __device__ Particle& operator*=(const R& scalar)
			{
				position *= scalar;
				momentum *= scalar;
				return *this;
			}

			__host__ __device__ Particle& operator+=(const Particle& other)
			{
				position += other.position;
				momentum += other.momentum;
				return *this;
			}

			__host__ __device__ Particle operator*(const R& scalar) const
			{
				Particle ret = *this;
				ret *= scalar;
				return ret;
			}

			__host__ __device__ Particle operator+(const Particle& other) const
			{
				Particle ret = *this;
				ret += other;
				return ret;
			}
	};	

	template<uint plus_dim, uint minus_dim, uint zero_dim, typename R>
	__host__ __device__ Particle<plus_dim, minus_dim, zero_dim, R> operator*(const R& scalar, const Particle<plus_dim, minus_dim, zero_dim, R>& part)
	{
		return part*scalar;
	}

	template<uint plus_dim, uint minus_dim, uint zero_dim, typename R>
	__host__ __device__ Particle<plus_dim, minus_dim, zero_dim, R> makeParticle(const mv::SingleGradedMultivector<1U, plus_dim, minus_dim, zero_dim, R>& pos, const mv::SingleGradedMultivector<1U, plus_dim, minus_dim, zero_dim, R>& mom)
	{
		return Particle<plus_dim, minus_dim, zero_dim, R>(pos, mom);
	}
}

#endif
