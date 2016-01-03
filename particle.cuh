#ifndef PARTICLE_HDR
#define PARTICLE_HDR
#include "clifford-static.cuh"

namespace pt
{
	template<typename R>
	struct Particle
	{
		private:
			__host__ __device__ constexpr static uint dimensions()
			{
				return 4u;
			}

			using Vector = multi::Vector<R>;
		public:
			Vector position, momentum;

			__host__ __device__ Particle(const R (&posarr)[dimensions()], const R (&momarr)[dimensions()])
			{
				position = Vector(posarr);
				momentum = Vector(momarr);
			}

			__host__ __device__ Particle(const Vector& pos, const Vector& mom) : position(pos), momentum(mom) {}

			__host__ __device__ Particle() : position(Vector()), momentum(Vector()) {}

			__host__ __device__ Particle& operator*=(const R& scalar)
			{
				position = position*scalar;
				momentum = momentum*scalar;
				return *this;
			}

			__host__ __device__ Particle& operator+=(const Particle& other)
			{
				position = position+other.position;
				momentum = momentum+other.momentum;
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

	template<typename R>
	__host__ __device__ Particle<R> operator*(const R& scalar, const Particle<R>& part)
	{
		return part*scalar;
	}

	/*
	template<typename R>
	__host__ __device__ Particle<R> makeParticle(const mv::SingleGradedMultivector<1U, plus_dim, minus_dim, zero_dim, R>& pos, const mv::SingleGradedMultivector<1U, plus_dim, minus_dim, zero_dim, R>& mom)
	{
		return Particle<plus_dim, minus_dim, zero_dim, R>(pos, mom);
	}
	*/
}

#endif
