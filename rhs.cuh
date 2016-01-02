#ifndef RHS_HDR
#define RHS_HDR
#include "particle.cuh"
#include "integrator.cuh"
#include "types.cuh"
#include "coord_systems.cuh"

namespace ray
{
	struct FlatRHS
	{
		template<typename R>
			__host__ __device__ Particle<R> operator()(const Particle<R>& data, const R& proper_time) const
			{
				const auto momrhs = Vector<R>();

				const auto posrhs = data.momentum;
				const auto rhs = pt::makeParticle(posrhs, momrhs);

				return rhs;
			}
	};


	template<typename R>
		__host__ __device__ R inline doranBeta(const R& cosh_mu, const R& sin_nu)
		{
			return atanh(sin_nu/cosh_mu);
		}

	template<typename R>
		__host__ __device__ inline Vector<R> doranVectorV(const R& beta, const R& sin_phi, const R& cos_phi, const Vector<R>& that, const Vector<R>& phihat)
		{
			return that*cosh(beta) + phihat*sinh(beta);
		}

	template<typename R>
		__host__ __device__ inline Vector<R> doranPositionGauge(const R& sinh_mu, const Vector<R>& emuhat, const R& scale_factor_a, const Vector<R>& doran_vec_v, const Vector<R>& vec_arg)
		{
			const auto rootfactor = R(sqrt(2.*sinh_mu/scale_factor_a/(1.+sinh_mu*sinh_mu)));
			const auto dot_product = (vec_arg*doran_vec_v).scalarPart();
			return vec_arg + rootfactor*dot_product*emuhat;
		}

	template<typename R>
		__host__ __device__ inline Bivector<R> doranRotationGauge(const R& sinh_mu, const R& cos_nu, const Vector<R>& muhat, const Vector<R>& nuhat, const Vector<R>& phihat, const Vector<R>& that, const R& beta, const Vector<R>& doran_v, const R& scale_factor_a, const Vector<R>& vec_arg)
		{
			const R alpha = R(-sqrt(2.*sinh_mu/(scale_factor_a*(sinh_mu*sinh_mu + cos_nu*cos_nu))));

			const auto common_denom_inverse = (scale_factor_a*(Multivector<R>(sinh_mu) + Multivector<R>::makePseudoscalar(cos_nu)));
			const auto common_denom = common_denom_inverse.inverse();

			const auto arg_dot_mu = (vec_arg*muhat).scalarPart();
			const auto arg_dot_nu = (vec_arg*nuhat).scalarPart();
			const auto arg_dot_phi = (vec_arg*phihat).scalarPart();

			const auto uterm = R(1./alpha)*Bivector<R>(common_denom*common_denom*Bivector<R>(muhat*doran_v));
			const auto nuterm = alpha*Bivector<R>(-common_denom*Bivector<R>(nuhat*doran_v));
			const auto phiterm = alpha/cosh(beta)*Bivector<R>(-common_denom*Bivector<R>(phihat*that));

			return arg_dot_mu*uterm + arg_dot_nu*nuterm + arg_dot_phi*phiterm;
		}

	struct DoranRHS
	{
		template<typename R>
			__host__ __device__ Particle<R> operator()(const Particle<R>& data, const R& affine_param, const R& scale_factor_a) const
			{
				const auto spheroidal_coords = spheroidalCoordinatesFromCartesian(scale_factor_a, data.position);

				const R mu = spheroidal_coords.extractComponent(0);
				const R nu = spheroidal_coords.extractComponent(1);
				const R phi = spheroidal_coords.extractComponent(2);

				const R sinh_mu = sinh(mu);
				const R cosh_mu = cosh(mu);
				const R sin_nu = sin(nu);
				const R cos_nu = cos(nu);
				const R sin_phi = sin(phi);
				const R cos_phi = cos(phi);

				const auto muhat = spheroidalBasisVectorEmu(sinh_mu, cosh_mu, sin_nu, cos_nu, sin_phi, cos_phi);
				const auto nuhat = spheroidalBasisVectorEnu(sinh_mu, cosh_mu, sin_nu, cos_nu, sin_phi, cos_phi);
				const auto phihat = spheroidalBasisVectorPhi(sin_phi,cos_phi);
				const auto that = spheroidalBasisVectorT<R>();
				const auto beta = doranBeta(cosh_mu, sin_nu);
				const auto doran_vector_v = doranVectorV(beta, sin_phi, cos_phi, that, phihat);

				const auto posrhs = doranPositionGauge(sinh_mu, muhat, scale_factor_a, doran_vector_v, data.momentum);
				const auto momrhs = -Vector<R>(doranRotationGauge(sinh_mu, cos_nu, muhat, nuhat, phihat, that, beta, doran_vector_v, scale_factor_a, data.momentum)*data.momentum);
				return pt::makeParticle(posrhs, momrhs);
			}
	};
}
#endif
