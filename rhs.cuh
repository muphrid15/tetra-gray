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
				const auto rhs = Particle<R>(posrhs, momrhs);

				return rhs;
			}
	};
	/*
	   Functions for Doran spinning black holes

	   These equations are detailed in section 14.7.3 of Doran and Lasenby, i.e. Geometric Algebra for Physicists
	  */

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
			const auto dot_product = (vec_arg|doran_vec_v);
			return vec_arg + emuhat*rootfactor*dot_product;
		}

	template<typename R>
		__host__ __device__ inline Bivector<R> doranRotationGauge(const R& sinh_mu, const R& cos_nu, const Vector<R>& muhat, const Vector<R>& nuhat, const Vector<R>& phihat, const Vector<R>& that, const R& beta, const Vector<R>& doran_v, const R& scale_factor_a, const Vector<R>& vec_arg)
		{
			const R alpha = R(-sqrt(2.*sinh_mu/(scale_factor_a*(sinh_mu*sinh_mu + cos_nu*cos_nu))));

			/*
			const auto common_denom_inverse = (scale_factor_a*(Versor<R>(sinh_mu) + Versor<R>::makePseudoscalar(cos_nu)));
			const auto common_denom = common_denom_inverse.inverse();
			*/

			const auto arg_dot_mu = (vec_arg|muhat);
			const auto arg_dot_nu = (vec_arg|nuhat);
			const auto arg_dot_phi = (vec_arg|phihat);

//			const auto uterm = Bivector<R>(common_denom*common_denom*Bivector<R>(muhat^doran_v))*R(1./alpha);
			const auto muterm = (muhat^doran_v)*R(1./alpha);
//			const auto nuterm = Bivector<R>(-common_denom*Bivector<R>(nuhat*doran_v))*alpha;
			const auto nuterm = (-nuhat^doran_v)*alpha;
//			const auto phiterm = Bivector<R>(-common_denom*Bivector<R>(phihat^that))*(alpha/cosh(beta));
			const auto phiterm = -(phihat^that)*(alpha/cosh(beta));

			const auto common_scalar = scale_factor_a*sinh_mu;
			const auto common_pseudo = scale_factor_a*cos_nu;
			const auto common_denom = (common_scalar*common_scalar + common_pseudo*common_pseudo);

			const auto mu_scalar = (common_scalar*common_scalar - common_pseudo*common_pseudo)/(common_denom*common_denom);
			const auto mu_pseudo = R(2.*common_scalar*common_pseudo)/(common_denom*common_denom);

			return (muterm*mu_scalar + (~muterm)*mu_pseudo)*arg_dot_mu/(common_denom*common_denom) + ((nuterm*common_scalar + (~nuterm)*common_pseudo)*arg_dot_nu + (phiterm*common_scalar + (~phiterm)*common_pseudo)*arg_dot_phi)/(common_denom);
		}

	struct DoranRHS
	{
		template<typename R>
			__host__ __device__ Particle<R> operator()(const Particle<R>& data, const R& affine_param, const R& scale_factor_a) const
			{
				const auto spheroidal_coords = spheroidalCoordinatesFromCartesian(scale_factor_a, data.position);

				const R mu = spheroidal_coords[1];
				const R nu = spheroidal_coords[2];
				const R phi = spheroidal_coords[3];

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

				//The geodesic equation: x' = h(k), k' = -omega(k)|k (cartesian coords, no derivatives of basis vectors)
				const auto posrhs = doranPositionGauge(sinh_mu, muhat, scale_factor_a, doran_vector_v, data.momentum);
				const auto momrhs = -(doranRotationGauge(sinh_mu, cos_nu, muhat, nuhat, phihat, that, beta, doran_vector_v, scale_factor_a, data.momentum)|data.momentum);
				return Particle<R>(posrhs, momrhs);
			}
	};
}
#endif
