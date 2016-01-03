#ifndef ODE_HDR
#define ODE_HDR

namespace ode
{
	//param is arbitrary, need not represent time (and doesn't in GR)
	template<class A, class Real>
		struct ODEData
		{
			A value;
			Real param, dparam;
			__host__ __device__ constexpr ODEData(const A& ai, const Real& ti, const Real& dti) : value(ai), param(ti), dparam(dti) {}

			__host__ __device__ constexpr ODEData() : value(), param(), dparam() {}
		};


	//Summary of required function signatures:
	//A: whatever data we're evolving
	//R: the real data type (float, double, etc.)
	//O<A,R>: the ode data
	//Stepper: O<A,R> x (AxR -> A) -> O<A,R>
	//RHS: AxR -> A
	//Stop: O<A,R> -> 2 (bool)
	struct ODEIntegrator
	{
		template<typename A, typename RHS, typename Stepper, typename Stop, typename Real>
			__host__ __device__ ODEData<A,Real> operator()(const ODEData<A, Real>& ode0, const Stepper& step, const RHS& rhs, const Stop& stop) const
			{
				ODEData<A, Real> odenow = ode0;
				while(!stop(odenow))
				{
					odenow = step(odenow, rhs);
				}
				return odenow;
			}
	};

	struct RK4
	{
		template<typename A, typename Real, typename RHS>
			__host__ __device__ ODEData<A,Real> operator()(const ODEData<A,Real>& ode0, const RHS& rhs) const
			{
				const auto k1 = rhs(ode0.value, ode0.param);
				const auto k2 = rhs(ode0.value+Real(ode0.dparam/2.)*k1, ode0.param+Real(ode0.dparam/2.));
				const auto k3 = rhs(ode0.value+Real(ode0.dparam/2.)*k2, ode0.param+Real(ode0.dparam/2.));
				const auto k4 = rhs(ode0.value+ode0.dparam*k3, ode0.param+ode0.dparam);
				return ODEData<A,Real>(ode0.value+Real(ode0.dparam/6.)*(k1+Real(2.)*k2+Real(2.)*k3+k4),ode0.param+ode0.dparam, ode0.dparam);
			}
	};
}

#endif
