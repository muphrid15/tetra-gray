template<class A, class Real>
struct ODEData
{
	A value;
	Real param, dparam;
	__host__ __device__ constexpr ODEData(const A& ai, const Real& ti, const Real& dti) : value(ai), param(ti), dparam(dti) {}
};


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
		const auto k2 = rhs(ode0.value+k1*ode0.dparam/2., ode0.param+ode0.dparam/2.);
		const auto k3 = rhs(ode0.value+k2*ode0.dparam/2., ode0.param+ode0.dparam/2.);
		const auto k4 = rhs(ode0.value+k3*ode0.dparam, ode0.param+ode0.dparam);
		return ODEData<A,Real>(ode0.value+(k1+2.*k2+2.*k3+k4)*ode0.dparam/6.,ode0.param+ode0.dparam, ode0.dparam);
	}
};