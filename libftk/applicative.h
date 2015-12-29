#ifndef FTK_APPLICATIVE_INTERFACE
#define FTK_APPLICATIVE_INTERFACE
#include "functor.h"

namespace ftk
{
	template<template<typename> class Tc>
		class Applicative : public Functor<Tc>
		{
			public:
			
			struct PureFunctor
			{
				template<typename A>
					constexpr Tc<A> operator()(const A& a) const
					{
						return Tc<A>::pure(a);
					}
			};

			struct ApplyFunctor
			{
				template<typename F, typename A>
					constexpr auto operator()(const Tc<F>& af, const Tc<A>& aa) const -> Tc<decltype(std::declval<F>()(std::declval<A>()))>
					{
						return Tc<A>::apply(af, aa);
					}
			};

			constexpr static PureFunctor pure = PureFunctor();
			constexpr static ApplyFunctor apply = ApplyFunctor();
		};

	template<typename Aa>
		struct ApplicativeData : public Applicative<Parameterizable<Aa>::template Typecons> {};

	struct FtkApplyFunctor
	{
		template<typename Af, typename Aa>
			constexpr auto operator()(const Af& af, const Aa& aa) -> typename Parameterizable<Af>::template Typecons<decltype(std::declval<typename Parameterizable<Af>::BaseType>()(std::declval<typename Parameterizable<Aa>::BaseType>()))>
			{
				return aa | ApplicativeData<Aa>::apply >> af;
			}
	};

	constexpr FtkApplyFunctor apply = FtkApplyFunctor();

}

#endif
