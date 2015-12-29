#ifndef FTK_FUNCTOR_INTERFACE
#define FTK_FUNCTOR_INTERFACE
#include <utility>
#include "operator.h"
#include "parameterized.h"

namespace ftk
{

	template<template<typename> class Tc>
		class Functor
		{
			public:
				struct FmapFunctor
				{
					template<typename F, typename A>
						constexpr auto operator()(const F& func, const Tc<A>& ta) const -> Tc<decltype(func(std::declval<A>()))>
						{
							return Tc<A>::fmap(func, ta);
						}
				};

				constexpr static FmapFunctor fmap = FmapFunctor();
		};

	template<typename Fa>
		class FunctorData : public Functor<Parameterizable<Fa>::template Typecons> {};

	
	struct FtkFmapFunctor
	{
		template<typename F, typename Fa>
			constexpr auto operator()(const F& f, const Fa& fa) -> typename Parameterizable<Fa>::template Typecons<decltype(f(std::declval<typename Parameterizable<Fa>::BaseType>()))>
			{
				return fa | FunctorData<Fa>::fmap >> f;
			}
	};

	constexpr FtkFmapFunctor fmap = FtkFmapFunctor();
}

#endif
