#ifndef FTK_FUNCTOR_INTERFACE
#define FTK_FUNCTOR_INTERFACE
#include <utility>
#include "operator.h"
#include "parameterized.h"

namespace ftk
{

	/*
	 * Welcome to the pit of insanity.
	 *
	 * "Functor" here means functor in the category theory/Haskell sense: for any type, the functor defines a new type. In c++, that's just a template.
	 * But a functor does more than that: it also says, for any types A and B, a function f: A->B, a functor F also defines a corresponding function fmap f: F<A> -> F<B>
	 * This is familar from std::transform, for instance.  Vector type containers are functors in this sense.
	 * 
	 * class Functor is a thin wrapper: it allows us to write the templated function FmapFunctor::operator() without knowing the concrete datatype A that the underlying functor (here, Tc, for "type constructor") is acting on
	 *
	 * This is more necessary for applicative functors and monads, where the type constructor can't be deduced through argument deduction; this is only provided for consistency, or to be able to guarantee that a given type constructor satisfies the functor constraints
	 */
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

	
	//The FtkFmapFunctor is a convenient way to invoke the higher-order function fmap without explicitly stating the underlying Functor type (or even its underlying data type).
	//Note that fmap (the object) is constexpr and can't have its address taken. It can only be used with the operators in "operators.h".
	//FtkFmapFunctor() can be used in all contexts.
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
