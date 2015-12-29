#ifndef IDENTITY_MONAD_HDR
#define IDENTITY_MONAD_HDR
#include "monad.h"
/*
 * I think we can do this better.
 * All this specification of nested types in function arguments can confuse the compiler. We should help the compiler by making arguments general types and then check to see if that type is parameterized.
 * That means implementing a Parameterizable template that simply checks whether the datatype defines a type called BaseType.
 * Further, if it's a Functor of any kind, it will define a FunctorInterface, which the individual types of functor will check through to ensure the existence (and matching signatures) of the various functor operations.
 * If we don't do something like this, consider a State monad.  If we want to execState, and we follow the design of Identity, then we would have to specify the state monad's type parameters to get the execState function.
 * If instead we allow a general argument to be passed in, and then we see if there is a State Monad instance for that argument...
 * But of course we must know there is a state monad instance for that; that has to be encoded in the FunctorInterface typedef.
 *
 * In other words, when we use one of these parameterized types, like IdentityType<A>, the computation should look like this:
 * fmap(f, I_A)
 * fmap inspects I_A, asks if there is a functor interface for it (so we create a template FunctorInterface and specialize it for all Identity<A> to be the Identity Functor interface)
 * Using the functor interface, we check that this is indeed a monad (we try to instantiate Monad with the functor interface)
 * If it is, we use the function from monad to compute fmap
 *
 * For a function like runIdentity, it goes something like this
 * All Identity-like classes are "Identity-like": we can use a traits class here to designate the type of the function called run
 * Therefore runIdentity will look at IdentityType<A>, instantiate the IdentityLike class on this to see if it has a function called run, and if so, use that
 */

/*
namespace ftk
{
	template<template<typename> class Id>
	struct IdentityLike
	{
		public:
			template<typename A>
			constexpr static A run(const Id<A>& ia)
			{
				return ia.run();
			}
	};

	struct RunIdentityFunctor
	{
		template<typename Ida>
		constexpr auto operator()(const Ida& ida) const -> typename Parameterizable<Ida>::BaseType
		{
			return IdentityLike<Parameterizable<Ida>::template Typecons>::run(ida);
		}
	};

	constexpr RunIdentityFunctor runIdentity = RunIdentityFunctor();

	template<typename A>
		struct Identity
		{
			private: 
				A value;

			public:
				constexpr Identity(const A& a) : value(a) {}
				constexpr A run() const { return value; }
		};

	template<typename A>
		struct Parameterizable<Identity<A> >
		{
			template<typename B>
			using Typecons = Identity<B>;

			using BaseType = A;
		};

	template<template<typename> class Tc>
	struct IdentityLikeMonadInstance
	{
		using ILike = IdentityLike<Tc>;

		template<typename A>
			constexpr static Tc<A> unit(const A& a)
			{
				return Tc<A>(a);
			}

		template<typename F, typename A>
			constexpr static auto fmap(const F& f, const Tc<A>& ta) -> Tc<decltype(f(std::declval<A>()))>
			{
				return unit(f(ILike::run(ta)));
			}

		template<typename A>
			constexpr static auto join(const Tc<Tc<A> >& tta) -> Tc<A>
			{
				return unit(ILike::run(ILike::run(tta)));
			}
	};

	template<>
	struct MonadInstance<Identity> : public IdentityLikeMonadInstance<Identity> {};

	template<>
	struct ApplicativeInstance<Identity> : public MonadApplicativeDefault<Identity> {};

	template<>
	struct FunctorInstance<Identity> : public MonadFunctorDefault<Identity> {};

}
*/

namespace ftk
{
	template<typename Ida>
	struct IdentityLikeData 
	{
		public:
			constexpr static typename Parameterizable<Ida>::BaseType run(const Ida& ida)
			{
				return ida.run();
			}
	};

	template<typename Ida>
		struct IdentityLikeDataMonad : public MonadDefault<Ida>
		{
			private:
				using A = typename Parameterizable<Ida>::BaseType;

				template<typename B>
				using Tc = typename Parameterizable<Ida>::template Typecons<B>;
			public:
				constexpr static Ida unit(const A& a)
				{
					return Ida(a);
				}

				template<typename F>
					constexpr static auto fmap(const F& f, const Ida& ta) -> Tc<decltype(f(std::declval<A>()))>
					{
						using B = decltype(f(std::declval<A>()));
						return IdentityLikeDataMonad<Tc<B> >::unit(f(IdentityLikeData<Ida>::run(ta)));
					}

				constexpr static auto join(const Tc<Ida>& tta) -> Ida
				{
					return unit(
							IdentityLikeData<Ida>::run(
								IdentityLikeData<Tc<Ida> >::run(tta)));
				}
		};


	template<typename A>
		struct Identity : public IdentityLikeDataMonad<Identity<A> >
		{
			private: 
				A value;

			public:
				constexpr Identity(const A& a) : value(a) {}

				//Required by IdentityLikeData
				constexpr A run() const { return value; }

				/*
				//Required by Parameterizable
				template<typename B>
					using Typecons = Identity<B>;
				using BaseType = A;
				*/
		};
	
	struct RunIdentityFunctor
	{
		template<typename Ida>
		constexpr auto operator()(const Ida& ida) const -> typename Parameterizable<Ida>::BaseType
		{
			return IdentityLikeData<Ida>::run(ida);
		}
	};

	constexpr RunIdentityFunctor runIdentity = RunIdentityFunctor();

	template<typename A>
		struct Parameterizable<Identity<A> >
		{
			template<typename B>
			using Typecons = Identity<B>;

			using BaseType = A;
		};
}

#endif
