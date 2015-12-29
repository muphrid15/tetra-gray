#ifndef FTK_MONAD_INTERFACE
#define FTK_MONAD_INTERFACE
#include "applicative.h"

namespace ftk
{
	template<template<typename> class Tc>
		class Monad : public Applicative<Tc>
		{
			public:

			struct JoinFunctor
			{
				template<typename A>
					constexpr Tc<A> operator()(const Tc<Tc<A> >& tta) const
					{
						return Tc<A>::join(tta);
					}

			};

			struct BindFunctor
			{
				template<typename F, typename A>
					constexpr auto operator()(const F& func, const Tc<A>& ta) const -> decltype(func(std::declval<A>()))
					{
						return Tc<A>::join(Tc<A>::fmap(func, ta));
					}
			};

			private:
			template<typename Mb>
				struct ThenFunctorHelper
				{
					Mb mb;
					constexpr ThenFunctorHelper(const Mb& in) : mb(in) {}

					template<typename A>
						constexpr Mb operator()(const A& a) const
						{
							return mb;
						}
				};

			public:
			struct ThenFunctor
			{
				template<typename B, typename A>
				constexpr auto operator()(const Tc<B>& mb, const Tc<A>& ma) -> Tc<B>
				{
					return BindFunctor()(ThenFunctorHelper<Tc<B> >(mb), ma);
				}
			};

			constexpr static typename Applicative<Tc>::PureFunctor unit = Applicative<Tc>::pure;
			constexpr static JoinFunctor join = JoinFunctor();
			constexpr static BindFunctor bind = BindFunctor();
			constexpr static ThenFunctor then = ThenFunctor();
		};

	template<typename Ma>
		class MonadData : public Monad<Parameterizable<Ma>::template Typecons> {};


	template<typename Ma>
		struct MonadDefault
		{
			private:
				template<typename A>
			using Typecons = typename Parameterizable<Ma>::template Typecons<A>;
			using BaseType = typename Parameterizable<Ma>::BaseType;

			struct ApplyHelper
			{
				template<typename F>
					constexpr auto operator()(const Ma& ta, const F& f) const -> Typecons<decltype(f(std::declval<BaseType>()))>
					{
						return Ma::fmap(f, ta);
					}
			};


			public:
			template<typename Mf>
				constexpr static auto apply(const Mf& tf, const Ma& ta) -> typename Parameterizable<Mf>::template Typecons<decltype(std::declval<typename Parameterizable<Mf>::BaseType>()(std::declval<BaseType>()))>
				{
					return Parameterizable<Ma>::template Typecons<decltype(std::declval<typename Parameterizable<Mf>::BaseType>()(std::declval<BaseType>()))>::join(Mf::fmap(ApplyHelper() >> ta, tf));
				}

			template<typename A>
				constexpr static Ma pure(const A& a)
				{
					return Ma::unit(a);
				}
		};

	struct FtkJoinFunctor
	{
		template<typename Mma>
			constexpr auto operator()(const Mma& mma) const -> typename Parameterizable<Mma>::template Typecons<typename Parameterizable<typename Parameterizable<Mma>::BaseType>::BaseType> 
			{
				return mma | MonadData<Mma>::join;
			}
	};

	constexpr FtkJoinFunctor join = FtkJoinFunctor();

	struct FtkBindFunctor
	{
		template<typename F, typename Ma>
			constexpr auto operator()(const F& f, const Ma& ma) -> decltype(f(std::declval<typename Parameterizable<Ma>::BaseType>()))
			{
				return ma | MonadData<Ma>::bind >> f;
			}
	};

	constexpr FtkBindFunctor bind = FtkBindFunctor();
	
	struct FtkThenFunctor
	{
		template<typename Mb, typename Ma>
			constexpr auto operator()(const Mb& mb, const Ma& ma ) -> Mb
			{
				return ma | MonadData<Ma>::then >> mb;
			}
	};

	constexpr FtkThenFunctor then = FtkThenFunctor();
}
#endif
