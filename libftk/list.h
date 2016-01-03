#ifndef FTK_LIST_HDR
#define FTK_LIST_HDR
#include <vector>
#include <numeric>
#include <algorithm>
#include "monad.h"

namespace ftk
{
	//ListLike describes properties of a type constructor (Li), not data
	//Hence, range is a function that is characteristic of the type constructor
	//This is a means to access range using template argument deduction
	template<template<typename> class Li>
		struct ListLike
		{
			public:
				struct RangeFunctor
				{
					template<typename A>
						constexpr Li<A> operator()(const A& a0, const int& len)
						{
							return Li<A>::range(a0, len);
						}
				};

				constexpr static RangeFunctor range = RangeFunctor();
		};

	//Most other functions are functions on data
	//ListLikeData is a thin, generic wrpaper that concrete implementations should follow
	template<typename Lia>
		struct ListLikeData : public ListLike<Parameterizable<Lia>::template Typecons>
		{
			private:
				using A = typename Parameterizable<Lia>::BaseType;

				template<typename B>
				using Tc = typename Parameterizable<Lia>::template Typecons<B>;

			public:
				constexpr static Lia concat(const Lia& l1, const Lia& l2)
				{
					return l1.concat(l2);
				}

				template<typename F>
				constexpr static A fold(const F& f, const A& a0, const Lia& la)
				{
					return la.fold(f, a0);
				}

				constexpr static int length(const Lia& la)
				{
					return la.length();

				}

				template<typename F>
					static auto map(const F& f, const Lia& la) -> Tc<decltype(f(std::declval<A>()))>
					{
						return la.map(f);
					}
		};

	//The following functors (as in function objects, not Hasekll-type functors) enable the use of lists with template argument deduction
	struct ConcatFunctor
	{
		template<typename Lia>
		constexpr Lia operator()(const Lia& l1, const Lia& l2) const
		{
			return ListLikeData<Lia>::concat(l1, l2);
		}
	};

	constexpr ConcatFunctor concat = ConcatFunctor();

	struct FoldFunctor
	{
		template<typename F, typename A0, typename Lia>
			constexpr A0 operator()(const F& f, const A0& a0, const Lia& la)
		{
			return ListLikeData<Lia>::fold(f, a0, la);
		}
	};

	constexpr FoldFunctor fold = FoldFunctor();

	struct LengthFunctor
	{
		template<typename Lia>
			constexpr int operator()(const Lia& la)
			{
				return ListLikeData<Lia>::length(la);
			}
	};

	constexpr LengthFunctor length = LengthFunctor();
	
	//Any list can be made into a monad if it supports the operations
	//This is using curiously recurring template pattern
	template<typename Lia>
	struct ListLikeDataMonad : public MonadDefault<Lia>
	{
		private:
			//These typedefs are for brevity, and therefore not public
			using A = typename Parameterizable<Lia>::BaseType;

			template<typename B>
				using Tc = typename Parameterizable<Lia>::template Typecons<B>;

			using LLike = ListLikeData<Lia>;

		public:
			constexpr static Lia unit(const A& a)
			{
				return Lia(a);
			}

		template<typename F>
			constexpr static auto fmap(const F& f, const Lia& ta) -> Tc<decltype(f(std::declval<A>()))>
			{
				return LLike::map(f, ta);
			}

			constexpr static auto join(const Tc<Lia>& tta) -> Lia
			{
				return ListLikeData<Tc<Lia> >::fold(concat, Lia(), tta);
			}
	};

	template<typename A> 
	class List : public ListLikeDataMonad<List<A> >
	{
		private:
			std::vector<A> values;

		public:
			List() : values(std::vector<A>()) {}
			List(const A& dt) : values(std::vector<A>(1,dt)) {}
			List(const std::vector<A>& dts) : values(dts) {}
					
			List<A> concat(const List<A>& l2) const
			{
				auto d3 = values;
				d3.insert(d3.end(), l2.values.begin(), l2.values.end());
				return List<A>(d3);
			}

			template<typename F>
				A fold(const F& func, const A& initval) const
				{
					return std::accumulate(values.begin(), values.end(), initval, func);
				}

			template<typename F>
				auto map(const F& f) const -> List<decltype(f(std::declval<A>()))>
				{
					using B = decltype(f(A()));
					std::vector<B> newvals(values.size());
					std::transform(values.begin(), values.end(), newvals.begin(), f);
					return List<B>(newvals);
				}

			int length() const
			{
				return values.size();
			}

			static List<A> range(const A& start, const int& len)
			{
				std::vector<A> invec(len);
				std::iota(invec.begin(), invec.end(), start);
				return List<A>(invec);
			}

	};
	
	template<typename A>
		struct Parameterizable<List<A> >
		{
			template<typename B>
			using Typecons = List<B>;

			using BaseType = A;
		};
}

#endif
