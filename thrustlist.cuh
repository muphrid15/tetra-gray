#ifndef THRUST_LIST_HDR
#define THRUST_LIST_HDR
#include "libftk/list.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

namespace cudaftk
{
	//Thrust vector types are just default templates; alloc is required
	//see libftk/list.h for model example that this class is imitating
	template<template<typename, typename> class V, template<typename> class Alloc, typename A>
		class ThrustList
		{
			private:
				template<typename B>
					using Vector = V<B, Alloc<B> >;
				Vector<A> values;

			public:
				ThrustList() : values(Vector<A>()) {}
				ThrustList(const A& a) : values(Vector<A>(1, a)) {}

				template<template<typename, typename> class W, template<typename> class Walloc>
					ThrustList(const W<A, Walloc<A> >& wa) : values(wa) {}

				Vector<A> unpack() const
				{
					return values;
				}

				ThrustList concat(const ThrustList& l2) const
				{
					Vector<A> d3 = values;
					d3.insert(d3.end(), l2.values.begin(), l2.values.end());
					return ThrustList(d3);
				}

				template<typename F>
					A fold(const F& func, const A& init) const
					{
						return thrust::reduce(values.begin(), values.end(), init, func);
					}

				template<typename F>
					auto map(const F& f) const -> ThrustList<V, Alloc, decltype(f(std::declval<A>()))>
					{
						using B = decltype(f(std::declval<A>()));
						Vector<B> newvals(values.size());
						thrust::transform(values.begin(), values.end(), newvals.begin(), f);
						return ThrustList<V, Alloc, B>(newvals);
					}

				template<typename F>
					static auto fmap(const F& f, const ThrustList& la) -> ThrustList<V, Alloc, decltype(f(std::declval<A>()))>
					{
						return la.map(f);
					}

				int length() const
				{
					return values.size();
				}

				static ThrustList range(const A& start, const int& len)
				{
					Vector<A> invec(len);
					thrust::sequence(invec.begin(), invec.end(), start);
					return ThrustList(invec);
				}
		};

	template<typename A>
		using CPUList = ThrustList<thrust::template host_vector, std::template allocator, A>;
	template<typename A>
		using GPUList = ThrustList<thrust::template device_vector, thrust::template device_malloc_allocator, A>;


	struct UnpackFunctor
	{
		template<template<typename, typename> class V, template<typename> class Alloc, typename A>
		V<Alloc<A>, A> operator()(const ThrustList<V, Alloc, A>& ta) const
		{
			return ta.unpack();
		}
	};

	constexpr UnpackFunctor unpack = UnpackFunctor();
}

namespace ftk
{
	template<template<typename, typename> class V, template<typename> class Alloc, typename A>
		struct Parameterizable<cudaftk::ThrustList<V, Alloc, A> >
		{
			template<typename B>
				using Typecons = cudaftk::ThrustList<V, Alloc, B>;

			using BaseType = A;
		};
}



#endif

