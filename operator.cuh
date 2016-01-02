#ifndef CUDA_OPERATOR_HDR
#define CUDA_OPERATOR_HDR

namespace cudaftk
{
	template<typename F, typename A>
		class CurryFunctor
		{
			private:
				F func;
				A arg1;

			public:
				__host__ __device__ CurryFunctor(F f, A a) : func(f), arg1(a) {}

				template<typename... Args>
					__host__ __device__ constexpr auto operator()(Args... args) const -> decltype(func(arg1, args...)) 
					{
						return func(arg1, args...);
					}
		};

	template<typename F, typename A>
		class ReverseCurryFunctor
		{
			private:
				F func;
				A arg1;

			public:
				__host__ __device__ ReverseCurryFunctor(F f, A a) : func(f), arg1(a) {}

				template<typename... Args>
					__host__ __device__ constexpr auto operator()(Args... args) const -> decltype(func(args..., arg1)) 
					{
						return func(args..., arg1);
					}
		};

	template<typename F, typename G>
		struct ComposeFunctor
		{
			F first;
			G second;
			__host__ __device__ ComposeFunctor(F f, G g) : first(f), second(g) {}

			template<typename... Args>
				__host__ __device__ constexpr auto operator()(Args... a) const -> decltype(second(first(a...)))
				{
					return second(first(a...));
				}
		};
	/*
	template<typename F, typename A>
		__host__ __device__ auto operator|(A a, F f) -> decltype(f(a))
		{
			return f(a);
		}
		*/

	//curry left: partially applies a function on an argument, placing that argument on the left of the argument list
	template<typename F, typename A>
		__host__ __device__ constexpr auto operator%(F f, A a) -> CurryFunctor<F,A>
		{
			return CurryFunctor<F, A>(f, a);
		}

	//curry right: partially applies a function on an argument, placing that argument on the right of the argument list
	template<typename F, typename A>
		__host__ __device__ constexpr auto operator/(F f, A a) -> ReverseCurryFunctor<F,A>
		{
			return ReverseCurryFunctor<F, A>(f, a);
		}

	//compose: compose two functions
	template<typename F, typename G>
		__host__ __device__ constexpr auto operator*(F f, G g) -> ComposeFunctor<F,G>
		{
			return ComposeFunctor<F,G>(f, g);
		}

}

#endif
