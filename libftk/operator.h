#ifndef FTK_OPERATORS
#define FTK_OPERATORS

namespace ftk
{
	template<typename F, typename A>
		class CurryFunctor
		{
			private:
				F func;
				A arg1;

			public:
				constexpr CurryFunctor(F f, A a) : func(f), arg1(a) {}

				template<typename... Args>
					constexpr auto operator()(Args... args) const -> decltype(func(arg1, args...)) 
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
				constexpr ReverseCurryFunctor(F f, A a) : func(f), arg1(a) {}

				template<typename... Args>
					constexpr auto operator()(Args... args) const -> decltype(func(args..., arg1)) 
					{
						return func(args..., arg1);
					}
		};

	template<typename F, typename G>
		struct ComposeFunctor
		{
			F first;
			G second;
			constexpr ComposeFunctor(F f, G g) : first(f), second(g) {}

			template<typename A>
				constexpr auto operator()(A a) const -> decltype(second(first(a)))
				{
					return second(first(a));
				}
		};
	template<typename F, typename A>
		constexpr auto operator|(A a, F f) -> decltype(f(a))
		{
			return f(a);
		}

	//curry left: partially applies a function on an argument, placing that argument on the left of the argument list
	template<typename F, typename A>
		constexpr auto operator>>(F f, A a) -> CurryFunctor<F,A>
		{
			return CurryFunctor<F, A>(f, a);
		}

	//curry right: partially applies a function on an argument, placing that argument on the right of the argument list
	template<typename F, typename A>
		constexpr auto operator<<(F f, A a) -> ReverseCurryFunctor<F,A>
		{
			return ReverseCurryFunctor<F, A>(f, a);
		}

	//compose: compose two functions
	template<typename F, typename G>
		constexpr auto operator&(F f, G g) -> ComposeFunctor<F,G>
		{
			return ComposeFunctor<F,G>(f, g);
		}

}

#endif
