#ifndef FTK_PROCEDURE_HDR
#define FTK_PROCEDURE_HDR
#include <functional>
#include <utility>
#include "monad.h"

namespace ftk
{

	struct Empty {};

	template<typename A>
	struct Procedure;

	struct MakeProcedureFunctor
	{
		template<typename F>
		auto operator()(const F& f) const -> Procedure<decltype(f())>
		{
			using A = decltype(f());
			return Procedure<A>(std::function<A()>(f));
		}
	};

	constexpr MakeProcedureFunctor makeProcedure = MakeProcedureFunctor();

	template<typename Pra>
		struct ProcedureMonad : MonadDefault<Pra>
		{
			private:
				using A = typename Parameterizable<Pra>::BaseType;

				template<typename B>
				using Tc = typename Parameterizable<Pra>::template Typecons<B>;

			public:
				static Tc<A> unit(const A& a)
				{
					return Tc<A>(a);
				}

			template<typename F>
				static auto fmap(const F& f, const Tc<A>& ta) -> Tc<decltype(f(std::declval<A>()))>
				{
					return [ta, f]() { return f(ta.run()); } | makeProcedure;
				}

				static auto join(const Tc<Pra>& tta) -> Pra
				{
					return [tta]() { return tta.run().run(); } | makeProcedure;
				}

		};
	
	template<typename A>
		struct Procedure : ProcedureMonad<Procedure<A> >
		{
			public:
				using ProcedureType = std::function<A()>;
			private:
				ProcedureType proc;
			public:
				Procedure(const ProcedureType& pc) : proc(pc) {}
				Procedure(const A& a)
				{
					proc = [a]() { return a; };
				}

				A run() const
				{
					return proc();
				}
		};

	template<typename A>
		struct Parameterizable<Procedure<A> >
		{
			template<typename B>
			using Typecons = Procedure<B>;

			using BaseType = A;
		};
		
}
#endif
