#include <cstddef>
#include <iostream>

__host__ __device__ constexpr uint ipow(const uint& base, const uint& radix)
{
	return (radix == 0) ? 1 : base*ipow(base, radix-1);
}

__host__ __device__ constexpr uint choose(const uint& top, const uint& bot)
{
	return (bot > top || bot == 0 || top == 0) ? 1 : top*choose(top-1, bot-1)/bot;
}

/*
__host__ __device__ constexpr uint recsum(const uint& val)
{
	return (val == 0) : 0 ? val+recsum(val-1);
}
*/

__host__ __device__ int popcount(const uint& val)
{
#ifdef CUDA_ARCH
	return __popc(val);
#else
	return __builtin_popcount(val);
#endif
}

__host__ __device__ int contractionSign(const uint& plus_dim, const uint& minus_dim, const uint& zero_dim, const uint& contracted_mask)
{
	const uint zero_mask = (ipow(2,zero_dim)-1)*ipow(2,plus_dim+minus_dim);
	const uint minus_mask = (ipow(2,minus_dim)-1)*ipow(2,plus_dim);
	return popcount(zero_mask & contracted_mask) > 0 ? 0 : popcount(minus_mask & contracted_mask) % 2 == 1 ? -1 : 1;
}

__host__ __device__ int permutationSign(const uint& left_mask, const uint& right_mask, const uint& max_bits)
{
	int result = 0;
	for(uint test_mask_log = 1; test_mask_log < max_bits; test_mask_log++)
	{
		uint count_mask = ipow(2,test_mask_log) - 1;
		result += (popcount(left_mask & ipow(2,test_mask_log)) == 0) ? 0 : popcount(count_mask & right_mask);
	}
	return (result % 2 == 0) ? 1 : -1;
}

template<typename Real, typename F, std::size_t components>
__host__ __device__ void transformArray(Real (&cfs)[components], const F& f)
{
	for(std::size_t i = 0; i < components; i++)
	{
		cfs[i] = f(i,cfs[i]);
	}
}


template<uint plus_dim, uint minus_dim = 0, uint zero_dim = 0, typename Real = double>
class Multivector
{
	public:
		__host__ __device__ static constexpr uint dimensions()
		{
			return plus_dim + minus_dim + zero_dim;
		}

		__host__ __device__ static constexpr uint components()
		{
			return ipow(2, dimensions());
		}
		Real coeffs[components()];
		
		template<uint grade, uint len = choose(components(), grade)>
		__host__ __device__ static Multivector makeMultivectorFromGrade(const Real (&gradecoeffs)[len])
		{
			Multivector ret;
			uint grade_components_taken = 0;
			for(uint i = 0; i < components(); i++)
			{
				if(popcount(i) == grade)
				{
					ret.coeffs[i] = gradecoeffs[grade_components_taken];
					grade_components_taken++;
				}
				else
				{
					ret.coeffs[i] = 0.;
				}
			}
			return ret;
		}

	private:
		struct Zeroer
		{
			constexpr __host__ __device__ Real operator()(const uint& i, const Real& comp) const
			{
				return 0.;
			}
		};

		struct Printer
		{
			__host__ Real operator()(const uint& i, const Real& comp) const
			{
				std::cout << comp << " ";
				return comp;
			}
		};
		
	public:
		__host__ __device__ Multivector()
		{
			transformArray(coeffs, Zeroer());
		}

		__host__ void print()
		{
			std::cout << "[";
			transformArray(coeffs, Printer());
			std::cout << "]";
		}
		/*
	
	private:
		struct Adder
		{
			private:
				Multivector mthis;

			public:
				__host__ __device__ Adder(const Multivector& mt) : mthis(mt) {}

				constexpr __host__ __device__ Real operator()(const uint& i, const Real& comp) const
				{
					return mthis.coeffs[i] + comp;
				}
		};

		struct ScalarMultiplier
		{
			private:
				Real scalar;

			public:
				__host__ __device__ ScalarMultiplier(const Real& sc) : scalar(sc) {}

				constexpr __host__ __device__ Real operator()(const uint& i, const Real& comp) const
				{
					return comp*scalar;
				}
		};

		struct GradeProjector
		{
			private:
				uint grade;

			public:
				__host__ __device__ GradeProjector(const uint& gr) : grade(gr) {}

				__host__ __device__ Real operator()(const uint& i, const Real& comp) const
				{
					return (popcount(i) == grade) ? comp : 0.;
				}
		};

	public:
		__host__ __device__ constexpr Multivector(const Multivector& mv) : coeffs(mv.coeffs) {}

		__host__ __device__ Multivector& operator+=(const Multivector& mo) 
		{
		}

		__host__ __device__ Multivector& operator*=(const Real& scalar)
		{
		}
		*/

		__host__ __device__ Multivector operator*(const Multivector& mo) const
		{
			auto res = Multivector();			

			for(uint i = 0; i < components(); i++)
			{
				for(uint j = 0; j < components(); j++)
				{
					const uint result_mask = i^j;
					res.coeffs[result_mask] += coeffs[i]*mo.coeffs[j]*contractionSign(plus_dim, minus_dim, zero_dim, i&j)*permutationSign(i,j, plus_dim+minus_dim+zero_dim);
				}
			}
			return res;
		}

		//% here is the grade projection operator
		/*
		__host__ __device__ Multivector% operator%=(const uint& grade)
		{
		}
		*/
};
