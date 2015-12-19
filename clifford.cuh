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

__host__ __device__ constexpr uint componentsForGrade(const uint& dimensions, const uint& grade)
{
	return choose(dimensions, grade);
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

	template<typename R, typename F, std::size_t components>
__host__ __device__ void transformArray(R (&cfs)[components], const F& f)
{
	for(std::size_t i = 0; i < components; i++)
	{
		cfs[i] = f(i,cfs[i]);
	}
}

template<typename R>
struct Zeroer
{
	constexpr __host__ __device__ R operator()(const uint& i, const R& comp) const
	{
		return 0.;
	}
};

template<typename A>
struct Printer
{
	__host__ A operator()(const uint& i, const A& comp) const
	{
		std::cout << comp << " ";
		return comp;
	}
};

template<typename A, uint len>
struct Adder
{
	private:
		A mthis[len];

	public:
		__host__ __device__ Adder(const A (&mt)[len])
		{
			for(uint i = 0; i < len; i++)
			{
				mthis[i] = mt[i];
			}
		}

		constexpr __host__ __device__ A operator()(const uint& i, const A& comp) const
		{
			return (i >= len) ? 0 : mthis[i] + comp;
		}
};

template<typename R>
struct ScalarMultiplier
{
	private:
		R scalar;

	public:
		__host__ __device__ ScalarMultiplier(const R& sc) : scalar(sc) {}

		constexpr __host__ __device__ R operator()(const uint& i, const R& comp) const
		{
			return comp*scalar;
		}
};

template<uint plus_dim, uint minus_dim = 0, uint zero_dim = 0, typename R = double>
class Multivector;

template<uint grade, uint plus_dim, uint minus_dim = 0, uint zero_dim = 0, typename R = double>
class SingleGradedMultivector
{

	private:
		using MV = Multivector<plus_dim, minus_dim, zero_dim, R>;

	public:
		__host__ __device__ static constexpr uint components()
		{
			return componentsForGrade(plus_dim + minus_dim + zero_dim, grade);
		}

	private:
		R coeffs[components()];

	public:
		__host__ __device__ SingleGradedMultivector(const R (&other)[components()])
		{
			for(uint i = 0; i < components(); i++)
			{
				coeffs[i] = other[i];
			}
		}

		__host__ void print()
		{
			std::cout << "[";
			transformArray(coeffs, Printer<R>());
			std::cout << "]";
		}

		friend class Multivector<plus_dim, minus_dim, zero_dim, R>;
		__host__ __device__ SingleGradedMultivector& operator+=(const SingleGradedMultivector& mo) 
		{
			transformArray(coeffs, Adder<R, components()>(mo.coeffs));	
			return *this;
		}

		__host__ __device__ SingleGradedMultivector operator+(const SingleGradedMultivector& mo)
		{
			SingleGradedMultivector ret = *this;
			ret += mo;
			return ret;
		}

		__host__ __device__ SingleGradedMultivector& operator*=(const R& scalar)
		{
			transformArray(coeffs, ScalarMultiplier<R>(scalar));
			return *this;
		}

		__host__ __device__ SingleGradedMultivector operator*(const R& scalar) const
		{
			SingleGradedMultivector ret = *this;
			ret *= scalar;
			return ret;
		}

		__host__ __device__ MV operator*(const MV& mv) const
		{
			return MV::makeMultivectorFromGrade(*this)*mv;
		}

		template<uint ograde>
		__host__ __device__ MV operator*(const SingleGradedMultivector<ograde, plus_dim, minus_dim, zero_dim, R>& osmv) const
		{
			return MV::makeMultivectorFromGrade(*this)*osmv;
		}
};

template<uint plus_dim, uint minus_dim, uint zero_dim, typename R>
class Multivector
{

	template<uint grade>
	using SMV = SingleGradedMultivector<grade, plus_dim, minus_dim, zero_dim, R>;

	public:
		__host__ __device__ constexpr static uint dimensions()
		{
			return plus_dim + minus_dim + zero_dim;
		}

		__host__ __device__ constexpr static uint components()
		{
			return ipow(2, dimensions());
		}

		__host__ __device__ constexpr static uint gradeComponents(const uint grade)
		{
			return choose(dimensions(), grade);
		}
		R coeffs[components()];

		template<uint grade>
			__host__ __device__ static Multivector makeMultivectorFromGrade(const SMV<grade>& sg)
			{
				Multivector ret;
				uint grade_components_taken = 0;
				for(uint i = 0; i < components(); i++)
				{
					if(popcount(i) == grade)
					{
						ret.coeffs[i] = sg.coeffs[grade_components_taken];
						grade_components_taken++;
					}
					else
					{
						ret.coeffs[i] = 0.;
					}
				}
				return ret;
			}

		template<uint grade>
			__host__ __device__ SMV<grade> compressToGrade() const
			{
				const Multivector other = *this % grade;	
				uint grade_components_taken= 0;
				const uint num_compressed_components = choose(dimensions(), grade);
				R compressed_components[num_compressed_components];
				for(uint i = 0; i < components(); i++)
				{
					if(grade_components_taken >= num_compressed_components)
					{
						break;
					}
					else if(popcount(i) == grade)
					{
						compressed_components[grade_components_taken] = other.coeffs[i];
						grade_components_taken++;
					}
				}
				return SMV<grade>(compressed_components);
			}
	public:
		__host__ __device__ Multivector()
		{
			transformArray(coeffs, Zeroer<R>());
		}

		__host__ void print()
		{
			std::cout << "[";
			transformArray(coeffs, Printer<R>());
			std::cout << "]";
		}

	private:
		struct GradeProjector
		{
			private:
				uint grade;

			public:
				__host__ __device__ GradeProjector(const uint& gr) : grade(gr) {}

				__host__ __device__ R operator()(const uint& i, const R& comp) const
				{
					return (popcount(i) == grade) ? comp : 0.;
				}
		};

	public:
		//__host__ __device__ constexpr Multivector(const Multivector& mv) : coeffs(mv.coeffs) {}

		__host__ __device__ Multivector& operator+=(const Multivector& mo) 
		{
			transformArray(coeffs, Adder<R, components()>(mo.coeffs));	
			return *this;
		}

		__host__ __device__ Multivector operator+(const Multivector& mo)
		{
			Multivector ret = *this;
			ret += mo;
			return ret;
		}

		__host__ __device__ Multivector& operator*=(const R& scalar)
		{
			transformArray(coeffs, ScalarMultiplier<R>(scalar));
			return *this;
		}

		__host__ __device__ Multivector operator*(const R& scalar) const
		{
			Multivector ret = *this;
			ret *= scalar;
			return ret;
		}

		template<uint grade>
		__host__ __device__ Multivector operator*(const SMV<grade>& smv) const
		{
			return *this*(makeMultivectorFromGrade<grade>(smv));
		}

		__host__ __device__ Multivector& operator%=(const uint& grade)
		{
			transformArray(coeffs, GradeProjector(grade));
			return *this;
		}

		__host__ __device__ Multivector operator%(const uint& grade) const
		{
			Multivector ret = *this;
			ret %= grade;
			return ret;
		}

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

};
