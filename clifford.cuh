#ifndef MV_HDR
#define MV_HDR
#include <cstddef>
#include <iostream>

namespace mv
{
	namespace impl
	{
		__host__ __device__ constexpr uint ipow2(const uint& radix)
		{
			return 1u << radix;
		}
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

		__host__ __device__ inline int popcount(const uint val)
		{
#ifdef __CUDA_ARCH__
			return __popc(val);
#else
			return __builtin_popcount(val);
#endif
		}

		__host__ __device__ inline int contractionSign(const uint& plus_dim, const uint& minus_dim, const uint& zero_dim, const uint& contracted_mask)
		{
			const uint zero_mask = (ipow2(zero_dim)-1)*ipow2(plus_dim+minus_dim);
			const uint minus_mask = (ipow2(minus_dim)-1)*ipow2(plus_dim);
			return popcount(zero_mask & contracted_mask) > 0 ? 0 : popcount(minus_mask & contracted_mask) % 2 == 1 ? -1 : 1;
		}

		__host__ __device__ inline int permutationSign(const uint& left_mask, const uint& right_mask, const uint& max_bits)
		{
			int result = 0;
			for(uint test_mask_log = 1; test_mask_log < max_bits; test_mask_log++)
			{
				uint count_mask = ipow2(test_mask_log) - 1;
				result += (popcount(left_mask & ipow2(test_mask_log)) == 0) ? 0 : popcount(count_mask & right_mask);
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

			struct Reverser
			{
				template<typename A>
				__host__ __device__ A operator()(const uint& i, const A& comp) const
				{
					const int pc = popcount(i);
					const int expo = (pc*pc-pc)/2;
					return (expo % 2 == 0) ? comp : -comp;
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
	}

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
						return impl::componentsForGrade(plus_dim + minus_dim + zero_dim, grade);
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

					__host__ __device__ SingleGradedMultivector()
					{
						impl::transformArray(coeffs, impl::Zeroer<R>());
					}


					__host__ void print()
					{
						std::cout << "[";
						impl::transformArray(coeffs, impl::Printer<R>());
						std::cout << "]";
					}


					friend class Multivector<plus_dim, minus_dim, zero_dim, R>;
					__host__ __device__ SingleGradedMultivector(const MV& multi)
					{
						*this = SingleGradedMultivector();
						int last = -1;
						for(int j = 0; j < components(); j++)
						{
							bool set = false;
							for(int i = 0; i < MV::components(); i++)
							{
								if(impl::popcount(i) == int(grade) && i > last && !set)
								{
									this->coeffs[j] = multi.coeffs[i];
									last = i;
									set = true;
								}
							}
						}
					}
					
					__host__ __device__ R extractComponent(const uint& idx) const
					{
						return (idx < components()) ? coeffs[idx] : 0;
					}
					__host__ __device__  MV promote() const
					{
						return MV::makeMultivectorFromGrade(*this);
					}

					__host__ __device__ SingleGradedMultivector& operator+=(const SingleGradedMultivector& mo) 
					{
						impl::transformArray(coeffs, impl::Adder<R, components()>(mo.coeffs));	
						return *this;
					}

					__host__ __device__ SingleGradedMultivector operator+(const SingleGradedMultivector& mo) const
					{
						SingleGradedMultivector ret = *this;
						ret += mo;
						return ret;
					}

					__host__ __device__ SingleGradedMultivector& operator*=(const R& scalar)
					{
						impl::transformArray(coeffs, impl::ScalarMultiplier<R>(scalar));
						return *this;
					}

					__host__ __device__ SingleGradedMultivector operator*(const R& scalar) const
					{
						SingleGradedMultivector ret = *this;
						ret *= scalar;
						return ret;
					}

					__host__ __device__ SingleGradedMultivector operator-() const
					{
						auto ret = *this;
						ret *= -1;
						return ret;
					}

					__host__ __device__ SingleGradedMultivector& operator-=(const SingleGradedMultivector& sgother)
					{
						*this += -sgother;
						return *this;
					}

					__host__ __device__ SingleGradedMultivector operator-(const SingleGradedMultivector& rhs) const
					{
						auto ret = *this;
						ret -= rhs;
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

					__host__ __device__ MV normalize() const
					{
						return MV::makeMultivectorFromGrade(*this).normalize();
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
					return impl::ipow2( dimensions());
				}

				__host__ __device__ constexpr static uint gradeComponents(const uint grade)
				{
					return impl::choose(dimensions(), grade);
				}
				R coeffs[components()];

				template<uint grade>
					__host__ __device__ static Multivector makeMultivectorFromGrade(const SMV<grade>& sg)
					{
						Multivector ret;
						uint grade_components_taken = 0;
						for(uint i = 0; i < components(); i++)
						{
							if(impl::popcount(i) == grade)
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


				__host__ __device__ Multivector()
				{
					impl::transformArray(coeffs, impl::Zeroer<R>());
				}
				
				__host__ __device__ static Multivector makePseudoscalar(const R& val)
				{
					auto ret = Multivector();
					ret.coeffs[components()-1u] = val;
					return ret;
				}

				__host__ __device__ Multivector(const R& scalar)
				{
					impl::transformArray(coeffs, impl::Zeroer<R>());
					coeffs[0] = scalar;
				}

				__host__ void print()
				{
					std::cout << "[";
					impl::transformArray(coeffs, impl::Printer<R>());
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
							return (impl::popcount(i) == grade) ? comp : 0.;
						}
				};

				public:
				//__host__ __device__ constexpr Multivector(const Multivector& mv) : coeffs(mv.coeffs) {}

				__host__ __device__ Multivector& operator+=(const Multivector& mo) 
				{
					impl::transformArray(coeffs, impl::Adder<R, components()>(mo.coeffs));	
					return *this;
				}

				__host__ __device__ Multivector operator+(const Multivector& mo) const
				{
					Multivector ret = *this;
					ret += mo;
					return ret;
				}

				__host__ __device__ Multivector& operator*=(const R& scalar)
				{
					impl::transformArray(coeffs, impl::ScalarMultiplier<R>(scalar));
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
					impl::transformArray(coeffs, GradeProjector(grade));
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
								res.coeffs[result_mask] += coeffs[i]*mo.coeffs[j]*impl::contractionSign(plus_dim, minus_dim, zero_dim, i&j)*impl::permutationSign(i,j, plus_dim+minus_dim+zero_dim);
						}
					}
					return res;
				}

				__host__ __device__ Multivector& operator/=(const R& scalar)
				{
					*this *= (1./scalar);
					return *this;
				}

				__host__ __device__ Multivector operator/(const R& scalar) const
				{
					Multivector ret = *this;
					ret /= scalar;
					return ret;
				}

				__host__ __device__ Multivector operator-() const
				{
					Multivector ret = *this;
					ret *= -1;
					return ret;
				}

				__host__ __device__ Multivector& operator-=(const Multivector& rhs)
				{
					*this += (-rhs);
					return *this;
				}

				__host__ __device__ Multivector operator-(const Multivector& rhs) const
				{
					Multivector ret = *this;
					ret -= rhs;
					return ret;
				}


				__host__ __device__ Multivector reverse() const
				{
					Multivector copy = *this;
					impl::transformArray(copy.coeffs, impl::Reverser());
					return copy;
				}

				__host__ __device__ R norm2() const
				{
					return (*this*(this->reverse())).coeffs[0];
				}
				__host__ __device__ R absnorm2() const
				{
					return abs(this->norm2());
				}

				__host__ __device__ R norm() const
				{
					return sqrt(this->absnorm2());
				}

				__host__ __device__ R scalarPart() const
				{
					return coeffs[0];
				}

				__host__ __device__ Multivector normalize() const
				{
					return (*this/this->norm());
				}

				__host__ __device__ Multivector inverse() const
				{
					const auto rev = this->reverse();
					return rev/(this->norm2());
				}
			};

		template<uint plus_dim, uint minus_dim, uint zero_dim, typename R>
		__host__ __device__ Multivector<plus_dim, minus_dim, zero_dim, R> operator*(const R& scalar, const Multivector<plus_dim, minus_dim, zero_dim, R>& multi)
		{
			return multi*scalar;
		}

		template<uint grade, uint plus_dim, uint minus_dim, uint zero_dim, typename R>
		__host__ __device__ SingleGradedMultivector<grade, plus_dim, minus_dim, zero_dim, R> operator*(const R& scalar, const SingleGradedMultivector<grade, plus_dim, minus_dim, zero_dim, R>& multi)
		{
			return multi*scalar;
		}
}
#endif
