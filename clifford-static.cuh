#ifndef CLIFFORD_STATIC_HDR
#define CLIFFORD_STATIC_HDR
#include <iostream>

namespace multi
{
	namespace impl
	{
		//We're using curiously recurring template pattern here, to conveniently generate operators for Vector, Bivector, etc.
		//Don't yet have a solution that naturally defines compound assignment operators
		template<typename Derived, typename R, uint length>
			class NumericalArray
			{
				public:
					R values[length];
				protected:
					template<typename C, typename F>
						__host__ __device__ static C transformArray(const C& c, const F& f)
						{
							C copy;
							for(uint i = 0; i < length; i++)
							{
								copy.values[i] = f(i,c.values[i]);
							}
							return copy;
						}

				public:
					__host__ __device__ NumericalArray(const R(&other)[length])
					{
						for(uint i = 0; i < length; i++)
						{
							values[i] = other[i];
						}
					}
					__host__ __device__ NumericalArray()
					{
						for(uint i = 0; i < length; i++)
						{
							values[i] = R(0.);
						}
					}

					__host__ void print() const
					{
						std::cout << "[";
						transformArray(*this, [] (const uint& idx, const R& val) -> R
								{
								std::cout << val << " ";
								return val;
								});
						std::cout << "]" << std::endl;
					}

					__host__ __device__ Derived operator+(const Derived& other) const
					{
						return Derived(transformArray(*this, [other] __host__ __device__ (const uint& idx, const R& val)
								{
								return val+other.values[idx];
								}));
					}

					__host__ __device__ Derived operator*(const R& scalar) const
					{
						return Derived(transformArray(*this, [scalar] __host__ __device__ (const uint& idx, const R& val)
								{
								return val*scalar;
								}));
					}

					/*
					__host__ __device__ Derived operator*(const R& scalar) const
					{
						auto copy = *this;
						copy *= scalar;
						return copy;
					}
					*/

					__host__ __device__ Derived operator-() const
					{
						auto copy = *this;
						return copy*R(-1.);
					}
					
					/*
					__host__ __device__ Derived& operator-=(const Derived& other)
					{
						*this = *this + (-other);
						return *this;
					}
					*/

					__host__ __device__ Derived operator-(const Derived& other) const
					{
						auto copy = *this;
						return copy + (-other);
					}

					/*
					__host__ __device__ Derived& operator/=(const R& scalar)
					{
						*this *= R(1./scalar);
						return *this;
					}
					*/

					__host__ __device__ Derived operator/(const R& scalar) const
					{
						auto copy = *this;
						return copy * R(1./scalar);
					}

					__host__ __device__ R operator[](const int idx) const
					{
						return values[idx];
					}
			};
	}

	template<typename R>
	class Vector : public impl::NumericalArray<Vector<R>, R, 4u>
	{
		private:
			using BaseType = impl::NumericalArray<Vector<R>, R, 4u>;
		public:
			__host__ __device__ Vector() : BaseType() {}
			__host__ __device__ Vector(const R (&comps)[4]) : BaseType(comps) {}
			__host__ __device__ Vector(const BaseType& bt) : BaseType(bt) {}
	};

	template<typename R>
	class Bivector : public impl::NumericalArray<Bivector<R>, R, 6u>
	{
		private:
			using BaseType = impl::NumericalArray<Bivector<R>, R, 6u>;
		public:
			__host__ __device__ Bivector() : BaseType() {}
			__host__ __device__ Bivector(const R (&comps)[6]) : BaseType(comps) {}
			__host__ __device__ Bivector(const BaseType& bt) : BaseType(bt) {}
	};
	
	template<typename R>
	class Trivector : public impl::NumericalArray<Trivector<R>, R, 4u>
	{
		private:
			using BaseType = impl::NumericalArray<Trivector<R>, R, 4u>;
		public:
			__host__ __device__ Trivector() : BaseType() {}
			__host__ __device__ Trivector(const R (&comps)[4]) : BaseType(comps) {}
			__host__ __device__ Trivector(const BaseType& bt) : BaseType(bt) {}
	};
		
	//Versor here meaning "even-graded multivector"
	template<typename R>
	class Versor : public impl::NumericalArray<Versor<R>, R, 8u>
	{
		private:
			using BaseType = impl::NumericalArray<Versor<R>, R, 8u>;
		public:
			__host__ __device__ Versor() : BaseType() {}
			__host__ __device__ Versor(const R (&comps)[4]) : BaseType(comps) {}
			__host__ __device__ Versor(const BaseType& bt) : BaseType(bt) {}
			__host__ __device__ Versor(const R& scalar) : BaseType()
			{
				BaseType::values[0] = scalar;
			}

			__host__ __device__ Versor(const Bivector<R>& bivec) : BaseType()
			{
				BaseType::values[1] = bivec[0];
				BaseType::values[2] = bivec[1];
				BaseType::values[3] = bivec[2];
				BaseType::values[4] = bivec[3];
				BaseType::values[5] = bivec[4];
				BaseType::values[6] = bivec[5];
			}
			__host__ __device__ static Versor makePseudoscalar(const R& pseudo)
			{
				Versor vers = Versor();
				vers.values[7] = pseudo;
				return vers;
			}
				
			__host__ __device__ Bivector<R> bivectorPart() const
			{
				//const R comps[6] = { values[1], values[2], values[3], values[4], values[5], values[6] };
				R ret[6];
				ret[0] = this->values[1];
				ret[1] = this->values[2];
				ret[2] = this->values[3];
				ret[3] = this->values[4];
				ret[4] = this->values[5];
				ret[5] = this->values[6];
				return Bivector<R>(ret);
			}

			/*
			__host__ __device__ Versor inverse() const
			{
				Versor inv = *this;
				inv.values[1] = -values[1];
				inv.values[2] = -values[2];
				inv.values[3] = -values[3];
				inv.values[4] = -values[4];
				inv.values[5] = -values[5];
				inv.values[6] = -values[6];
				return inv/(values[0]*values[0] - values[7]*values[7] - (bivectorPart()|bivectorPart());
			}
			*/
	};

	/*
	template<typename R>
	class OddMultivector : impl::NumericalArray<OddMultivector<R>, R , 8u>
	{
	};
	*/
	/*
		Product operatios from clifford algebra
		Some comments here may use notation like "0123 13 = -02"
		This is shorthand for the product e_0 e_1 e_2 e_3 e_1 e_3 = -e_0 e_2
		(where e_i is the ith basis vector)
		(product is clifford, or geometric, product)
		The products themselves define the reference ordering for components
		However, as a general rule, components are ordered in "bit" ordering
		Example: bivectors have 6 components here
		e_0 e_1 corresponds to bit sequence 0011 (base 2) = 3 (base 10)
		e_2 e_3 corresponds to bit sequence 1100 (base 2) = 12 (base 10)
		Components are ordered by bit mask representation
		Canonical order of basis vectors is ascending always, outside of this.
		(that is, e_0 e_1 is correct order, e_1 e_0 is incorrect order)
	   */

	//Norms
	template<typename R>
	__host__ __device__ R operator|(const Vector<R>& v1, const Vector<R>& v2)
	{
		return -v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] + v1[3]*v2[3];
	}

	template<typename R>
	__host__ __device__ R operator|(const Bivector<R>& b1, const Bivector<R>& b2)
	{
		return b1[0]*b2[0] + b1[1]*b2[1] - b1[2]*b2[2] + b1[3]*b2[3] - b1[4]*b2[4] - b1[5]*b2[5];
	}

	template<typename R>
	__host__ __device__ R operator|(const Versor<R>& e1, const Versor<R>& e2)
	{
		return e1[0]*e2[0] - e1[7]*e2[7] + (e1.bivectorPart()|e2.bivectorPart());
	}
	
	//Other contractions
	template<typename R>
	__host__ __device__ Vector<R> operator|(const Bivector<R>& b1, const Vector<R>& v2)
	{
		R ret[4];
		ret[0] = b1[0]*v2[1] + b1[1]*v2[2] + b1[3]*v2[3]; //0 = 01 1 + 02 2 + 03 3
		ret[1] = b1[0]*v2[0] + b1[2]*v2[2] + b1[4]*v2[3]; //1 = 01 0 + 12 2 + 13 3
		ret[2] = b1[1]*v2[0] - b1[2]*v2[1] + b1[5]*v2[3]; //2 = 02 0 - 12 1 + 23 3
		ret[3] = b1[3]*v2[0] - b1[4]*v2[1] - b1[5]*v2[2]; //3 = 03 0 - 13 1 - 23 2
		return Vector<R>(ret);
	}
	
	template<typename R>
	__host__ __device__ Vector<R> operator|(const Vector<R>& v1, const Bivector<R>& b2)
	{
		return -b2|v1;
	}
	
	template<typename R>
	__host__ __device__ Vector<R> operator|(const Trivector<R>& t1, const Bivector<R>& b2)
	{
		return -(~((~t1)^b2));
	}
	
	template<typename R>
	__host__ __device__ Vector<R> operator|(const Bivector<R>& b1, const Trivector<R>& t2)
	{
		return t2|b1;
	}

	//Wedges
	template<typename R>
	__host__ __device__ Bivector<R> operator^(const Vector<R>& v1, const Vector<R>& v2)
	{
		R ret[6];
		ret[0] = v1[0]*v2[1] - v1[1]*v2[0]; //bit 3
		ret[1] = v1[0]*v2[2] - v1[2]*v2[0]; //bit 5
		ret[2] = v1[1]*v2[2] - v1[2]*v2[1]; //bit 6
		ret[3] = v1[0]*v2[3] - v1[3]*v2[0]; //bit 9
		ret[4] = v1[1]*v2[3] - v1[3]*v2[1]; //bit 10
		ret[5] = v1[2]*v2[3] - v1[3]*v2[2]; //bit 12
		return Bivector<R>(ret);
	}

	template<typename R>
	__host__ __device__ Trivector<R> operator^(const Bivector<R>& b1, const Vector<R>& v2)
	{
		R ret[4];
		ret[0] = b1[0]*v2[2] - b1[1]*v2[1] + b1[2]*v2[0]; //012 = 01 2 - 02 1 + 12 0 
		ret[1] = b1[0]*v2[3] - b1[3]*v2[1] + b1[4]*v2[0]; //013 = 01 3 - 03 1 + 13 0
		ret[2] = b1[1]*v2[3] - b1[3]*v2[2] + b1[5]*v2[0]; //023 = 02 3 - 03 2 + 23 0
		ret[3] = b1[2]*v2[3] - b1[4]*v2[2] + b1[5]*v2[1]; //123 = 12 3 - 13 2 + 23 1
		return Trivector<R>(ret);
	}

	template<typename R>
	__host__ __device__ Trivector<R> operator^(const Vector<R>& v1, const Bivector<R>& b2)
	{
		return b2^v1;
	}

	//Duals (contractions with pseudoscalars on the left)
	template<typename R>
	__host__ __device__ Bivector<R> operator~(const Bivector<R>& b1)
	{
		R ret[6];
		ret[0] = -b1[5]; //0123 23 = -01
		ret[1] = b1[4]; //0123 13 = +02
		ret[2] = b1[3]; //0123 03 = +12
		ret[3] = -b1[2];  //0123 12 = -03
		ret[4] = -b1[1]; //0123 02 = -13
		ret[5] = b1[0]; //0123 01 = +23
		return Bivector<R>(ret);
	}

	template<typename R>
	__host__ __device__ Trivector<R> operator~(const Vector<R>& v1)
	{
		R ret[4];
		ret[0] = v1[3]; //012, bit 7: 0123 3 = 012
		ret[1] = -v1[2]; //013, bit 11: 0123 2 = -013
		ret[2] = v1[1]; //023, bit 13: 0123 1 = 023
		ret[3] = v1[0]; //123, bit 14: 0123 0 = 123
		return Trivector<R>(ret);
	}

	template<typename R>
	__host__ __device__ Vector<R> operator~(const Trivector<R>& t1)
	{
		R ret[4];
		ret[0] = -t1[3]; //0123123 = -0
		ret[1] = -t1[2]; //0123023 = -1
		ret[2] = t1[1];  //0123013 = +2
		ret[3] = -t1[0]; //0123012 = -3
		return Vector<R>(ret);
	}
	
	//general products
	template<typename R>
	__host__ __device__ Versor<R> operator*(const Bivector<R>& e1, const Bivector<R>& e2)
	{
		R ret[8];
		ret[0] = (e1|e2);
		ret[7] = e1[0]*e2[5] - e1[1]*e2[4] + e1[2]*e2[3] + e1[3]*e2[2] - e1[4]*e2[1] + e1[5]*e2[0];
		ret[1] = -e1[1]*e2[2] - e1[3]*e2[4] + e1[2]*e2[1] + e1[4]*e2[3]; //01
		ret[2] = e1[0]*e2[2] - e1[3]*e2[5] - e1[2]*e2[0] + e1[5]*e2[3];   //02
		ret[3] = e1[0]*e2[1] - e1[4]*e2[5] - e1[1]*e2[0] + e1[5]*e2[4];//12
		ret[4] = e1[0]*e2[4] + e1[1]*e2[5] - e1[4]*e2[0] - e1[5]*e2[1];//03
		ret[5] = e1[0]*e2[3] + e1[2]*e2[5] - e1[3]*e2[0] - e1[5]*e2[2];//13
		ret[6] = e1[1]*e2[3] - e1[2]*e2[4] - e1[3]*e2[1] + e1[4]*e2[2]; //23

		return Versor<R>(ret);
	}

	template<typename R>
	__host__ __device__ Versor<R> operator*(const Versor<R>& e1, const Versor<R>& e2)
	{
		Versor<R> ret;
		ret.values[0] = e1[0]*e2[0] - e1[7]*e2[7];
		ret.values[7] = e1[0]*e2[7] + e1[7]*e2[0];
		const auto b1 = e1.bivectorPart();
		const auto b2 = e2.bivectorPart();
		return ret + b1*b2 + Versor<R>(b1*e2[0] + b2*e1[0] + (~b1)*e2[7] + (~b2)*e1[7]);
	}

	//Decompose a versor as scalar + bivector + pseudoscalar
	//Resulting product of versor*vector*(versor inverse) has 9 terms
	//Two terms are automatically zero (scalar vector pseudoscalar, and pseudo vector scalar)
	//Two pairs of terms can be condensed, as the pairs are actually identical (see "partc" and "partf")
	template<typename R>
	__host__ __device__ Vector<R> bilinearMultiply(const Versor<R>& e1, const Vector<R>& v2)
	{
		const auto bivec = e1.bivectorPart();
		const auto scalar = e1[0];
		const auto pseudo = e1[7];
		const auto bdotv = (bivec|v2);
		const auto bwedgev = bivec^v2;
		const auto parta = v2*scalar*scalar;
		const auto partb = v2*pseudo*pseudo;
		const auto partc = bdotv*R(2.)*scalar;
		const auto partd = (bdotv|bivec);
		const auto parte = bwedgev|bivec;
		const auto partf = ~bwedgev*pseudo*R(2.);
		const auto norm = scalar*scalar - pseudo*pseudo - (bivec|bivec);
		return (parta + partb + partc - partd - parte + partf)/norm;
//		return v2*scalar*scalar + v2*pseudo*pseudo + (bivec|v2)*R(2.)*scalar - ((bivec|v2)|bivec) - (((bivec^v2)|bivec) - ((~(bivec^v2)))/(e1|e1)*R(2.));
	}
}


#endif

