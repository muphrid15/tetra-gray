#ifndef CLIFFORD_STATIC_HDR
#define CLIFFORD_STATIC_HDR

namespace multi
{
	namespace impl
	{
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
		
	template<typename R>
	class Versor : public impl::NumericalArray<Versor<R>, R, 8u>
	{
		private:
			using BaseType = impl::NumericalArray<Versor<R>, R, 8u>;
		public:
			__host__ __device__ Versor() : BaseType() {}
			__host__ __device__ Versor(const R (&comps)[4]) : BaseType(comps) {}
			__host__ __device__ Versor(const BaseType& bt) : BaseType(bt) {}
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
	};

	/*
	template<typename R>
	class OddMultivector : impl::NumericalArray<OddMultivector<R>, R , 8u>
	{
	};
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
		ret[0] = b1[0]*v2[1] + b1[1]*v2[2] + b1[3]*v2[3];
		ret[1] = b1[0]*v2[0] + b1[2]*v2[2] + b1[3]*v2[3];
		ret[2] = b1[1]*v2[0] - b1[2]*v2[1] + b1[5]*v2[3];
		ret[3] = b1[3]*v2[0] - b1[4]*v2[1] - b1[5]*v2[2];
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
		ret[0] = b1[0]*v2[2] - b1[1]*v2[1] + b1[2]*v2[0];
		ret[1] = b1[0]*v2[3] - b1[3]*v2[1] + b1[4]*v2[0];
		ret[2] = b1[1]*v2[3] - b1[3]*v2[2] + b1[5]*v2[0];
		ret[3] = b1[2]*v2[3] - b1[4]*v2[2] + b1[5]*v2[1];
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
		ret[0] = b1[5];
		ret[1] = -b1[4];
		ret[2] = -b1[3];
		ret[3] = b1[2];
		ret[4] = b1[1];
		ret[5] = -b1[0];
		return Bivector<R>(ret);
	}

	template<typename R>
	__host__ __device__ Trivector<R> operator~(const Vector<R>& v1)
	{
		R ret[4];
		ret[0] = v1[3]; //012, bit 7
		ret[1] = -v1[2]; //013, bit 11
		ret[2] = v1[1]; //023, bit 13
		ret[3] = v1[0]; //123, bit 14
		return Trivector<R>(ret);
	}

	template<typename R>
	__host__ __device__ Vector<R> operator~(const Trivector<R>& t1)
	{
		R ret[4];
		ret[0] = -t1[3];
		ret[1] = -t1[2];
		ret[2] = t1[1];
		ret[3] = -t1[0];
		return Vector<R>(ret);
	}

	template<typename R>
	__host__ __device__ Vector<R> bilinearMultiply(const Versor<R>& e1, const Vector<R>& v2)
	{
		const auto bivec = e1.bivectorPart();
		const auto scalar = e1[0];
		const auto pseudo = e1[7];
		const auto parta = v2*scalar*scalar;
		const auto partb = v2*pseudo*pseudo;
		const auto partc = (bivec|v2)*R(2.)*scalar;
		const auto partd = ((bivec|v2)|bivec);
		const auto parte = (bivec^v2)|bivec;
		const auto partf = ~(bivec^v2)*pseudo*R(2.);
		const auto norm = scalar*scalar - pseudo*pseudo - (bivec|bivec);
		return (parta + partb + partc - partd - parte - partf)/norm;
//		return v2*scalar*scalar + v2*pseudo*pseudo + (bivec|v2)*R(2.)*scalar - ((bivec|v2)|bivec) - (((bivec^v2)|bivec) - ((~(bivec^v2)))/(e1|e1)*R(2.));
	}
}


#endif

