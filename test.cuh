#include <iostream>
#include <cstddef>

template<typename Real>
struct TransformTester
{
	constexpr __host__ __device__ Real operator()(const uint& i, const Real& comp) const
	{
		return comp + i;
	}
};

template<typename A, std::size_t length>
void printArray(const A (&a)[length])
{
	std::cout << "("; std::cout << a[0];
	for(std::size_t i = 1; i < length; i++)
	{
		std::cout << "," << a[i];
	}
	std::cout << ")";
}



