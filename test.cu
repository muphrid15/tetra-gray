#include <iostream>
#include "integrator.cuh"

double rk4_test();
int main(void)
{
	std::cout << "rk4_test(): " << rk4_test() << std::endl;

	return 0;
}

double rk4_test()
{
	using Data = ODEData<double, double>;
	return ODEIntegrator()(Data(0., -1., .2), RK4(), [](double val, double param) { return -2.*val+param;}, [](Data od) { return od.param >= 1.; }).value;
	//test result: .263753, per wolfram alpha https://www.wolframalpha.com/input/?i=fourth%E2%80%90order+Runge%E2%80%90Kutta+method+{y%27[x]+%3D%3D+x+-+2+y[x]%2C+y[-1.]+%3D%3D+0}+from+-1.+to+1.+stepsize+%3D+0.2&lk=1
}
