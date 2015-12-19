#include <iostream>
#include "integrator.cuh"
#include "clifford.cuh"
#include "test.cuh"

double rk4_test();
int popcount_test();
uint ipow_test();
int contractionSign_test_zero();
int contractionSign_test_minus();
int contractionSign_no_match_test();
int permutationSign_test();
void transformArray_test();
int choose_test();
void multivector_zero_test();
void multivector_vector_test();
void multivector_multiply_test();
void multivector_add_test();
void multivector_scalar_multiply_test();
void multivector_grade_project_test();
void single_multiply_test_left();
void single_multiply_test_right();

int main(void)
{
	std::cout << "rk4_test(): " << rk4_test() << std::endl;
	std::cout << "popcount_test(): " << popcount_test() << std::endl;
	std::cout << "ipow_test(): " << ipow_test() << std::endl;
	std::cout << "contractionSign_test_zero(): " << contractionSign_test_zero() << std::endl;
	std::cout << "contractionSign_test_minus(): " << contractionSign_test_minus() << std::endl;
	std::cout << "contractionSign_no_match_test(): " << contractionSign_no_match_test() << std::endl;
	std::cout << "permutationSign_test(): " << permutationSign_test() << std::endl;
	std::cout << "transformArray_test(): "; transformArray_test(); std::cout << std::endl;
	std::cout << "choose_test(): " << choose_test() << std::endl;
	std::cout << "multivector_zero_test(): "; multivector_zero_test(); std::cout << std::endl;
	std::cout << "multivector_vector_test(): "; multivector_vector_test(); std::cout << std::endl;
	std::cout << "multivector_multiply_test(): "; multivector_multiply_test(); std::cout << std::endl;
	std::cout << "multivector_add_test(): "; multivector_add_test(); std::cout << std::endl;
	std::cout << "multivector_scalar_multiply_test(): "; multivector_scalar_multiply_test(); std::cout << std::endl;
	std::cout << "multivector_grade_project_test(): "; multivector_grade_project_test(); std::cout << std::endl;
	std::cout << "single_multiply_test_left(): "; single_multiply_test_left(); std::cout << std::endl;
	std::cout << "single_multiply_test_right(): "; single_multiply_test_right(); std::cout << std::endl;

	return 0;
}

double rk4_test()
{
	using Data = ODEData<double, double>;
	return ODEIntegrator()(Data(0., -1., .2), RK4(), [](double val, double param) { return -2.*val+param;}, [](Data od) { return od.param >= 1.; }).value;
	//test result: .263753, per wolfram alpha https://www.wolframalpha.com/input/?i=fourth%E2%80%90order+Runge%E2%80%90Kutta+method+{y%27[x]+%3D%3D+x+-+2+y[x]%2C+y[-1.]+%3D%3D+0}+from+-1.+to+1.+stepsize+%3D+0.2&lk=1
}

int popcount_test()
{
	return popcount(7); //3
}

uint ipow_test()
{
	return ipow(3, 4); //81
}

int contractionSign_test_zero()
{
	return contractionSign(2,1,1,9); //1
}

int permutationSign_test()
{
	return permutationSign(4,2,4); //0
}

int contractionSign_test_minus()
{
	return contractionSign(3,2,4,10); //-1
}
int contractionSign_no_match_test()
{
	return contractionSign(2, 4, 3, 0U); //0
}

void transformArray_test()
{
	double testar[4] = { 1., 3., 5., 7.};

	transformArray(testar, TransformTester<double>());
	printArray(testar); //(1,4,7,10)
}

int choose_test()
{
	return choose(4,2); //6
}

void multivector_zero_test()
{
	Multivector<4>().print(); //0 0 0...
}

void multivector_vector_test()
{
	double testar[4] = {1., 3., 5., 7.}; //0 1 3 0 5 0 0 0 7 0 ...
	Multivector<3,1,0, double>::makeMultivectorFromGrade<1>(testar).print();
}


void multivector_multiply_test()
{
	using Mv = Multivector<3,1,0,double>;
	double veca[4] = {1., 3., 5., 7.};
	double vecb[4] = {2., 4., 6., 8.};
	(Mv::makeMultivectorFromGrade<1>(veca)*Mv::makeMultivectorFromGrade<1>(vecb)).print(); //[-12 0 0 -2 0 -4 -2 0 0 -6 -4 0 -2 0 0 0 ]
}

void multivector_add_test()
{
	using Mv = Multivector<2,0,0,double>;
	double veca[2] = {1., 3.};
	double vecb[2] = {2, -1};

	const auto smva = SingleGradedMultivector<1,2,0,0,double>(veca);
	const auto smvb = SingleGradedMultivector<1,2,0,0,double>(vecb);

	const auto mva = Mv::makeMultivectorFromGrade<1>(smva);
	const auto mvb = Mv::makeMultivectorFromGrade<1>(smvb);

	(mva*mvb+mva).print(); //[-1 1 3 -7]
}

void multivector_scalar_multiply_test()
{
	using Mv = Multivector<2,0,0,double>;
	double veca[2] = {1., 3.};

	const auto mva = Mv::makeMultivectorFromGrade<1>(veca);
	(mva*5.).print(); //[0 5 15 0]
}

void multivector_grade_project_test()
{
	using Mv = Multivector<2,0,0,double>;
	double veca[2] = {1., 3.};
	double vecb[2] = {2, -1};

	const auto smva = SingleGradedMultivector<1,2,0,0,double>(veca);
	const auto smvb = SingleGradedMultivector<1,2,0,0,double>(vecb);
	const auto mva = Mv::makeMultivectorFromGrade<1>(smva);
	const auto mvb = Mv::makeMultivectorFromGrade<1>(smvb);

	((mva*mvb+mva) % 2U).print(); //[0 0 0 -7]
}

void single_multiply_test_left()
{
	using Mv = Multivector<2,0,0,double>;
	double veca[2] = {1., 3.};
	double vecb[2] = {2, -1};

	const auto smva = SingleGradedMultivector<1,2,0,0,double>(veca);
	const auto smvb = SingleGradedMultivector<1,2,0,0,double>(vecb);
	const auto mvb = Mv::makeMultivectorFromGrade<1>(smvb);
	(smva*mvb).print(); //[-1 0 0 -7]
}

void single_multiply_test_right()
{
	using Mv = Multivector<2,0,0,double>;
	double veca[2] = {1., 3.};
	double vecb[2] = {2, -1};

	const auto smva = SingleGradedMultivector<1,2,0,0,double>(veca);
	const auto smvb = SingleGradedMultivector<1,2,0,0,double>(vecb);
	const auto mva = Mv::makeMultivectorFromGrade<1>(smva);
	const auto mvb = Mv::makeMultivectorFromGrade<1>(smvb);
	(mva*smvb).print(); //[-1 0 0 -7]
}

