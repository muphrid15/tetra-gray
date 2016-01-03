#include <iostream>
#include "integrator.cuh"
#include "clifford.cuh"
#include "test.cuh"
#include "thrustlist.cuh"
#include "operator.cuh"
#include "thrust/functional.h"
#include "particle.cuh"
#include "rhs.cuh"
#include "orientation.cuh"
#include "types.cuh"
#include "raytracer.cuh"
#include "image.cuh"
#include "libftk/functor.h"
#include "libftk/operator.h"
#include "image_id.cuh"
#include "clifford-static.cuh"

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
int host_list_concat_fold_test();
int host_list_range_map_test();
int host_list_fmap_test();
int device_list_concat_fold_test();
int device_list_fmap_test();
void single_particle_evolve_test();
void multivector_rotate_test();
void multivector_trig_rotate_test();
void compress_to_grade_test();
void single_photon_test();
void single_photon_test_position();
void camera_id_test();
void camera_id_host_test();
void single_photon_test_position_doran();
void clifford_static_rotation_test();
void clifford_static_rotation_test2();
void doran_rhs_test();

int main(void)
{
	//integrator tests
	std::cout << "rk4_test(): " << rk4_test() << std::endl;

	//multivector/clifford algebra tests
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
	
	//thrust list tests
	std::cout << "host_list_concat_fold_test(): " << host_list_concat_fold_test() << std::endl;
	std::cout << "host_list_range_map_test(): " << host_list_range_map_test() << std::endl;
	std::cout << "host_list_fmap_test(): " << host_list_fmap_test() << std::endl;
	std::cout << "device_list_concat_fold_test(): " << device_list_concat_fold_test() << std::endl;
	std::cout << "device_list_fmap_test(): " << device_list_fmap_test() << std::endl;

	std::cout << "single_particle_evolve_test(): "; single_particle_evolve_test(); std::cout << std::endl;
	std::cout << "multivector_rotate_test(): "; multivector_rotate_test(); std::cout << std::endl;
	std::cout << "multivector_trig_rotate_test(): "; multivector_trig_rotate_test(); std::cout << std::endl;
	std::cout << "compress_to_grade_test(): "; compress_to_grade_test(); std::cout << std::endl;
	std::cout << "single_photon_test(): "; single_photon_test(); std::cout << std::endl;
	std::cout << "single_photon_test_position(): "; single_photon_test_position(); std::cout << std::endl;
	std::cout << "camera_id_test(): "; camera_id_test(); std::cout << std::endl;
	std::cout << "camera_id_host_test(): "; camera_id_host_test(); std::cout << std::endl;
	std::cout << "single_photon_test_position_doran(): "; single_photon_test_position_doran(); std::cout << std::endl;
	
	std::cout << "clifford_static_rotation_test(): "; clifford_static_rotation_test(); std::cout << std::endl;
	std::cout << "clifford_static_rotation_test2(): "; clifford_static_rotation_test2(); std::cout << std::endl;
	std::cout << "doran_rhs_test(): "; doran_rhs_test(); std::cout << std::endl;
	return 0;
}

double rk4_test()
{
	using namespace ode;
	using Data = ODEData<double, double>;
	return ODEIntegrator()(Data(0., -1., .2), RK4(), [](double val, double param) { return -2.*val+param;}, [](Data od) { return od.param >= 1.; }).value;
	//test result: .263753, per wolfram alpha https://www.wolframalpha.com/input/?i=fourth%E2%80%90order+Runge%E2%80%90Kutta+method+{y%27[x]+%3D%3D+x+-+2+y[x]%2C+y[-1.]+%3D%3D+0}+from+-1.+to+1.+stepsize+%3D+0.2&lk=1
}

int popcount_test()
{
	return mv::impl::popcount(7); //3
}

uint ipow_test()
{
	return mv::impl::ipow(3, 4); //81
}

int contractionSign_test_zero()
{
	return mv::impl::contractionSign(2,1,1,9); //1
}

int permutationSign_test()
{
	return mv::impl::permutationSign(4,2,4); //0
}

int contractionSign_test_minus()
{
	return mv::impl::contractionSign(3,2,4,10); //-1
}
int contractionSign_no_match_test()
{
	return mv::impl::contractionSign(2, 4, 3, 0U); //0
}

void transformArray_test()
{
	double testar[4] = { 1., 3., 5., 7.};

	mv::impl::transformArray(testar, TransformTester<double>());
	printArray(testar); //(1,4,7,10)
}

int choose_test()
{
	return mv::impl::choose(4,2); //6
}

void multivector_zero_test()
{
	mv::Multivector<4>().print(); //0 0 0...
}

void multivector_vector_test()
{
	double testar[4] = {1., 3., 5., 7.}; //0 1 3 0 5 0 0 0 7 0 ...
	mv::Multivector<3,1,0, double>::makeMultivectorFromGrade<1>(testar).print();
}


void multivector_multiply_test()
{
	using Mv = mv::Multivector<3,1,0,double>;
	double veca[4] = {1., 3., 5., 7.};
	double vecb[4] = {2., 4., 6., 8.};
	(Mv::makeMultivectorFromGrade<1>(veca)*Mv::makeMultivectorFromGrade<1>(vecb)).print(); //[-12 0 0 -2 0 -4 -2 0 0 -6 -4 0 -2 0 0 0 ]
}

void multivector_add_test()
{
	using Mv = mv::Multivector<2,0,0,double>;
	using SMv = mv::SingleGradedMultivector<1,2,0,0,double>;
	double veca[2] = {1., 3.};
	double vecb[2] = {2, -1};

	const auto smva = SMv(veca);
	const auto smvb = SMv(vecb);

	const auto mva = Mv::makeMultivectorFromGrade<1>(smva);
	const auto mvb = Mv::makeMultivectorFromGrade<1>(smvb);

	(mva*mvb+mva).print(); //[-1 1 3 -7]
}

void multivector_scalar_multiply_test()
{
	using Mv = mv::Multivector<2,0,0,double>;
	double veca[2] = {1., 3.};

	const auto mva = Mv::makeMultivectorFromGrade<1>(veca);
	(mva*5.).print(); //[0 5 15 0]
}

void multivector_grade_project_test()
{
	using Mv = mv::Multivector<2,0,0,double>;
	double veca[2] = {1., 3.};
	double vecb[2] = {2, -1};

	const auto smva = mv::SingleGradedMultivector<1,2,0,0,double>(veca);
	const auto smvb = mv::SingleGradedMultivector<1,2,0,0,double>(vecb);
	const auto mva = Mv::makeMultivectorFromGrade<1>(smva);
	const auto mvb = Mv::makeMultivectorFromGrade<1>(smvb);

	((mva*mvb+mva) % 2U).print(); //[0 0 0 -7]
}

void single_multiply_test_left()
{
	using Mv = mv::Multivector<2,0,0,double>;
	double veca[2] = {1., 3.};
	double vecb[2] = {2, -1};

	const auto smva = mv::SingleGradedMultivector<1,2,0,0,double>(veca);
	const auto smvb = mv::SingleGradedMultivector<1,2,0,0,double>(vecb);
	const auto mvb = Mv::makeMultivectorFromGrade<1>(smvb);
	(smva*mvb).print(); //[-1 0 0 -7]
}

void single_multiply_test_right()
{
	using Mv = mv::Multivector<2,0,0,double>;
	double veca[2] = {1., 3.};
	double vecb[2] = {2, -1};

	const auto smva = mv::SingleGradedMultivector<1,2,0,0,double>(veca);
	const auto smvb = mv::SingleGradedMultivector<1,2,0,0,double>(vecb);
	const auto mva = Mv::makeMultivectorFromGrade<1>(smva);
	const auto mvb = Mv::makeMultivectorFromGrade<1>(smvb);
	(mva*smvb).print(); //[-1 0 0 -7]
}

int host_list_concat_fold_test(void)
{
	//tests the constructor, concat, and fold
	return cudaftk::CPUList<int>(2) | ftk::concat >> cudaftk::CPUList<int>(1) | ftk::fold >> thrust::plus<int>() >> 0; //3
}

int host_list_range_map_test(void)
{
	return ftk::ListLikeData<cudaftk::CPUList<int> >::map(thrust::negate<int>(), 3 | ftk::ListLike<cudaftk::CPUList>::range >> -2) | ftk::fold >> thrust::plus<int>() >> 0; //3
}
int host_list_fmap_test(void)
{
	return -(3 | ftk::ListLike<cudaftk::GPUList>::range >> 1 | ftk::fmap >> thrust::negate<int>() | ftk::fold >> thrust::plus<int>() >> 0)/2; //3
}

int device_list_concat_fold_test(void)
{
	//tests the constructor, concat, and fold
	return cudaftk::GPUList<int>(2) | ftk::concat >> cudaftk::GPUList<int>(1) | ftk::fold >> thrust::plus<int>() >> 0; //3
}

/*
  TODO: this triggers an internal compiler error; try to avoid it
int device_list_range_map_test(void)
{
	return ftk::ListLikeData<cudaftk::GPUList<int> >::map(thrust::negate<int>(), 3 | ftk::ListLike<cudaftk::GPUList>::range >> -2) | ftk::fold >> thrust::plus<int>() >> 0;
}
*/
int device_list_fmap_test(void)
{
	return -(3 | ftk::ListLike<cudaftk::GPUList>::range >> 1 | ftk::fmap >> thrust::negate<int>() | ftk::fold >> thrust::plus<int>() >> 0)/2; //3
}

void single_particle_evolve_test()
{
	using R = double;
	using Pt = pt::Particle<R>;

	const R posarr[4] = {0., 0., 0., 0.};
	const R momarr[4] = {1., 0., 0., 0.};

	const auto pinit = Pt(posarr, momarr);	

	using Data = ode::ODEData<Pt, R>;

	(ode::ODEIntegrator()(Data(pinit, 0., .1), ode::RK4(), ray::FlatRHS(), [] (Data dat) { return dat.param >= 2.; })).value.position.print(); 
}

void multivector_rotate_test()
{
	/*
	using Mv = mv::Multivector<2,0,0,double>;
	using SMv = mv::SingleGradedMultivector<1,2,0,0,double>;
	double veca[2] = {1., 0.};
	double vecb[2] = {1., 1.};

	const auto smva = SMv(veca);
	const auto smvb = SMv(vecb);

	double vecc[2] = {0., 1.};
	const auto smvc = SMv(vecc);


	const auto rotor = ray::simpleRotor(smva, smvb);
	ray::bilinearMultiply(rotor, Mv::makeMultivectorFromGrade(smvc)).print();
	*/

	std::cout << "Obsolete" << std::endl;
}

void multivector_trig_rotate_test()
{
	using Mv = multi::Versor<double>;
	using SMv = multi::Vector<double>;
	double veca[4] = {0., 1., 0, 0.};
	double vecb[4] = {0., 0., 1., 0.};

	const auto smva = SMv(veca);
	const auto smvb = SMv(vecb);

	double vecc[4] = {0., 1., 0., 0.};
	const auto smvc = SMv(vecc);
	const double angle = 1.5708; //PI/2


	const auto rotor = ray::simpleRotorFromAngle(smva, smvb, angle);
	multi::bilinearMultiply(rotor, smvc).print();
}

void compress_to_grade_test()
{
	using Mv = mv::Multivector<2,0,0,double>;
	using SMv = mv::SingleGradedMultivector<1,2,0,0,double>;
	double veca[2] = {1., 2.};
	double vecb[2] = {4., 7.};

	const auto smva = SMv(veca);
	const auto smvb = SMv(vecb);

	const auto mvc = smva*smvb;
	auto smvc = SMv(mvc);
	smvc.print();
}

void single_photon_test()
{
	using cudaftk::operator/;
	using cudaftk::operator%;
	using ftk::operator|;
	using ftk::operator>>;
	using cudaftk::operator*;
	const float campos[4] = {0.f, 20.f, 0.f, 0.f};
	const float extract_radius = 50.f;
	const auto camposvec = ray::Vector<float>(campos);
	const float pho_mom[4] = {-1.f, -.707f, -.707f, 0.f};
	const auto photon_momentum = ray::Vector<float>(pho_mom);

	const auto odata = ray::ParticleData<float>(ray::Particle<float>(camposvec, photon_momentum), 0.f, 1.f);
	const auto olist = cudaftk::GPUList<ray::ParticleData<float> >(odata);
	auto result = olist | ftk::fmap >>
	   	((ode::ODEIntegrator() / (ray::CombinedStopCondition() % extract_radius % 1000.f)
	   	/ ray::FlatRHS()
		/ ode::RK4())
		 * (ray::SphericalColormap() % extract_radius));

	thrust::host_vector<uint> hostresult = result.unpack();
	std::cout << hostresult[0];
}

void single_photon_test_position()
{
	using cudaftk::operator/;
	using cudaftk::operator%;
	using ftk::operator|;
	using ftk::operator>>;
	using cudaftk::operator*;
	const float campos[4] = {0.f, 20.f, 0.f, 0.f};
	const float extract_radius = 50.f;
	const auto camposvec = ray::Vector<float>(campos);
	const float pho_mom[4] = {-1.f, -.707f, -.707f, 0.f};
	const auto photon_momentum = ray::Vector<float>(pho_mom);
	const auto odata = ray::ParticleData<float>(ray::Particle<float>(camposvec, photon_momentum), 0.f, 1.f);

	const auto olist = cudaftk::GPUList<ray::ParticleData<float> >(odata);
	auto result = olist | ftk::fmap >>
	   	((ode::ODEIntegrator() / (ray::CombinedStopCondition() % extract_radius % 1000.f)
	   	/ ray::FlatRHS()
		/ ode::RK4()));

	thrust::host_vector<ray::ParticleData<float> > hostresult = result.unpack();
	std::cout << "Position:"; hostresult[0].value.position.print(); std::cout << std::endl;
	std::cout << "Momentum:"; hostresult[0].value.momentum.print(); std::cout << std::endl;
}

void camera_id_test()
{
	using cudaftk::operator/;
	using cudaftk::operator%;
	using ftk::operator|;
	using ftk::operator>>;
	using cudaftk::operator*;
	const float campos[4] = {20.f, 20.f, 0.f, 0.f};
	const float extract_radius = 50.f;
	const auto camposvec = ray::Vector<float>(campos);
	const float pho_mom[4] = {-1.f, -1.f, 0.f, 0.f};
	const auto photon_momentum = ray::Vector<float>(pho_mom);
	const auto odata = ray::ParticleData<float>(ray::Particle<float>(camposvec, photon_momentum), 0.f, 1.f);

//	const auto olist = cudaftk::GPUList<ray::ParticleData<float> >(odata);
	const auto zero_list = cudaftk::GPUList<uint>(0u);
	auto result = zero_list | ftk::fmap >>
		((ray::ImageInitialDataSolver() % camposvec % ray::Versor<float>(1.f) % 1280u % 720u % float(ray::PI/4.f) % 1.f));

	thrust::host_vector<ray::ParticleData<float> > hostresult = result.unpack();
	std::cout << "Position:"; hostresult[0].value.position.print(); std::cout << std::endl;
	std::cout << "Momentum:"; hostresult[0].value.momentum.print(); std::cout << std::endl;
	std::cout << "Color:" << ray::SphericalColormap()(extract_radius, hostresult[0]) << std::endl;
}

void camera_id_host_test()
{
	using cudaftk::operator/;
	using cudaftk::operator%;
	using ftk::operator|;
	using ftk::operator>>;
	using cudaftk::operator*;
	const float campos[4] = {0.f, 20.f, 0.f, 0.f};
	const float extract_radius = 50.f;
	const auto camposvec = ray::Vector<float>(campos);
	const float pho_mom[4] = {-1.f, -1.f, 0.f, 0.f};
	const auto photon_momentum = ray::Vector<float>(pho_mom);
	const auto odata = ray::ParticleData<float>(ray::Particle<float>(camposvec, photon_momentum), 0.f, 1.f);

//	const auto olist = cudaftk::GPUList<ray::ParticleData<float> >(odata);
	const auto zero_list = cudaftk::CPUList<uint>(0u);
	auto result = zero_list | ftk::fmap >>
		((ray::ImageInitialDataSolver() % camposvec % ray::Versor<float>(1.f) % 1280u % 720u % float(ray::PI/4.f) % 1.f));

	thrust::host_vector<ray::ParticleData<float> > hostresult = result.unpack();
	std::cout << "Position:"; hostresult[0].value.position.print(); std::cout << std::endl;
	std::cout << "Momentum:"; hostresult[0].value.momentum.print(); std::cout << std::endl;
	std::cout << "Color:" << ray::SphericalColormap()(extract_radius, hostresult[0]) << std::endl;
}

void single_photon_test_position_doran()
{
	using cudaftk::operator/;
	using cudaftk::operator%;
	using ftk::operator|;
	using ftk::operator>>;
	using cudaftk::operator*;
	const float campos[4] = {0.f, 20.f, 0.f, 0.f};
	const float extract_radius = 50.f;
	const auto camposvec = ray::Vector<float>(campos);
	const float pho_mom[4] = {-1.f, -.707f, -0.707f, 0.f};
	const auto photon_momentum = ray::Vector<float>(pho_mom);
	const auto odata = ray::ParticleData<float>(ray::Particle<float>(camposvec, photon_momentum), 0.f, .1f);

	const auto olist = cudaftk::CPUList<ray::ParticleData<float> >(odata);
	auto result = olist | ftk::fmap >>
	   	((ode::ODEIntegrator() / (ray::CombinedStopCondition() % extract_radius % 100.f)
	   	/ (ray::DoranRHS() / .5f)
		/ ode::RK4()));

	thrust::host_vector<ray::ParticleData<float> > hostresult = result.unpack();
	std::cout << "Position:"; hostresult[0].value.position.print(); std::cout << std::endl;
	std::cout << "Momentum:"; hostresult[0].value.momentum.print(); std::cout << std::endl;
}

void clifford_static_rotation_test()
{
	using R = double;
	const R evencomps[8] = { .923879, 0., 0., -.382683, 0., 0., 0., 0. };	
	const auto rotor = multi::Versor<R>(evencomps);
	const R veccomps[4] = {0., 1., 0., 0.};
	const auto vec = multi::Vector<R>(veccomps);
	multi::bilinearMultiply(rotor, vec).print(); //[0 0 -1 0 ] is correct; the +.707 in the bivector part means we rotate clockwise
}

void clifford_static_rotation_test2()
{
	using R = double;
	const R evencomps[8] = {1.038, 0., 0., 0., 0.278, 0., 0., 0. };	
	const auto rotor = multi::Versor<R>(evencomps);
	const R veccomps[4] = {0., 0., 1., 0.};
	const auto vec = multi::Vector<R>(veccomps);
	multi::bilinearMultiply(rotor, vec).print(); //[0 0 -1 0 ] is correct; the +.707 in the bivector part means we rotate clockwise
}

void doran_rhs_test()
{
	const float campos[4] = {0.f, 20.f, 0.f, 0.f};
	const auto camposvec = ray::Vector<float>(campos);
	const float pho_mom[4] = {-1.f, -1.f, 0.f, 0.f};
	const auto photon_momentum = ray::Vector<float>(pho_mom);

	const auto particle = ray::Particle<float>(camposvec, photon_momentum);
	const auto particle_rhs = ray::DoranRHS()(particle, 0.f, .5f);
	
	std::cout << "Position rhs:"; particle_rhs.position.print(); std::cout << std::endl;
	std::cout << "Momentum rhs:"; particle_rhs.momentum.print(); std::cout << std::endl;
}



