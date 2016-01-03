#include "raytracer.cuh"
#include "rhs.cuh"
#include "image.cuh"
#include "image_id.cuh"
#include "stepsize.cuh"

int main(void)
{
	using cudaftk::operator%;
	using cudaftk::operator/;
	using cudaftk::operator*;
	const float campos[4] = {0.f, 20.f, 0.f, 0.f};
	const float extract_radius = 50.f;
	const float stepsize = 0.05f;
	const float max_step_ratio = 40.f;
	const float max_time = 500.0f;
	const float xdir[4] = {0.f, 1.f, 0.f, 0.f};
	const float ydir[4] = {0.f, 0.f, 1.f, 0.f};
	const auto camposvec = ray::Vector<float>(campos);
	//auto stop_cond = [max_step_ratio, extract_radius, max_time]  __device__ (ray::ParticleData<float> data) { return ray::DynamicStepsizeStopCondition()(max_step_ratio, data) && ray::CombinedStopCondition()(extract_radius, max_time, data); };
	ray::raytrace(
			1280U,
			720U,
			float(ray::PI/2.f),
			camposvec,
//			ray::simpleRotorFromAngle(ray::Vector<float>(xdir), ray::Vector<float>(ydir), float(ray::PI/2.f)),
			ray::Versor<float>(1.f),
			"doran.png",
			stepsize,
			ray::ImageInitialDataSolver(),
			ode::RK4() * (ray::DynamicStepsizeAdjuster() % stepsize),
			ray::DoranRHS() / .5f,
//			[max_step_ratio, extract_radius, max_time]  __device__ (const ray::ParticleData<float>& data) -> bool { return ray::DynamicStepsizeStopCondition()(max_step_ratio, data) && ray::CombinedStopCondition()(extract_radius, max_time, data); } ,
			ray::CombinedDynamicStepsizeStopCondition() % extract_radius % max_time % max_step_ratio % stepsize,
			ray::SphericalColormap() % extract_radius,
			ray::PngppImageWriter());
}
