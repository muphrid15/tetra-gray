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
	const float campos[4] = {20.f, 0.f, 0.f, 0.f};
	const float extract_radius = 50.f;
	const float stepsize = .05f;
	const uint max_step_ratio = 2500;
	const float max_time = max_step_ratio*stepsize;
	const auto camposvec = ray::Vector<float>(campos);
	//auto stop_cond = [max_step_ratio, extract_radius, max_time]  __device__ (ray::ParticleData<float> data) { return ray::DynamicStepsizeStopCondition()(max_step_ratio, data) && ray::CombinedStopCondition()(extract_radius, max_time, data); };
	ray::raytrace(
			1280U,
			720U,
			float(ray::PI/2.f),
			camposvec,
			ray::Multivector<float>(1.f),
			"doran.png",
			stepsize,
			ray::ImageInitialDataSolver(),
			ode::RK4() * (ray::DynamicStepsizeAdjuster() % stepsize),
			ray::DoranRHS() / .5f,
//			[max_step_ratio, extract_radius, max_time]  __device__ (const ray::ParticleData<float>& data) -> bool { return ray::DynamicStepsizeStopCondition()(max_step_ratio, data) && ray::CombinedStopCondition()(extract_radius, max_time, data); } ,
			ray::CombinedDynamicStepsizeStopCondition() % extract_radius % max_time % max_step_ratio,
			ray::SphericalColormap() % extract_radius,
			ray::PngppImageWriter());
}
