#include "raytracer.cuh"
#include "rhs.cuh"
#include "image.cuh"
#include "image_id.cuh"

int main(void)
{
	using cudaftk::operator%;
	using cudaftk::operator/;
	const float campos[4] = {20.f, 0.f, 0.f, 0.f};
	const float extract_radius = 50.f;
	const auto camposvec = ray::Vector<float>(campos);
	ray::raytrace(
			1280U,
			720U,
			float(ray::PI/4.f),
			camposvec,
			ray::Multivector<float>(1.f),
			"doran.png",
			.5f,
			ray::ImageInitialDataSolver(),
			ode::RK4(),
			ray::DoranRHS() / .5f,
			ray::CombinedStopCondition() % extract_radius % .5f,
			ray::SphericalColormap() % extract_radius,
			ray::PngppImageWriter());
}
