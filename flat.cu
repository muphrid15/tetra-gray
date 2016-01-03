#include "raytracer.cuh"
#include "rhs.cuh"
#include "image.cuh"
#include "image_id.cuh"

int main(void)
{
	using cudaftk::operator%;
	const float campos[4] = {0.f, 20.f, 0.f, 0.f};
	const float extract_radius = 50.f;
	const auto camposvec = ray::Vector<float>(campos);
	ray::raytrace(
			1280U,
			720U,
			float(ray::PI/2.f),
			camposvec,
			ray::Versor<float>(1.f),
			"flat.png",
			.05f,
			ray::ImageInitialDataSolver(),
			ode::RK4(),
			ray::FlatRHS(),
			ray::CombinedStopCondition() % extract_radius % 200.f,
			ray::SphericalColormap() % extract_radius,
			ray::PngppImageWriter());
}
