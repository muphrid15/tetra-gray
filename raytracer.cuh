#ifndef RAYTRACER_HDR
#define RAYTRACER_HDR
#include "types.cuh"
#include "thrustlist.cuh"
#include "operator.cuh"
#include "libftk/operator.h"
#include "libftk/functor.h"

namespace ray
{
	//Required signatures:
	//R: some real data type (double, float)
	//Z+: uint
	//M<R>: versor
	//V<R>: vector
	//X: arb type
	//2: bool
	//(): void
	//S: string (char*)
	//O<X,R>: ode::ODEData<X,R> (i.e. (X, R, R) )
	//ImageID: V<R> x M<R> x Z+ x Z+ x R x R x Z+ -> X
	//RHS: XxR -> X
	//Integrator: O<X,R> x RHS -> O<X,R>
	//Stop: O<X,R> -> 2
	//Colormap: O<X,R> -> Z+
	//ImageWriter: Z+ x Z+ x Z+ x S -> ()
	template<typename R, typename Integrator, typename ImageID, typename RHS, typename Stop, typename Colormap, typename ImageWriter>
	void raytrace(
			const uint& img_width_px,
			const uint& img_height_px,
			const R& horizontal_fov,
			const Vector<R>& camera_position,
			const Versor<R>& camera_orientation,
			const char* out_filename,
			const R& stepsize,
			const ImageID& img_id,
			const Integrator& integrator,
			const RHS& rhs,
			const Stop& stop,
		   	const Colormap& cmap,
			const ImageWriter& writer)
	{
		const uint img_size = img_width_px * img_height_px;
		using cudaftk::operator%;
		using cudaftk::operator/;
		using cudaftk::operator*;
		using ftk::operator|;
		using ftk::operator<<;
		img_size
			| ftk::ListLike<cudaftk::GPUList>::range >> 0U
			| ftk::fmap >> 
			((ImageID() % camera_position % camera_orientation % img_width_px % img_height_px % horizontal_fov % stepsize)
			* (ode::ODEIntegrator() / stop / rhs / integrator)
			* cmap)
			| writer << out_filename << img_height_px << img_width_px;
	}

}
#endif
