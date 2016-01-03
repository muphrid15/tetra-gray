#ifndef IMAGE_ID_HDR
#define IMAGE_ID_HDR
#include "clifford.cuh"
#include "particle.cuh"
#include "orientation.cuh"

namespace ray
{
	struct ImageInitialDataSolver
	{
		public:
			template<typename R>
				__host__ __device__ ParticleData<R> operator()(const Vector<R>& position, const Versor<R>& center_orientation, const uint& img_width_px, const uint& img_height_px, const R& horizontal_fov, const R& dparam, const uint& pixel_idx) const
				{
					//the camera is canonically considered to point in the -x direction, with width along the +y direction and height along the +z direction
					//this is chosen so that larger (more positive) rotation angles in the out-left and up-out oriented planes correspond to increased indices in the width/height dimensions.
					//the 0th pixel is in the top-left corner of the image
					//we rotate about the camera's up direction first, then about the camera's left direction.
					//this converts to rotating about the baseline left, then baseline up (converting body rotations to fixed axis rotations is always a reversed sequence of rotations)
					//hence, we use the baseline orientation directions here
					const R baseline_time_dir[4] = {1., 0., 0., 0.};
					const R baseline_out_dir[4] = {0., -1., 0., 0.};
					const R baseline_left_dir[4] = {0., 0., 1., 0.};
					const R baseline_up_dir[4] = {0., 0., 0., 1.};

					const auto vec_out_dir = Vector<R>(baseline_out_dir);
					const auto vec_left_dir = Vector<R>(baseline_left_dir);
					const auto vec_up_dir = Vector<R>(baseline_up_dir);
					const auto vec_time_dir = Vector<R>(baseline_time_dir);

					const R da = horizontal_fov/img_width_px;
					const uint height_idx = pixel_idx/img_width_px;
					const uint width_idx = pixel_idx - height_idx*img_width_px;

					const R width_angle = (int(width_idx) - int(img_width_px/2) + .5) * da; //.5 offset so that the centerline of the image is the boundary between pixels
					const auto rotor_left_right = simpleRotorFromAngle(vec_out_dir, vec_left_dir, width_angle);

					const R height_angle = (int(height_idx) - int(img_height_px/2) + .5)*da;
					const auto rotor_up_down = simpleRotorFromAngle(vec_up_dir, vec_out_dir, height_angle);

					//again, rotating left-right first and then up-down according to the rotated axis corresponds to rotating up-down first and then left-right according to the baseline axes
					//hence, we put the left-right rotor on the right of this multiply
					//we also put the orientation passed in on the left, requiring that it describes a rotation that would otherwise be performed first to the camera
					const Vector<R> vec_central_four_momentum = vec_out_dir - vec_time_dir;
					const auto total_rotor = center_orientation*rotor_left_right*rotor_up_down;
					const auto momentum = multi::bilinearMultiply(total_rotor, vec_central_four_momentum);
					return ParticleData<R>(Particle<R>(position, momentum), R(0.), dparam);
				}
	};

	//Test purposes only
	struct DoNothingImageIDSolver
	{
		public:
			template<typename R>
				__host__ __device__ ParticleData<R> operator()(const Vector<R>& position, const Versor<R>& center_orientation, const uint& img_width_px, const uint& img_height_px, const R& horizontal_fov, const R& dparam, const uint& pixel_idx) const
				{
					return ParticleData<R>(Particle<R>(position, Vector<R>()), 0, dparam);
				}
	};
}
#endif
