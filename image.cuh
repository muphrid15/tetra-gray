#ifndef IMAGE_HDR
#define IMAGE_HDR
#include "types.cuh"
#include "coord_systems.cuh"
#include <png++/png.hpp>
#include "thrustlist.cuh"

namespace ray
{
	struct MaxRadiusStopCondition
	{
		template<typename R>
			__host__ __device__ bool operator()(const R& radius, const ParticleData<R>& pdata) const
			{
				return sphericalCoordinatesFromCartesian(pdata.value.position).extractComponent(0) >= radius;
			}	
	};

	struct MaxParameterStopCondition
	{
		template<typename R>
			__host__ __device__ bool operator()(const R& max_parameter, const ParticleData<R>& pdata) const
			{
				return pdata.param >= max_parameter;
			}
	};

	struct CombinedStopCondition
	{
		template<typename R>
			__host__ __device__ bool operator()(const R& radius, const R& max_parameter, const ParticleData<R>& pdata) const
			{
				return MaxRadiusStopCondition()(radius, pdata) || MaxParameterStopCondition()(max_parameter, pdata);
			}	
	};

	constexpr __host__ __device__ uint rgbToSingle(const uint& red, const uint& green, const uint& blue)
	{
		return red*256*256 + green*256 + blue;
	}

	template<typename R>
		__host__ __device__ bool testStripe(const R& angle, const R& stripe_interval, const R& stripe_half_width_ratio)
		{
			const auto stripe_number = angle/stripe_interval;
			const auto stripe_remainder = stripe_number - lrint(stripe_number);
			return abs(stripe_remainder) <= stripe_half_width_ratio;
		}

	struct SphericalColormap
	{
		template<typename R>
			__host__ __device__ uint operator()(const R& escape_radius, ParticleData<R>& pdata) const
			{
				if(MaxRadiusStopCondition()(escape_radius, pdata))
				{
					const auto spherical_coords = sphericalCoordinatesFromCartesian(pdata.value.position);	
					const auto theta = spherical_coords.extractComponent(1);
					const auto phi = spherical_coords.extractComponent(2);

					const R stripe_half_width_ratio = .05;
					const R stripe_interval = PI/18.; //10 degrees

					const uint black = 0;
					if(testStripe(theta, stripe_interval, stripe_half_width_ratio) || testStripe(phi, stripe_interval, stripe_half_width_ratio))
					{
						return black;
					}

					const auto x = pdata.value.position.extractComponent(0);
					const auto y = pdata.value.position.extractComponent(1);
					const auto z = pdata.value.position.extractComponent(2);

					if(y > R(0.) && z > R(0.))
					{
						return rgbToSingle(255, 0, 0);
					}
					if(y < 0. && z > 0.)
					{
						return rgbToSingle(0, 255, 0);
					}
					if(y > 0. && z < 0.)
					{
						return rgbToSingle(0, 0, 255);
					}
					else
					{
						return rgbToSingle(255, 255, 0);
					}
				}
				return rgbToSingle(0, 255, 255);
			}
	};

	struct PngppImageWriter
	{
		private:
			png::rgb_pixel singleToRgbPixel(const uint& pixval) const;

		public:
			void operator()(const cudaftk::GPUList<uint>& single_colors, const uint& img_width_px, const uint& img_height_px, const char* filename) const;
	};

}

#endif
