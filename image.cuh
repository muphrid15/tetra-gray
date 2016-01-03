#ifndef IMAGE_HDR
#define IMAGE_HDR
#include <png++/png.hpp>
#include "types.cuh"
#include "coord_systems.cuh"
#include "thrustlist.cuh"

namespace ray
{
	struct MaxRadiusStopCondition
	{
		template<typename R>
			__host__ __device__ bool operator()(const R& radius, const ParticleData<R>& pdata) const
			{
				return sphericalCoordinatesFromCartesian(pdata.value.position)[1] >= radius;
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

	//implementation details for Spherical Colormap
	namespace impl
	{
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
	}

	//Draws a lat-long grid of lines at the given radius, and paints four areas in solid colors
	//Paints pixels that do not reach the extraction radius in a distinct color
	//This is more or less exactly like the map used in Bohn(2014) - http://arxiv.org/abs/1410.7775v2
	struct SphericalColormap
	{
		template<typename R>
			__host__ __device__ uint operator()(const R& escape_radius, ParticleData<R>& pdata) const
			{
				if(MaxRadiusStopCondition()(escape_radius, pdata))
				{
					const auto spherical_coords = sphericalCoordinatesFromCartesian(pdata.value.position);	
					const auto theta = spherical_coords[2];
					const auto phi = spherical_coords[3];

					const R stripe_half_width_ratio = .05;
					const R stripe_interval = PI/18.; //10 degrees

					const uint black = 0;
					if(impl::testStripe(theta, stripe_interval, stripe_half_width_ratio) || impl::testStripe(phi, stripe_interval, stripe_half_width_ratio))
					{
						return black;
					}

					const auto x = pdata.value.position[1];
					const auto y = pdata.value.position[2];
					const auto z = pdata.value.position[3];

					if(y > R(0.) && z > R(0.))
					{
						return impl::rgbToSingle(255, 0, 0); //red
					}
					if(y < 0. && z > 0.)
					{
						return impl::rgbToSingle(0, 255, 0); //green
					}
					if(y > 0. && z < 0.)
					{
						return impl::rgbToSingle(0, 0, 255); //blue
					}
					else
					{
						return impl::rgbToSingle(255, 255, 0); //yellow
					}
				}
				return impl::rgbToSingle(0, 255, 255); //cyan
			}
	};

	//not templates
	struct PngppImageWriter
	{
		private:
			png::rgb_pixel singleToRgbPixel(const uint& pixval) const;

		public:
			void operator()(const cudaftk::GPUList<uint>& single_colors, const uint& img_width_px, const uint& img_height_px, const char* filename) const;
	};

}

#endif
