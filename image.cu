#include "image.cuh"

namespace ray
{
	png::rgb_pixel PngppImageWriter::singleToRgbPixel(const uint& pixval) const
	{
		const uint leading_val = pixval / (256U*256U);
		//uints larger than the rgb range are silently capped to the range, with red given the max value
		const uint red = (leading_val > 255U) ? 255U : leading_val;
		const uint green = (pixval - leading_val*256U*256U)/256U;
		const uint blue = (pixval - leading_val*256U*256U - green*256U);
		return png::rgb_pixel(red, green, blue);
	}

	void PngppImageWriter::operator()(const cudaftk::GPUList<uint>& single_colors, const uint& img_width_px, const uint& img_height_px, const char* filename) const
	{
		const thrust::host_vector<uint> raw_list = single_colors.unpack();
		auto img = png::image<png::rgb_pixel>(img_width_px, img_height_px);
		for(int j = 0; j < img_height_px; j++)
		{
			for(int i = 0; i < img_width_px; i++)
			{
				img.set_pixel(i, j, singleToRgbPixel(raw_list[i + img_width_px*j]));
			}
		}

		img.write(filename);
	}
}
