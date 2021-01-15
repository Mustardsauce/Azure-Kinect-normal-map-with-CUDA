#include <cstdio>

#include "cuda_normal.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat get_mat_from_k4a(k4a::image& src, bool deep_copy=true);
cv::Mat k4a_get_mat(k4a_image_t& src, bool deep_copy = true);
void release_k4a_capture(k4a_capture_t& c);

int main(void)
{
		
	const float depth_scale = 0.001f;
	const int depth_scale_for_visualization = 100;

	k4a_device_t device = NULL;
	uint32_t device_count = k4a_device_get_installed_count();
		
	if (device_count == 0)
	{
		printf("No K4A devices found\n");
		return 0;
	}

	if (K4A_RESULT_SUCCEEDED !=
		k4a_device_open(K4A_DEVICE_DEFAULT, &device))
	{
		printf("Failed to open device\n");
		goto Exit;
	}

	k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
	config.color_format = K4A_IMAGE_FORMAT_COLOR_MJPG;
	config.color_resolution = K4A_COLOR_RESOLUTION_720P;
	config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
	config.camera_fps = K4A_FRAMES_PER_SECOND_30;

	if (K4A_RESULT_SUCCEEDED !=
		k4a_device_start_cameras(device, &config))
	{
		printf("Failed to start device\n");
		goto Exit;
	}

	k4a_calibration_t calibration;
	if (K4A_RESULT_SUCCEEDED !=
		k4a_device_get_calibration(device, config.depth_mode, config.color_resolution, &calibration))
	{
		printf("Failed to get the calibration\n");
		k4a_device_close(device);
		goto Exit;
	}
	
	const auto& depth_calibration = calibration.depth_camera_calibration;

	const int depth_width = calibration.depth_camera_calibration.resolution_width;
	const int depth_height = calibration.depth_camera_calibration.resolution_height;

	cuda_normal_map* normal_generator = new cuda_normal_map(depth_width, depth_height, depth_scale, calibration);
	k4a_capture_t capture = NULL;

	while (true)
	{
		release_k4a_capture(capture);
		if (k4a_device_get_capture(device, &capture, 0) == K4A_WAIT_RESULT_SUCCEEDED)
		{
			k4a_image_t depth_image = k4a_capture_get_depth_image(capture);				
	
			if (depth_image == NULL)	
			{
				continue;
			}
									
			cv::Mat distorted_depthFrame = k4a_get_mat(depth_image);
			cv::Mat distorted_normal,distort_normal_colorization;

			cudaEvent_t start, stop;
			float gpu_time = 0.0f;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start, 0);

			normal_generator->generate(distorted_depthFrame, distorted_normal);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);

			cudaEventElapsedTime(&gpu_time, start, stop);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			printf("Processing time : %f ms\n", gpu_time);

			normal_generator->colorization(distorted_normal, distort_normal_colorization);
			
			cv::imshow("distort_depth", distorted_depthFrame*depth_scale_for_visualization);
			cv::imshow("distorted_normal", distort_normal_colorization);

			k4a_image_release(depth_image);			
		}

		//
		const int key = cv::waitKey(1);
	}


Exit:
	if (device != NULL)
	{
		k4a_device_close(device);
	}

	return 0;
}


cv::Mat k4a_get_mat(k4a_image_t& src, bool deep_copy)
{
	k4a_image_reference(src);
	return get_mat_from_k4a(k4a::image(src), deep_copy);
}

cv::Mat get_mat_from_k4a(k4a::image& src, bool deep_copy)
{
	assert(src.get_size() != 0);

	cv::Mat mat;
	const int32_t width = src.get_width_pixels();
	const int32_t height = src.get_height_pixels();

	const k4a_image_format_t format = src.get_format();
	switch (format)
	{
	case k4a_image_format_t::K4A_IMAGE_FORMAT_COLOR_MJPG:
	{
		// NOTE: this is slower than color formats.
		std::vector<uint8_t> buffer(src.get_buffer(), src.get_buffer() + src.get_size());
		mat = cv::imdecode(buffer, cv::IMREAD_COLOR);
		//cv::cvtColor(mat, mat, cv::COLOR_RGB2XYZ);
		break;
	}
	case k4a_image_format_t::K4A_IMAGE_FORMAT_COLOR_NV12:
	{
		cv::Mat nv12 = cv::Mat(height + height / 2, width, CV_8UC1, src.get_buffer()).clone();
		cv::cvtColor(nv12, mat, cv::COLOR_YUV2BGR);
		break;
	}
	case k4a_image_format_t::K4A_IMAGE_FORMAT_COLOR_YUY2:
	{
		cv::Mat yuy2 = cv::Mat(height, width, CV_8UC2, src.get_buffer()).clone();
		cv::cvtColor(yuy2, mat, cv::COLOR_YUV2BGR);
		break;
	}
	case k4a_image_format_t::K4A_IMAGE_FORMAT_COLOR_BGRA32:
	{
		mat = deep_copy ? cv::Mat(height, width, CV_8UC4, src.get_buffer()).clone()
			: cv::Mat(height, width, CV_8UC4, src.get_buffer());
		cv::cvtColor(mat, mat, cv::COLOR_BGRA2BGR);
		break;
	}
	case k4a_image_format_t::K4A_IMAGE_FORMAT_DEPTH16:
	case k4a_image_format_t::K4A_IMAGE_FORMAT_IR16:
	{
		mat = deep_copy ? cv::Mat(height, width, CV_16UC1, reinterpret_cast<uint16_t*>(src.get_buffer())).clone()
			: cv::Mat(height, width, CV_16UC1, reinterpret_cast<uint16_t*>(src.get_buffer()));
		break;
	}
	case k4a_image_format_t::K4A_IMAGE_FORMAT_CUSTOM8:
	{
		mat = cv::Mat(height, width, CV_8UC1, src.get_buffer()).clone();
		break;
	}
	case k4a_image_format_t::K4A_IMAGE_FORMAT_CUSTOM:
	{
		// NOTE: This is opencv_viz module format (cv::viz::WCloud).
		const int16_t* buffer = reinterpret_cast<int16_t*>(src.get_buffer());
		mat = cv::Mat(height, width, CV_32FC3, cv::Vec3f::all(std::numeric_limits<float>::quiet_NaN()));
		mat.forEach<cv::Vec3f>(
			[&](cv::Vec3f& point, const int32_t* position) {
			const int32_t index = (position[0] * width + position[1]) * 3;
			point = cv::Vec3f(buffer[index + 0], buffer[index + 1], buffer[index + 2]);
		}
		);
		break;
	}
	default:
		throw k4a::error("Failed to convert this format!");
		break;
	}
	return mat;
}

void release_k4a_capture(k4a_capture_t& c)
{
	if (c != NULL)						
	{
		k4a_capture_release(c);			
		c = NULL;						
	}
}