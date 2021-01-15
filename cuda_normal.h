#ifndef CUDA_NORMAL_H
#define CUDA_NORMAL_H

#include <k4a/k4a.hpp>
#include <opencv2/core.hpp>

#pragma comment (lib ,"cudart.lib")
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <device_functions.h>
#include <vector_functions.h>
#include <vector_types.h>

#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>

struct cuda_intrinsics
{
	int width;
	int height;

	float fx;
	float fy;
	float cx;
	float cy;

	float k1;
	float k2;
	float k3;
	float k4;
	float k5;
	float k6;
	float codx; // center of distortion is set to 0 for Brown Conrady model
	float cody;
	float p1;
	float p2;
	float metric_radius;

	cuda_intrinsics()
	{
		width = height = fx = fy = cx = cy = k1 = k2 = k3 = k4 = k5 = k6 = codx = cody = p1 = p2 = metric_radius = 0.0;
	}

	cuda_intrinsics(const k4a_calibration_camera_t& in)
	{
		this->width = in.resolution_width;
		this->height = in.resolution_height;

		this->fx = in.intrinsics.parameters.param.fx;
		this->fy = in.intrinsics.parameters.param.fy;
		this->cx = in.intrinsics.parameters.param.cx;
		this->cy = in.intrinsics.parameters.param.cy;

		this->k1 = in.intrinsics.parameters.param.k1;
		this->k2 = in.intrinsics.parameters.param.k2;
		this->k3 = in.intrinsics.parameters.param.k3;
		this->k4 = in.intrinsics.parameters.param.k4;
		this->k5 = in.intrinsics.parameters.param.k5;
		this->k6 = in.intrinsics.parameters.param.k6;
		this->codx = in.intrinsics.parameters.param.codx; // center of distortion is set to 0 for Brown Conrady model
		this->cody = in.intrinsics.parameters.param.cody;
		this->p1 = in.intrinsics.parameters.param.p1;
		this->p2 = in.intrinsics.parameters.param.p2;

		this->metric_radius = in.metric_radius;
	}
};


class cuda_normal_map
{
public:
	cuda_normal_map(const int w, const int h, const float scale, const k4a_calibration_t& calibration)
	{
		this->update(w, h, scale, calibration);
	}

	void update(const int w, const int h, const float scale, const k4a_calibration_t& calibration);

	void generate(const cv::Mat& depth, cv::Mat& normal);
	void colorization(const cv::Mat& in, cv::Mat& out);

private:	

	int width;
	int height;

	float depth_scale;

	std::shared_ptr<cuda_intrinsics> d_depth_intrinsic;
	std::shared_ptr<uint16_t> d_depth;
	std::shared_ptr<float3> d_pointcloud;
	std::shared_ptr<float3> d_normal_map;

};



#endif
