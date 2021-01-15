#include "cuda_normal.h"

#define BLOCK_SIZE (16)

__device__ __forceinline__ float3
f3_add(const float3& f1, const float3& f2) {
	return make_float3(f2.x + f1.x, f2.y + f1.y, f2.z + f1.z);
}

__device__ __forceinline__ float3
f3_sub(const float3& f1, const float3& f2) {
	return make_float3(f1.x - f2.x, f1.y - f2.y, f1.z - f2.z);
}

__device__ __forceinline__ float3
f3_div_elem(const float3& f1, const float3& f2) {
	return make_float3(f1.x / f2.x, f1.y / f2.y, f1.z / f2.z);
}

__device__ __host__ __forceinline__ float3
f3_div_elem(const float3& f, const dim3& i) {
	return make_float3(f.x / i.x, f.y / i.y, f.z / i.z);
}

__device__ __host__ __forceinline__ float3
f3_div_elem(const float3& f, const int& i) {
	return f3_div_elem(f, dim3(i, i, i));
}

__device__ __host__ __forceinline__ float
f3_norm(float3 vec) {
	return sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);
}

__device__ __forceinline__ float
f3_inner_product(const float3& vec1, const float3& vec2) {
	return (vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z);
}

__device__ __forceinline__ float3
f3_cross_product(const float3& vec1, const float3& vec2) {
	return make_float3(vec1.y * vec2.z - vec1.z * vec2.y
		, vec1.z * vec2.x - vec1.x * vec2.z
		, vec1.x * vec2.y - vec1.y * vec2.x);
}

__device__ __forceinline__ float3
f3_normalization(const float3& vec) {
	const float l = sqrt(vec.x*vec.x + vec.y*vec.y + vec.z*vec.z);

	if (l == 0)
		return make_float3(0, 0, 0);
	else
		return make_float3(vec.x / l, vec.y / l, vec.z / l);
}


__device__ __forceinline__ void
computeRoots2(const float& b, const float& c, float3& roots)
{
	roots.x = 0.f;
	float d = b * b - 4.f * c;
	if (d < 0.f) // no real roots!!!! THIS SHOULD NOT HAPPEN!
		d = 0.f;

	float sd = sqrtf(d);

	roots.z = 0.5f * (b + sd);
	roots.y = 0.5f * (b - sd);
}

__device__ __forceinline__ void
swap(float& a, float& b)
{
	const float temp = a;
	a = b;
	b = temp;
}

__device__ __forceinline__ void
computeRoots3(float c0, float c1, float c2, float3& roots)
{
	if (abs(c0) < 1.192092896e-07F)// one root is 0 -> quadratic equation
	{
		computeRoots2(c2, c1, roots);
	}
	else
	{
		const float s_inv3 = 1.f / 3.f;
		const float s_sqrt3 = sqrtf(3.f);
		// Construct the parameters used in classifying the roots of the equation
		// and in solving the equation for the roots in closed form.
		float c2_over_3 = c2 * s_inv3;
		float a_over_3 = (c1 - c2*c2_over_3)*s_inv3;
		if (a_over_3 > 0.f)
			a_over_3 = 0.f;

		float half_b = 0.5f * (c0 + c2_over_3 * (2.f * c2_over_3 * c2_over_3 - c1));

		float q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
		if (q > 0.f)
			q = 0.f;

		// Compute the eigenvalues by solving for the roots of the polynomial.
		float rho = sqrtf(-a_over_3);
		float theta = atan2(sqrtf(-q), half_b)*s_inv3;
		float cos_theta = __cosf(theta);
		float sin_theta = __sinf(theta);
		roots.x = c2_over_3 + 2.f * rho * cos_theta;
		roots.y = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
		roots.z = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);

		// Sort in increasing order.
		if (roots.x >= roots.y)
			swap(roots.x, roots.y);

		if (roots.y >= roots.z)
		{
			swap(roots.y, roots.z);

			if (roots.x >= roots.y)
				swap(roots.x, roots.y);
		}
		if (roots.x <= 0) // eigenval for symmetric positive semi-definite matrix can not be negative! Set it to 0
			computeRoots2(c2, c1, roots);
	}
}

__device__  __forceinline__ static bool
isMuchSmallerThan(float x, float y)
{
	// copied from <eigen>/include/Eigen/src/Core/NumTraits.h
	const float prec_sqr = 1.192092896e-07F * 1.192092896e-07F;
	return x * x <= prec_sqr * y * y;
}

static __forceinline__ __device__ float3
unitOrthogonal(const float3& src)
{
	float3 perp;
	/* Let us compute the crossed product of *this with a vector
	* that is not too close to being colinear to *this.
	*/

	/* unless the x and y coords are both close to zero, we can
	* simply take ( -y, x, 0 ) and normalize it.
	*/
	if (!isMuchSmallerThan(src.x, src.z) || !isMuchSmallerThan(src.y, src.z))
	{
		float invnm = rsqrtf(src.x*src.x + src.y*src.y);
		perp.x = -src.y * invnm;
		perp.y = src.x * invnm;
		perp.z = 0.0f;
	}
	/* if both x and y are close to zero, then the vector is close
	* to the z-axis, so it's far from colinear to the x-axis for instance.
	* So we take the crossed product with (1,0,0) and normalize it.
	*/
	else
	{
		float invnm = rsqrtf(src.z * src.z + src.y * src.y);
		perp.x = 0.0f;
		perp.y = -src.z * invnm;
		perp.z = src.y * invnm;
	}

	return perp;
}

__device__ void
solve_eigen_decomposition(float cov[6], float evecs[9], float3& evals)
{
	float max01 = fmaxf(abs(cov[0]), abs(cov[1]));
	float max23 = fmaxf(abs(cov[2]), abs(cov[3]));
	float max45 = fmaxf(abs(cov[4]), abs(cov[5]));
	float m0123 = fmaxf(max01, max23);
	float scale = fmaxf(max45, m0123);

	if (scale <= FLT_MIN)
		scale = 1.f;

	cov[0] /= scale;
	cov[1] /= scale;
	cov[2] /= scale;
	cov[3] /= scale;
	cov[4] /= scale;
	cov[5] /= scale;

	float c0 = cov[0] * cov[3] * cov[5]
		+ 2.f * cov[1] * cov[2] * cov[4]
		- cov[0] * cov[4] * cov[4]
		- cov[3] * cov[2] * cov[2]
		- cov[5] * cov[1] * cov[1];

	float c1 = cov[0] * cov[3] -
		cov[1] * cov[1] +
		cov[0] * cov[5] -
		cov[2] * cov[2] +
		cov[3] * cov[5] -
		cov[4] * cov[4];

	float c2 = cov[0] + cov[3] + cov[5];

	computeRoots3(c0, c1, c2, evals);

	if (evals.z - evals.x <= 1.192092896e-07F)
	{
		evecs[0] = evecs[4] = evecs[8] = 1.f;
		evecs[1] = evecs[2] = evecs[3] = evecs[5] = evecs[6] = evecs[7] = 0.f;
	}
	else if (evals.y - evals.x <= 1.192092896e-07F)
	{
		float3 row_tmp[3];
		row_tmp[0] = make_float3(cov[0] - evals.z, cov[1], cov[2]);
		row_tmp[1] = make_float3(cov[1], cov[3] - evals.z, cov[4]);
		row_tmp[2] = make_float3(cov[2], cov[4], cov[5] - evals.z);

		float3 vec_tmp_0 = f3_cross_product(row_tmp[0], row_tmp[1]);
		float3 vec_tmp_1 = f3_cross_product(row_tmp[0], row_tmp[2]);
		float3 vec_tmp_2 = f3_cross_product(row_tmp[1], row_tmp[2]);

		float len1 = f3_inner_product(vec_tmp_0, vec_tmp_0);
		float len2 = f3_inner_product(vec_tmp_1, vec_tmp_1);
		float len3 = f3_inner_product(vec_tmp_2, vec_tmp_2);

		if (len1 >= len2 && len1 >= len3)
		{
			const float sqr_len = rsqrtf(len1);

			evecs[6] = vec_tmp_0.x * sqr_len;
			evecs[7] = vec_tmp_0.y * sqr_len;
			evecs[8] = vec_tmp_0.z * sqr_len;
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			const float sqr_len = rsqrtf(len2);

			evecs[6] = vec_tmp_1.x * sqr_len;
			evecs[7] = vec_tmp_1.y * sqr_len;
			evecs[8] = vec_tmp_1.z * sqr_len;
		}
		else
		{
			const float sqr_len = rsqrtf(len3);

			evecs[6] = vec_tmp_2.x * sqr_len;
			evecs[7] = vec_tmp_2.y * sqr_len;
			evecs[8] = vec_tmp_2.z * sqr_len;
		}

		float3 evecs_2 = make_float3(evecs[6], evecs[7], evecs[8]);

		float3 evecs_1 = unitOrthogonal(evecs_2);

		evecs[3] = evecs_1.x;
		evecs[4] = evecs_1.y;
		evecs[5] = evecs_1.z;

		float3 evecs_0 = f3_cross_product(evecs_1, evecs_2);

		evecs[0] = evecs_0.x;
		evecs[1] = evecs_0.y;
		evecs[2] = evecs_0.z;

	}
	else if (evals.z - evals.y <= 1.192092896e-07F)
	{
		float3 row_tmp[3];
		row_tmp[0] = make_float3(cov[0] - evals.x, cov[1], cov[2]);
		row_tmp[1] = make_float3(cov[1], cov[3] - evals.x, cov[4]);
		row_tmp[2] = make_float3(cov[2], cov[4], cov[5] - evals.x);

		float3 vec_tmp_0 = f3_cross_product(row_tmp[0], row_tmp[1]);
		float3 vec_tmp_1 = f3_cross_product(row_tmp[0], row_tmp[2]);
		float3 vec_tmp_2 = f3_cross_product(row_tmp[1], row_tmp[2]);

		float len1 = f3_inner_product(vec_tmp_0, vec_tmp_0);
		float len2 = f3_inner_product(vec_tmp_1, vec_tmp_1);
		float len3 = f3_inner_product(vec_tmp_2, vec_tmp_2);

		if (len1 >= len2 && len1 >= len3)
		{
			const float sqr_len = rsqrtf(len1);

			evecs[0] = vec_tmp_0.x * sqr_len;
			evecs[1] = vec_tmp_0.y * sqr_len;
			evecs[2] = vec_tmp_0.z * sqr_len;

			//evecs[0] = vec_tmp[0] * rsqrtf(len1);
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			const float sqr_len = rsqrtf(len2);

			evecs[0] = vec_tmp_1.x * sqr_len;
			evecs[1] = vec_tmp_1.y * sqr_len;
			evecs[2] = vec_tmp_1.z * sqr_len;

			//evecs[0] = vec_tmp[1] * rsqrtf(len2);
		}
		else
		{
			const float sqr_len = rsqrtf(len3);

			evecs[0] = vec_tmp_2.x * sqr_len;
			evecs[1] = vec_tmp_2.y * sqr_len;
			evecs[2] = vec_tmp_2.z * sqr_len;

			//evecs[0] = vec_tmp[2] * rsqrtf(len3);
		}

		float3 evecs_0 = make_float3(evecs[0], evecs[1], evecs[2]);

		float3 evecs_1 = unitOrthogonal(evecs_0);

		evecs[3] = evecs_1.x;
		evecs[4] = evecs_1.y;
		evecs[5] = evecs_1.z;

		float3 evecs_2 = f3_cross_product(evecs_0, evecs_1);

		evecs[6] = evecs_2.x;
		evecs[7] = evecs_2.y;
		evecs[8] = evecs_2.z;

	}
	else
	{
		float3 row_tmp[3];
		row_tmp[0] = make_float3(cov[0] - evals.z, cov[1], cov[2]);
		row_tmp[1] = make_float3(cov[1], cov[3] - evals.z, cov[4]);
		row_tmp[2] = make_float3(cov[2], cov[4], cov[5] - evals.z);

		float3 vec_tmp_0 = f3_cross_product(row_tmp[0], row_tmp[1]);
		float3 vec_tmp_1 = f3_cross_product(row_tmp[0], row_tmp[2]);
		float3 vec_tmp_2 = f3_cross_product(row_tmp[1], row_tmp[2]);

		float len1 = f3_inner_product(vec_tmp_0, vec_tmp_0);
		float len2 = f3_inner_product(vec_tmp_1, vec_tmp_1);
		float len3 = f3_inner_product(vec_tmp_2, vec_tmp_2);

		float mmax[3];

		unsigned int min_el = 2;
		unsigned int max_el = 2;
		if (len1 >= len2 && len1 >= len3)
		{
			mmax[2] = len1;
			const float sqr_len = rsqrtf(len1);

			evecs[6] = vec_tmp_0.x * sqr_len;
			evecs[7] = vec_tmp_0.y * sqr_len;
			evecs[8] = vec_tmp_0.z * sqr_len;

			//evecs[2] = vec_tmp[0] * rsqrtf(len1);
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			mmax[2] = len2;

			const float sqr_len = rsqrtf(len2);

			evecs[6] = vec_tmp_1.x * sqr_len;
			evecs[7] = vec_tmp_1.y * sqr_len;
			evecs[8] = vec_tmp_1.z * sqr_len;

			//evecs[2] = vec_tmp[1] * rsqrtf(len2);
		}
		else
		{
			mmax[2] = len3;

			const float sqr_len = rsqrtf(len3);

			evecs[6] = vec_tmp_2.x * sqr_len;
			evecs[7] = vec_tmp_2.y * sqr_len;
			evecs[8] = vec_tmp_2.z * sqr_len;

			//evecs[2] = vec_tmp[2] * rsqrtf(len3);
		}

		row_tmp[0] = make_float3(cov[0] - evals.y, cov[1], cov[2]);
		row_tmp[1] = make_float3(cov[1], cov[3] - evals.y, cov[4]);
		row_tmp[2] = make_float3(cov[2], cov[4], cov[5] - evals.y);

		vec_tmp_0 = f3_cross_product(row_tmp[0], row_tmp[1]);
		vec_tmp_1 = f3_cross_product(row_tmp[0], row_tmp[2]);
		vec_tmp_2 = f3_cross_product(row_tmp[1], row_tmp[2]);

		len1 = f3_inner_product(vec_tmp_0, vec_tmp_0);
		len2 = f3_inner_product(vec_tmp_1, vec_tmp_1);
		len3 = f3_inner_product(vec_tmp_2, vec_tmp_2);


		if (len1 >= len2 && len1 >= len3)
		{
			mmax[1] = len1;

			const float sqr_len = rsqrtf(len1);

			evecs[3] = vec_tmp_0.x * sqr_len;
			evecs[4] = vec_tmp_0.y * sqr_len;
			evecs[5] = vec_tmp_0.z * sqr_len;

			//evecs[1] = vec_tmp[0] * rsqrtf(len1);

			min_el = len1 <= mmax[min_el] ? 1 : min_el;
			max_el = len1  > mmax[max_el] ? 1 : max_el;
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			mmax[1] = len2;

			const float sqr_len = rsqrtf(len2);

			evecs[3] = vec_tmp_1.x * sqr_len;
			evecs[4] = vec_tmp_1.y * sqr_len;
			evecs[5] = vec_tmp_1.z * sqr_len;

			//evecs[1] = vec_tmp[1] * rsqrtf(len2);
			min_el = len2 <= mmax[min_el] ? 1 : min_el;
			max_el = len2  > mmax[max_el] ? 1 : max_el;
		}
		else
		{
			mmax[1] = len3;

			const float sqr_len = rsqrtf(len3);

			evecs[3] = vec_tmp_2.x * sqr_len;
			evecs[4] = vec_tmp_2.y * sqr_len;
			evecs[5] = vec_tmp_2.z * sqr_len;

			//evecs[1] = vec_tmp[2] * rsqrtf(len3);
			min_el = len3 <= mmax[min_el] ? 1 : min_el;
			max_el = len3 >  mmax[max_el] ? 1 : max_el;
		}

		row_tmp[0] = make_float3(cov[0] - evals.x, cov[1], cov[2]);
		row_tmp[1] = make_float3(cov[1], cov[3] - evals.x, cov[4]);
		row_tmp[2] = make_float3(cov[2], cov[4], cov[5] - evals.x);

		vec_tmp_0 = f3_cross_product(row_tmp[0], row_tmp[1]);
		vec_tmp_1 = f3_cross_product(row_tmp[0], row_tmp[2]);
		vec_tmp_2 = f3_cross_product(row_tmp[1], row_tmp[2]);

		len1 = f3_inner_product(vec_tmp_0, vec_tmp_0);
		len2 = f3_inner_product(vec_tmp_1, vec_tmp_1);
		len3 = f3_inner_product(vec_tmp_2, vec_tmp_2);


		if (len1 >= len2 && len1 >= len3)
		{
			mmax[0] = len1;

			const float sqr_len = rsqrtf(len1);

			evecs[0] = vec_tmp_0.x * sqr_len;
			evecs[1] = vec_tmp_0.y * sqr_len;
			evecs[2] = vec_tmp_0.z * sqr_len;

			min_el = len3 <= mmax[min_el] ? 0 : min_el;
			max_el = len3  > mmax[max_el] ? 0 : max_el;
		}
		else if (len2 >= len1 && len2 >= len3)
		{
			mmax[0] = len2;

			const float sqr_len = rsqrtf(len2);

			evecs[0] = vec_tmp_1.x * sqr_len;
			evecs[1] = vec_tmp_1.y * sqr_len;
			evecs[2] = vec_tmp_1.z * sqr_len;

			min_el = len3 <= mmax[min_el] ? 0 : min_el;
			max_el = len3  > mmax[max_el] ? 0 : max_el;
		}
		else
		{
			mmax[0] = len3;

			const float sqr_len = rsqrtf(len3);

			evecs[0] = vec_tmp_2.x * sqr_len;
			evecs[1] = vec_tmp_2.y * sqr_len;
			evecs[2] = vec_tmp_2.z * sqr_len;

			min_el = len3 <= mmax[min_el] ? 0 : min_el;
			max_el = len3  > mmax[max_el] ? 0 : max_el;
		}

		unsigned mid_el = 3 - min_el - max_el;

		const int min_el_1 = ((min_el + 1) % 3) * 3;
		const int min_el_2 = ((min_el + 2) % 3) * 3;
		const int mid_el_1 = ((mid_el + 1) % 3) * 3;
		const int mid_el_2 = ((mid_el + 2) % 3) * 3;

		float3 evecs_min_el = f3_normalization(f3_cross_product(
			make_float3(evecs[min_el_1 + 0], evecs[min_el_1 + 1], evecs[min_el_1 + 2])
			, make_float3(evecs[min_el_2 + 0], evecs[min_el_2 + 1], evecs[min_el_2 + 2])));

		float3 evecs_mid_el = f3_normalization(f3_cross_product(
			make_float3(evecs[mid_el_1 + 0], evecs[mid_el_1 + 1], evecs[mid_el_1 + 2])
			, make_float3(evecs[mid_el_2 + 0], evecs[mid_el_2 + 1], evecs[mid_el_2 + 2])));

		evecs[min_el * 3 + 0] = evecs_min_el.x;
		evecs[min_el * 3 + 1] = evecs_min_el.y;
		evecs[min_el * 3 + 2] = evecs_min_el.z;

		evecs[mid_el * 3 + 0] = evecs_mid_el.x;
		evecs[mid_el * 3 + 1] = evecs_mid_el.y;
		evecs[mid_el * 3 + 2] = evecs_mid_el.z;

	}

	evals = make_float3(evals.x * scale, evals.y * scale, evals.z * scale);
}



__device__ bool
transformation_project(const cuda_intrinsics* intrinsic,
	const float xy[2],
	float uv[2],
	int& valid,
	float J_xy[2 * 2])
{	
	const float fx = intrinsic->fx;
	const float cx = intrinsic->cx;
	const float fy = intrinsic->fy;
	const float cy = intrinsic->cy;

	const float k1 = intrinsic->k1;
	const float k2 = intrinsic->k2;
	const float k3 = intrinsic->k3;
	const float k4 = intrinsic->k4;
	const float k5 = intrinsic->k5;
	const float k6 = intrinsic->k6;

	const float codx = intrinsic->codx; // center of distortion is set to 0 for Brown Conrady model
	const float cody = intrinsic->cody;

	const float p1 = intrinsic->p1;
	const float p2 = intrinsic->p2;

	const float max_radius_for_projection = intrinsic->metric_radius;

	valid = 1;

	float xp = xy[0] - codx;
	float yp = xy[1] - cody;

	float xp2 = xp * xp;
	float yp2 = yp * yp;
	float xyp = xp * yp;
	float rs = xp2 + yp2;
	if (rs > max_radius_for_projection * max_radius_for_projection)
	{
		valid = 0;
		return true;
	}
	float rss = rs * rs;
	float rsc = rss * rs;
	float a = 1.f + k1 * rs + k2 * rss + k3 * rsc;
	float b = 1.f + k4 * rs + k5 * rss + k6 * rsc;
	float bi;
	if (b != 0.f)
	{
		bi = 1.f / b;
	}
	else
	{
		bi = 1.f;
	}
	float d = a * bi;

	float xp_d = xp * d;
	float yp_d = yp * d;

	float rs_2xp2 = rs + 2.f * xp2;
	float rs_2yp2 = rs + 2.f * yp2;

	bool RATIONAL_6KT_MODE = false;

	if (RATIONAL_6KT_MODE)
	{
		xp_d += rs_2xp2 * p2 + xyp * p1;
		yp_d += rs_2yp2 * p1 + xyp * p2;
	}
	else
	{
		// the only difference from Rational6ktCameraModel is 2 multiplier for the tangential coefficient term xyp*p1
		// and xyp*p2
		xp_d += rs_2xp2 * p2 + 2.f * xyp * p1;
		yp_d += rs_2yp2 * p1 + 2.f * xyp * p2;
	}

	float xp_d_cx = xp_d + codx;
	float yp_d_cy = yp_d + cody;

	uv[0] = xp_d_cx * fx + cx;
	uv[1] = yp_d_cy * fy + cy;

	if (J_xy == 0)
	{
		return true;
	}

	// compute Jacobian matrix
	float dudrs = k1 + 2.f * k2 * rs + 3.f * k3 * rss;
	// compute d(b)/d(r^2)
	float dvdrs = k4 + 2.f * k5 * rs + 3.f * k6 * rss;
	float bis = bi * bi;
	float dddrs = (dudrs * b - a * dvdrs) * bis;

	float dddrs_2 = dddrs * 2.f;
	float xp_dddrs_2 = xp * dddrs_2;
	float yp_xp_dddrs_2 = yp * xp_dddrs_2;
	// compute d(u)/d(xp)
	if (RATIONAL_6KT_MODE)
	{
		J_xy[0] = fx * (d + xp * xp_dddrs_2 + 6.f * xp * p2 + yp * p1);
		J_xy[1] = fx * (yp_xp_dddrs_2 + 2.f * yp * p2 + xp * p1);
		J_xy[2] = fy * (yp_xp_dddrs_2 + 2.f * xp * p1 + yp * p2);
		J_xy[3] = fy * (d + yp * yp * dddrs_2 + 6.f * yp * p1 + xp * p2);
	}
	else
	{
		J_xy[0] = fx * (d + xp * xp_dddrs_2 + 6.f * xp * p2 + 2.f * yp * p1);
		J_xy[1] = fx * (yp_xp_dddrs_2 + 2.f * yp * p2 + 2.f * xp * p1);
		J_xy[2] = fy * (yp_xp_dddrs_2 + 2.f * xp * p1 + 2.f * yp * p2);
		J_xy[3] = fy * (d + yp * yp * dddrs_2 + 6.f * yp * p1 + 2.f * xp * p2);
	}

	return true;
}

__device__ float3 
deproject_pixel(const int target_x, const int target_y, const float depth, const cuda_intrinsics* intrinsic)
{
	const float fx = intrinsic->fx;
	const float cx = intrinsic->cx;
	const float fy = intrinsic->fy;
	const float cy = intrinsic->cy;

	const float k1 = intrinsic->k1;
	const float k2 = intrinsic->k2;
	const float k3 = intrinsic->k3;
	const float k4 = intrinsic->k4;
	const float k5 = intrinsic->k5;
	const float k6 = intrinsic->k6;

	const float codx = intrinsic->codx; // center of distortion is set to 0 for Brown Conrady model
	const float cody = intrinsic->cody;

	const float p1 = intrinsic->p1;
	const float p2 = intrinsic->p2;

	const float max_radius_for_projection = intrinsic->metric_radius;

	float xp_d = (float)(target_x - cx) / fx - codx;
	float yp_d = (float)(target_y - cy) / fy - cody;

	float rs = xp_d * xp_d + yp_d * yp_d;
	float rss = rs * rs;
	float rsc = rss * rs;
	float a = 1.f + k1 * rs + k2 * rss + k3 * rsc;
	float b = 1.f + k4 * rs + k5 * rss + k6 * rsc;
	float ai;
	if (a != 0.f)
	{
		ai = 1.f / a;
	}
	else
	{
		ai = 1.f;
	}
	float di = ai * b;

	float xy[2];

	xy[0] = xp_d * di;
	xy[1] = yp_d * di;

	// approximate correction for tangential params
	float two_xy = 2.f * xy[0] * xy[1];
	float xx = xy[0] * xy[0];
	float yy = xy[1] * xy[1];

	xy[0] -= (yy + 3.f * xx) * p2 + two_xy * p1;
	xy[1] -= (xx + 3.f * yy) * p1 + two_xy * p2;

	// add on center of distortion
	xy[0] += codx;
	xy[1] += cody;


	int valid = 1;
	const int max_passes = 20;
	float Jinv[2 * 2];
	float best_xy[2] = { 0.f, 0.f };
	float best_err = FLT_MAX;

	for (unsigned int pass = 0; pass < max_passes; ++pass)
	{
		float p[2];
		float J[2 * 2];

		if (!transformation_project(intrinsic, xy, p, valid, J) || (valid == 0))
		{
			return float3{ 0.0f,0.0f,0.0f };
		}

		float err_x = target_x - p[0];
		float err_y = target_y - p[1];
		float err = err_x * err_x + err_y * err_y;
		if (err >= best_err)
		{
			xy[0] = best_xy[0];
			xy[1] = best_xy[1];
			break;
		}

		best_err = err;
		best_xy[0] = xy[0];
		best_xy[1] = xy[1];

		float detJ = J[0] * J[3] - J[1] * J[2];
		float inv_detJ = 1.f / detJ;

		Jinv[0] = inv_detJ * J[3];
		Jinv[3] = inv_detJ * J[0];
		Jinv[1] = -inv_detJ * J[1];
		Jinv[2] = -inv_detJ * J[2];

		if (pass + 1 == max_passes || best_err < 1e-22f)
		{
			break;
		}

		float dx = Jinv[0] * err_x + Jinv[1] * err_y;
		float dy = Jinv[2] * err_x + Jinv[3] * err_y;

		xy[0] += dx;
		xy[1] += dy;
	}

	if (best_err > 1e-6f)
	{
		valid = 0;
	}

	float3 output = float3{ xy[0] * depth, xy[1] * depth, depth };

	return output;
}

__global__ void
depth2pointcloud(float3* pointcloud_map, const uint16_t *depth, const cuda_intrinsics* intrinsic, const float depth_scale, const int im_width, const int im_height)
{
	// pixel index
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	const int pixel_margin = 1;
	const int img_size = im_width*im_height;

	if ((pixel_margin <= x) && (x <= im_width - 1 - pixel_margin) && (pixel_margin <= y) && (y <= im_height - 1 - pixel_margin))
	{
		pointcloud_map[y* im_width + x] = deproject_pixel(x, y, depth[y * im_width + x] * depth_scale, intrinsic);
	}
}


__global__ void
pointcloud2normal_map(float3* normal_map, const float3* pointcloud_map, const int im_width, const int im_height)
{
	// pixel index
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	const int kx = 7;
	const int ky = 7;
	const int STEP = 1;

	const int camera_idx = blockIdx.z;

	const int img_size = im_width*im_height;
	const int img_offset = (camera_idx*img_size);

	const int pixel_margin = 1;

	if ((pixel_margin <= x) && (x <= im_width - 1 - pixel_margin) && (pixel_margin <= y) && (y <= im_height - 1 - pixel_margin))
	{
		const int ty = min(y - ky / 2 + ky, im_height - 1);
		const int tx = min(x - kx / 2 + kx, im_width - 1);

		float3 centroid = make_float3(0.f, 0.f, 0.f);
		int counter = 0;
		for (int cy = max(y - ky / 2, 0); cy < ty; cy += STEP)
		{
			for (int cx = max(x - kx / 2, 0); cx < tx; cx += STEP)
			{
				const float3 vertices = pointcloud_map[img_offset + (cy)* im_width + (cx)];

				if (vertices.z > 0)
				{
					centroid = f3_add(centroid, vertices);
					++counter;
				}
			}
		}

		if (counter < kx * ky / 2)
		{
			normal_map[img_offset + (y)* im_width + (x)] = make_float3(0.f, 0.f, 0.f);
			return;
		}

		centroid = f3_div_elem(centroid, counter);

		float cov[] = { 0, 0, 0, 0, 0, 0 };

		for (int cy = max(y - ky / 2, 0); cy < ty; cy += STEP)
		{
			for (int cx = max(x - kx / 2, 0); cx < tx; cx += STEP)
			{
				const float3 vertices = pointcloud_map[img_offset + (cy)* im_width + (cx)];

				if (vertices.z > 0)
				{
					float3 d = f3_sub(vertices, centroid);

					cov[0] += d.x * d.x;               //cov (0, 0)
					cov[1] += d.x * d.y;               //cov (0, 1)
					cov[2] += d.x * d.z;               //cov (0, 2)
					cov[3] += d.y * d.y;               //cov (1, 1)
					cov[4] += d.y * d.z;               //cov (1, 2)
					cov[5] += d.z * d.z;               //cov (2, 2)
				}
			}
		}

		float evecs[9];
		float3 evals;
		solve_eigen_decomposition(cov, evecs, evals);

		float3 n = f3_normalization(make_float3(evecs[0], evecs[1], evecs[2]));

		float3 point_vec = f3_normalization(pointcloud_map[img_offset + y* im_width + x]);

		float inner = f3_inner_product(n, point_vec);

		if (inner < 0) {
			normal_map[(camera_idx*img_size) + y* im_width + x] = f3_normalization(make_float3(evecs[0], evecs[1], evecs[2]));
		}
		else
		{
			normal_map[(camera_idx*img_size) + y* im_width + x] = f3_normalization(make_float3(-evecs[0], -evecs[1], -evecs[2]));
		}		

	}
}


static inline int divUp(int total, int grain) {
	return (total + grain - 1) / grain;
}

template<typename  T>
std::shared_ptr<T> make_device_copy(T obj)
{
	T* d_data;
	auto res = cudaMalloc(&d_data, sizeof(T));
	if (res != cudaSuccess)
		throw std::runtime_error("cudaMalloc failed status: " + res);
	cudaMemcpy(d_data, &obj, sizeof(T), cudaMemcpyHostToDevice);
	return std::shared_ptr<T>(d_data, [](T* data) { cudaFree(data); release_memory(data); });
}

template<typename  T>
std::shared_ptr<T> alloc_dev(int elements)
{
	T* d_data;
	auto res = cudaMalloc(&d_data, sizeof(T) * elements);
	if (res != cudaSuccess)
		throw std::runtime_error("cudaMalloc failed status: " + res);
	return std::shared_ptr<T>(d_data, [](T* p) { cudaFree(p); release_memory(p); });
}

template<class T>
void release_memory(T& obj)
{
	obj = nullptr;
}

void cuda_normal_map::update(const int w, const int h, const float scale, const k4a_calibration_t& calibration)
{
	this->width = w;
	this->height = h;
	this->depth_scale = scale;

	this->d_depth = alloc_dev<uint16_t>(this->width * this->height);
	this->d_pointcloud = alloc_dev<float3>(this->width * this->height);
	this->d_normal_map = alloc_dev<float3>(this->width * this->height);

	cuda_intrinsics depth_intrinsic(calibration.depth_camera_calibration);
	this->d_depth_intrinsic = make_device_copy(depth_intrinsic);
}

void cuda_normal_map::generate(const cv::Mat& depth, cv::Mat& normal)
{
	const int img_width = this->width;
	const int img_height = this->height;

	normal.create(cv::Size(img_width, img_height), CV_32FC3);

	thrust::device_ptr<float3> ptr_pointcloud(this->d_pointcloud.get());
	thrust::fill(ptr_pointcloud, ptr_pointcloud + (img_width * img_height), float3{ 0.0f, 0.0f ,0.0f });

	cudaMemcpy(this->d_depth.get(), (uint16_t*)depth.data, sizeof(uint16_t)*(img_width * img_height), cudaMemcpyHostToDevice);

	dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	depth2pointcloud << <dim3(divUp(img_width, block.x), divUp(img_height, block.y)), block >> >
		(this->d_pointcloud.get(), this->d_depth.get(), this->d_depth_intrinsic.get(), this->depth_scale, img_width, img_height);

	pointcloud2normal_map << <dim3(divUp(img_width, block.x), divUp(img_height, block.y)), block >> >
		(this->d_normal_map.get(), this->d_pointcloud.get(), img_width, img_height);

	cudaMemcpy((float3*)normal.data, this->d_normal_map.get(), sizeof(float3)*(img_width * img_height), cudaMemcpyDeviceToHost);	

}


void cuda_normal_map::colorization(const cv::Mat& in, cv::Mat& out)
{
	const int img_width = in.cols;
	const int img_height = in.rows;
	const int img_size = img_width*img_height;

	out.create(cv::Size(img_width, img_height), CV_8UC3);
	
	for (int idx = 0; idx < img_size; ++idx)
	{
		const cv::Vec3f normal_vector = in.at<cv::Vec3f>(idx);

		out.at<cv::Vec3b>(idx) = cv::Vec3b((normal_vector.val[0] + 1.0f) * 0.5f * 255, (normal_vector.val[1] + 1.0f) * 0.5f * 255, (normal_vector.val[2] + 1.0f) * 0.5f * 255);
	}

}