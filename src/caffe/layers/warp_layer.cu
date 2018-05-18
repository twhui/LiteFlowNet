#include <vector>
#include <iostream>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/warp_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ForwardGPU(const int nthreads, const int C, const int W, const int H,
		const Dtype* coords, const Dtype* U, Dtype* V)
{
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int HW = H*W;
		const int w = index % W;
		const int h = (index / W) % H;
		const int c = (index / HW) % C;
		const int n = index / (HW * C);

    const int coords_index = 2*n*HW + h*W + w;
		const Dtype x = coords[coords_index];
	  const Dtype y = coords[coords_index + HW];
    const Dtype* U_slice = U + (n * C + c) * HW;
		int px, py;

	  px = floor(x);
		py = floor(y);
	  if(px >= 0 && px < W && py >= 0 && py < H) {
	  	V[index] += (1 - (x - px)) * (1 - (y - py)) * U_slice[py * W + px];
	  }

	  px = floor(x) + 1;
		py = floor(y);
	  if(px >= 0 && px < W && py >= 0 && py < H) {
	  	V[index] += (1 - (px - x)) * (1 - (y - py)) * U_slice[py * W + px];
	  }

	  px = floor(x);
		py = floor(y) + 1;
	  if(px >= 0 && px < W && py >= 0 && py < H) {
	  	V[index] += (1 - (x - px)) * (1 - (py - y)) * U_slice[py * W + px];
	  }

	  px = floor(x) + 1;
		py = floor(y) + 1;
	  if(px >= 0 && px < W && py >= 0 && py < H) {
	  	V[index] += (1 - (px - x)) * (1 - (py - y)) * U_slice[py * W + px];
	  }
  }
}

template <typename Dtype>
void WarpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	const Dtype* U = bottom[0]->gpu_data();
	const Dtype* coords = bottom[1]->gpu_data();
	Dtype* V = top[0]->mutable_gpu_data();

	const int N = bottom[0]->num();
	const int C = bottom[0]->channels();
	const int W = bottom[0]->width();
	const int H = bottom[0]->height();
	const int count = top[0]->count();

	caffe_gpu_set(count, (Dtype)0., V);

	ForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		count, C, W, H, coords, U, V);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void BackwardGPU_dU(const int nthreads, const int C, const int W, const int H,
	const Dtype* coords, const Dtype* dV, Dtype* dU) {

	CUDA_KERNEL_LOOP(index, nthreads) {

		const int HW = H*W;
		const int w = index % W;
		const int h = (index / W) % H;
		const int c = (index / HW) % C;
		const int n = index / (HW * C);

    const int coords_index = 2*n*HW + h*W + w;
		const Dtype x = coords[coords_index];
	  const Dtype y = coords[coords_index + HW];
    Dtype* dU_slice = dU + (n * C + c) * HW;
		int px, py;

	  px = floor(x);
		py = floor(y);
	  if(px >= 0 && px < W && py >= 0 && py < H) {
			dU_slice[py * W + px] += (1 - (x - px)) * (1 - (y - py)) * dV[index];
	  }

	  px = floor(x) + 1;
		py = floor(y);
	  if(px >= 0 && px < W && py >= 0 && py < H) {
			dU_slice[py * W + px] += (1 - (px - x)) * (1 - (y - py)) * dV[index];
	  }

	  px = floor(x);
		py = floor(y) + 1;
	  if(px >= 0 && px < W && py >= 0 && py < H) {
			dU_slice[py * W + px] += (1 - (x - px)) * (1 - (py - y)) * dV[index];
	  }

	  px = floor(x) + 1;
		py = floor(y) + 1;
	  if(px >= 0 && px < W && py >= 0 && py < H) {

			dU_slice[py * W + px] += (1 - (px - x)) * (1 - (py - y)) * dV[index];
	  }
	}
}

template <typename Dtype>
__global__ void BackwardGPU_dp(const int nthreads, const int c, const int C, const int W, const int H,
		const Dtype* coords, const Dtype* dV, const Dtype* U, Dtype* dp)
{
	CUDA_KERNEL_LOOP(index_, nthreads) {

		const int HW = H*W;
		const int w = index_ % W;
		const int h = (index_ / W) % H;
		const int n = index_ / HW;
		const int index = (n*C + c)*HW + (h*W + w);

    const int coords_index = 2*n*HW+ h*W + w;
		const Dtype x = coords[coords_index];
	  const Dtype y = coords[coords_index + HW];
    const Dtype* U_slice = U + (n * C + c) * HW;
		int px, py;
		Dtype dpx = (Dtype)0.;
		Dtype dpy = (Dtype)0.;

		px = floor(x);
		py = floor(y);
		if(px >= 0 && px < W && py >= 0 && py < H) {
			dpx -= (1 - (y - py)) * U_slice[py * W + px];
			dpy -= (1 - (x - px)) * U_slice[py * W + px];
		}

		px = floor(x);
		py = floor(y) + 1;
		if(px >= 0 && px < W && py >= 0 && py < H) {
			dpx -= (1 - (py - y)) * U_slice[py * W + px];
			dpy += (1 - (x - px)) * U_slice[py * W + px];
		}

		px = floor(x) + 1;
		py = floor(y);
		if(px >= 0 && px < W && py >= 0 && py < H) {
			dpx += (1 - (y - py)) * U_slice[py * W + px];
			dpy -= (1 - (px - x)) * U_slice[py * W + px];
		}

		px = floor(x) + 1;
		py = floor(y) + 1;
		if(px >= 0 && px < W && py >= 0 && py < H) {
			dpx += (1 - (py - y)) * U_slice[py * W + px];
			dpy += (1 - (px - x)) * U_slice[py * W + px];
		}

		const int dp_index = 2 * n * HW + h * W + w;

		dp[dp_index] += dpx * dV[index];
		dp[dp_index + HW] += dpy * dV[index];
	}
}

template <typename Dtype>
void WarpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype* dV = top[0]->gpu_diff();
	const Dtype* coords = bottom[1]->gpu_data();
	const Dtype* U = bottom[0]->gpu_data();

	const int N = bottom[0]->num();
	const int C = bottom[0]->channels();
	const int W = bottom[0]->width();
	const int H = bottom[0]->height();

	if(propagate_down[0]) {
		Dtype* dU = bottom[0]->mutable_gpu_diff();
		caffe_gpu_set(bottom[0]->count(), (Dtype)0., dU);

		BackwardGPU_dU<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
			top[0]->count(), C, W, H, coords, dV, dU);
		CUDA_POST_KERNEL_CHECK;
	}

	if(propagate_down[1]) {
		Dtype* dp = bottom[1]->mutable_gpu_diff();
		caffe_gpu_set(bottom[1]->count(), (Dtype)0., dp);
		const int count = N*H*W;

		for (int c = 0; c < C; c++) {
			BackwardGPU_dp<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				count, c, C, W, H, coords, dV, U, dp);
			CUDA_POST_KERNEL_CHECK;
		}
	}

}

INSTANTIATE_LAYER_GPU_FUNCS(WarpLayer);

}	// namespace caffe
