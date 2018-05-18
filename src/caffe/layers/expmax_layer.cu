#include <vector>
#include <cfloat>

#include "caffe/layers/expmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
  __global__ void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

template <typename Dtype>
void ExpMaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  Dtype* scale_data = scale_.mutable_gpu_data();
  int channels = top[0]->shape(1); // softmax_axis_ = 1
  const int W = bottom[0]->width();
	const int H = bottom[0]->height();
  int outer_num_ = 1;
  int inner_num_ = W*H;
  caffe_copy(count, bottom_data, top_data);

  // We need to subtract the max to avoid numerical issues, compute the exp.
  kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
      scale_data);

  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
      scale_data, top_data);
  
  if (inner_scale_ == Dtype(1)) {
    caffe_gpu_exp(count, top_data, top_data);
  } else {
    caffe_gpu_scale(count, inner_scale_, top_data, top_data);
    caffe_gpu_exp(count, top_data, top_data);
  }
  if (outer_scale_ != Dtype(1)) {
    caffe_gpu_scal(count, outer_scale_, top_data);
  }
}

template <typename Dtype>
void ExpMaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const int count = bottom[0]->count();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_mul(count, top_data, top_diff, bottom_diff);
  if (inner_scale_ != Dtype(1)) {
    caffe_gpu_scal(count, inner_scale_, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ExpMaxLayer);


}  // namespace caffe
