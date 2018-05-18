#include <vector>
#include <iostream>
#include "caffe/layers/warp_layer.hpp"

namespace caffe {

template<typename Dtype>
void WarpLayer< Dtype >::LayerSetUp(const vector< Blob< Dtype >* >& bottom, const vector< Blob< Dtype >* >& top)
{
	// Nothing to setup for
}

template<typename Dtype>
void WarpLayer< Dtype >::Reshape(const vector< Blob< Dtype >* >& bottom, const vector< Blob< Dtype >* >& top)
{
	CHECK_EQ(bottom.size(), 2) << "warp only supports two bottoms";
	CHECK_EQ(top.size(), 1) << "warp only supports one top";
	CHECK_EQ(bottom[0]->num(),
		bottom[1]->num()) << "warp requires num to be the same for both bottom blobs";
	CHECK_EQ(bottom[1]->channels(), 2) << "warp requires coords blob (bottom[1]) to have only 2 channels";
	top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
		bottom[1]->height(), bottom[1]->width());
}

template<typename Dtype>
inline Dtype getvalue(Dtype* V, int x, int y, int c, int n, const int C, const int W, const int H) {
	if ( x < 0 || x > W - 1 || y < 0 || y > H - 1 )
		return 0;
	return V[((n*C+c)*H+y)*W+x];
}

template<typename Dtype>
inline void addvalue(Dtype* V, int x, int y, int c, int n, const int C, const int W, const int H, const Dtype v) {
	if ( x < 0 || x > W - 1 || y < 0 || y > H - 1 )
		return;
	V[((n*C+c)*H+y)*W+x] += v;
}

template <typename Dtype>
void WarpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	const Dtype* Vb = bottom[0]->cpu_data();
	const Dtype* coords = bottom[1]->cpu_data();
	Dtype* Vt = top[0]->mutable_cpu_data();
	const int N  = bottom[0]->num();
	const int C  = bottom[0]->channels();
	const int W = bottom[0]->width();
	const int H = bottom[0]->height();
	float x, y, wx, wy, w00, w01, w10, w11, v00, v01, v10, v11;
	int x0, y0, x1, y1;
	int index;

	const int HW = H*W;

	for ( int n = 0; n < N; n++ ) {
		for ( int h = 0; h < H; h++ ) {
			for ( int w = 0; w < W; w++ ) {
				index = (n*2*H+h)*W+w;

				x = coords[index];
				y = coords[index + HW];
				x0 = floor(x);
				y0 = floor(y);
				x1 = x0 + 1;
				y1 = y0 + 1;
				wx = x - x0;
				wy = y - y0;
				w00 = (1 - wx) * (1 - wy);
				w01 = (1 - wx) * wy;
				w10 = wx * (1 - wy);
				w11 = wx * wy;

				for ( int c = 0; c < C; c++ ) {
					v00 = getvalue(Vb, x0, y0, c, n, C, W, H);
					v01 = getvalue(Vb, x0, y1, c, n, C, W, H);
					v10 = getvalue(Vb, x1, y0, c, n, C, W, H);
					v11 = getvalue(Vb, x1, y1, c, n, C, W, H);
					Vt[((n*C+c)*H+h)*W+w] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
				}
			}
		}
	}
}

template <typename Dtype>
void WarpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{

const Dtype* Vb = bottom[0]->cpu_data();
	const Dtype* coords = bottom[1]->cpu_data();
	const Dtype* top_diff = top[0]->cpu_diff();

	if (propagate_down[0]) {
		caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
	}

	if (propagate_down[1]) {
		caffe_set(bottom[1]->count(), Dtype(0), bottom[1]->mutable_cpu_diff());
	}

	const int N = bottom[0]->num();
	const int C = bottom[0]->channels();
	const int W = bottom[0]->width();
	const int H = bottom[0]->height();

	float x, y, wx, wy, v00, v01, v10, v11;
	float dx, dy, dv00, dv01, dv10, dv11;
	int x0, y0, x1, y1;
	int index, index1;

	const int HW = H*W;

	for ( int n = 0; n < N; n++ ) {
		for ( int h = 0; h < H; h++ ) {
			for ( int w = 0; w < W; w++ ) {
				index = (n*2*H+h)*W+w;

				x = coords[index];
				y = coords[index + HW];
				x0 = floor(x);
				y0 = floor(y);
				x1 = x0 + 1;
				y1 = y0 + 1;
				wx = x - x0;
				wy = y - y0;

				dv00 = (1-wx)*(1-wy);
				dv01 = (1-wx)*wy;
				dv10 = wx*(1-wy);
				dv11 = wx*wy;

				for ( int c = 0; c < C; c++ ) {
					index1 = ((n*C+c)*H+h)*W+w;

					v00 = getvalue(Vb, x0, y0, c, n, C, W, H);
					v01 = getvalue(Vb, x0, y1, c, n, C, W, H);
					v10 = getvalue(Vb, x1, y0, c, n, C, W, H);
					v11 = getvalue(Vb, x1, y1, c, n, C, W, H);

					dx = (wy-1)*v00 - wy*v01 + (1-wy)*v10 + wy*v11;
					dy = (wx-1)*v00 - wx*v10 + (1-wx)*v01 + wx*v11;

					if (propagate_down[0]) {
						Dtype* b0_diff = bottom[0]->mutable_cpu_diff();
						addvalue(b0_diff, x0, y0, c, n, C, W, H, dv00*top_diff[index1]);
						addvalue(b0_diff, x0, y1, c, n, C, W, H, dv01*top_diff[index1]);
						addvalue(b0_diff, x1, y0, c, n, C, W, H, dv10*top_diff[index1]);
						addvalue(b0_diff, x1, y1, c, n, C, W, H, dv11*top_diff[index1]);
					}
					if (propagate_down[1]) {

						Dtype* b1_diff = bottom[1]->mutable_cpu_diff();
						addvalue(b1_diff, w,  h,  0, n, 2, W, H, dx*top_diff[index1]);
						addvalue(b1_diff, w,  h,  1, n, 2, W, H, dy*top_diff[index1]);
					}
				}
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(WarpLayer);
#endif

INSTANTIATE_CLASS(WarpLayer);
REGISTER_LAYER_CLASS(Warp);

}  // namespace caffe
