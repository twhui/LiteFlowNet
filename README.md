This page is now under construction!

# LiteFlowNet
LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation, CVPR18 (Spotlight)

This repository is the release of <strong>LiteFlowNet</strong> for our paper <strong>LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation</strong></a> in CVPR18.

It comes as a fork of the modified caffe master branch from <a href="https://github.com/lmb-freiburg/flownet2">FlowNet2</a>.

# Prerequisites
For opencv 3+, you may need to change <code>opencv2/gpu/gpu.hpp</code> to <code>opencv2/cudaarithm.hpp</code> in <code>liteflownet/src/caffe/layersresample_layer.cu</code>.

Installation was tested under Ubuntu 14.04.5 and 16.04.2 with cuDNN v5.1 and CUDA 8.0. 

If your machine installed a newer version of cuDNN, you do not need to downgrade it. One trick is to rename <code>cudnn.h</code> in the extracted folder of <code>cudnn-8.0-linux-x64-v5.1.tgz</code> to <code>cudnn-5.1.h</code> and replace <code>#include <cudnn.h></code> to <code>#include <cudnn-5.1.h></code> in the folder <code>liteflownet/include/caffe/util/cudnn.hpp</code>. 

# Compiling
<pre><code>$ make -j 8 all tools pycaffe</code></pre>

# Trained models, training and testing codes
(To appear)

# License and Citation
All code is provided for research purposes only and without any warranty. Any commercial use requires our consent. If our work helps your research or you use the code in your research, please cite the following paper:

<pre><code>@InProceedings{hui18liteflownet,  
  author = {Tak-Wai Hui and Xiaoou Tang and Chen Change Loy},  
  title  = {LiteFlowNet: A Lightweight Convolutional Neural Network for Optical Flow Estimation},  
  booktitle  = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  
  year = {2018},  
  url = {http://mmlab.ie.cuhk.edu.hk/projects/LiteFlowNet/}
}</code></pre>
