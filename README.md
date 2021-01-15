# Azure Kinect normal map with CUDA

I've done a normal map generation from depth image(Azure kinect dk) with CUDA. I tested this on Windows 10. If you wanna use this on Linux, you could compile this to nvcc without cmake (with dependency).

The normal map generation is based on Eigen value decomposition. It is very fast.

# Environment
- Azure kinect : <Color : [1280 x 720], Depth : [640 x 576]>
- OS : Windows 10 (NOT Linux)
- IDE : Visual studio 2015 community
- CPU : Intel(R) Core(TM) i7-9700K (3.60GHz)
- GPU : Geforce RTX 2080 ti
- RAM : 64 GB

# Dependency
- Opencv : 4.3.0 (just for visualizing)
- Azure kinect SDK
- CUDA : 10.1

# Example
![ezgif com-gif-maker (3)](https://user-images.githubusercontent.com/23024027/104707143-60a70400-575f-11eb-816a-375360cd2581.gif)
