#pragma once

#if defined(__CUDACC__)

//! @brief time a generic unary function
template <class F> float timeGpu(F &&f) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, cudaStreamDefault);

  f();

  cudaEventRecord(stop, cudaStreamDefault);
  cudaEventSynchronize(stop);

  float t0;
  cudaEventElapsedTime(&t0, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return t0 * 1000; // convert ms to us
}

#endif

template <class F> float timeCpu(F &&f) {
  auto t0 = std::chrono::high_resolution_clock::now();
  f();
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}
