add_executable(bench_gemm bench_gemm.cu)
target_link_libraries(bench_gemm PRIVATE ozaki ozaki_warnings CUDA::cublas)
