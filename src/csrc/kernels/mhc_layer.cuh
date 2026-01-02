#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../include/mhc_types.h"
#include "rmsnorm.cuh"
#include "sinkhorn_knopp.cuh"
#include "stream_ops.cuh"

namespace mhc {

struct MHCLayerConfig {
    int batch_size;
    int hidden_dim;
    int expansion_rate;
    int sinkhorn_iters;
    float eps;
    float alpha_init;
    bool use_pdl;

    MHCLayerConfig()
        : batch_size(0), hidden_dim(0), expansion_rate(4), sinkhorn_iters(20), eps(1e-5f),
          alpha_init(0.01f), use_pdl(true) {}
};

struct MHCLayerWeights {
    floatX* rmsnorm_weight;
    float* H_pre;
    float* H_post;
    float* H_res;

    bool initialized;
    int hidden_dim;
    int expansion_rate;

    MHCLayerWeights() : initialized(false) {}

    void init(int C, int n) {
        hidden_dim = C;
        expansion_rate = n;

        CHECK_CUDA(cudaMalloc(&rmsnorm_weight, C * sizeof(floatX)));
        CHECK_CUDA(cudaMalloc(&H_pre, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&H_post, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&H_res, n * n * sizeof(float)));

        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;

        cudaFree(rmsnorm_weight);
        cudaFree(H_pre);
        cudaFree(H_post);
        cudaFree(H_res);

        initialized = false;
    }
};

struct MHCLayerBuffers {
    float* x_expanded;
    floatX* x_aggregated_bf16;
    float* x_aggregated_f32;
    float* rms_values;
    floatX* layer_out_bf16;
    float* layer_out_f32;
    float* y_distributed;
    float* sinkhorn_M;
    float* x_mixed;
    float* output;

    bool initialized;
    int batch_size;
    int hidden_dim;
    int expansion_rate;

    MHCLayerBuffers() : initialized(false), x_mixed(nullptr) {}

    void init(int B, int C, int n, bool needs_x_mixed = false) {
        batch_size = B;
        hidden_dim = C;
        expansion_rate = n;

        CHECK_CUDA(cudaMalloc(&x_expanded, B * n * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&x_aggregated_bf16, B * C * sizeof(floatX)));
        CHECK_CUDA(cudaMalloc(&x_aggregated_f32, B * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&rms_values, B * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&layer_out_bf16, B * C * sizeof(floatX)));
        CHECK_CUDA(cudaMalloc(&layer_out_f32, B * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&y_distributed, B * n * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&sinkhorn_M, n * n * sizeof(float)));
        if (needs_x_mixed) {
            CHECK_CUDA(cudaMalloc(&x_mixed, B * n * C * sizeof(float)));
        }
        CHECK_CUDA(cudaMalloc(&output, B * n * C * sizeof(float)));

        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;

        cudaFree(x_expanded);
        cudaFree(x_aggregated_bf16);
        cudaFree(x_aggregated_f32);
        cudaFree(rms_values);
        cudaFree(layer_out_bf16);
        cudaFree(layer_out_f32);
        cudaFree(y_distributed);
        cudaFree(sinkhorn_M);
        if (x_mixed)
            cudaFree(x_mixed);
        cudaFree(output);

        initialized = false;
    }
};

struct MHCLayerGradients {
    float* d_x_expanded;
    float* d_H_pre;
    float* d_rmsnorm_weight;
    float* d_H_post;
    float* d_H_res;
    float* d_x_aggregated;
    float* d_layer_out;
    float* d_y_distributed;
    float* d_x_mixed;
    float* d_M;

    bool initialized;

    MHCLayerGradients() : initialized(false) {}

    void init(int B, int C, int n) {
        CHECK_CUDA(cudaMalloc(&d_x_expanded, B * n * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_H_pre, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_rmsnorm_weight, C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_H_post, n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_H_res, n * n * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_x_aggregated, B * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_layer_out, B * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_y_distributed, B * n * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_x_mixed, B * n * C * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_M, n * n * sizeof(float)));

        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;

        cudaFree(d_x_expanded);
        cudaFree(d_H_pre);
        cudaFree(d_rmsnorm_weight);
        cudaFree(d_H_post);
        cudaFree(d_H_res);
        cudaFree(d_x_aggregated);
        cudaFree(d_layer_out);
        cudaFree(d_y_distributed);
        cudaFree(d_x_mixed);
        cudaFree(d_M);

        initialized = false;
    }

    void zero_weight_grads(int C, int n, cudaStream_t stream = nullptr) {
        CHECK_CUDA(cudaMemsetAsync(d_H_pre, 0, n * sizeof(float), stream));
        CHECK_CUDA(cudaMemsetAsync(d_rmsnorm_weight, 0, C * sizeof(float), stream));
        CHECK_CUDA(cudaMemsetAsync(d_H_post, 0, n * sizeof(float), stream));
        CHECK_CUDA(cudaMemsetAsync(d_H_res, 0, n * n * sizeof(float), stream));
    }
};

template<int BLOCK_SIZE>
__global__ void float_to_bf16_kernel(floatX* __restrict__ out, const float* __restrict__ inp,
                                     int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = (floatX)inp[idx];
    }
}

template<int BLOCK_SIZE>
__global__ void bf16_to_float_kernel(float* __restrict__ out, const floatX* __restrict__ inp,
                                     int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = (float)inp[idx];
    }
}

inline void float_to_bf16(floatX* out, const float* inp, int size, cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float_to_bf16_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, size);
}

inline void bf16_to_float(float* out, const floatX* inp, int size, cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bf16_to_float_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, size);
}

struct MHCLayer {
    MHCLayerConfig config;
    MHCLayerWeights weights;
    MHCLayerBuffers buffers;
    MHCLayerGradients grads;

    StreamMixTC stream_mix_tc;
    bool use_tc_mix;
    bool backward_enabled;

    cudaStream_t stream;
    bool owns_stream;
    bool initialized;

    MHCLayer()
        : stream(nullptr), owns_stream(false), initialized(false), use_tc_mix(false),
          backward_enabled(false) {}

    void init(const MHCLayerConfig& cfg, cudaStream_t s = nullptr, bool enable_backward = false) {
        config = cfg;
        int B = cfg.batch_size;
        int C = cfg.hidden_dim;
        int n = cfg.expansion_rate;

        use_tc_mix = (n >= STREAM_MIX_TC_THRESHOLD);
        backward_enabled = enable_backward;

        weights.init(C, n);
        buffers.init(B, C, n, use_tc_mix || backward_enabled);

        if (use_tc_mix) {
            stream_mix_tc.init(B, n, C);
        }

        if (backward_enabled) {
            grads.init(B, C, n);
        }

        if (s == nullptr) {
            CHECK_CUDA(cudaStreamCreate(&stream));
            owns_stream = true;
        } else {
            stream = s;
            owns_stream = false;
        }

        initialized = true;
    }

    void destroy() {
        if (!initialized)
            return;

        weights.destroy();
        buffers.destroy();
        if (backward_enabled) {
            grads.destroy();
        }

        if (use_tc_mix) {
            stream_mix_tc.destroy();
        }

        if (owns_stream && stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }

        initialized = false;
    }

    void set_weights(const floatX* h_rmsnorm_weight, const float* h_H_pre, const float* h_H_post,
                     const float* h_H_res) {
        int C = config.hidden_dim;
        int n = config.expansion_rate;

        CHECK_CUDA(cudaMemcpyAsync(weights.rmsnorm_weight, h_rmsnorm_weight, C * sizeof(floatX),
                                   cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(weights.H_pre, h_H_pre, n * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(weights.H_post, h_H_post, n * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(weights.H_res, h_H_res, n * n * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    }

    void forward(const float* x_expanded) {
        int B = config.batch_size;
        int C = config.hidden_dim;
        int n = config.expansion_rate;

        CHECK_CUDA(cudaMemcpyAsync(buffers.x_expanded, x_expanded, B * n * C * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));

        stream_aggregate_bf16(buffers.x_aggregated_bf16, buffers.x_expanded, weights.H_pre, B, n, C,
                              stream);

        rmsnorm_forward_with_rms(buffers.layer_out_bf16, buffers.rms_values,
                                 buffers.x_aggregated_bf16, weights.rmsnorm_weight, B, C,
                                 config.eps, stream);

        stream_distribute_from_bf16(buffers.y_distributed, buffers.layer_out_bf16, weights.H_post,
                                    B, n, C, stream);

#ifdef MHC_ENABLE_PDL
        if (config.use_pdl) {
            sinkhorn_knopp_forward_pdl(buffers.sinkhorn_M, weights.H_res, n, n,
                                       config.sinkhorn_iters, config.eps, stream);
        } else
#endif
        {
            sinkhorn_knopp_forward(buffers.sinkhorn_M, weights.H_res, n, n, config.sinkhorn_iters,
                                   config.eps, stream);
        }

        if (use_tc_mix) {
            stream_mix_tc.forward(buffers.x_mixed, buffers.x_expanded, buffers.sinkhorn_M, stream);
            stream_add(buffers.output, buffers.x_mixed, buffers.y_distributed, B * n * C, stream);
        } else {
            stream_mix_add(buffers.output, buffers.x_expanded, buffers.y_distributed,
                           buffers.sinkhorn_M, B, n, C, stream);
        }
    }

    void forward_device(const float* d_x_expanded) {
        int B = config.batch_size;
        int C = config.hidden_dim;
        int n = config.expansion_rate;

        CHECK_CUDA(cudaMemcpyAsync(buffers.x_expanded, d_x_expanded, B * n * C * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));

        stream_aggregate_bf16(buffers.x_aggregated_bf16, buffers.x_expanded, weights.H_pre, B, n, C,
                              stream);

        rmsnorm_forward_with_rms(buffers.layer_out_bf16, buffers.rms_values,
                                 buffers.x_aggregated_bf16, weights.rmsnorm_weight, B, C,
                                 config.eps, stream);

        stream_distribute_from_bf16(buffers.y_distributed, buffers.layer_out_bf16, weights.H_post,
                                    B, n, C, stream);

#ifdef MHC_ENABLE_PDL
        if (config.use_pdl) {
            sinkhorn_knopp_forward_pdl(buffers.sinkhorn_M, weights.H_res, n, n,
                                       config.sinkhorn_iters, config.eps, stream);
        } else
#endif
        {
            sinkhorn_knopp_forward(buffers.sinkhorn_M, weights.H_res, n, n, config.sinkhorn_iters,
                                   config.eps, stream);
        }

        if (use_tc_mix) {
            stream_mix_tc.forward(buffers.x_mixed, buffers.x_expanded, buffers.sinkhorn_M, stream);
            stream_add(buffers.output, buffers.x_mixed, buffers.y_distributed, B * n * C, stream);
        } else {
            stream_mix_add(buffers.output, buffers.x_expanded, buffers.y_distributed,
                           buffers.sinkhorn_M, B, n, C, stream);
        }
    }

    float* get_output() { return buffers.output; }

    float* get_rms_values() { return buffers.rms_values; }

    MHCLayerGradients& get_gradients() { return grads; }

    void backward(const float* d_output) {
        if (!backward_enabled) {
            fprintf(stderr, "MHCLayer::backward called but backward not enabled\n");
            return;
        }

        int B = config.batch_size;
        int C = config.hidden_dim;
        int n = config.expansion_rate;

        grads.zero_weight_grads(C, n, stream);

        stream_mix_backward(grads.d_x_mixed, grads.d_M, d_output, buffers.x_expanded,
                            buffers.sinkhorn_M, B, n, C, stream);

        CHECK_CUDA(cudaMemcpyAsync(grads.d_y_distributed, d_output, B * n * C * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));

        sinkhorn_knopp_backward(grads.d_H_res, grads.d_M, buffers.sinkhorn_M, weights.H_res, n,
                                config.sinkhorn_iters, config.eps, stream);

        bf16_to_float(buffers.layer_out_f32, buffers.layer_out_bf16, B * C, stream);

        stream_distribute_backward(grads.d_layer_out, grads.d_H_post, grads.d_y_distributed,
                                   buffers.layer_out_f32, weights.H_post, B, n, C, stream);

        bf16_to_float(buffers.x_aggregated_f32, buffers.x_aggregated_bf16, B * C, stream);

        rmsnorm_backward(grads.d_x_aggregated, grads.d_rmsnorm_weight, grads.d_layer_out,
                         buffers.x_aggregated_bf16, weights.rmsnorm_weight, buffers.rms_values, B,
                         C, stream);

        stream_aggregate_backward(grads.d_x_expanded, grads.d_H_pre, grads.d_x_aggregated,
                                  buffers.x_expanded, weights.H_pre, B, n, C, stream);
    }

    float* get_dx() { return grads.d_x_expanded; }

    void sync() { CHECK_CUDA(cudaStreamSynchronize(stream)); }
};
} // namespace mhc
