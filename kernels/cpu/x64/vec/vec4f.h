#pragma once

#include <immintrin.h>

#include <utility>

#include "sse_mathfun.h"

namespace kernels {
namespace cpu {

struct Vec4f {
  __m128 value;
  Vec4f() {}
  Vec4f(const float v) { value = _mm_set1_ps(v); }
  Vec4f(const float* addr) { value = _mm_set1_ps(*addr); }
  Vec4f(const __m128& v) { value = v; }
  Vec4f(const __m128&& v) { value = std::move(v); }
  Vec4f(const Vec4f& lr) { value = lr.value; }
  Vec4f(const Vec4f&& lr) { value = std::move(lr.value); }

  // void set_lane(float v, int i) {
  //     value[i] = v;
  // }

  // const float operator[](const int i) const {
  //     return value[i];
  // }

  static Vec4f load(const float* addr) {
    Vec4f v;
    v.value = _mm_load_ps(addr);
    return v;
  }
  static Vec4f loadu(const float* addr) {
    Vec4f v;
    v.value = _mm_loadu_ps(addr);
    return v;
  }
  static void save(float* addr, const Vec4f& v) { _mm_store_ps(addr, v.value); }
  static void saveu(float* addr, const Vec4f& v) {
    _mm_storeu_ps(addr, v.value);
  }
  // mla_231
  static void mla(Vec4f& v1, const Vec4f& v2, const Vec4f& v3) {
    v1.value = _mm_add_ps(v1.value, _mm_mul_ps(v2.value, v3.value));
  }
  static void mla_123(Vec4f& v1, const Vec4f& v2, const Vec4f& v3) {
    v1.value = _mm_add_ps(v3.value, _mm_mul_ps(v1.value, v2.value));
  }
  static void mls(Vec4f& v1, const Vec4f& v2, const Vec4f& v3) {
    v1.value = _mm_sub_ps(v1.value, _mm_mul_ps(v2.value, v3.value));
  }
  static Vec4f bsl_cle(const Vec4f& c1, const Vec4f& c2, const Vec4f& v1,
                       const Vec4f& v2) {
    Vec4f dst;
    dst.value =
        _mm_blendv_ps(v2.value, v1.value, _mm_cmpge_ps(c2.value, c1.value));
    return dst;
  }
  static Vec4f bsl_clt(const Vec4f& c1, const Vec4f& c2, const Vec4f& v1,
                       const Vec4f& v2) {
    Vec4f dst;
    dst.value =
        _mm_blendv_ps(v2.value, v1.value, _mm_cmpgt_ps(c2.value, c1.value));
    return dst;
  }
  static Vec4f bsl_cge(const Vec4f& c1, const Vec4f& c2, const Vec4f& v1,
                       const Vec4f& v2) {
    Vec4f dst;
    dst.value =
        _mm_blendv_ps(v2.value, v1.value, _mm_cmpge_ps(c1.value, c2.value));
    return dst;
  }
  static Vec4f bsl_cgt(const Vec4f& c1, const Vec4f& c2, const Vec4f& v1,
                       const Vec4f& v2) {
    Vec4f dst;
    dst.value =
        _mm_blendv_ps(v2.value, v1.value, _mm_cmpgt_ps(c1.value, c2.value));
    return dst;
  }
  static Vec4f max(const Vec4f& v1, const Vec4f& v2) {
    Vec4f dst;
    dst.value = _mm_max_ps(v1.value, v2.value);
    return dst;
  }
  static Vec4f min(const Vec4f& v1, const Vec4f& v2) {
    Vec4f dst;
    dst.value = _mm_min_ps(v1.value, v2.value);
    return dst;
  }
  static Vec4f add(const Vec4f& v1, const Vec4f& v2) {
    Vec4f dst;
    dst.value = _mm_add_ps(v1.value, v2.value);
    return dst;
  }
  static Vec4f sub(const Vec4f& v1, const Vec4f& v2) {
    Vec4f dst;
    dst.value = _mm_sub_ps(v1.value, v2.value);
    return dst;
  }
  static Vec4f mul(const Vec4f& v1, const Vec4f& v2) {
    Vec4f dst;
    dst.value = _mm_mul_ps(v1.value, v2.value);
    return dst;
  }
  static Vec4f div(const Vec4f& v1, const Vec4f& v2) {
    Vec4f dst;
    dst.value = _mm_div_ps(v1.value, v2.value);
    return dst;
  }
  static float reduce_add(const Vec4f& v) {
    Vec4f dst;
    dst.value = _mm_hadd_ps(v.value, v.value);
    dst.value = _mm_hadd_ps(dst.value, dst.value);
    float rst[4];
    _mm_store_ps(rst, dst.value);
    return rst[0];
  }
  static Vec4f neg(const Vec4f& v) {
    Vec4f dst;
    dst.value = _mm_xor_ps(v.value, *(__m128*)_ps_sign_mask);
    return dst;
  }
  static Vec4f abs(const Vec4f& v) {
    Vec4f dst;
    dst.value = _mm_max_ps(_mm_sub_ps(_mm_setzero_ps(), v.value), v.value);
    return dst;
  }
  static Vec4f sqrt(const Vec4f& v) {
    Vec4f dst;
    dst.value = _mm_sqrt_ps(v.value);
    return dst;
  }
  static Vec4f sigmoid(const Vec4f& v) {
    Vec4f dst;
    const __m128 one = _mm_set1_ps(1.0f);
    dst.value = _mm_div_ps(
        one, _mm_add_ps(one, exp_ps(_mm_sub_ps(_mm_setzero_ps(), v.value))));
    return dst;
  }
  static Vec4f exp(const Vec4f& v) {
    Vec4f dst;
    dst.value = exp_ps(v.value);
    return dst;
  }
  static Vec4f log(const Vec4f& v) {
    Vec4f dst;
    dst.value = log_ps(v.value);
    return dst;
  }
  static Vec4f tanh(const Vec4f& v) {
    Vec4f dst;
    dst.value = tanh_ps(v.value);
    return dst;
  }
  Vec4f operator+(const Vec4f& lr) const {
    Vec4f dst;
    dst.value = _mm_add_ps(value, lr.value);
    return dst;
  }
  Vec4f operator-(const Vec4f& lr) const {
    Vec4f dst;
    dst.value = _mm_sub_ps(value, lr.value);
    return dst;
  }
  Vec4f operator*(float lr) const {
    Vec4f dst;
    __m128 tmp = _mm_set1_ps(lr);
    dst.value = _mm_mul_ps(value, tmp);
    return dst;
  }
  Vec4f operator*(const Vec4f& lr) const {
    Vec4f dst;
    dst.value = _mm_mul_ps(value, lr.value);
    return dst;
  }
  Vec4f& operator=(const Vec4f& lr) {
    value = lr.value;
    return *this;
  }
  Vec4f& operator=(const Vec4f&& lr) {
    value = std::move(lr.value);
    return *this;
  }
  Vec4f operator-() const {
    Vec4f dst;
    dst.value = _mm_sub_ps(_mm_setzero_ps(), value);
    return dst;
  }
};

struct Vec4fx4 {
  __m128 value[4];

  static Vec4fx4 ld4u(const float* addr) {
    Vec4fx4 v;
    v.value[0] = _mm_loadu_ps(addr);
    v.value[1] = _mm_loadu_ps(addr + 4);
    v.value[2] = _mm_loadu_ps(addr + 8);
    v.value[3] = _mm_loadu_ps(addr + 12);
    _MM_TRANSPOSE4_PS(v.value[0], v.value[1], v.value[2], v.value[3]);
    return v;
  }
  static Vec4fx4 loadu(const float* addr) {
    Vec4fx4 v;
    v.value[0] = _mm_loadu_ps(addr);
    v.value[1] = _mm_loadu_ps(addr + 4);
    v.value[2] = _mm_loadu_ps(addr + 8);
    v.value[3] = _mm_loadu_ps(addr + 12);
    return v;
  }
  static Vec4fx4 ld4(const float* addr) {
    Vec4fx4 v;
    v.value[0] = _mm_load_ps(addr);
    v.value[1] = _mm_load_ps(addr + 4);
    v.value[2] = _mm_load_ps(addr + 8);
    v.value[3] = _mm_load_ps(addr + 12);
    _MM_TRANSPOSE4_PS(v.value[0], v.value[1], v.value[2], v.value[3]);
    return v;
  }
  static Vec4fx4 load(const float* addr) {
    Vec4fx4 v;
    v.value[0] = _mm_load_ps(addr);
    v.value[1] = _mm_load_ps(addr + 4);
    v.value[2] = _mm_load_ps(addr + 8);
    v.value[3] = _mm_load_ps(addr + 12);
    return v;
  }
  void get_lane(Vec4f& v, int index) { v.value = value[index]; }
};

}  // namespace cpu
}  // namespace kernels