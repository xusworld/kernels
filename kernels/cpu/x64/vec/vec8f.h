#pragma once

#include <utility>

#include "avx_mathfun.h"
#include "sse_mathfun.h"

namespace kernels {
namespace cpu {

struct Vec8f {
  __m256 value;
  Vec8f() {}
  Vec8f(const float v) { value = _mm256_set1_ps(v); }
  Vec8f(const float* addr) { value = _mm256_broadcast_ss(addr); }
  Vec8f(const __m256& v) { value = v; }
  Vec8f(const __m256&& v) { value = std::move(v); }
  Vec8f(const Vec8f& lr) { value = lr.value; }
  Vec8f(const Vec8f&& lr) { value = std::move(lr.value); }

  // void set_lane(float v, int i) {
  //     value[i] = v;
  // }

  // const float operator[](const int i) const {
  //     return value[i];
  // }

  static Vec8f load(const float* addr) {
    Vec8f v;
    v.value = _mm256_load_ps(addr);
    return v;
  }
  static Vec8f loadu(const float* addr) {
    Vec8f v;
    v.value = _mm256_loadu_ps(addr);
    return v;
  }
  static void save(float* addr, const Vec8f& v) {
    _mm256_store_ps(addr, v.value);
  }
  static void saveu(float* addr, const Vec8f& v) {
    _mm256_storeu_ps(addr, v.value);
  }
  // mla_231
  static void mla(Vec8f& v1, const Vec8f& v2, const Vec8f& v3) {
#ifdef __AVX2__
    v1.value = _mm256_fmadd_ps(v2.value, v3.value, v1.value);
#else
    v1.value = _mm256_add_ps(_mm256_mul_ps(v2.value, v3.value), v1.value);
#endif
  }
  static void mla_123(Vec8f& v1, const Vec8f& v2, const Vec8f& v3) {
#ifdef __AVX2__
    v1.value = _mm256_fmadd_ps(v1.value, v2.value, v3.value);
#else
    v1.value = _mm256_add_ps(_mm256_mul_ps(v1.value, v2.value), v3.value);
#endif
  }
  static void mls(Vec8f& v1, const Vec8f& v2, const Vec8f& v3) {
#ifdef __AVX2__
    v1.value = _mm256_fnmadd_ps(v2.value, v3.value, v1.value);
#else
    v1.value = _mm256_sub_ps(_mm256_mul_ps(v2.value, v3.value), v1.value);
#endif
  }
  static Vec8f bsl_cle(const Vec8f& c1, const Vec8f& c2, const Vec8f& v1,
                       const Vec8f& v2) {
    Vec8f dst;
    dst.value = _mm256_blendv_ps(v1.value, v2.value,
                                 _mm256_cmp_ps(c1.value, c2.value, _CMP_GE_OQ));
    return dst;
  }
  static Vec8f bsl_clt(const Vec8f& c1, const Vec8f& c2, const Vec8f& v1,
                       const Vec8f& v2) {
    Vec8f dst;
    dst.value = _mm256_blendv_ps(v1.value, v2.value,
                                 _mm256_cmp_ps(c1.value, c2.value, _CMP_GT_OQ));
    return dst;
  }
  static Vec8f bsl_cge(const Vec8f& c1, const Vec8f& c2, const Vec8f& v1,
                       const Vec8f& v2) {
    Vec8f dst;
    dst.value = _mm256_blendv_ps(v1.value, v2.value,
                                 _mm256_cmp_ps(c1.value, c2.value, _CMP_LE_OQ));
    return dst;
  }
  static Vec8f bsl_cgt(const Vec8f& c1, const Vec8f& c2, const Vec8f& v1,
                       const Vec8f& v2) {
    Vec8f dst;
    dst.value = _mm256_blendv_ps(v1.value, v2.value,
                                 _mm256_cmp_ps(c1.value, c2.value, _CMP_LT_OQ));
    return dst;
  }
  static Vec8f max(const Vec8f& v1, const Vec8f& v2) {
    Vec8f dst;
    dst.value = _mm256_max_ps(v1.value, v2.value);
    return dst;
  }
  static Vec8f min(const Vec8f& v1, const Vec8f& v2) {
    Vec8f dst;
    dst.value = _mm256_min_ps(v1.value, v2.value);
    return dst;
  }
  static Vec8f add(const Vec8f& v1, const Vec8f& v2) {
    Vec8f dst;
    dst.value = _mm256_add_ps(v1.value, v2.value);
    return dst;
  }
  static Vec8f sub(const Vec8f& v1, const Vec8f& v2) {
    Vec8f dst;
    dst.value = _mm256_sub_ps(v1.value, v2.value);
    return dst;
  }
  static Vec8f mul(const Vec8f& v1, const Vec8f& v2) {
    Vec8f dst;
    dst.value = _mm256_mul_ps(v1.value, v2.value);
    return dst;
  }
  static Vec8f div(const Vec8f& v1, const Vec8f& v2) {
    Vec8f dst;
    dst.value = _mm256_div_ps(v1.value, v2.value);
    return dst;
  }
  static float reduce_add(const Vec8f& v) {
    Vec8f tmp;
    tmp.value = _mm256_hadd_ps(v.value, v.value);
    tmp.value = _mm256_hadd_ps(tmp.value, tmp.value);
    float rst[8];
    _mm256_store_ps(rst, tmp.value);
    return rst[0];
  }
  static Vec8f neg(const Vec8f& v) {
    Vec8f dst;
    dst.value = _mm256_xor_ps(v.value, *(__m256*)_ps256_sign_mask);
    return dst;
  }
  static Vec8f abs(const Vec8f& v) {
    Vec8f dst;
    dst.value =
        _mm256_max_ps(_mm256_sub_ps(_mm256_setzero_ps(), v.value), v.value);
    return dst;
  }
  static Vec8f sqrt(const Vec8f& v) {
    Vec8f dst;
    dst.value = _mm256_sqrt_ps(v.value);
    return dst;
  }
  static Vec8f sigmoid(const Vec8f& v) {
    Vec8f dst;
    const __m256 one = _mm256_set1_ps(1.0f);
    dst.value = _mm256_div_ps(
        one, _mm256_add_ps(
                 one, exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), v.value))));
    return dst;
  }
  static Vec8f exp(const Vec8f& v) {
    Vec8f dst;
    dst.value = exp256_ps(v.value);
    return dst;
  }
  static Vec8f log(const Vec8f& v) {
    Vec8f dst;
    dst.value = log256_ps(v.value);
    return dst;
  }
  static Vec8f tanh(const Vec8f& v) {
    Vec8f dst;
    __m128 low = _mm256_extractf128_ps(v.value, 0);
    __m128 high = _mm256_extractf128_ps(v.value, 1);
    low = tanh_ps(low);
    high = tanh_ps(high);
    dst.value = _mm256_castps128_ps256(low);
    dst.value = _mm256_insertf128_ps(dst.value, high, 1);
    return dst;
  }
  Vec8f operator+(const Vec8f& lr) const {
    Vec8f dst;
    dst.value = _mm256_add_ps(value, lr.value);
    return dst;
  }
  Vec8f operator-(const Vec8f& lr) const {
    Vec8f dst;
    dst.value = _mm256_sub_ps(value, lr.value);
    return dst;
  }
  Vec8f operator*(float lr) const {
    Vec8f dst;
    __m256 tmp = _mm256_set1_ps(lr);
    dst.value = _mm256_mul_ps(value, tmp);
    return dst;
  }
  Vec8f operator*(const Vec8f& lr) const {
    Vec8f dst;
    dst.value = _mm256_mul_ps(value, lr.value);
    return dst;
  }
  Vec8f& operator=(const Vec8f& lr) {
    value = lr.value;
    return *this;
  }
  Vec8f& operator=(const Vec8f&& lr) {
    value = std::move(lr.value);
    return *this;
  }
  Vec8f operator-() const {
    Vec8f dst;
    dst.value = _mm256_sub_ps(_mm256_setzero_ps(), value);
    return dst;
  }
};

}  // namespace cpu
}  // namespace kernels
