#include <xbyak/xbyak.h>

constexpr int kMaxCodeSize = 256 * 1024;

class AVX2FMACode : Xbyak::CodeGenerator {
 public:
  AVX2FMACode(int loop_times, size_t code_size = kMaxCodeSize)
      : Xbyak::CodeGenerator(code_size, Xbyak::AutoGrow) {
    Preamble();
#ifdef XBYAK32
#else
    vxorps(ymm0, ymm0, ymm0);
    vxorps(ymm1, ymm1, ymm1);
    vxorps(ymm2, ymm2, ymm2);
    vxorps(ymm3, ymm3, ymm3);
    vxorps(ymm4, ymm4, ymm4);
    vxorps(ymm5, ymm5, ymm5);
    vxorps(ymm6, ymm6, ymm6);
    vxorps(ymm7, ymm7, ymm7);
    vxorps(ymm8, ymm8, ymm8);
    vxorps(ymm9, ymm9, ymm9);
    L("x86.fma.fp32.L1");
    vfmadd231ps(ymm0, ymm0, ymm0);
    vfmadd231ps(ymm1, ymm1, ymm1);
    vfmadd231ps(ymm2, ymm2, ymm2);
    vfmadd231ps(ymm3, ymm3, ymm3);
    vfmadd231ps(ymm4, ymm4, ymm4);
    vfmadd231ps(ymm5, ymm5, ymm5);
    vfmadd231ps(ymm6, ymm6, ymm6);
    vfmadd231ps(ymm7, ymm7, ymm7);
    vfmadd231ps(ymm8, ymm8, ymm8);
    vfmadd231ps(ymm9, ymm9, ymm9);
    sub(reg_loop_, 1);
    jne("x86.fma.fp32.L1");
#endif
    Postamble();
  }

  virtual ~AVX2FMACode() = default;

 public:
  void operator=(const AVX2FMACode &);

  // TODO: add return status
  virtual void CreateKernel() {
    GenerateCode();
    jit_kernel_ = GetCode();
  }

  template <typename... KernelArgs>
  void operator()(KernelArgs... args) const {
    using jit_kernel_func_t = void (*)(const KernelArgs... args);
    auto *fptr = (jit_kernel_func_t)jit_kernel_;
    (*fptr)(std::forward<KernelArgs>(args)...);
  }

  template <typename... KernelArgs>
  void Forward(KernelArgs... args) const {
    using jit_kernel_func_t = void (*)(const KernelArgs... args);
    auto *fptr = (jit_kernel_func_t)jit_kernel_;
    (*fptr)(std::forward<KernelArgs>(args)...);
  }

  int (*get() const)(int) { return getCode<int (*)(int)>(); }

  void Preamble() {
    push(rbx);
    push(rbp);
    push(r12);
    push(r13);
    push(r14);
    push(r15);
  }

  void Postamble() {
    pop(r15);
    pop(r14);
    pop(r13);
    pop(r12);
    pop(rbp);
    pop(rbx);
    ret();
  }

 protected:
  virtual void GenerateCode() = 0;
  const Xbyak::uint8 *jit_kernel_;

 private:
  const Xbyak::uint8 *GetCode() {
    // Xbyak::AutoGrow should call ready() before to run!
    // mode = Read/Write/Exec
    this->ready();
    const Xbyak::uint8 *code = CodeGenerator::getCode();
    return code;
  }

 private:
  Xbyak::Reg64 reg_loop_ = rdi;
};

int main() {
  AVX2FMACode code(1000);
  int (*func)(int) = code.get();
  int ret = func(2000);
  std::cout << "ret: " << ret << std::endl;
  return 0;
}