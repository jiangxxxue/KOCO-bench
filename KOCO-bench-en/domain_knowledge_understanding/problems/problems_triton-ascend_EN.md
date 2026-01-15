# Triton-Ascend Multiple Choice Problems

## triton-ascend - Problem 1 (Single Choice)
When designing a Triton kernel for Ascend NPU that performs both matrix multiplication and element-wise operations, what is the fundamental architectural trade-off that determines the choice of mix_mode?

- (A) AICore (cube) mode provides higher throughput for matrix operations but cannot execute vector operations, requiring separate kernel launches for element-wise operations
- (B) VectorCore (aiv) mode offers better energy efficiency but lacks the tensor core acceleration needed for matrix multiplication, making it unsuitable for mixed workloads
- (C) The mix_mode selection determines which physical compute units are utilized, with "aic" targeting cube cores for matrix ops and "aiv" targeting vector cores, but a single kernel cannot dynamically switch between them during execution
- (D) Mixed mode ("mix") allows runtime scheduling between cube and vector cores based on operation type, but introduces synchronization overhead that may negate performance benefits for small workloads

**Correct Answer: C**

**Explanation:** This tests understanding of Ascend NPU's heterogeneous architecture. From compiler.py's _parse_linalg_metadata and the architecture documentation, mix_mode is a compile-time decision that determines which physical compute units (AICore cube vs vector) will execute the kernel. (C) is correct because the mode is fixed at compile time and determines resource allocation. (A) is incorrect - AICore can execute vector operations through its vector units. (B) misrepresents the capabilities - vector cores can perform matrix operations, just less efficiently. (D) is incorrect - there's no dynamic runtime switching within a single kernel; the mode is statically determined. This requires synthesizing knowledge from compiler.py's metadata parsing, driver.py's core count queries, and the architectural constraints documented in HighPerformanceGuide.md.

---

## triton-ascend - Problem 2 (Multiple Choice)
In the Triton-to-Linalg compilation lowering pipeline, which design decisions enable the framework to preserve high-level semantic information while enabling hardware-specific optimizations?

- (A) The triton-to-annotation pass attaches domain-specific attributes (like tensor_kind and mix_mode) to MLIR operations before structural lowering occurs
- (B) The discrete-mask-access-conversion pass transforms irregular memory access patterns into structured forms that can be analyzed by subsequent optimization passes
- (C) The triton-to-unstructure pass deliberately removes control flow structure to enable more aggressive instruction scheduling
- (D) The bubble-up-operation pass reorders operations to expose data dependencies that inform the HIVM scheduler's decision-making
- (E) The triton-to-linalg pass directly generates CANN runtime calls without intermediate representations

**Correct Answer: A, B, D**

**Explanation:** This tests deep understanding of the compilation pipeline's design philosophy. From compiler.py's ttir_to_linalg function, the pass sequence is carefully ordered: (A) is correct - triton-to-annotation preserves semantic metadata that would be lost in pure structural lowering. (B) is correct - discrete-mask-access-conversion normalizes memory patterns for optimization. (D) is correct - bubble-up-operation exposes parallelism opportunities. (C) is incorrect - triton-to-unstructure converts to unstructured control flow for lowering, not for scheduling. (E) is incorrect - the pipeline goes through Linalg and HIVM, not directly to CANN. This requires understanding the interplay between ascend/backend/compiler.py's pass ordering, ascend/triton-adapter's transformation logic, and the architectural constraints that motivate each pass.

---

## triton-ascend - Problem 3 (Multiple Choice)
Which factors influence the HIVM intermediate virtual machine's ability to achieve efficient Cube-Vector core load balancing in a mixed-mode kernel?

- (A) The enable-hivm-auto-cv-balance compilation flag that enables automatic workload distribution analysis
- (B) The ratio of tl.dot operations to element-wise operations in the kernel source code
- (C) The tile_mix_vector_loop and tile_mix_cube_loop parameters that control loop tiling strategies for each core type
- (D) The physical proximity of cube and vector cores on the NPU die, which affects data transfer latency
- (E) The tensor_kind annotations that indicate which tensors are read-only vs read-write, affecting scheduling constraints

**Correct Answer: A, B, C**

**Explanation:** This tests understanding of HIVM's role in heterogeneous compute orchestration. From compiler.py's linalg_to_bin_enable_npu_compile, (A) is correct - enable_hivm_auto_cv_balance is explicitly provided as a tuning knob. (C) is correct - tile_mix_vector_loop and tile_mix_cube_loop control how loops are partitioned for different core types. (B) is correct in principle - the operation mix determines the theoretical workload split, though this is implicit rather than a direct parameter. (D) is incorrect - while physical layout matters, it's not a configurable factor in HIVM. (E) is incorrect - tensor_kind is for profiling, not scheduling. This requires understanding the interaction between compiler.py's NPUOptions, the HIVM compilation flags, and the architectural constraints from HighPerformanceGuide.md.

---

## triton-ascend - Problem 4 (Single Choice)
What is the fundamental design rationale behind triton-ascend's decision to use Linalg dialect as an intermediate representation rather than lowering directly from TTIR to LLVM IR?

- (A) Linalg provides a structured representation that enables polyhedral optimization techniques, which are essential for exploiting the regular memory access patterns of NPU architectures
- (B) Linalg dialect is required by the CANN runtime's binary format, which expects structured loop nests rather than arbitrary control flow
- (C) The HIVM virtual machine can only consume Linalg IR, making it a mandatory intermediate step in the compilation pipeline
- (D) Linalg enables easier integration with MLIR's existing transformation infrastructure while providing a level of abstraction suitable for hardware-specific passes before final lowering

**Correct Answer: D**

**Explanation:** This tests understanding of the compilation architecture's design philosophy. From compiler.py's add_stages and the triton-adapter pipeline, (D) is correct - Linalg serves as a sweet spot between high-level semantics and hardware-specific concerns. It allows triton-adapter to perform NPU-specific transformations (like HIVM lowering) while leveraging MLIR's ecosystem. (A) is partially true but overstates the role of polyhedral optimization. (B) is incorrect - CANN consumes binary code, not IR. (C) is incorrect - HIVM is part of the bishengir toolchain that processes Linalg, but this is an implementation detail, not a fundamental requirement. This requires synthesizing knowledge from compiler.py's pipeline design, the triton-adapter passes in ttir_to_linalg, and understanding the broader MLIR compilation philosophy.

---

## triton-ascend - Problem 5 (Multiple Choice)
In the context of multi-buffer pipeline optimization, which statements correctly describe the trade-offs and implementation mechanisms?

- (A) The multibuffer API increases memory consumption by buffer_num times but enables overlapping of data movement and computation through software pipelining
- (B) The limit-auto-multi-buffer-only-for-local-buffer flag restricts multi-buffering to on-chip memory to avoid exceeding the 192KB limit
- (C) The set-workspace-multibuffer parameter controls multi-buffering for the per-block workspace memory allocated by the runtime
- (D) Multi-buffering is automatically applied by the compiler to all tensors when enable-auto-multi-buffer is set, without requiring explicit API calls
- (E) The limit-auto-multi-buffer-of-local-buffer parameter provides fine-grained control over which specific local buffers receive multi-buffering optimization

**Correct Answer: A, C, E**

**Explanation:** This tests deep understanding of multi-buffer optimization mechanisms. From compiler.py's NPUOptions and linalg_to_bin_enable_npu_compile, and HighPerformanceGuide.md: (A) is correct - this is the fundamental trade-off stated in the documentation. (C) is correct - set_workspace_multibuffer is a distinct parameter for workspace memory. (E) is correct - this parameter provides selective control. (B) is incorrect - the flag controls where auto-multi-buffering is applied, not a hard memory limit. (D) is incorrect - enable-auto-multi-buffer enables automatic analysis, but the multibuffer API provides explicit control. This requires synthesizing knowledge from compiler.py's compilation options, HighPerformanceGuide.md's API documentation, and understanding the memory hierarchy implications.

---

## triton-ascend - Problem 6 (Single Choice)
When implementing inter-core parallelism with block-level task distribution, what is the critical constraint that determines the maximum effective parallelism achievable on Ascend NPU?

- (A) The CANN runtime's maximum grid size limit of 65535 blocks, which caps the total number of parallel tasks
- (B) The physical core count (AICore or VectorCore depending on kernel type), beyond which additional blocks incur batch scheduling overhead
- (C) The L2 cache coherency protocol's ability to maintain consistency across cores, which limits scalability
- (D) The HIVM scheduler's maximum thread count, which is determined by the available register file size

**Correct Answer: B**

**Explanation:** This tests understanding of inter-core parallelism constraints. From HighPerformanceGuide.md and driver.py's NPUUtils: (B) is correct - the guide explicitly states that exceeding physical core count causes batch scheduling with additional overhead, and recommends setting parallel tasks equal to core count. (A) is a hard limit but not the critical constraint for performance. (C) is incorrect - NPU architecture doesn't use traditional cache coherency. (D) is incorrect - HIVM doesn't use a thread model in this sense. This requires synthesizing knowledge from HighPerformanceGuide.md's parallelism recommendations, driver.py's get_aicore_num/get_aivector_core_num functions, and understanding the NPU's execution model where blocks map to physical cores.

---

## triton-ascend - Problem 7 (Multiple Choice)
Which mechanisms does the CANN runtime integration layer use to bridge between Triton's execution model and Ascend NPU's hardware capabilities?

- (A) The rtKernelLaunch API that accepts a packed argument structure containing kernel parameters, grid configuration, and runtime metadata
- (B) The OpCommand task queue system that integrates with PyTorch NPU's execution graph for better scheduling
- (C) The syncBlockLock mechanism that provides inter-block synchronization primitives allocated in device memory
- (D) The workspace memory allocation that provides per-block temporary storage managed by the NPU caching allocator
- (E) The MsprofApi profiling hooks that report kernel execution timing and tensor metadata to the profiling infrastructure

**Correct Answer: A, B, C, D, E**

**Explanation:** This tests understanding of CANN runtime integration mechanisms. From driver.py's generate_npu_wrapper_src: (A) is correct - rtKernelLaunch is the core launch API. (B) is correct - OpCommand is used when TRITON_ENABLE_TASKQUEUE is enabled. (C) is correct - syncBlockLock is allocated for inter-block synchronization. (D) is correct - workspace memory is allocated per-block. (E) is correct - MsprofApi hooks are integrated for profiling. All five mechanisms are present in the generated wrapper code and serve distinct purposes in the runtime integration. This requires detailed understanding of driver.py's wrapper generation, the CANN runtime APIs, and how they map to Triton's execution model.

---

## triton-ascend - Problem 8 (Multiple Choice)
When implementing intra-core parallelism through sub-block tiling, which factors must be balanced to achieve optimal performance?

- (A) The sub-block size must be small enough that all input, output, and intermediate tensors fit within the 192KB on-chip memory limit
- (B) The sub-block size should be chosen to maximize data reuse within the on-chip memory across loop iterations
- (C) The number of sub-blocks must be a power of 2 to align with the NPU's SIMD width requirements
- (D) The sub-block tiling strategy must account for the compiler's automatic multi-buffering, which multiplies memory requirements
- (E) The sub-block size should be tuned using autotune to find the optimal balance between memory pressure and computational efficiency

**Correct Answer: A, B, D, E**

**Explanation:** This tests understanding of intra-core parallelism and memory optimization. From HighPerformanceGuide.md's sub-block discussion: (A) is correct - this is the fundamental constraint. (B) is correct - data reuse is a key optimization goal. (D) is correct - multi-buffering multiplies memory usage, affecting sub-block sizing. (E) is correct - the guide explicitly recommends using autotune for sub-block sizing. (C) is incorrect - there's no power-of-2 requirement for sub-blocks. This requires synthesizing knowledge from HighPerformanceGuide.md's tiling strategies, compiler.py's multi-buffer options, and understanding the memory hierarchy's impact on performance.

---

## triton-ascend - Problem 9 (Single Choice)
What is the architectural reason why the triton-adapter's triton-to-hivm pass is necessary before generating NPU binaries, rather than directly compiling Linalg to machine code?

- (A) HIVM provides a virtual instruction set that abstracts over different Ascend NPU generations, enabling forward compatibility
- (B) HIVM serves as an intermediate layer that performs NPU-specific optimizations like cube-vector workload balancing and memory hierarchy management that cannot be expressed in standard Linalg
- (C) The CANN runtime can only execute HIVM bytecode, not native machine instructions
- (D) HIVM enables dynamic compilation and adaptive optimization based on runtime profiling data

**Correct Answer: B**

**Explanation:** This tests understanding of HIVM's architectural role. From compiler.py's ttir_to_linalg and linalg_to_bin_enable_npu_compile: (B) is correct - HIVM is where NPU-specific optimizations occur, including the cube-vector balance, memory management, and other hardware-specific transformations that are beyond Linalg's abstraction level. (A) is partially true but not the primary reason. (C) is incorrect - CANN executes native binaries generated from HIVM. (D) is incorrect - HIVM is a compile-time intermediate representation, not a runtime system. This requires understanding the compilation flow from compiler.py, the HIVM compilation options, and the architectural constraints that necessitate this intermediate layer.

---

## triton-ascend - Problem 10 (Multiple Choice)
Which compilation pipeline design decisions enable triton-ascend to maintain compatibility with upstream Triton while supporting Ascend-specific optimizations?

- (A) The use of a separate triton_patch directory that overrides specific runtime and compiler components without modifying upstream code
- (B) The backend plugin architecture that allows registering AscendBackend as an alternative to GPU backends
- (C) The preservation of GPU-related MLIR dialects (NVVM, AMDGPU) in the build system to maintain upstream compatibility
- (D) The use of environment variables like TRITON_ASCEND_COMPILE_SPEED_OPT to control Ascend-specific behavior
- (E) The complete reimplementation of Triton's frontend in Python to support NPU-specific language extensions

**Correct Answer: A, B, C, D**

**Explanation:** This tests understanding of the compatibility architecture. From setup.py, CMakeLists.txt, and the repository structure: (A) is correct - triton_patch provides selective overrides. (B) is correct - the backend plugin system enables clean separation. (C) is correct - maintaining GPU dialects minimizes divergence. (D) is correct - environment variables provide Ascend-specific controls. (E) is incorrect - the frontend is not reimplemented; extensions are added through patches. This requires understanding setup.py's package structure, the backend registration mechanism in compiler.py, and the overall architecture that balances upstream compatibility with Ascend-specific needs.

---

## triton-ascend - Problem 11 (Multiple Choice)
Which aspects of the on-chip memory constrained programming model require coordination between the Triton kernel author, the compiler, and the runtime system?

- (A) The kernel author must design block and sub-block sizes that respect memory limits, which the compiler validates during HIVM compilation
- (B) The compiler must insert memory allocation and deallocation operations that the runtime executes to manage the 192KB on-chip memory
- (C) The runtime must allocate workspace memory based on compiler-provided metadata, which the kernel accesses through generated wrapper code
- (D) The kernel author uses autotune to explore the memory-performance trade-off space, which the compiler evaluates by attempting compilation with different configurations
- (E) The compiler automatically partitions large tensors across multiple blocks, which the runtime coordinates through inter-block synchronization

**Correct Answer: A, C, D**

**Explanation:** This tests understanding of the multi-layer memory management system. From HighPerformanceGuide.md, compiler.py, and driver.py: (A) is correct - authors design sizes, compiler validates. (C) is correct - runtime allocates workspace based on compiler metadata. (D) is correct - autotune explores configurations through compilation attempts. (B) is incorrect - on-chip memory is managed by HIVM/hardware, not explicit runtime operations. (E) is incorrect - tensor partitioning is a kernel design decision, not automatic. This requires understanding the interaction between HighPerformanceGuide.md's programming model, compiler.py's workspace_size metadata, driver.py's memory allocation, and the autotune mechanism.

---

## triton-ascend - Problem 12 (Multiple Choice)
In the context of CANN runtime integration, which design patterns enable efficient kernel launching while maintaining compatibility with PyTorch's execution model?

- (A) The generated C++ launcher wrapper that marshals Python arguments into the packed structure expected by rtKernelLaunch
- (B) The OpCommand task queue integration that allows kernels to participate in PyTorch's graph optimization and scheduling
- (C) The use of torch_npu's NPUCachingAllocator for workspace and synchronization memory to enable memory reuse across kernel launches
- (D) The MsProf profiling integration that reports kernel execution to PyTorch's profiler infrastructure
- (E) The automatic conversion of Triton tensors to torch_npu tensors through the data_ptr() method

**Correct Answer: A, B, C, E**

**Explanation:** This tests understanding of PyTorch integration mechanisms. From driver.py's generate_npu_wrapper_src and make_npu_launcher_stub: (A) is correct - the wrapper bridges Python and C++ calling conventions. (B) is correct - OpCommand integrates with PyTorch's execution model. (C) is correct - using NPUCachingAllocator enables memory pooling. (E) is correct - the wrapper extracts data_ptr() from tensor objects. (D) is incorrect - MsProf is CANN's profiler, not PyTorch's. This requires understanding driver.py's launcher generation, the OpCommand integration when TRITON_ENABLE_TASKQUEUE is enabled, and how the wrapper code bridges between Python/PyTorch and CANN runtime.

---

## triton-ascend - Problem 13 (Multiple Choice)
Which factors determine whether a Triton kernel can successfully exploit multi-buffer pipeline optimization to overlap computation and data movement?

- (A) The kernel must have sufficient computational intensity that computation time exceeds data movement time, making overlap beneficial
- (B) The on-chip memory must be large enough to hold multiple buffer copies plus intermediate computation results
- (C) The kernel's memory access pattern must be predictable enough for the compiler to insert prefetch operations
- (D) The HIVM scheduler must be able to identify independent loop iterations that can be pipelined
- (E) The kernel must explicitly use the triton.language.multibuffer API to enable the optimization

**Correct Answer: A, B, D**

**Explanation:** This tests understanding of multi-buffer optimization requirements. From HighPerformanceGuide.md and compiler.py's multi-buffer options: (A) is correct - overlap only helps if there's computation to overlap with. (B) is correct - memory must accommodate multiple buffers. (D) is correct - the compiler must identify pipelineable iterations. (C) is incorrect - prefetching is not the mechanism; multi-buffering works through software pipelining. (E) is incorrect - both explicit API and automatic enable-auto-multi-buffer can trigger the optimization. This requires synthesizing knowledge from HighPerformanceGuide.md's multibuffer discussion, compiler.py's automatic and manual multi-buffer options, and understanding the general requirements for software pipelining optimizations.

---

## triton-ascend - Problem 14 (Multiple Choice)
Which compilation pipeline stages are responsible for transforming Triton's high-level memory abstractions (pointers, loads, stores) into NPU-specific memory operations?

- (A) The discrete-mask-access-conversion pass that transforms masked memory operations into structured access patterns
- (B) The triton-to-linalg pass that converts pointer arithmetic and memory operations into memref operations
- (C) The HIVM compilation stage that maps memref operations to NPU's memory hierarchy (global memory, L1 cache, registers)
- (D) The LLVM IR generation stage that converts memref operations into LLVM load/store instructions
- (E) The rtMemcpy calls in the generated launcher that handle host-device memory transfers

**Correct Answer: A, B, C**

**Explanation:** This tests understanding of memory abstraction lowering. From compiler.py's pipeline: (A) is correct - this pass normalizes memory access patterns. (B) is correct - triton-to-linalg converts Triton's pointer model to MLIR's memref abstraction. (C) is correct - HIVM maps to physical memory hierarchy. (D) is incorrect - NPU backend doesn't go through LLVM IR for memory operations; HIVM handles this. (E) is incorrect - rtMemcpy is for initialization, not kernel memory operations. This requires understanding the transformation sequence in compiler.py's ttir_to_linalg and linalg_to_bin_enable_npu_compile, and how memory abstractions are progressively lowered through the pipeline.

---

## triton-ascend - Problem 15 (Single Choice)
When designing a FlashAttention kernel for Ascend NPU, what is the primary architectural challenge that differs from GPU implementations?

- (A) The lack of shared memory atomics on NPU requires redesigning the attention score accumulation to use only local memory operations
- (B) The 192KB on-chip memory limit requires more aggressive block tiling compared to GPU's larger shared memory, potentially impacting the algorithm's ability to keep attention matrices on-chip
- (C) The separation of cube and vector cores requires splitting the attention computation into separate matrix multiplication and softmax kernels
- (D) The CANN runtime's lack of dynamic parallelism support prevents implementing the adaptive block scheduling used in GPU FlashAttention

**Correct Answer: B**

**Explanation:** This tests understanding of architectural constraints in complex algorithms. From HighPerformanceGuide.md's memory constraints and the FlashAttention example in tutorials: (B) is correct - the 192KB limit is significantly smaller than GPU shared memory (e.g., 48-164KB per SM on NVIDIA GPUs, but with many SMs), requiring more careful tiling to keep the working set on-chip. This is the fundamental challenge. (A) is incorrect - atomics are available. (C) is incorrect - mixed operations can execute in a single kernel. (D) is incorrect - the block-level parallelism model is similar. This requires understanding HighPerformanceGuide.md's memory constraints, the FlashAttention algorithm's memory requirements, and comparing NPU vs GPU memory hierarchies.

---

## triton-ascend - Problem 16 (Multiple Choice)
Which mechanisms enable the triton-ascend compiler to provide meaningful error messages when compilation fails due to resource constraints?

- (A) The HIVM compiler's error reporting that indicates which resources (memory, registers) exceeded limits
- (B) The triton-adapter's validation passes that check tensor sizes against known hardware limits before HIVM compilation
- (C) The TRITON_ASCEND_COMPILE_SPEED_OPT environment variable that controls whether compilation failures are treated as errors or warnings
- (D) The debug mode that dumps intermediate IR at each compilation stage, allowing developers to identify where resource allocation fails
- (E) The autotune framework that automatically retries compilation with smaller block sizes when resource limits are exceeded

**Correct Answer: A, C, D**

**Explanation:** This tests understanding of error handling and debugging mechanisms. From compiler.py and utils.py: (A) is correct - HIVM reports resource constraint violations. (C) is correct - this env var controls error handling behavior. (D) is correct - debug mode enables IR dumping for diagnosis. (B) is incorrect - triton-adapter doesn't validate against hardware limits; HIVM does. (E) is incorrect - autotune explores configurations but doesn't automatically retry on failures. This requires understanding compiler.py's debug mode and dump_manager, utils.py's environment variable handling, and the overall error propagation through the compilation pipeline.

---

## triton-ascend - Problem 17 (Multiple Choice)
Which design decisions in the Linalg-to-HIVM lowering process enable efficient mapping to Ascend NPU's memory hierarchy?

- (A) The bubble-up-operation pass that reorders operations to expose opportunities for keeping data in on-chip memory across operations
- (B) The enable-hivm-inject-barrier-all-sync flag that controls insertion of memory barriers to ensure correct ordering
- (C) The enable-nd2nz-on-vector flag that controls data layout transformations for vector operations
- (D) The tile-mix-vector-loop and tile-mix-cube-loop parameters that control how loops are tiled for different memory levels
- (E) The one-shot-bufferize pass that determines which tensors are allocated in on-chip vs global memory

**Correct Answer: A, B, C, D**

**Explanation:** This tests understanding of memory hierarchy mapping. From compiler.py's compilation options and passes: (A) is correct - bubble-up exposes optimization opportunities. (B) is correct - barrier insertion affects memory ordering. (C) is correct - data layout impacts memory efficiency. (D) is correct - loop tiling directly affects memory hierarchy usage. (E) is incorrect - one-shot-bufferize is in the CPU backend path (linalg_to_llir), not the NPU path. This requires understanding compiler.py's ttir_to_linalg passes, the HIVM compilation options in linalg_to_bin_enable_npu_compile, and how these transformations affect memory hierarchy utilization.

---

## triton-ascend - Problem 18 (Single Choice)
Why does the tensor kind metadata system distinguish between INPUT_OUTPUT (2) tensors and separate INPUT (0) + OUTPUT (1) tensors in the profiling infrastructure?

- (A) INPUT_OUTPUT tensors require special memory allocation that supports both read and write operations, while INPUT/OUTPUT tensors can use read-only or write-only memory
- (B) The profiling system reports INPUT_OUTPUT tensors twice (once as input, once as output) to accurately reflect their contribution to both input and output bandwidth
- (C) INPUT_OUTPUT tensors indicate in-place operations that have different performance characteristics and memory traffic patterns than separate input/output tensors
- (D) The CANN runtime uses tensor kind to determine whether to perform memory prefetching, which is only beneficial for INPUT tensors

**Correct Answer: B**

**Explanation:** This tests understanding of the tensor kind system's semantics. From driver.py's MsprofTensorInfo reporting: (B) is correct - the code explicitly shows that INPUT_OUTPUT tensors are reported twice, once as MSPROF_GE_TENSOR_TYPE_INPUT and once as MSPROF_GE_TENSOR_TYPE_OUTPUT, to accurately reflect their dual role in bandwidth analysis. (A) is incorrect - memory allocation doesn't depend on tensor_kind. (C) is partially true but not the primary reason for the distinction. (D) is incorrect - tensor_kind is for profiling, not runtime optimization. This requires understanding driver.py's profiling code that iterates over tensor_kinds and reports them to MsProf.

---

## triton-ascend - Problem 19 (Multiple Choice)
Which factors influence the effectiveness of the HIVM scheduler's automatic cube-vector balance optimization?

- (A) The ratio of compute-intensive operations (matrix multiplications) to memory-intensive operations (element-wise ops) in the kernel
- (B) The enable-hivm-auto-cv-balance compilation flag that enables the analysis and optimization
- (C) The availability of sufficient on-chip memory to buffer data being transferred between cube and vector cores
- (D) The kernel's control flow structure, as complex branching limits the scheduler's ability to statically analyze workload distribution
- (E) The unit-flag synchronization parameter that controls fine-grained synchronization between cube and vector operations

**Correct Answer: A, B, D**

**Explanation:** This tests understanding of HIVM's load balancing mechanisms. From compiler.py's HIVM options: (A) is correct - the operation mix determines the theoretical optimal balance. (B) is correct - this flag enables the optimization. (D) is correct - static analysis is limited by control flow complexity. (C) is incorrect - while memory matters, it's not specific to CV balance. (E) is incorrect - unit-flag is for synchronization, not load balancing. This requires understanding compiler.py's enable_hivm_auto_cv_balance and related options, and the general principles of static workload analysis in compilers.

---

## triton-ascend - Problem 20 (Multiple Choice)
Which aspects of the compilation pipeline are affected by the choice between reg-based and non-reg-based HIVM compilation modes?

- (A) The instruction scheduling strategy used by the HIVM backend
- (B) The memory allocation strategy for intermediate values
- (C) The compilation flags passed to bishengir-compile (--reg-based vs --enable-hivm-compile)
- (D) The binary format of the generated kernel object file
- (E) The runtime performance characteristics of the compiled kernel

**Correct Answer: A, B, C, E**

**Explanation:** This tests understanding of HIVM compilation modes. From compiler.py's linalg_to_bin_enable_npu_compile and utils.py's _check_bishengir_is_regbased: (C) is correct - different flags are passed. (A), (B), (E) are all correct - reg-based vs non-reg-based represents fundamentally different compilation strategies that affect scheduling, allocation, and performance. (D) is incorrect - the binary format (kernel.o vs kernel_reloc.o) is determined by API version, not reg-based mode. This requires understanding compiler.py's bishengir_hivm_opt selection and the implications of different HIVM compilation strategies.

---

## triton-ascend - Problem 21 (Single Choice)
When the OpCommand task queue integration is enabled (TRITON_ENABLE_TASKQUEUE=true), what is the primary benefit compared to direct rtKernelLaunch calls?

- (A) OpCommand enables automatic kernel fusion by batching multiple kernel launches into a single NPU submission
- (B) OpCommand integrates with PyTorch's execution graph, enabling better scheduling and overlap with other PyTorch operations
- (C) OpCommand provides automatic error recovery and retry mechanisms for failed kernel launches
- (D) OpCommand reduces kernel launch overhead by caching compiled kernels more efficiently

**Correct Answer: B**

**Explanation:** This tests understanding of OpCommand integration. From driver.py's generate_npu_wrapper_src: (B) is correct - OpCommand integrates with PyTorch NPU's execution model (at_npu::native::OpCommand), allowing the kernel to participate in PyTorch's graph scheduling and enabling better overlap with other operations. (A) is incorrect - fusion is not automatic. (C) is incorrect - no automatic retry. (D) is incorrect - caching is separate. This requires understanding driver.py's conditional OpCommand usage and the broader PyTorch NPU execution model.

---

## triton-ascend - Problem 22 (Multiple Choice)
Which compilation pipeline stages must successfully complete before the workspace_size metadata can be determined for a kernel?

- (A) The triton-to-linalg pass that converts high-level operations into structured loop nests
- (B) The HIVM compilation that analyzes memory requirements and generates the workspace inference callback
- (C) The bishengir-compile invocation that produces the libkernel.so containing workspace inference functions
- (D) The LLVM IR generation that determines stack frame sizes
- (E) The kernel binary generation that finalizes the memory layout

**Correct Answer: B, C**

**Explanation:** This tests understanding of workspace size determination. From compiler.py's linalg_to_bin_enable_npu_compile: (B) and (C) are correct - workspace_size is determined by calling the _infer_workspace_shape_function callback from libkernel.so, which is generated by bishengir-compile during HIVM compilation. (A) is incorrect - Linalg conversion doesn't determine workspace size. (D) is incorrect - LLVM IR is not in the NPU path. (E) is incorrect - workspace size is determined before final binary generation. This requires understanding compiler.py's __get_metadata_attr_by_callback mechanism and when libkernel.so becomes available in the compilation pipeline.

---

