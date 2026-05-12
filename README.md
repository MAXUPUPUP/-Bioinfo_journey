# Ai_infra_journey
Let's do AI Infra!!!
## AI Infra 学习路线(2 年完整版)

### 一、先明确目标岗位画像

**AI Infra 工程师在大厂干什么:**

- 大模型训练框架开发(分布式训练、并行策略)
- 推理引擎开发与优化(vLLM、SGLang 这类)
- GPU 算子开发(CUDA Kernel 优化)
- 训练/推理集群调度与资源管理
- 模型压缩、量化、蒸馏的工程化

**对应招聘要求(校招 SP/SSP 级):**

- 扎实的 C++ 和 Python 工程能力
- CUDA 编程经验
- 熟悉至少一个深度学习框架的内部机制(PyTorch 优先)
- 有开源贡献或硬核项目经历
- 操作系统、计算机体系结构、并行计算基础扎实

---

### 二、技术栈全景

```
应用层      vLLM / SGLang / DeepSpeed / Megatron-LM
框架层      PyTorch 内核 / Triton / TensorRT-LLM  
算子层      CUDA / cuDNN / cuBLAS / NCCL
硬件层      GPU 架构 / 内存层级 / NVLink / RDMA
基础层      C++ / Python / Linux / 计算机体系结构
```

**学习顺序:从下往上打,不能反。**

---

### 三、分阶段路线(共 6 个阶段)

#### 阶段 1:夯实基础底座(4 个月)

**目标**:让你具备啃硬骨头的能力。这个阶段不碰 AI,纯打地基。

**1.1 C++ 进阶**(必须,优先级最高)

- 教材:《Effective Modern C++》+《C++ Primer》关键章节
- 重点:RAII、智能指针、移动语义、模板、并发(std::thread、原子操作、内存模型)
- 实战:用 C++ 实现一个线程池 + 一个简单的内存池
- 时间:每天 1.5 小时,持续 2 个月

**1.2 计算机体系结构**(AI Infra 的命脉)

- 教材:CSAPP(深入理解计算机系统),重点啃第 3、5、6、9、12 章
- 课程:CMU 15-213(CSAPP 配套课程,B 站有)
- 重点:CPU 流水线、Cache 层级、内存模型、虚拟内存、并发
- **为什么重要**:GPU 优化的所有思想都源自 CPU 体系结构,Cache 命中、流水线、并行这些概念是通用的

**1.3 Linux 系统编程**

- 教材:《Linux 高性能服务器编程》或《UNIX 环境高级编程》关键章节
- 重点:进程/线程、IO 模型、内存管理、性能分析工具(perf、strace、gdb)
- 实战:用 epoll 写一个简单的 HTTP 服务器

**1.4 Python 工程化**(并行进行)

- 不是只会写脚本,要懂:类型注解、async/await、装饰器、上下文管理器、C 扩展(ctypes/pybind11)
- 实战:用 pybind11 把一个 C++ 函数封装成 Python 模块

**阶段产出**:一个用 C++ 实现的小型项目(线程池 + 内存池 + 简易日志库),代码上 GitHub。

---

#### 阶段 2:深度学习核心原理(3 个月)

**目标**:理解模型,但**不是为了做算法,而是为了理解优化的对象**。

**2.1 深度学习基础**

- 课程:李沐《动手学深度学习》(PyTorch 版),代码全部跟着敲一遍
- 不需要看完所有章节,重点:基础+CNN+RNN+Transformer+优化器
- 时间:1 个月,每天 2 小时

**2.2 Transformer 深度理解**

- 论文:Attention Is All You Need(精读)
- 实战:**从零手写一个 Transformer**(不看现成代码),实现 Multi-Head Attention、Position Encoding、训练一个小语言模型在 TinyShakespeare 上跑通
- 参考:Karpathy 的 nanoGPT(读完源码)、minGPT
- **这一步极其重要,跳过就是后面所有 LLM 推理优化的认知断层**

**2.3 PyTorch 核心机制**

- 必须搞懂:autograd 机制、计算图、Tensor 内存布局、CUDA Stream、torch.compile 原理概览
- 实战:实现一个自定义 autograd Function、写一个自定义 nn.Module 并测试反向传播

**阶段产出**:GitHub 上一个手写 Transformer 项目 + 一篇博客解释 Attention 的实现细节。

**阶段 1,2 检查点(8 月底)**:

- ✅ GitHub 上有 2-3 个有质量的仓库
- ✅ C++ 能熟练写并发代码
- ✅ 手写过 Transformer,理解每一行
- ✅ 完成一段日常实习或硬核自学
- ❌ 如果上面任何一项没达成,**警报**

---

#### 阶段 3:CUDA 编程(3 个月,集中突破)

**这是 AI Infra 的核心壁垒,也是你和绝大多数候选人拉开差距的地方。**

**3.1 CUDA 入门**

- 教材:《CUDA C Programming Guide》(NVIDIA 官方)+《Professional CUDA C Programming》
- 课程:**Programming Massively Parallel Processors**(PMPP,这是 CUDA 圣经,B 站有伊利诺伊大学的课程视频)
- 重点概念:线程层级(grid/block/warp/thread)、内存层级(global/shared/register)、Warp 调度、同步、Occupancy

**3.2 经典 Kernel 实战**(按顺序写)

每一个都自己实现,然后和 cuBLAS 对比性能,持续优化:

1. 向量加法(Hello CUDA)
2. 矩阵转置(理解 Coalesced Access)
3. **Reduce / Sum**(经典优化案例,从 naive 到 warp shuffle 七版优化)
4. **GEMM 矩阵乘法**(CUDA 的 "Hello World",至少做到 cuBLAS 的 70% 性能)
5. **Softmax**(理解 LLM 推理为什么要 Online Softmax)
6. **LayerNorm / RMSNorm**
7. **Flash Attention v1**(读论文,然后自己实现简化版)

**3.3 性能分析工具**

- Nsight Compute:分析 Kernel 性能瓶颈
- Nsight Systems:分析整体时间线
- nvprof / ncu 命令行工具

**3.4 推荐资源**

- 推荐 GitHub 仓库:`NVIDIA/cuda-samples`、`Bruce-Lee-LY/cuda_hgemm`(中文社区高质量 GEMM 实现)
- 跟着这些项目一行行写、改、调,比看十本书都管用

**阶段产出**:GitHub 仓库,包含 7 个 Kernel 的逐步优化版本,每个都有 README 说明优化思路和性能数据。**这个仓库就是你 AI Infra 求职最值钱的简历。**


**阶段 3检查点(12 月底)**:

- ✅ CUDA Kernel 仓库:7+ 个 Kernel,每个有性能数据和优化思路
- ✅ 简化版 Flash Attention 跑通
- ✅ 至少 2 篇 CUDA 主题深度博客
- ✅ vLLM 源码读过核心模块
- ❌ 如果 CUDA 这关没过,1 月必须紧急补救,否则放弃 AI Infra 主路线

---

#### 阶段 4:LLM 推理引擎(4 个月)

**目标**:深入理解一个工业级推理引擎,有能力贡献代码。

**4.1 LLM 推理基础**

必须吃透的概念:

- Prefill vs Decode 阶段的本质差异
- KV Cache 的原理、内存占用计算、PageAttention(vLLM 核心)
- Continuous Batching(动态批处理)
- 投机解码(Speculative Decoding)
- 量化:INT8、INT4、AWQ、GPTQ、FP8 的原理和取舍
- 并行策略:Tensor Parallel、Pipeline Parallel、Expert Parallel 的差异

**推荐论文**(精读,不要只看博客):

- Flash Attention v1 / v2 / v3
- PagedAttention(vLLM)
- Efficient Memory Management for Large Language Model Serving(vLLM 论文)
- SGLang 的 RadixAttention 论文

**4.2 选定一个推理引擎深入**

**强烈推荐 vLLM**(代码相对清晰、社区活跃、招聘市场认可度高)

学习路径:

1. 从 README 开始,跑通几个 Demo
2. 读源码:从 `LLMEngine` 入口往下,理解调度器(Scheduler)→ Worker → ModelRunner → Attention Backend 的整条路径
3. 读完 PagedAttention 的 CUDA Kernel 实现(`vllm/csrc/attention/`)
4. 跟踪一次完整请求的生命周期,画出时序图
5. 关注 PR,看大家在改什么、为什么改

**4.3 自己造轮子**(可选但极加分)

实现一个 mini 推理引擎,支持:

- 一个开源小模型(Qwen2-0.5B 或 TinyLlama)
- KV Cache
- Continuous Batching
- 一个简单的 PagedAttention

代码量约 2000-5000 行,从 0 写完后,你对 LLM 推理的理解会甩开 99% 的应届生。

**阶段产出**:

- vLLM 至少 1 个 PR(哪怕修文档、改测试、加小特性都行,**有 PR 被 merge 这件事本身在简历上就是稀缺资产**)
- 自己的 mini 推理引擎(GitHub 项目)
- 2-3 篇深度博客:vLLM 调度器源码解析、PagedAttention 实现解析、KV Cache 优化

---

#### 阶段 5:分布式训练 + 实习冲刺(4 个月)

**这个阶段双线并行:技术深化 + 暑期实习准备**

**5.1 分布式训练基础**

核心概念:

- 数据并行(DP / DDP / FSDP)
- 张量并行(Megatron-LM 风格)
- 流水线并行(GPipe / PipeDream / 1F1B)
- 3D 并行的组合
- ZeRO 优化(Stage 1/2/3)
- 通信原语:AllReduce、AllGather、ReduceScatter,以及 NCCL

**资源**:

- 论文:Megatron-LM 系列、ZeRO 系列、GPipe
- 框架:DeepSpeed(读 README + 关键模块代码)、PyTorch FSDP 文档
- 视频课程:Stanford CS336(Language Modeling from Scratch),里面有非常硬核的训练系统讲解

**5.2 实战**

- 多卡跑通一个小模型的 DDP 训练(单机 2-4 卡,本地或云上租)
- 用 DeepSpeed ZeRO-3 训练一个略大的模型,体会显存优化效果
- 写一篇博客对比 DDP vs FSDP vs DeepSpeed 的差异

**5.3 暑期实习冲刺**(从 2 月开始投递)

**目标公司分层**:

- 第一档:字节(AML/Seed)、阿里(PAI/通义)、腾讯(混元/机器学习平台)、华为(昇腾/盘古)、Moonshot/智谱/Minimax/百川等大模型公司
- 第二档:商汤、旷视、英伟达中国区、AMD 中国区
- 第三档:各家云厂商的 AI 平台团队

**面试准备**:

- 算法刷题:LeetCode Hot 100 + 剑指 Offer 必须刷透
- 八股:操作系统、计算机网络、C++、Python、CUDA 八股
- 项目深挖:你的 CUDA Kernel 项目和 vLLM PR 必须能讲到每一行细节
- 系统设计:能设计一个简单的推理服务架构
- 论文准备:能讲清楚 Flash Attention、PagedAttention 的原理

**阶段产出**:拿到大厂 AI Infra 暑期实习 offer。

**阶段 5检查点(8 月底)**:

- ✅ 拿到 return offer 口头/书面承诺
- ✅ 实习经历可以在简历上写出 3-4 个具体亮点
- ✅ 实习中接触到的真实问题,可以讲到非常细的程度
- ❌ 如果没拿 return,9 月秋招更要加倍准备

---

#### 阶段 6:暑期实习 + 秋招收割(4 个月)

**6.1 实习期间的核心任务**

- 把分配到的项目做出彩,**目标是拿 return offer**
- 主动接复杂任务,哪怕加班也要争
- 和 mentor、leader 多沟通,了解团队判断你的标准
- 学习真实工业级代码库的设计模式
- 实习结束前主动要 return offer 评价

**6.2 秋招阶段**

- 9 月开始铺简历(实习还没结束就要开始)
- 优先级:return offer > 头部大厂 SSP > 大模型公司 SP > 其他
- 多投不要押宝,但准备时间要分配好
- 谈 offer 的时候用 return offer 当锚

---

### 四、关键资源清单(精挑细选,不堆砌)

**书**:

- CSAPP(必读)
- 《Effective Modern C++》
- 《Professional CUDA C Programming》

**课程**:

- CMU 15-213(CSAPP)
- PMPP(CUDA 圣经课)
- 李沐《动手学深度学习》
- Stanford CS336(Language Modeling from Scratch)
- MIT 6.5940(TinyML and Efficient Deep Learning)

**论文(按阅读顺序)**:

- Attention Is All You Need
- Flash Attention v1/v2
- Efficient Memory Management for LLM Serving(vLLM)
- Megatron-LM
- ZeRO
- GPipe

**GitHub 项目(必读源码)**:

- nanoGPT(入门 Transformer)
- vLLM(LLM 推理)
- llama.cpp(纯 C++ 推理实现,代码风格不同但思路启发大)
- DeepSpeed(分布式训练)

**信息源**:

- GitHub Trending(每周看)
- 知乎话题:CUDA、AI Infra
- 公众号:GiantPandaCV、机器之心
- Twitter/X:关注 vLLM、SGLang、PyTorch 核心开发者

---

### 五、几个生死攸关的提醒

**1. CUDA 是命门,不要绕开。** 很多人想做 AI Infra 但不想啃 CUDA,最后只能去做 MLOps 或调度系统。**真正核心、薪资最高的岗位都要 CUDA**。这是壁垒,啃下来就是护城河。

**2. 必须有可证明的产出。** GitHub 仓库 + 开源 PR + 博客,这三样是你的硬通货。AI Infra 招聘极看重"硬核证据",八股答得再好,不如一个被 merge 的 vLLM PR。

**3. 数学基础够用就行,别陷进去。** 你是做 Infra 的不是做算法的,线性代数、微积分、概率论懂基本概念即可。把时间花在系统优化和工程能力上。

**4. 不要追新框架。** 半年一个新框架,但底层原理(Attention、KV Cache、并行策略、CUDA 优化)是稳定的。**学原理,框架自然就会了。**

**5. 持续高强度,不要佛。** AI Infra 的招聘门槛在涨,2028 年的竞争只会比现在更激烈。这条路线安排是高强度的,平均每天 4-6 小时学习时间,周末更多。**有这个觉悟再走这条路,否则换个方向。**

**6. 找到同路人。** 进 CUDA / vLLM / 高性能计算的技术群,在知乎和 GitHub 上关注同方向的人。这个圈子小,水平高,信息密度大。


### 时间分配的硬约束

整个 16 个月,你必须接受几个**残酷的现实**:

**1. 周均学习时间不能低于 40 小时**(除考试周和大型项目截止周)

- 工作日:每天 4-5 小时
- 周末:每天 8-10 小时
- 寒暑假:全天高强度

**2. 算法刷题不能停**

- LeetCode 每天 1-2 题,16 个月 600+ 题保底
- Hot 100 + 剑指 Offer 必须刷透
- 临近实习/秋招前 1 个月集中模拟

**3. 英语必须过关**

- 每周至少读 1 篇英文论文 / 技术博客
- 论文阅读不流畅 = AI Infra 死路

**4. 输出必须持续**

- GitHub 每周必有 commit
- 每月至少 1 篇博客
- 这两件事 16 个月不间断

### 几个生死线判断

**5 月底生死线**:C++ 进度。如果连 C++ 并发都吃力,后面 CUDA 直接劝退,**这种情况建议把主线切到 Java 后端 + AI 应用方向**。

**12 月底生死线**:CUDA 突破。如果 CUDA Kernel 仓库做不出来,3 月份你投不进任何头部 AI Infra 团队,**这种情况要么再赌一波 1-2 月寒假冲刺,要么主线切换**。

**3 月底生死线**:实习 offer。如果 3 月底连面试都没几场,要么是简历有问题(项目分量不够),要么是赛道太挤,**这种情况立刻把 Java 后端实习当主目标**。

### 心态准备

我必须很直白地告诉你:**这条路线 16 个月走下来,大概率会有 30% 的人坚持不下来,40% 的人完成度不到 60%,真正打满全程的不到 30%**。

你会经历:

- CUDA 写不出来想砸键盘的崩溃
- 论文看不懂的挫败
- 投实习被拒的焦虑
- 同学保研直博、出国、躺平,你还在啃 GEMM 优化的孤独
- 寒暑假别人玩你学的不平衡

**能熬过这些的人,2027 年秋招会非常好看**。熬不过的,中途切到 Java 后端也不是失败,是合理的战略调整。




