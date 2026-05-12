# 🧬 Bioinfo_journey

Let's do AI for Science!!!

## AI for Science / Bioinfo 学习路线(2 年完整版 · 28卒)

> 东大津田研 M1 (28卒) | 武汉理工软工本科 | 从 CS 跨入 BioInfo
> 📅 Started: 2026.05 | 🎯 Target: 2028.04 入职 Research Engineer
> 🔬 Lab: Tsuda Lab (机器学习 × 分子设计 × 生物信息)

---

### 一、先明确目标岗位画像

**Research Engineer (AI for Science 方向) 在公司干什么:**

- 把研究方向的算法工程化、落地化(论文 → production code)
- 蛋白质/分子模型的训练与推理优化(AlphaFold、ESM、RFdiffusion生态)
- 科学计算工作流的工程实现(数据pipeline、实验自动化)
- 模型评估、benchmark、可视化工具
- 研究员和工程团队之间的桥梁角色

**对应招聘要求(校招新卒,Preferred Networks / Isomorphic Labs / 字节AML / 大药企AI):**

- 扎实的 Python + PyTorch 工程能力
- 至少一个 AI for Science 领域的深度积累(分子 / 蛋白 / 基因组 / 量子化学)
- 能读懂顶会论文,且能复现核心idea
- 有研究室paper产出 或 高质量开源项目
- 数学基础(线代/概率/优化)够用
- **学术圈和工业界的双重背书**(导师推荐 + GitHub作品)

**和纯 SWE / 纯 PhD 的关键区别:**

```
纯 SWE 路线:    LeetCode 800+ → 进 Mercari 做后端
纯 PhD 路线:    顶会论文 → 读博 → 5年后进 Research Scientist
你的路线:      研究够扎实 + 工程够能打 → 1年内进 Research Engineer
```

**这条路的稀缺性 = 你的护城河。** 纯 PhD 不会写 production code,纯 SWE 看不懂论文,**你在两者交界处独占一块市场**。

---

### 二、技术栈全景

```
应用层      AlphaFold / ESM / RFdiffusion / Boltz / ChemTS(你导师产品)
模型层      Transformer / GNN / Diffusion / Equivariant Networks
框架层      PyTorch / PyTorch Geometric / DGL / JAX(可选)
科学库      RDKit(分子) / Biopython(序列) / OpenMM(动力学) / ASE(材料)
数据层      HuggingFace / ProteinNet / PDB / ChEMBL / Materials Project
工程层      Python / Docker / 实验跟踪(W&B) / 多GPU训练
研究层      论文阅读 / 实验设计 / 修論 / Workshop投稿
```

**学习顺序:Python → PyTorch → 选定一个垂直方向深耕 → 跨工程到产品**

---

### 三、分阶段路线(共 6 个阶段)

#### 阶段 1:Python + ML 基础 + 进入研究室(3 个月,5-7 月)

**目标**:让你具备读 paper 和跑代码的能力,同时融入研究室。

**1.1 Python 工程化**(必须扎实)

- 教材:《Effective Python》(Brett Slatkin)+《Fluent Python》关键章节
- 重点:类型注解、async、装饰器、上下文管理器、numpy/pandas 熟练
- **不是只会写脚本**,要懂模块化、testing(pytest)、virtualenv/poetry、调试技巧
- 实战:写一个能在 GitHub 公开的小工具(任何方向都行)
- 时间:每天 1.5 小时,持续 1 个月

**1.2 PyTorch 深度掌握**(优先级最高)

- 教程:PyTorch 官方 60-min tutorial → 完成
- 课程:**李沐《动手学深度学习》PyTorch版**(B站有,跟着敲代码)
- 必须懂:autograd、nn.Module 设计、DataLoader、CUDA Tensor、训练循环
- 实战:从零写一个 MLP 分类器 → CNN 做 MNIST → 简单的序列模型
- **每一行代码自己敲过,不要 copy paste**

**1.3 数学补漏**(够用即可,别陷进去)

- 线性代数:矩阵乘法、特征值、SVD(够用)
- 概率论:贝叶斯、分布、采样(够用)
- 优化:梯度下降、Adam、二阶方法基本概念
- 资源:3Blue1Brown 的可视化课(够你建立直觉)
- **警告**:你是 Research Engineer 不是数学家,深陷数学黑洞是常见死法

**1.4 进入津田研**(同样优先级 S 级)

- **每周参加 zemi**(研究室讨论会),不缺席
- 找 2-3 个 sempai 混熟(中国人优先,中文交流快)
- 读你导师 Google Scholar 最近 20 篇 paper 的标题 + abstract,精读 3 篇
- **5月内必须做的事**:预约和导师的第一次 1-on-1,说清楚你的 plan(双线策略、不读博)
- 加入研究室的 Slack/邮件列表/git仓库

**阶段产出**:

- GitHub 上至少 1 个用 PyTorch 写的小项目(MNIST 之外的)
- 能用 Python 流畅处理 numpy/pandas 数据
- 在研究室认识 3+ 个人
- 读懂导师至少 3 篇 paper

---

#### 阶段 2:选定垂直方向 + 第一个研究尝试(3 个月,8-10 月)

**目标**:不再"什么都学一点",而是**选一个垂直方向深耕**,这是你简历的差异化所在。

**2.1 决定你的"研究方向"**(8 月底前必须决定)

津田研可选方向:

- **方向 A:分子/材料设计**(导师组主推方向)
  - 工具栈:RDKit + DeepChem + PyTorch Geometric + ChemTS
  - 核心论文:ChemTS、JT-VAE、Diffusion for molecules

- **方向 B:蛋白质**(全球最热,但津田研非主线)
  - 工具栈:Biopython + ESM-2 + AlphaFold/OpenFold + ProteinMPNN
  - 核心论文:AlphaFold 2/3、ESM-2、RFdiffusion

- **方向 C:量子计算 + ML**(津田研独特方向,稀缺性极高)
  - 工具栈:Qiskit / PennyLane + PyTorch
  - 核心论文:VQE、QAOA、量子机器学习综述

**建议:选 A 或 C**,因为这是你导师真正的方向,推荐信和资源都更强。**B 最热但你在导师那里支持度不如 A/C**。

**2.2 跑通研究室一个开源代码**

- 优先:你导师组 GitHub(`tsudalab`)上的项目
- 推荐:**ChemTS**(津田老师的 baby,跑通它在面试时是巨大加分项)
- 跑通 → 改一个参数 → 改一个小模块 → 复现 paper 中的某个实验
- **不只是跑,要读懂每个文件做什么**

**2.3 深度学习核心 — Transformer**

- 论文:Attention Is All You Need(精读 3 遍)
- 实战:**从零手写一个 Transformer**(参考 Karpathy 的 nanoGPT,但自己写)
- 在小数据集(SMILES 分子序列 或 蛋白质序列)上训练一个小模型
- **这一步极其重要,跳过就是后面所有理解的认知断层**

**2.4 第一个研究小尝试**

- 在你方向上找一个小课题(和导师商量)
- 不需要原创,可以是"在数据集 X 上复现方法 Y"
- 目的:**为修論积累原材料,顺便练习实验方法**

**阶段产出**:

- GitHub 上一个完整的方向相关项目(可 deploy 或可复现)
- 一个手写的 mini Transformer 项目
- 在研究方向上有第一个 baseline 比较实验

---

**阶段 1, 2 检查点(2026 年 10 月底)**:

- ✅ 研究方向已确定,且和导师对齐
- ✅ GitHub 上有 2-3 个高质量项目
- ✅ 手写过 Transformer,理解每一行
- ✅ 读懂方向内 5-10 篇核心论文
- ✅ 和导师有过至少 2 次 1-on-1
- ❌ 如果上面任何一项没达成,**警报**。如果连方向都没定,11 月必须紧急决定

---

#### 阶段 3:深度技术栈 + 投递暑期实习(3 个月,11月-1月)

**这是你拉开差距的阶段。同期普通东大M1还在刷LeetCode,你已经在做研究方向的真实代码。**

**3.1 方向核心技术深入**(按你方向选)

**如果选了分子方向(方向A)**:

- 必学:**Graph Neural Network**(分子表达的核心)
  - 课程:Stanford CS224W(Machine Learning with Graphs)
  - 框架:PyTorch Geometric → 跑通 GCN/GAT/MPNN
- 论文精读(按顺序):
  1. **SMILES + Transformer**(ChemBERTa,基础)
  2. **Message Passing Neural Networks**(MPNN,GNN分子化的开山)
  3. **JT-VAE**(分子生成的经典)
  4. **ChemTS**(你导师的!必看)
  5. **GeoDiff / DiffDock**(diffusion for molecules)
  6. **RFdiffusion**(蛋白质生成,但思路通用)
- 实战:用 PyTorch Geometric 在 QM9 或 MoleculeNet 上训练一个分子属性预测模型

**如果选了蛋白质方向(方向B)**:

- 必学:**蛋白质语言模型 + 结构预测**
- 论文精读:
  1. **ProtTrans / ESM-1b**(蛋白质语言模型基础)
  2. **AlphaFold 2**(必读,2-3 遍)
  3. **ESM-2**(Meta 的蛋白质 LLM)
  4. **ProteinMPNN**(蛋白质设计)
  5. **RFdiffusion**(蛋白质 diffusion 生成)
  6. **AlphaFold 3**(2024,扩展到任意分子)
- 实战:用 ESM-2 提取蛋白质 embedding,做下游任务

**如果选了量子+ML(方向C)**:

- 必学:**Qiskit 或 PennyLane**
- 论文精读:量子机器学习综述 + 你导师的相关论文
- 实战:复现一个 VQE 算法

**3.2 PyTorch 内部机制深入**(所有方向都需要)

- 必懂:autograd 机制、计算图、CUDA Stream、torch.compile 概览
- 实战:实现一个自定义 autograd Function、写一个自定义 nn.Module 并测试反向传播
- **不需要懂 CUDA Kernel,但要懂 PyTorch 怎么调用 CUDA**

**3.3 LeetCode + 工程基础**(保底战线)

- LeetCode 累计 150-200 题(够用,不需要 500)
- Top 100 高频题刷透
- 重点:Array, HashMap, Two Pointers, Tree, DP 基础
- **你的简历优势不在 LeetCode,但不能完全不会**

**3.4 暑期实习投递期**(11 月-12 月集中投)

**目标公司分层(应届 Summer Intern)**:

- 🥇 第一档(用导师推荐信):
  - **Preferred Networks**(津田研嫡系,几乎保送)
  - **理研 AIP**(你导师本人挂职那里!)
  - **NIMS**(物质材料研究机构,导师合作单位)

- 🥈 第二档(简历背书 + 推荐信):
  - **NVIDIA Tokyo**(BioNeMo 团队招 AI for Science 实习)
  - **字节日本 AML**(中文母语优势)
  - **米哈游日本 AI**(游戏 AI / 渲染 AI)
  - **武田制药 AI 部门**

- 🥉 第三档(国际远程,门槛极高但试试):
  - **Isomorphic Labs**(DeepMind 子公司,远程友好)
  - **Recursion / Insitro**(AI 制药独角兽)
  - **Anthropic**(刚收购 Coefficient Bio,在扩 bio 团队)

- 🏅 兜底(普通 ML/SWE 实习):
  - **Mercari ML Platform**
  - **PayPay Data Engineer**
  - **乐天**(你已经看过的 JD)

**关键策略**:**导师推荐信是你最大武器**。11 月底前必须找老师写好至少 2 封推荐信(英文版)。

**3.5 简历准备**(11 月底前定稿)

- 英文版 + 日文版各一份
- 核心卖点(按优先级):
  1. **Tsuda Lab @ UTokyo**(导师 + 学校是顶级背书)
  2. **方向相关项目**(GitHub)
  3. **手写 Transformer / GNN 项目**
  4. **武汉理工本科 + 数学/CS基础**
  5. LeetCode 和工程能力(放最后)

**阶段产出**:

- 至少 5 家"研究方向相关公司"实习已投递
- 至少 1 封导师推荐信
- GitHub 上有 1 个方向相关的硬核项目(GNN分子 / 蛋白质 embedding / 量子电路 等)

---

**阶段 3 检查点(2027 年 1 月底)**:

- ✅ 至少进入 2-3 家公司的面试
- ✅ Preferred Networks 或理研有面试 / 实习意向
- ✅ 方向内核心论文读懂 10+ 篇
- ✅ LeetCode 200+ 题
- ❌ 如果 1 月底连一个面试都没有,**警报**。要么简历问题,要么方向选错,必须调整

---

#### 阶段 4:面试冲刺 + 拿 Offer(3 个月,2-4 月)

**目标**:拿到至少 1 个高质量暑期实习 offer。

**4.1 Research Engineer 面试特点**

和普通 SWE 完全不同。面试官关注:

- **你的研究项目**(40%):能不能讲清楚你做了什么、为什么这样做、结果如何
- **方向知识**(30%):问你方向内的核心论文、技术原理
- **编程能力**(20%):写代码题,但难度比 SWE 低
- **科研思维**(10%):给你一个问题,问你怎么设计实验

**4.2 面试准备清单**

**A. 你的项目深挖**(最重要):
- 准备一个 20-30 分钟的英文 talk,介绍你的研究方向项目
- 每个技术决策都能说清"为什么这样,不那样"
- 数据 / 实验 / 失败 / 改进都能讲

**B. 方向论文准备**:
- 你方向的 5-10 篇核心论文,**能在白板上画出架构图**
- AlphaFold 2 / GNN MPNN / Diffusion / Transformer 必备

**C. Coding 准备**:
- LeetCode Hot 100 刷透
- 重点:数组、字符串、树、DP、图(因为 GNN 相关)
- 编程语言:Python 流畅 + 能写 PyTorch 代码

**D. 系统设计 / ML System**(轻量):
- 如何设计一个分子搜索系统?
- 如何评估一个蛋白质预测模型?
- 简单的 ML pipeline 设计

**E. Behavioral**:
- 为什么对 AI for Science 感兴趣
- 为什么不读博
- 长期职业规划
- 用英语流畅讲

**4.3 论文写作 / Workshop 投稿**(可选但加分)

- 如果你的研究方向小课题有结果,考虑投 Workshop
- 推荐:NeurIPS Workshop, ICML Workshop, ICLR Workshop(2027 年的 deadline 通常在 6-8 月)
- 哪怕没投上,**简历上写"submitted to XXX Workshop"也是加分**

**阶段产出**:

- 拿到至少 1 个暑期实习 offer
- 在 1-2 家顶级 AI for Science 公司有面试经验

---

**阶段 4 检查点(2027 年 4 月底)**:

- ✅ 至少拿到 1 个暑期实习 offer
- ✅ Preferred Networks / 理研 / 字节 AML 至少 1 个在手
- ✅ 修論方向已和导师明确
- ❌ 如果没有 offer,5 月前必须降档投 Mercari/PayPay 等保底岗位

---

#### 阶段 5:暑期实习 + return offer(3 个月,7-9 月)

**这 3 个月是你 2028 年 4 月去向的决战期。研究室任务可以暂停。**

**5.1 实习期间的核心目标**

- **拿 return offer**(首要目标)
- 学到 production-grade 工程实践
- 建立行业人脉(实习同事是 5 年内最重要的关系)
- 把实习内容**转化为修論话题**(双赢)

**5.2 战术清单**

- 第 1 周:快速熟悉 codebase,主动 ask sempai/manager
- 第 2-4 周:主动接有难度的任务,不要做"小任务专业户"
- 中期(6-8 周):和 manager 1-on-1,**明确表达"想拿 return offer"**
- 后期(实习结束前):完成有可见成果的项目,让 team 记住你
- 结束:主动要 feedback,转化为修論的素材

**5.3 实习平行的事**

- LeetCode 不停(每天 30 分钟)
- 关注秋招信息(以防 return 拿不到)
- 维护和导师的沟通(每 2 周一次进度汇报)

**阶段产出**:

- Return offer 或 内定承诺
- 1-2 个可在简历上写的实习项目
- 修論的实质性进展(实习内容能用)

---

**阶段 5 检查点(2027 年 9 月底)**:

- ✅ 拿到 return offer 口头/书面承诺
- ✅ 实习经历可以在简历上写出 3 个具体亮点
- ✅ 修論进度跟得上
- ❌ 如果没拿 return,10 月秋招更要加倍准备

---

#### 阶段 6:秋招收割 + 修論收尾(6 个月,2027 年 10 月 - 2028 年 3 月)

**6.1 秋招阶段**(如有 return,这阶段轻松)

- 9 月开始铺简历(实习还没结束就要开始)
- 优先级:return offer > Isomorphic / Anthropic / 国际顶级 > Preferred Networks > 字节AML
- 多投不要押宝
- 谈 offer 时用 return offer 当锚

**目标公司(秋招 28卒 正式岗位)**:

- 🥇 国际顶级 Research Engineer 岗:
  - DeepMind / Isomorphic Labs
  - Anthropic (生物科技团队)
  - Google Research AI for Science
  - NVIDIA BioNeMo

- 🥈 日本本土最强:
  - Preferred Networks(实习的话基本 return offer)
  - Sakana AI(日本新兴 AI 公司)
  - 理研 / NIMS 研究系特任研究员

- 🥉 大药企 AI / 中国 AI 制药:
  - 武田制药 / 中外制药 / Astellas AI
  - 百图生科 / 晶泰科技(中国,可远程)

**6.2 修論收尾**

- 10 月-12 月:全力推进研究产出
- 1-2 月:写修論
- 3 月:答辩 + 毕业准备

---

### 四、关键资源清单(精挑细选,不堆砌)

**书**:

- **《Deep Learning for the Life Sciences》**(O'Reilly,生物 + DL 入门最佳)
- **《Bioinformatics Algorithms: An Active Learning Approach》**(Compeau & Pevzner,生信经典)
- **《Effective Python》**(Brett Slatkin)
- **《动手学深度学习》李沐**(中文最佳 PyTorch 入门)
- **《Designing Machine Learning Systems》Chip Huyen**(MLOps 必读)

**课程**:

- **李沐《动手学深度学习》**(必看,跟着敲代码)
- **Stanford CS224W**(Machine Learning with Graphs,GNN 圣经课)
- **Stanford CS25**(Transformers United,前沿 Transformer)
- **MIT 6.S897 / 6.871**(Deep Learning for Life Sciences)
- **Hugging Face NLP Course**(免费,实操强)
- **Andrej Karpathy YouTube**(Neural Networks: Zero to Hero,从零写 GPT)

**论文(按阅读顺序)**:

**通用基础**:
- Attention Is All You Need
- BERT
- 一篇 Diffusion Model 综述(选最近的)

**分子方向**:
- Message Passing Neural Networks for Quantum Chemistry(MPNN)
- Junction Tree Variational Autoencoder(JT-VAE)
- ChemTS(**你导师的!必看**)
- GeoDiff: Geometric Diffusion Model for Molecular Conformation Generation
- DiffDock: Diffusion Steps for Molecular Docking

**蛋白质方向**:
- ProtTrans / ESM-2(Meta 的蛋白质 LLM)
- Highly accurate protein structure prediction with AlphaFold(AF2)
- AlphaFold 3(2024 Nature)
- ProteinMPNN(蛋白质设计)
- RFdiffusion(蛋白质 diffusion 生成)

**量子 + ML 方向**:
- Variational Quantum Eigensolver(VQE 原文)
- Quantum Machine Learning 综述(选最新的)
- 你导师在量子 + ML 上的论文(看 Google Scholar)

**GitHub 项目(必读 / 必跑源码)**:

- **`tsudalab/ChemTS`**(你导师的,必读)
- **`facebookresearch/esm`**(蛋白质 LLM)
- **`deepmind/alphafold`** 或 **`aqlaboratory/openfold`**(AlphaFold 开源版)
- **`pyg-team/pytorch_geometric`**(GNN 框架)
- **`huggingface/transformers`**(Transformer 生态)
- **`karpathy/nanoGPT`**(理解 Transformer)
- **`RosettaCommons/RFdiffusion`**(蛋白质 diffusion)

**数据集**:

- **QM9 / MoleculeNet**(分子属性预测)
- **PDB**(蛋白质结构)
- **ChEMBL**(药物分子库)
- **Materials Project**(材料数据)
- **Hugging Face Datasets**(各种 ML 数据集)

**信息源**:

- arXiv q-bio.BM(每周扫)
- Twitter/X 关注:@MoAlQuraishi, @sokrypton, @prinsterja, @sashaw
- 知乎话题:AI 制药 / 蛋白质设计 / 分子机器学习
- 公众号:智药局、药明康德、机器之心 BioAI
- 你研究室的 Slack/邮件列表(最重要的本地信号)

---

### 五、几个生死攸关的提醒

**1. 你导师就是你最大的护城河,但要主动经营。** 津田老师在 AI for Science 圈是顶级人物,他的推荐信能让你跳过简历筛选。**但东大研究室常态是:很多学生毕业时导师都叫不出名字。每月至少 1 次 1-on-1,每周参加 zemi,主动汇报进度,这些都是基本盘。**

**2. 不要走纯研究路线,也不要走纯工程路线。** 你目标是 Research Engineer。纯研究路线要求顶会一作,你 2 年时间和应届毕业卡死了不可能。纯工程路线放弃你导师的资源太亏。**走"中间路线":研究做扎实但不极致(workshop 级别即可),工程能打但不卷 LeetCode 500 题,两边都够用就是最优解。**

**3. 必须有可证明的产出。** GitHub 仓库 + 研究室项目贡献 + 1 篇 workshop 论文,这是你的硬通货。AI for Science 招聘看重:**你导师是谁 + 你做了什么 + 你能讲多深**。Behavioral 答得再好,不如一个 ChemTS 的优化 PR。

**4. CUDA / 推理优化不是你的核心,但要知道。** 你方向是 Research Engineer,不是 AI Infra Engineer。CUDA Kernel 不需要会写,但要懂 PyTorch 怎么调用 CUDA、KV Cache 是什么、为什么 batch size 重要。**够用即可,深入是浪费时间**。

**5. 英语 > 日语。** AI for Science 顶级公司(Preferred Networks 内部英语化、Isomorphic / Anthropic 全英语、理研国际化)对你日语 N3 完全够用。**把学日语的时间投到读 paper 和写 GitHub 上,回报高 100 倍**。

**6. 不要被"AI Infra 高薪"诱惑跑偏。** 你看到那些"字节 AML SSP 100 万"的 offer 是给清北博士 + 顶级实习的人。**你的差异化不在那里,在"津田研 + AI for Science 工程化"**。坚持自己的路径。

**7. 持续高强度,但允许 burnout。** 这条路 23 个月,周均 40 小时学习时间是底线。但**研究室不是 996,允许自己有节奏感**。和那种把自己耗到睡眠不足 6 小时的"卷王"不要比。**研究是马拉松,你的优势就是节奏感**。

---

### 时间分配的硬约束

**1. 周均学习时间约 40-50 小时**(整个 M1-M2)

- 研究室占 50%(zemi + 实验 + paper)
- 工程 + 求职占 35%(LeetCode + Side Project + 简历)
- 英语 / 软实力占 15%

**2. 算法刷题适度**

- LeetCode 累计 200-300 题(够 Research Engineer 面试用)
- Hot 100 必须刷透
- **不需要 500+**,你简历优势不在这

**3. 英语必须过关**

- 每周至少读 1 篇英文论文
- 每月 1 次英语 1-on-1 / Speaking 练习
- 论文阅读不流畅 = AI for Science 死路

**4. 输出必须持续**

- GitHub 每周必有 commit(可以是研究室代码或个人项目)
- 每月至少 1 篇博客 / Technical writeup
- **导师每月至少 1 次进度沟通**

---

### 几个生死线判断

**2026 年 7 月底生死线**:研究方向。如果还没选定方向,或者和导师没对齐,**8 月底前必须解决,否则M1 整年都在浪费**。

**2026 年 12 月底生死线**:第一个项目 + 投递准备。如果 GitHub 还是空白,简历还没定稿,实习还没开始投,**1 月份必须紧急启动**。

**2027 年 4 月底生死线**:暑期实习 offer。如果连面试都没几场,要么是项目分量不够,要么是简历问题,**这种情况立刻把 Mercari / PayPay / 普通 ML 岗位作为保底**。

**2027 年 9 月底生死线**:return offer。如果没拿到,10 月秋招要从普通 SWE 岗位也投起来,**Research Engineer 不能成为唯一选项**。

---

### 心态准备

直白说:**这条路 23 个月走下来,30% 的人会中途切到普通 SWE,40% 的人完成度不到 60%,真正打满全程拿到顶级 offer 的不到 30%**。

你会经历:

- 论文看不懂的挫折(AlphaFold 那种密度的 paper 第一次读是绝望)
- 跑代码 OOM 的崩溃
- 实验结果不显著的焦虑
- 同学拿了 Mercari SP offer 你还在调参数的不平衡
- 研究室任务和求职任务冲突的撕裂
- 修論进度落后的恐慌

**能熬过这些的人,2028 年 4 月会站在和大部分人不同的起跑线上**。熬不过的,中途切到普通 ML 工程师也不是失败,是合理的战略调整。

---

### 检查清单(每月对照)

```
研究线:
- [ ] 本月 zemi 全部参加
- [ ] 至少 1 次和导师 1-on-1
- [ ] 读完 4+ 篇 paper
- [ ] 研究小课题有进展

工程线:
- [ ] LeetCode 累计 +20 题
- [ ] GitHub 至少 4 次 commit
- [ ] 学了 1 个新工具/库

求职线:
- [ ] 简历更新
- [ ] 关注 3+ 家目标公司动态
- [ ] OB 对话或 LinkedIn networking

健康线:
- [ ] 睡眠平均 7 小时+
- [ ] 至少 1 次完整休息日
- [ ] 没有连续 3 天高压
```

---

### 一句话总结

> **你的竞争优势不是和东大 M1 卷工程,也不是和 PhD 卷 paper,而是站在 Tsuda Lab 的肩膀上做"会工程的研究者 / 会研究的工程师"。这个赛道全球都缺人,你正好在最好的入口。**

---

## 📚 References

- [Tsuda Lab](https://www.tsudalab.org/)
- [Preferred Networks Careers](https://www.preferred.jp/en/careers/)
- [RIKEN AIP](https://aip.riken.jp/)
- Original AI Infra Journey by @MAXUPUPUP (inspiration)
