# Darcy Flow in Bias_Aware_FNO

## 1. PDE

我们考虑定义在单位方形区域上的 Darcy 方程：

\[
\Omega = (0,1)^2,
\]

\[
-\nabla \cdot (a(x)\nabla u(x)) = f(x), \qquad x \in \Omega,
\]

\[
u(x) = 0, \qquad x \in \partial \Omega.
\]

这里：

- \(u(x)\) 是标量解；
- \(a(x)\) 是空间变化的渗透率系数场；
- 源项固定为
  \[
  f(x) = 1.
  \]

这与 FNO 论文中经典 Darcy benchmark 的物理问题保持一致，变化只在于数值求解器和数据组织形式。

## 2. 变分形式与有限元离散

对应的弱形式是：求 \(u \in H_0^1(\Omega)\)，使得对任意 \(v \in H_0^1(\Omega)\)，

\[
\int_\Omega a(x)\nabla u(x)\cdot \nabla v(x)\,dx
=
\int_\Omega v(x)\,dx.
\]

当前仓库中，Darcy 数据由 FEniCSx 的 `P1` 有限元生成：

- 粗网格解 \(u_H\)：在较粗的三角网格上求解；
- 细网格解 \(u_h\)：在较细的三角网格上求解；
- 系数场在粗/细网格上都表示为 `DG0` 分片常数场。

对同一个随机系数场样本，会分别求解一次粗网格和细网格解，然后把二者都评估到同一个规则网格上，形成 paired dataset。

## 3. 随机场生成方式

### 3.1 潜在高斯随机场

系数场不是直接采样，而是先生成一个潜在 Gaussian random field \(g(x)\)。

当前实现采用与 FNO-Darcy 风格一致的 DCT 构造。令 \(\xi_{k_1,k_2}\sim\mathcal N(0,1)\)，并定义频域系数

\[
c_{k_1,k_2}
=
\tau^{\alpha-1}
\left(\pi^2(k_1^2+k_2^2)+\tau^2\right)^{-\alpha/2}.
\]

然后构造

\[
\hat g_{k_1,k_2} = R \, c_{k_1,k_2}\, \xi_{k_1,k_2},
\]

其中 \(R\) 是分辨率因子，之后通过二维 inverse DCT 得到潜在场 \(g(x)\)。

代码对应实现见：

- [grf.py](/Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO/data_generation/darcy/grf.py)

默认参数是：

- `alpha = 2`
- `tau = 3`

它们分别控制随机场的平滑性和相关尺度。

### 3.2 两值阈值映射

得到潜在场 \(g(x)\) 后，再通过阈值映射得到最终系数场：

\[
a(x)=
\begin{cases}
a_{\text{high}}, & g(x)\ge 0,\\
a_{\text{low}}, & g(x)<0.
\end{cases}
\]

当前主实验默认取：

\[
a_{\text{low}} = 4,\qquad a_{\text{high}} = 12.
\]

因此最终的 \(a(x)\) 是一个两值、分段常数的随机介质场，这与原始 FNO-Darcy benchmark 的分布风格保持一致。

## 4. 数据集与双网格任务设置

对于每个样本，数据生成流程是：

1. 在统一规则网格上生成一个高分辨率系数场 `coeff`；
2. 将这个 `coeff` 映射到粗 FEM 网格和细 FEM 网格上的 `DG0` 系数表示；
3. 在粗网格上求解一次，得到粗解 \(u_H\)；
4. 在细网格上求解一次，得到细解 \(u_h\)；
5. 再把 \(u_H\) 和 \(u_h\) 都通过三角剖分上的点定位与有限元函数评估，统一采样到同一个规则网格。

主实验的目标设定是：

- 粗网格：`256 x 256`
- 细网格：`512 x 512`
- 统一规则采样网格：`421 x 421`

于是一个样本最终包含：

- `coeff`：规则网格上的系数张量
- `sol_coarse`：粗网格 FEM 解在统一规则网格上的采样
- `sol_fine`：细网格 FEM 解在统一规则网格上的采样
- `error_hf_lf = sol_fine - sol_coarse`

这里的 `sol_coarse` 和 `sol_fine` 都不是图像 resize 的结果，而是通过有限元解在规则点上的显式评估得到的。

## 5. FNO 任务定义

在这个仓库里，Darcy 的学习任务是一个静态算子学习问题：

\[
\mathcal G^\dagger : a(x) \mapsto u(x).
\]

但由于我们同时保存了粗/细两种 fidelity，训练和测试的监督目标不同：

- 训练输入：`coeff`
- 训练目标：`sol_coarse`
- 测试输入：`coeff`
- 测试目标：`sol_fine`

也就是说，当前 FNO 学的是：

\[
\text{coeff}_{421\times421} \longmapsto u_H^{421\times421},
\]

但测试时会拿预测结果与细网格参考解比较：

\[
\widehat u \quad \text{vs.} \quad u_h^{421\times421}.
\]

这对应一种“coarse-train / fine-test”的设置，用来研究：

- coarse-to-fine 泛化能力；
- 数值离散偏差带来的误差；
- 神经算子在不同 fidelity 监督下的表现。

## 6. 当前仓库中的对应关系

主要文件如下：

- 数据生成：
  [darcy_fem_fenicsx.py](/Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO/data_generation/darcy/darcy_fem_fenicsx.py)
- 随机场生成：
  [grf.py](/Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO/data_generation/darcy/grf.py)
- FNO 训练：
  [fourier_2d_darcy_fem.py](/Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO/fourier_2d_darcy_fem.py)
- FNO 评测：
  [eval_2d_darcy_fem.py](/Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO/scripts/eval_2d_darcy_fem.py)
- correction 模型：
  [darcy_correction.py](/Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO/darcy_correction.py)
- correction 训练：
  [fourier_2d_darcy_correction.py](/Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO/fourier_2d_darcy_correction.py)
- correction 评测：
  [eval_2d_darcy_correction.py](/Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO/scripts/eval_2d_darcy_correction.py)

一句话概括当前 Darcy 主线：

> 用 FEniCSx 在粗/细两张三角网格上分别求解同一个随机系数场对应的 Darcy 方程，再把两套解统一采样到同一规则网格上，用粗解训练 FNO、用细解测试 FNO。

## 7. Darcy Correction 分支

在 baseline Darcy-FNO 之外，这个仓库还新增了一条独立的 correction 实验分支：

\[
\tilde u_h = G_\phi(a_h), \qquad \hat u_h = \tilde u_h + b_h.
\]

这里：

- \(G_\phi\) 是现有的 Stage 1 Darcy-FNO backbone；
- \(\tilde u_h\) 是 backbone 在统一规则网格上的预测；
- \(b_h\) 是结构化 correction layer 解出的修正场；
- \(\hat u_h\) 是最终修正输出。

### 7.1 Stage 划分

该分支严格沿用现有 baseline 作为 Stage 1：

1. **Stage 1**
   用 [fourier_2d_darcy_fem.py](/Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO/fourier_2d_darcy_fem.py) 训练 backbone；
2. **Stage 2**
   固定 backbone，只训练 correction 模块；
3. **Stage 3**
   解冻 backbone，与 correction 联合微调。

当前约定是：

- Stage 1 使用完整训练集；
- Stage 2 / Stage 3 只使用训练集前 `100` 个样本；
- 若训练集不足 `100` 个样本，则 Stage 2 / Stage 3 使用全部训练样本。

### 7.2 correction 网格与离散算子

correction 不回到 FEM 自由度上，而是直接定义在当前共享规则网格上。主实验里这个网格是：

\[
421 \times 421.
\]

在该规则网格上，定义离散 Darcy 算子：

\[
A_h(a_h) = -D_h M_h(a_h) G_h,
\]

其中：

- \(G_h\)：cell-centered 到 face-centered 的离散梯度；
- \(M_h(a_h)\)：用 harmonic average 得到的面系数；
- \(D_h\)：face-centered 到 cell-centered 的离散散度。

correction 方程写成：

\[
A_h(a_h)b_h = D_h \tau_h,
\qquad
b_h|_{\partial D}=0.
\]

### 7.3 结构化 correction 思路

correction layer 不直接预测 \(b_h\)，而是先预测结构化系数场 \(\beta_h\)，再组装 correction source-flux \(\tau_h\)。当前实现中，correction net 输出四个通道：

- `beta_bulk`
- `beta_n`
- `beta_t`
- `beta_b`

它们分别用于：

- bulk correction
- interface normal correction
- interface tangential correction
- boundary correction

随后通过离散 Darcy 算子求解出 \(b_h\)。

### 7.4 Stage 2 / Stage 3 的监督

correction 训练同时使用 coarse 与 fine：

- coarse supervision 约束 backbone 输出接近 `sol_coarse`
- fine supervision 约束 corrected output 接近 `sol_fine`

也就是说，这个分支不是简单把 baseline 换成 fine supervision，而是显式利用 coarse / fine 双保真信息来训练 correction。

### 7.5 评测输出

correction run 的主要测试输出包括：

- `pred_backbone`
- `pred_corrected`
- `target_coarse`
- `target_fine`
- `residual_corrected`
- `flux_error_corrected`

## 8. Bayesian / VDN-lite 扩展

在当前 deterministic correction baseline 之上，这个仓库还支持一个并行的 Bayesian 分支。它不改数据，也不改 Stage 1，而是在现有 structured correction 上增加 stochastic latent bias 与逐点方差头。

核心思路是：

\[
\beta_h = \mu_\beta + \exp\!\left(\tfrac12 \log \sigma_\beta^2\right)\odot \varepsilon,
\qquad
\varepsilon \sim \mathcal N(0, I),
\]

\[
A_h(a_h) b_h = D_h \tau_h(\beta_h),
\qquad
\hat u_h = \tilde u_h + b_h.
\]

这里：

- backbone 预测 \(\tilde u_h\) 仍作为 clean mean；
- correction net 额外输出：
  - `beta_mu`
  - `beta_logvar`
- uncertainty head 输出：
  - `pred_logvar`
  - `pred_std = exp(0.5 * pred_logvar)`

在实现上，这是一个 VDN-lite 版本：

- 不引入新的潜变量语义；
- 不切换到新的 `exp(g)` 数据生成；
- 只是在现有 shared-grid correction operators 上增加 stochastic `beta` 和 observation variance。

## 9. 消融开关

correction 分支不再通过多份脚本做消融，而是统一通过 CLI flags 控制：

- `--ablation direct-bias`
  - 直接输出 \(b_h\)，跳过 structured \(\tau_h\) 和 correction PDE solve
- `--ablation direct-flux`
  - 直接输出 \(\tau_h\)，保留 correction PDE solve，但不经过 `beta_bulk / beta_n / beta_t / beta_b`
- `--disable-interface-correction`
  - 去掉 interface correction
- `--disable-boundary-correction`
  - 去掉 boundary correction
- `--disable-flux-loss`
  - 训练时移除 flux loss，但评测仍然继续报告 flux 指标

因此当前实验语义是：

- `deterministic + none`：structured correction baseline
- `bayesian + none`：VDN-lite Bayesian correction
- 其余 flags：结构化消融

## 10. 当前 correction 训练入口

当前 correction 训练脚本是：

- [fourier_2d_darcy_correction.py](/Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO/fourier_2d_darcy_correction.py)

它固定遵循：

1. Stage 1 完全沿用 baseline Darcy-FNO；
2. Stage 2 / Stage 3 只使用训练集前 `100` 个样本；
3. coarse target 约束 backbone；
4. fine target 约束 corrected output；
5. 若启用 Bayesian 变体，则在 Stage 2 / 3 中额外加入 `NLL + KL_beta + KL_var` 项。

对应评测脚本是：

- [eval_2d_darcy_correction.py](/Users/zhougc/Desktop/IID/LRTOR_project/Bias_Aware_FNO/scripts/eval_2d_darcy_correction.py)

其中：

- deterministic run 会继续输出 `pred_backbone` 和 `pred_corrected`
- Bayesian run 还会输出 `pred_mean`、`pred_std`、`pred_logvar`、`beta_mu`、`beta_logvar`
- `eval_metrics.json` 中会额外记录：
  - `nll_mean`
  - `calibration_bins`
  - `ece_like_error`
  - `variance_error_correlation`

并在 `eval_metrics.json` 中记录：

- `backbone_fine_l2_mean`
- `corrected_fine_l2_mean`
- `coarse_l2_mean`
- `flux_l2_mean`
- `pde_residual_mean`
- `correction_improvement_ratio`

一句话概括 correction 分支：

> 保留原始 Darcy-FNO backbone 作为 Stage 1，再在统一规则网格上引入一个结构化 flux correction layer，通过 coarse/fine 双监督提升对细网格参考解的逼近，并同时监控 PDE residual 和 flux error。
