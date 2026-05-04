# GOAL
在全新 WSL 环境中复现并优化 `03_AutoEncoder-minist.ipynb`：
- 成功运行原始代码
- 在数值指标（loss）上优于原结果
- 输出完整可复现环境说明

---

# CONSTRAINTS
- 允许改模型结构，但改动需合理且不过度（如 MLP → 简单 CNN）
- 优先保证“结果可解释”，避免盲目复杂化
- 不删除原始功能，只做增强
- 必须支持 GPU（CUDA）
- sudo密码:zhao

---

# TASK FLOW

## 1. 环境配置
- 安装 Python + PyTorch + CUDA
- 验证 GPU 可用

输出：`env_setup.md`

---

## 2. 基线复现
- 运行原 notebook
- 记录 loss 收敛结果

输出：`baseline_result.md`

---

## 3. 优化
从以下选择 2–3 项：
- 小幅调整网络结构（如增加层数 / CNN AE）
- 调整 latent 维度
- 修改 optimizer / learning rate
- 改进 loss（如 MSE / BCE）

---

## 4. 对比验证
- 优化后 loss 必须优于 baseline
- 给出对比数据（数值即可）

输出：`improved_result.md`

---

## 5. 最终整理
- `README.md`（包含环境 + 运行方式）
- `requirements.txt`

---

# OUTPUT RULE
每次输出必须包含：
1. 当前阶段
2. 做了什么
3. 结果
4. 下一步

---

# DEBUG
- 出错必须分析 + 修复 + 验证
- 不允许跳过报错

---

# SUCCESS
- 新环境可完整复现
- GPU 正常使用
- loss 优于原始结果
- 全流程有说明文档