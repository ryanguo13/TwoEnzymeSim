# 贝叶斯优化配置文件使用指南

## 概述

为了统一管理贝叶斯优化的各种参数，我们创建了 TOML 格式的配置文件系统。这使得参数调优更加方便，并支持不同的优化场景。

## 配置文件结构

配置文件位于 `config/bayesian_optimization_config.toml`，包含以下主要部分：

### 1. 基本优化配置

- **single_objective**: 单目标优化配置
- **multi_objective**: 多目标优化配置
- **acquisition_comparison**: 采集函数比较配置
- **constraint_optimization**: 约束优化配置

### 2. 参数空间配置

- **parameter_space.rates**: 反应速率常数范围
- **parameter_space.initial_conditions**: 初始条件范围
- **parameter_space.time**: 时间范围设置

### 3. 高级配置

- **gaussian_process**: GP模型超参数
- **constraints**: 热力学约束参数
- **output**: 结果保存设置

## 使用方法

### 1. 命令行使用

```bash
# 使用默认配置运行单目标优化
julia bayesian_optimization.jl --single

# 使用默认配置运行多目标优化
julia bayesian_optimization.jl --multi

# 指定自定义配置文件
julia bayesian_optimization.jl --config custom_config.toml single_objective

# 显示帮助信息
julia bayesian_optimization.jl --help
```

### 2. 在代码中使用

```julia
# 加载默认配置
config = load_bayesian_config()

# 加载指定配置文件和部分
config = load_bayesian_config("config/my_config.toml", "single_objective")

# 加载参数空间
param_space = load_parameter_space_from_config("config/my_config.toml")

# 使用配置创建工作流程
optimizer = create_bayesian_optimization_workflow("config/my_config.toml", "single_objective")
```

### 3. 示例演示

```julia
# 使用配置文件运行演示
demo_single_objective_optimization("config/bayesian_optimization_config.toml")

# 运行综合演示
comprehensive_bayesian_demo("config/bayesian_optimization_config.toml")
```

## 配置文件定制

### 1. 创建自定义配置

复制现有配置文件并修改：

```bash
cp config/bayesian_optimization_config.toml config/my_custom_config.toml
```

### 2. 主要配置参数说明

#### 单目标优化 [single_objective]
- `n_initial_points`: 初始探索点数量 (推荐15-30)
- `n_iterations`: 贝叶斯优化迭代次数 (推荐30-100)
- `acquisition_function`: 采集函数类型 ("ei", "ucb", "poi")
- `exploration_weight`: 探索权重 (UCB使用, 推荐1.0-3.0)

#### 多目标优化 [multi_objective]
- `multi_objectives`: 优化目标列表 ["C_final", "v1_mean"]
- `multi_weights`: 目标权重 [0.7, 0.3]
- `n_initial_points`: 多目标需要更多初始点 (推荐25-50)

#### 约束配置 [constraints]
- `keq_min/keq_max`: 平衡常数范围
- `rate_constant_min/max`: 速率常数范围
- `concentration_min/max`: 浓度范围

#### 参数空间 [parameter_space]
```toml
[parameter_space.rates]
k1f = { min = 0.1, max = 20.0 }
k1r = { min = 0.1, max = 20.0 }
# ... 其他速率常数

[parameter_space.initial_conditions]
A = { min = 0.1, max = 20.0 }
B = { min = 0.0, max = 5.0 }
# ... 其他初始条件
```

### 3. 实验配置模板

配置文件包含预定义的实验模板：

- `experiments.single_demo`: 单目标演示配置
- `experiments.multi_demo`: 多目标演示配置
- `experiments.acquisition_comp`: 采集函数比较配置
- `experiments.constraint_demo`: 约束优化配置

## 最佳实践

### 1. 参数调优建议

- **初始点数**: 维度数的1.5-2倍，但至少15个
- **迭代次数**: 根据计算预算，通常50-100次
- **采集函数**: EI适合平衡探索和利用，UCB适合需要高确信度的场景
- **探索权重**: 问题复杂时增加，接近收敛时减少

### 2. 约束设置

- 根据物理意义设置合理的参数范围
- 热力学约束确保结果的物理可行性
- 约束过严会降低优化效率

### 3. 多目标优化

- 权重设置反映目标的相对重要性
- 可以通过Pareto前沿分析探索不同的权重组合
- 多目标问题需要更多的初始点和迭代

## 配置验证

系统会自动验证配置文件：

- 检查必需参数的存在性
- 验证数值范围的合理性
- 提供默认值作为备选
- 输出配置加载状态信息

## 故障排除

### 1. 配置文件错误
- 检查TOML语法是否正确
- 确保所有必需的section存在
- 验证数值类型匹配

### 2. 参数范围问题
- 确保min < max
- 检查物理约束的合理性
- 验证初始条件在允许范围内

### 3. 性能问题
- 减少初始点数量
- 降低迭代次数
- 检查约束是否过于严格

## 扩展配置

可以根据需要添加新的配置section：

```toml
[my_custom_experiment]
objective_type = "single_objective"
target_variable = "custom_objective"
n_initial_points = 20
n_iterations = 40
# ... 其他参数
```

然后在代码中使用：

```julia
config = load_bayesian_config("config/bayesian_optimization_config.toml", "my_custom_experiment")
```

这种配置文件系统提供了灵活性和一致性，使得贝叶斯优化的使用更加标准化和可重现。