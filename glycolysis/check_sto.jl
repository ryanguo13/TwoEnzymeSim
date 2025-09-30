## 检查净化学计量：
## 1 Glucose + 2 ADP + 2 Pi + 2 NAD -> 2 Pyruvate + 2 ATP + 2 NADH + 2 H2O + 2 H

using Test
using Catalyst
using Unitful

include("main.jl")

macro showall(x)
    :(show(IOContext(stdout, :limit => false), $(esc(x))))
end

function compute_net_stoichiometry(glycolysis_network)
    rxns = reactions(glycolysis_network)
    @test length(rxns) == 20  # 10 步正向 + 10 步反向

    # 在 main.jl 中的定义顺序：每步是正向、随后反向，依次共10步
    forward_indices = collect(1:2:20)  # 1,3,5,...,19
    # 三碳阶段（步骤6-10）发生两次（来自裂解的两个 triose）
    forward_coefficients = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    @test length(forward_indices) == length(forward_coefficients)

    # 使用 Catalyst 提供的净化学计量矩阵：物种×反应
    N = Catalyst.netstoichmat(glycolysis_network)
    sp = species(glycolysis_network)

    # 聚合净计量向量
    net_vec = zeros(Float64, length(sp))
    for (idx, coeff) in zip(forward_indices, forward_coefficients)
        net_vec .+= coeff .* N[:, idx]
    end

    # 映射为 Symbol => 值 的字典
    net = Dict{Symbol, Float64}()
    for (i, s) in enumerate(sp)
        # species 名称形如 "Glucose(t)"，标准化为不含 (t) 的符号键
        sname = replace(string(s), "(t)" => "")
        net[Symbol(sname)] = net_vec[i]
    end
    return net
end

function check_sto()
    glycolysis_network = create_glycolysis_network()
    net = compute_net_stoichiometry(glycolysis_network)

    # 关键物种净计量
    @test isapprox(get(net, :Glucose, 0.0), -1.0; atol=0, rtol=0)
    @test isapprox(get(net, :Pyruvate, 0.0), 2.0; atol=0, rtol=0)

    @test isapprox(get(net, :ATP, 0.0), 2.0; atol=0, rtol=0)
    @test isapprox(get(net, :ADP, 0.0), -2.0; atol=0, rtol=0)
    @test isapprox(get(net, :Pi, 0.0), -2.0; atol=0, rtol=0)

    @test isapprox(get(net, :NAD, 0.0), -2.0; atol=0, rtol=0)
    @test isapprox(get(net, :NADH, 0.0), 2.0; atol=0, rtol=0)
    @test isapprox(get(net, :H, 0.0), 2.0; atol=0, rtol=0)
    @test isapprox(get(net, :H2O, 0.0), 2.0; atol=0, rtol=0)

    # 中间体应净为 0
    for sp in (:G6P, :F6P, :F16BP, :DHAP, :GAP, :BPG13, :PG3, :PG2, :PEP)
        @test isapprox(get(net, sp, 0.0), 0.0; atol=0, rtol=0)
    end

    # return the full results of the check 
    return net
end

@showall check_sto()
