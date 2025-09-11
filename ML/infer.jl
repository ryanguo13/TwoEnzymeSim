"""
代理模型推理脚本 infer.jl

用法:
  1) 直接运行进行演示（使用参数空间均值）
     julia infer.jl

  2) 指定模型文件与采样次数（可选）
     julia infer.jl --model /path/to/model.jld2 --samples 100

  3) 通过 key=value 传入参数（其余未提供参数将使用默认值）
     julia infer.jl k1f=8.2 k1r=9.1 k2f=7.0 k2r=6.8 k3f=10.0 k3r=11.0 k4f=9.5 k4r=10.2 A=12.0 B=1.0 C=0.5 E1=10.0 E2=10.0

参数顺序（13维，必须匹配训练时的顺序）:
  [:k1f, :k1r, :k2f, :k2r, :k3f, :k3r, :k4f, :k4r, :A, :B, :C, :E1, :E2]
"""

using JLD2

include("surrogate_model.jl")

const DEFAULT_MODEL_PATH = "/home/ryankwok/Documents/TwoEnzymeSim/ML/model/cuda_integrated_surrogate.jld2"

const ALL_PARAM_NAMES = [:k1f, :k1r, :k2f, :k2r, :k3f, :k3r, :k4f, :k4r, :A, :B, :C, :E1, :E2]

function build_parameter_vector(params_in::Dict{Symbol, Float64}, surrogate_model::SurrogateModel)
    # 使用默认值：取参数空间的均值
    ranges = [
        surrogate_model.param_space.k1f_range, surrogate_model.param_space.k1r_range,
        surrogate_model.param_space.k2f_range, surrogate_model.param_space.k2r_range,
        surrogate_model.param_space.k3f_range, surrogate_model.param_space.k3r_range,
        surrogate_model.param_space.k4f_range, surrogate_model.param_space.k4r_range,
        surrogate_model.param_space.A_range,  surrogate_model.param_space.B_range,
        surrogate_model.param_space.C_range,  surrogate_model.param_space.E1_range,
        surrogate_model.param_space.E2_range
    ]

    defaults = [mean(r) for r in ranges]
    vec = copy(defaults)

    for (i, name) in enumerate(ALL_PARAM_NAMES)
        if haskey(params_in, name)
            vec[i] = params_in[name]
        end
    end

    return vec
end

function parse_cli_kv(args::Vector{String})
    params = Dict{Symbol, Float64}()
    for a in args
        if occursin('=', a)
            k, v = split(a, '=')
            sym = Symbol(k)
            try
                params[sym] = parse(Float64, v)
            catch
                @warn "无法解析参数值为Float64" key=sym value=v
            end
        end
    end
    return params
end

function run_infer(; model_path::String=DEFAULT_MODEL_PATH, n_samples::Int=50, params::Dict{Symbol, Float64}=Dict{Symbol, Float64}())
    # 加载模型
    surrogate = load_surrogate_model(model_path)

    # 构造 1×13 输入
    x_vec = build_parameter_vector(params, surrogate)
    X = reshape(x_vec, 1, :)

    # 推理（带不确定性）
    y_mean, y_std = predict_with_uncertainty(surrogate, X, n_samples=n_samples)

    # 打包成命名结果
    target_vars = surrogate.config.target_variables
    result = Dict{Symbol, Float64}()
    result_std = Dict{Symbol, Float64}()
    for (j, name) in enumerate(target_vars)
        result[name] = y_mean[1, j]
        result_std[name] = y_std[1, j]
    end

    return (predictions=result, uncertainties=result_std)
end

function print_results(res)
    println("\n📌 推理结果:")
    for (k, v) in res.predictions
        σ = get(res.uncertainties, k, 0.0)
        println("  $(k): $(round(v, digits=6))  ±  $(round(σ, digits=6))")
    end
end

function main()
    # 解析简单的标志
    model_path = DEFAULT_MODEL_PATH
    n_samples = 50

    # 抽取 --model 与 --samples，其余视为 key=value
    remaining = String[]
    i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--model" && i < length(ARGS)
            model_path = ARGS[i+1]
            i += 2
            continue
        elseif a == "--samples" && i < length(ARGS)
            n_samples = parse(Int, ARGS[i+1])
            i += 2
            continue
        else
            push!(remaining, a)
            i += 1
        end
    end

    kv_params = parse_cli_kv(remaining)

    if isempty(kv_params)
        println("🧪 无显式参数，使用参数空间均值进行演示推理…")
    end

    println("📂 使用模型: $(model_path)")
    println("🔁 MC采样次数: $(n_samples)")

    res = run_infer(model_path=model_path, n_samples=n_samples, params=kv_params)
    print_results(res)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


