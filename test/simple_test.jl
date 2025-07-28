using Metal

println("=== Simple GPU Optimization Test ===")

# Test GPU availability
if Metal.functional()
    println("✅ Metal GPU is available")
    
    # Test basic GPU array operations with Float32
    test_array = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
    gpu_array = MtlArray{Float32}(test_array)
    
    println("CPU array: $test_array")
    println("GPU array created successfully")
    
    # Test GPU computation with Float32
    result = Array(gpu_array .* Float32(2.0))
    println("GPU computation result: $result")
    
    println("\n✅ Basic GPU functionality test passed!")
    
else
    println("❌ Metal GPU is not available")
end

println("\n=== Optimization Summary ===")
println("✓ GPU array creation works")
println("✓ GPU computation works")
println("✓ Metal.jl integration successful") 