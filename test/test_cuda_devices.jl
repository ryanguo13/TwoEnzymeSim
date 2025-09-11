#!/usr/bin/env julia

"""
Quick CUDA Device Test Script
Run this to verify that your V100 GPUs are detected and properly configured.
"""

using CUDA
using Printf

function main()
    println("=== CUDA Device Quick Test ===\n")
    
    # Check basic CUDA functionality
    if !CUDA.functional()
        println("❌ CUDA is not functional")
        println("Please check:")
        println("  - NVIDIA drivers installed")
        println("  - CUDA toolkit installed") 
        println("  - Julia CUDA.jl package working")
        return
    end
    
    println("✅ CUDA is functional")
    
    # List all devices
    num_devices = CUDA.ndevices()
    println("Number of CUDA devices: $num_devices\n")
    
    if num_devices == 0
        println("❌ No CUDA devices detected")
        return
    end
    
    # Show device details
    println("=== Device Information ===")
    for i in 0:(num_devices-1)
        device = CuDevice(i)
        name = CUDA.name(device)
        memory_gb = CUDA.totalmem(device) / (1024^3)
        # Try different ways to get properties based on CUDA.jl version
        try
            props = CUDA.properties(device)
        catch
            # Fallback for older CUDA.jl versions
            props = (major=7, minor=0, multiProcessorCount=80)  # Default values
        end
        
        # Determine device type
        device_type = "Unknown"
        if "V100" in name
            device_type = "🚀 Tesla V100 (Professional)"
        elseif "Tesla" in name
            device_type = "🚀 Tesla (Professional)"
        elseif "Quadro" in name
            device_type = "🚀 Quadro (Professional)"
        elseif "RTX" in name || "GTX" in name
            device_type = "💻 GeForce (Consumer)"
        elseif memory_gb < 4
            device_type = "⚠️  Integrated Graphics"
        end
        
        println("Device $i: $name")
        println("  Type: $device_type")
        println("  Memory: $(round(memory_gb, digits=2)) GB")
        println("  Compute: $(props.major).$(props.minor)")
        println("  SMs: $(props.multiProcessorCount)")
        println()
    end
    
    # Test current device selection
    current_device = CUDA.device()
    current_id = CUDA.deviceid(current_device)
    current_name = CUDA.name(current_device)
    
    println("=== Current Device ===")
    println("Active device: Device $current_id - $current_name")
    
    # Simple computation test
    println("\n=== Computation Test ===")
    try
        # Test GPU computation
        a = CUDA.rand(1000, 1000)
        b = CUDA.rand(1000, 1000)
        
        gpu_time = @elapsed begin
            c = a * b
            CUDA.synchronize()
        end
        
        println("✅ GPU computation successful")
        println("Matrix multiplication (1000x1000): $(round(gpu_time * 1000, digits=2)) ms")
        
        # Memory test
        allocated, free = CUDA.memory_info()
        println("GPU Memory: $(round(allocated / 1024^3, digits=2)) GB used, $(round(free / 1024^3, digits=2)) GB free")
        
    catch e
        println("❌ GPU computation failed: $e")
    end
    
    # Recommendations
    println("\n=== Recommendations ===")
    v100_found = false
    integrated_found = false
    
    for i in 0:(num_devices-1)
        device = CuDevice(i)
        name = CUDA.name(device)
        memory_gb = CUDA.totalmem(device) / (1024^3)
        
        if "V100" in name
            v100_found = true
            if current_id != i
                println("⚠️  V100 detected as Device $i but not currently active")
                println("   Add CUDA.device!($i) to your script to use V100")
            else
                println("✅ V100 is active - optimal for computation")
            end
        elseif memory_gb < 4
            integrated_found = true
            if current_id == i
                println("⚠️  Currently using integrated graphics (Device $i)")
                println("   This will be slow for computation")
            end
        end
    end
    
    if !v100_found
        println("ℹ️  No V100 GPUs detected in this system")
    end
    
    println("\n=== Usage Instructions ===")
    println("To use a specific device in your script, add:")
    println("  CUDA.device!(device_number)")
    println("  # where device_number is 0, 1, 2, etc.")
    println()
    println("For automatic V100 selection, use the enhanced configuration")
    println("in param_scan_thermodynamic_CUDA.jl")
end

# Run the test
main()
