#!/usr/bin/env julia

"""
Simple CUDA Device Test - Compatible with different CUDA.jl versions
"""

using CUDA

function main()
    println("=== Simple CUDA Test ===\n")
    
    if !CUDA.functional()
        println("❌ CUDA not functional")
        return
    end
    
    println("✅ CUDA is functional")
    num_devices = CUDA.ndevices()
    println("Number of devices: $num_devices\n")
    
    # List all devices with basic info
    for i in 0:(num_devices-1)
        device = CuDevice(i)
        name = CUDA.name(device)
        memory_gb = CUDA.totalmem(device) / (1024^3)
        
        # Determine device type based on name and memory
        if occursin("V100", name)
            device_type = "🚀 Tesla V100"
        elseif occursin("Tesla", name)
            device_type = "🚀 Tesla Professional"
        elseif memory_gb < 4
            device_type = "⚠️  Integrated Graphics"
        else
            device_type = "💻 Discrete GPU"
        end
        
        println("Device $i: $name")
        println("  Type: $device_type")
        println("  Memory: $(round(memory_gb, digits=2)) GB")
        println()
    end
    
    # Show current device
    current = CUDA.device()
    current_id = CUDA.deviceid(current)
    current_name = CUDA.name(current)
    
    println("Current active device: Device $current_id - $current_name")
    
    # Test computation
    println("\nTesting computation...")
    try
        a = CUDA.rand(100, 100)
        b = CUDA.rand(100, 100)
        c = a * b
        CUDA.synchronize()
        println("✅ GPU computation successful")
    catch e
        println("❌ GPU computation failed: $e")
    end
    
    # Find best device for V100
    println("\n=== Device Selection Recommendations ===")
    best_device = -1
    best_score = -1
    
    for i in 0:(num_devices-1)
        device = CuDevice(i)
        name = CUDA.name(device)
        memory_gb = CUDA.totalmem(device) / (1024^3)
        
        score = 0
        if occursin("V100", name)
            score += 100
            println("Device $i: V100 detected - EXCELLENT for computation")
        elseif occursin("Tesla", name)
            score += 80
            println("Device $i: Tesla detected - GOOD for computation")
        elseif memory_gb >= 8
            score += 50
            println("Device $i: High memory GPU - OK for computation")
        else
            println("Device $i: $(memory_gb)GB memory - may be slow")
        end
        
        if score > best_score
            best_score = score
            best_device = i
        end
    end
    
    if best_device != -1 && best_device != current_id
        println("\n⚠️  RECOMMENDATION: Switch to Device $best_device for better performance")
        println("Add this to your script: CUDA.device!($best_device)")
    elseif best_device == current_id
        println("\n✅ You're already using the best available device")
    end
end

main()
