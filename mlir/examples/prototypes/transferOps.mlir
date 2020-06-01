// cmake --build . --target mlir-rocm-runner && ./bin/mlir-rocm-runner --shared-libs=lib/librocm-runtime-wrappers.so,lib/libmlir_runner_utils.so --entry-point-result=void /root/tmp/mlir_tests/vecadd.mlir
// RUN: mlir-rocm-runner %s --shared-libs=%rocm_wrapper_library_dir/librocm-runtime-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

func @vecadd(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>, %arg2 : memref<?xf32>) {
  %cst = constant 1 : index
  %cst2 = dim %arg0, 0 : memref<?xf32>
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %cst, %grid_y = %cst, %grid_z = %cst)
             threads(%tx, %ty, %tz) in (%block_x = %cst2, %block_y = %cst, %block_z = %cst) {
    %f0 = constant 0.0: f32
    %base = constant 0 : index
    %f = vector.transfer_read %arg0[%base], %f0
        {permutation_map = affine_map<(d0) -> (d0)>} :
      memref<?xf32>, vector<4xf32>
    %res = vector.reduction "add", %f : vector<4xf32> into f32
    store %res, %arg2[%tx] : memref<?xf32>

    gpu.terminator
  }
  return
}

// CHECK: [4.92, 4.92, 4.92, 4.92]
func @main() {
  %cf1 = constant 1.0 : f32

  %arg0 = alloc() : memref<4xf32>
  %arg1 = alloc() : memref<4xf32>
  %arg2 = alloc() : memref<4xf32>

  %22 = memref_cast %arg0 : memref<4xf32> to memref<?xf32>
  %23 = memref_cast %arg1 : memref<4xf32> to memref<?xf32>
  %24 = memref_cast %arg2 : memref<4xf32> to memref<?xf32>


  %cast0 = memref_cast %22 : memref<?xf32> to memref<*xf32>
  %cast1 = memref_cast %23 : memref<?xf32> to memref<*xf32>
  %cast2 = memref_cast %24 : memref<?xf32> to memref<*xf32>

  call @mgpuMemHostRegisterFloat(%cast0) : (memref<*xf32>) -> ()
  call @mgpuMemHostRegisterFloat(%cast1) : (memref<*xf32>) -> ()
  call @mgpuMemHostRegisterFloat(%cast2) : (memref<*xf32>) -> ()

  %25 = call @mgpuMemGetDeviceMemRef1dFloat(%22) : (memref<?xf32>) -> (memref<?xf32>)
  %26 = call @mgpuMemGetDeviceMemRef1dFloat(%23) : (memref<?xf32>) -> (memref<?xf32>)
  %27 = call @mgpuMemGetDeviceMemRef1dFloat(%24) : (memref<?xf32>) -> (memref<?xf32>)

  call @vecadd(%25, %26, %27) : (memref<?xf32>, memref<?xf32>, memref<?xf32>) -> ()
  call @print_memref_f32(%cast2) : (memref<*xf32>) -> ()
  return
}

 

func @mgpuMemHostRegisterFloat(%ptr : memref<*xf32>)
func @mgpuMemGetDeviceMemRef1dFloat(%ptr : memref<?xf32>) -> (memref<?xf32>)
func @print_memref_f32(%ptr : memref<*xf32>)
