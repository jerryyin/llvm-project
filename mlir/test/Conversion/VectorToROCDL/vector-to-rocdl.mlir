// RUN: mlir-opt %s -convert-vector-to-rocdl | FileCheck %s

gpu.module @test_module{
func @transfer_readx2(%A : memref<?xf32>, %base: index) -> vector<4xf32> {
  %f0 = constant 0.0: f32
  %f = vector.transfer_read %A[%base], %f0
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32>, vector<4xf32>
  return %f: vector<4xf32>
}
// CHECK-LABEL: @transfer_readx2
// CHECK: rocdl.buffer.load {{.*}} !llvm<"<4 x float>">

func @transfer_readx4(%A : memref<?xf32>, %base: index) -> vector<4xf32> {
  %f0 = constant 0.0: f32
  %f = vector.transfer_read %A[%base], %f0
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32>, vector<4xf32>
  return %f: vector<4xf32>
}
// CHECK-LABEL: @transfer_readx4
// CHECK: rocdl.buffer.load {{.*}} !llvm<"<4 x float>">

func @transfer_read_dwordConfig(%A : memref<?xf32>, %base: index) -> vector<4xf32> {
  %f0 = constant 0.0: f32
  %f = vector.transfer_read %A[%base], %f0
      {permutation_map = affine_map<(d0) -> (d0)>} :
    memref<?xf32>, vector<4xf32>
  return %f: vector<4xf32>
}
// CHECK-LABEL: @transfer_read_dwordConfig
// CHECK: %[[gep:.*]] = llvm.getelementptr {{.*}}
// CHECK: [0, 0, -1, 159744]
// CHECK: %[[i64:.*]] = llvm.ptrtoint %[[gep]]
// CHECK: llvm.insertelement %[[i64]]

}

