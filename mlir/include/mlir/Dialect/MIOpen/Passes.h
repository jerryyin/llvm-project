//===- Passes.h - Linalg pass entry points ----------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIOPEN_PASSES_H_
#define MLIR_DIALECT_MIOPEN_PASSES_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

using LaunchDimensionCallback = std::function<void(int64_t, int64_t)>;

namespace mlir {
class Pass;

namespace miopen {

/// Create a pass to convert MIOpen conv2d operations to transform and
/// gridwise_gemm operations.
std::unique_ptr<Pass> createLowerMIOpenOpsStep1Pass();

/// Create a pass to convert MIOpen gridwise_gemm operations to blockwise
/// operations.
std::unique_ptr<Pass> createLowerMIOpenOpsStep2Pass();

/// Create a pass to convert MIOpen blockwise operations to threadwise
/// operations.
std::unique_ptr<Pass> createLowerMIOpenOpsStep3Pass();

/// Create a pass to convert MIOpen threadwise operations to other dialects.
std::unique_ptr<Pass> createLowerMIOpenOpsStep4Pass();

/// Create a pass to convert affine / loop to std dialect.
std::unique_ptr<Pass> createLowerMIOpenOpsStep5Pass();

/// Create a pass to convert transform operations to affine maps.
std::unique_ptr<Pass> createAffineTransformPass();

/// Create a pass to affix tuning parameters to gridwise gemm ops.
std::unique_ptr<Pass> createAffixTuningParametersPass(
    int64_t blockSizeOverride = 0,
    LaunchDimensionCallback launchDimCallback = nullptr);

} // namespace miopen
} // namespace mlir

#endif // MLIR_DIALECT_MIOPEN_PASSES_H_
