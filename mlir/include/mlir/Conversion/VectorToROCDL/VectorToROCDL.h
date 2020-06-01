//===- VectorToROCDL.h - Convert Vector to ROCDL dialect ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_VECTORTOROCDL_VECTORTOROCDL_H_
#define MLIR_CONVERSION_VECTORTOROCDL_VECTORTOROCDL_H_

#include "mlir/IR/MLIRContext.h"
#include <memory>

namespace mlir {
class LLVMTypeConverter;
class OwningRewritePatternList;

template <typename OpT>
class OperationPass;

/// Collect a set of patterns to convert from the GPU dialect to ROCDL.
void populateVectorToROCDLConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOROCDL_VECTORTOROCDL_H_
