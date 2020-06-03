//===- VectorToROCDL.cpp - Vector to ROCDL lowering passes ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate ROCDLIR operations for higher-level
// Vector operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToROCDL/VectorToROCDL.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::vector;

namespace {

static TransferReadOpOperandAdaptor
getTransferOpAdapter(TransferReadOp xferOp, ArrayRef<Value> operands) {
  return TransferReadOpOperandAdaptor(operands);
}

static TransferWriteOpOperandAdaptor
getTransferOpAdapter(TransferWriteOp xferOp, ArrayRef<Value> operands) {
  return TransferWriteOpOperandAdaptor(operands);
}

LogicalResult replaceTransferOpWithMubuf(
    ConversionPatternRewriter &rewriter, ArrayRef<Value> operands,
    LLVMTypeConverter &typeConverter, Location loc, TransferReadOp xferOp,
    LLVM::LLVMType &vecTy, Value &dwordConfig, Value &int32Zero,
    Value &offsetSizeInBytes, Value &glc, Value &slc) {
  rewriter.replaceOpWithNewOp<ROCDL::MubufLoadOp>(
      xferOp, vecTy, dwordConfig, int32Zero, offsetSizeInBytes, glc, slc);
  return success();
}

LogicalResult replaceTransferOpWithMubuf(
    ConversionPatternRewriter &rewriter, ArrayRef<Value> operands,
    LLVMTypeConverter &typeConverter, Location loc, TransferWriteOp xferOp,
    LLVM::LLVMType &vecTy, Value &dwordConfig, Value &int32Zero,
    Value &offsetSizeInBytes, Value &glc, Value &slc) {
  auto adaptor = TransferWriteOpOperandAdaptor(operands);
  rewriter.replaceOpWithNewOp<ROCDL::MubufStoreOp>(xferOp, adaptor.vector(),
                                                   dwordConfig, int32Zero,
                                                   offsetSizeInBytes, glc, slc);

  return success();
}

// Conversion pattern that converts a 1-D vector transfer read/write op in a
// sequence of:
template <typename ConcreteOp>
class VectorTransferConversion : public ConvertToLLVMPattern {
public:
  explicit VectorTransferConversion(MLIRContext *context,
                                    LLVMTypeConverter &typeConv)
      : ConvertToLLVMPattern(ConcreteOp::getOperationName(), context,
                             typeConv) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto xferOp = cast<ConcreteOp>(op);
    auto adaptor = getTransferOpAdapter(xferOp, operands);

    if (xferOp.getVectorType().getRank() > 1 ||
        llvm::size(xferOp.indices()) == 0)
      return failure();

    if (xferOp.permutation_map() !=
        AffineMap::getMinorIdentityMap(xferOp.permutation_map().getNumInputs(),
                                       xferOp.getVectorType().getRank(),
                                       op->getContext()))
      return failure();

    // Have it handled in vector->llvm conversion pass.
    if (!xferOp.isMaskedDim(0))
      return failure();

    auto toLLVMTy = [&](Type t) { return typeConverter.convertType(t); };
    LLVM::LLVMType vecTy =
        toLLVMTy(xferOp.getVectorType()).template cast<LLVM::LLVMType>();
    unsigned vecWidth = vecTy.getVectorNumElements();
    Location loc = op->getLoc();

    // The backend result vector scalarization have trouble scalarize
    // <1 x ty> result, exclude the x1 width from the lowering.
    if (vecWidth != 2 && vecWidth != 4)
      return failure();

    // Obtain dataPtr and elementType from the memref.
    MemRefType memRefType = xferOp.getMemRefType();
    auto elementType = memRefType.getElementType();
    auto convertedPtrType = typeConverter.convertType(elementType)
                                .template cast<LLVM::LLVMType>()
                                .getPointerTo(0);
    // Note that the dataPtr starts at the offset address specified by
    // indices, so no need to calculat offset size in bytes again in
    // the MUBUF instruction.
    Value dataPtr = getDataPtr(loc, memRefType, adaptor.memref(),
                               adaptor.indices(), rewriter, getModule());

    if (memRefType.getMemorySpace() != 0)
      dataPtr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, convertedPtrType,
                                                       dataPtr);

    // 1. Create and fill a <4 x i32> dwordConfig with:
    //    1st two elements holding the address of dataPtr.
    //    3rd element: -1.
    //    4th element: 0x27000.
    SmallVector<int32_t, 4> indices{0, 0, -1, 0x27000};
    Type i32Ty = rewriter.getIntegerType(32);
    VectorType i32Vecx4 = VectorType::get(4, i32Ty);
    Value constConfig = rewriter.create<ConstantOp>(
        loc, i32Vecx4,
        DenseElementsAttr::get(i32Vecx4, ArrayRef<int32_t>(indices)));
    constConfig = rewriter.create<LLVM::DialectCastOp>(loc, toLLVMTy(i32Vecx4),
                                                       constConfig);

    // Treat first two element of <4 x i32> as i64, and save the dataPtr
    // to it.
    Type i64Ty = rewriter.getIntegerType(64);
    Value i64x2Ty = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMType::getVectorTy(
            toLLVMTy(i64Ty).template cast<LLVM::LLVMType>(), 2),
        constConfig);
    Value dataPtrAsI64 = rewriter.create<LLVM::PtrToIntOp>(
        loc, toLLVMTy(i64Ty).template cast<LLVM::LLVMType>(), dataPtr);
    Value zero = createIndexConstant(rewriter, loc, 0);
    Value dwordConfig = rewriter.create<LLVM::InsertElementOp>(
        loc,
        LLVM::LLVMType::getVectorTy(
            toLLVMTy(i64Ty).template cast<LLVM::LLVMType>(), 2),
        i64x2Ty, dataPtrAsI64, zero);
    dwordConfig =
        rewriter.create<LLVM::BitcastOp>(loc, toLLVMTy(i32Vecx4), dwordConfig);

    // 2. Rewrite op as a buffer read or write.
    Value int1False = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerType(1),
        rewriter.getIntegerAttr(rewriter.getIntegerType(1), 0));
    Value int32Zero = rewriter.create<ConstantOp>(
        loc, i32Ty, rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));
    return replaceTransferOpWithMubuf(rewriter, operands, typeConverter, loc,
                                      xferOp, vecTy, dwordConfig, int32Zero,
                                      int32Zero, int1False, int1False);
  }
};
} // end anonymous namespace

void mlir::populateVectorToROCDLConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  MLIRContext *ctx = converter.getDialect()->getContext();
  patterns.insert<VectorTransferConversion<TransferReadOp>,
                  VectorTransferConversion<TransferWriteOp>>(ctx, converter);
}
