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

bool isMinorIdentity(AffineMap map, unsigned rank) {
  if (map.getNumResults() < rank)
    return false;
  unsigned startDim = map.getNumDims() - rank;
  for (unsigned i = 0; i < rank; ++i)
    if (map.getResult(i) != getAffineDimExpr(startDim + i, map.getContext()))
      return false;
  return true;
}

LogicalResult replaceTransferOpWithMubuf(
    ConversionPatternRewriter &rewriter, ArrayRef<Value> operands,
    LLVMTypeConverter &typeConverter, Location loc, TransferReadOp xferOp,
    LLVM::LLVMType &vecTy, Value &dwordConfig, Value &int32zero,
    Value &offsetSizeInBytes, Value &int1False) {
  rewriter.replaceOpWithNewOp<ROCDL::MubufLoadOp>(xferOp, vecTy, dwordConfig,
                                                  int32zero, offsetSizeInBytes,
                                                  int1False, int1False);
  return success();
}

LogicalResult replaceTransferOpWithMubuf(
    ConversionPatternRewriter &rewriter, ArrayRef<Value> operands,
    LLVMTypeConverter &typeConverter, Location loc, TransferWriteOp xferOp,
    LLVM::LLVMType &vecTy, Value &dwordConfig, Value &int32zero,
    Value &offsetSizeInBytes, Value &int1False) {
  auto adaptor = TransferWriteOpOperandAdaptor(operands);
  rewriter.replaceOpWithNewOp<ROCDL::MubufStoreOp>(
      xferOp, adaptor.vector(), dwordConfig, int32zero, offsetSizeInBytes,
      int1False, int1False);

  return success();
}

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

    if (!isMinorIdentity(xferOp.permutation_map(),
                         xferOp.getVectorType().getRank()))
      return failure();

    // Have it handled in vector->llvm conversion pass
    if (!xferOp.isMaskedDim(0))
      return failure();

    auto toLLVMTy = [&](Type t) { return typeConverter.convertType(t); };
    LLVM::LLVMType vecTy =
        toLLVMTy(xferOp.getVectorType()).template cast<LLVM::LLVMType>();
    unsigned vecWidth = vecTy.getVectorNumElements();
    Location loc = op->getLoc();

    if (vecWidth != 1 && vecWidth != 2 && vecWidth != 4)
      return failure();

    // Alloca a <4 x i32> vector on address space 5 and cast it back
    Type i32Ty = rewriter.getIntegerType(32);
    LLVM::LLVMType i32Vecx4Ty = LLVM::LLVMType::getVectorTy(
        toLLVMTy(i32Ty).template cast<LLVM::LLVMType>(), 4);
    auto i32x4TyPtr = i32Vecx4Ty.getPointerTo(5);
    LLVM::LLVMFuncOp func = xferOp.template getParentOfType<LLVM::LLVMFuncOp>();
    Block *entryBlock = &func.getBody().front();
    // Alloca to the begining of the function body
    rewriter.setInsertionPointToStart(entryBlock);
    auto attrOne = rewriter.getIntegerAttr(rewriter.getIntegerType(32), 1);
    Value numElements = rewriter.create<ConstantOp>(loc, i32Ty, attrOne);
    Value dwordConfigPtr = rewriter.create<LLVM::AllocaOp>(
        func.getBody().getLoc(), i32x4TyPtr, numElements, /*alignment=*/0);
    // Reset the insertion point back
    rewriter.setInsertionPoint(xferOp);
    dwordConfigPtr = rewriter.create<LLVM::AddrSpaceCastOp>(
        loc, i32Vecx4Ty.getPointerTo(0), dwordConfigPtr);

    // Typecast the x4 int32 array to [2 x float*]
    Type f32Ty = rewriter.getF32Type();
    auto ptrf32 =
        toLLVMTy(f32Ty).template cast<LLVM::LLVMType>().getPointerTo(0);
    LLVM::LLVMType f32Ptrx2Ty = LLVM::LLVMType::getArrayTy(ptrf32, 2);
    auto ptrFloatx2Ty = f32Ptrx2Ty.getPointerTo(0);
    Value ptrF32Ptrx2 =
        rewriter.create<LLVM::BitcastOp>(loc, ptrFloatx2Ty, dwordConfigPtr);

    // Store the memref source data pointer to the first element
    MemRefType memRefType = xferOp.getMemRefType();
    auto elementType = memRefType.getElementType();
    auto convertedPtrType = typeConverter.convertType(elementType)
                                .template cast<LLVM::LLVMType>()
                                .getPointerTo(0);
    Value dataPtr = getDataPtr(loc, memRefType, adaptor.memref(),
                               adaptor.indices(), rewriter, getModule());

    if (memRefType.getMemorySpace() != 0)
      dataPtr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, convertedPtrType,
                                                       dataPtr);

    Value zero = createIndexConstant(rewriter, loc, 0);
    Value firstElementPtr = rewriter.create<LLVM::GEPOp>(
        loc, ptrf32.getPointerTo(0), ptrF32Ptrx2, ValueRange{zero, zero});
    rewriter.create<LLVM::StoreOp>(loc, ValueRange{dataPtr, firstElementPtr});

    // store -1 to 3rd element of [4 x i32]
    auto attrMinusOne =
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), -1);
    Value minusOne = rewriter.create<ConstantOp>(loc, i32Ty, attrMinusOne);
    Value two = createIndexConstant(rewriter, loc, 2);
    Value secondElementPtr = rewriter.create<LLVM::GEPOp>(
        loc, toLLVMTy(i32Ty).template cast<LLVM::LLVMType>().getPointerTo(0),
        dwordConfigPtr, ValueRange{zero, two});
    rewriter.create<LLVM::StoreOp>(loc, minusOne, secondElementPtr);

    // store 159744(0x27000) to 4th element of [4 x i32]
    auto attr159744 =
        rewriter.getIntegerAttr(rewriter.getIntegerType(32), 159744);
    Value val159744 = rewriter.create<ConstantOp>(loc, i32Ty, attr159744);
    Value three = createIndexConstant(rewriter, loc, 3);
    Value thridElementPtr = rewriter.create<LLVM::GEPOp>(
        loc, toLLVMTy(i32Ty).template cast<LLVM::LLVMType>().getPointerTo(0),
        dwordConfigPtr, ValueRange{zero, three});
    rewriter.create<LLVM::StoreOp>(loc, val159744, thridElementPtr);

    // Access and calculate offset in byte
    unsigned lastIndex = llvm::size(xferOp.indices()) - 1;
    Value offsetIndex = *(xferOp.indices().begin() + lastIndex);
    // Compute the size of an individual element.
    Value nullPtr = rewriter.create<LLVM::NullOp>(loc, convertedPtrType);
    Value one = createIndexConstant(rewriter, loc, 1);
    Value elementSizeInPtr = rewriter.create<LLVM::GEPOp>(
        loc, convertedPtrType, ArrayRef<Value>{nullPtr, one});
    Value elementSize = rewriter.create<LLVM::PtrToIntOp>(loc, getIndexType(),
                                                          elementSizeInPtr);
    // offsetInByte = offset x elementSize
    Value offsetSizeInBytes = rewriter.create<LLVM::MulOp>(
        loc, getIndexType(), offsetIndex, elementSize);
    offsetSizeInBytes = rewriter.create<LLVM::TruncOp>(
        loc, toLLVMTy(i32Ty).template cast<LLVM::LLVMType>(),
        offsetSizeInBytes);

    Value dwordConfig = rewriter.create<LLVM::LoadOp>(loc, dwordConfigPtr);
    Value int1False = rewriter.create<ConstantOp>(
        loc, rewriter.getIntegerType(1),
        rewriter.getIntegerAttr(rewriter.getIntegerType(1), 0));
    Value int32zero = rewriter.create<ConstantOp>(
        loc, i32Ty, rewriter.getIntegerAttr(rewriter.getIntegerType(32), 0));

    return replaceTransferOpWithMubuf(rewriter, operands, typeConverter, loc,
                                      xferOp, vecTy, dwordConfig, int32zero,
                                      offsetSizeInBytes, int1False);
  }
};
} // end anonymous namespace

void mlir::populateVectorToROCDLConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  MLIRContext *ctx = converter.getDialect()->getContext();
  patterns.insert<VectorTransferConversion<TransferReadOp>,
                  VectorTransferConversion<TransferWriteOp>>(ctx, converter);
}
