//===- MIOpenOps.cpp - MIOpen MLIR Operations -----------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MIOpenOps/MIOpenOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Support/STLExtras.h"
#include "mlir/Transforms/SideEffectsInterface.h"

using namespace mlir;
using namespace mlir::miopen;

//===----------------------------------------------------------------------===//
// MIOpenOpsDialect Interfaces
//===----------------------------------------------------------------------===//
namespace {

} // namespace

//===----------------------------------------------------------------------===//
// MIOpenOpsDialect
//===----------------------------------------------------------------------===//

MIOpenOpsDialect::MIOpenOpsDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/MIOpenOps/MIOpenOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

static ParseResult parseConv2DOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, Conv2DOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(Conv2DOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// Conv2DBwdDataOp
//===----------------------------------------------------------------------===//

static ParseResult parseConv2DBwdDataOp(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, Conv2DBwdDataOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(Conv2DBwdDataOp op) { return success(); }

//===----------------------------------------------------------------------===//
// Conv2DBwdWeightOp
//===----------------------------------------------------------------------===//

static ParseResult parseConv2DBwdWeightOp(OpAsmParser &parser,
                                        OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, Conv2DBwdWeightOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(Conv2DBwdWeightOp op) { return success(); }


//===----------------------------------------------------------------------===//
// TransformOp
//===----------------------------------------------------------------------===//

static ParseResult parseTransformOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType src;
  Type srcType, dstType;
  return failure(
      parser.parseLParen() ||
      parser.parseOperand(src) ||
      parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(srcType) ||
      parser.resolveOperand(src, srcType, result.operands) ||
      parser.parseKeywordType("to", dstType) ||
      parser.addTypeToList(dstType, result.types));
  return success();
}

static void print(OpAsmPrinter &p, TransformOp op) {
  p << op.getOperationName() << "(" << op.getOperand() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperand()->getType() << " to " << op.getType();
}

static LogicalResult verify(TransformOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// GridwiseGemmOp
//===----------------------------------------------------------------------===//

static ParseResult parseGridwiseGemmOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, GridwiseGemmOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(GridwiseGemmOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// GridwiseGemmExOp
//===----------------------------------------------------------------------===//

static ParseResult parseGridwiseGemmExOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, GridwiseGemmExOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(GridwiseGemmExOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// GpuAllocOp
//===----------------------------------------------------------------------===//

static ParseResult parseGpuAllocOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  Type allocatedType;

  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(allocatedType) ||
      parser.resolveOperands(ops, {parser.getBuilder().getIndexType(), parser.getBuilder().getIndexType()}, parser.getNameLoc(), result.operands) ||
      parser.addTypeToList(allocatedType, result.types));
}

static void print(OpAsmPrinter &p, GpuAllocOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getType();
}

static LogicalResult verify(GpuAllocOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// SubviewOp
//===----------------------------------------------------------------------===//

static ParseResult parseSubviewOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType src, offset;
  Type srcType, dstType;
  return failure(
      parser.parseLParen() ||
      parser.parseOperand(src) ||
      parser.parseComma() ||
      parser.parseOperand(offset) ||
      parser.parseRParen() ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(srcType) ||
      parser.resolveOperand(src, srcType, result.operands) ||
      parser.resolveOperand(offset, parser.getBuilder().getIndexType(), result.operands) ||
      parser.parseKeywordType("to", dstType) ||
      parser.addTypeToList(dstType, result.types));
  return success();
}

static void print(OpAsmPrinter &p, miopen::SubviewOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperands()[0]->getType() << " to " << op.getType();
}

static LogicalResult verify(miopen::SubviewOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// FillOp
//===----------------------------------------------------------------------===//

static ParseResult parseFillOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::OperandType src, constantValue;
  Type srcType;

  return failure(
      parser.parseLParen() ||
      parser.parseOperand(src) ||
      parser.parseComma() ||
      parser.parseOperand(constantValue) ||
      parser.parseRParen() ||
      parser.parseColonType(srcType) ||
      parser.resolveOperand(src, srcType, result.operands) ||
      parser.resolveOperand(constantValue, parser.getBuilder().getIndexType(), result.operands));
}

static void print(OpAsmPrinter &p, FillOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p << " : " << op.getOperands()[0]->getType();
}

static LogicalResult verify(FillOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// LdsBarrierOp
//===----------------------------------------------------------------------===//

static ParseResult parseLdsBarrierOp(OpAsmParser &parser, OperationState &result) {
  return success();
}

static void print(OpAsmPrinter &p, LdsBarrierOp op) {
  p << op.getOperationName();
}

static LogicalResult verify(LdsBarrierOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// BlockwiseGemmOp
//===----------------------------------------------------------------------===//

static ParseResult parseBlockwiseGemmOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 3> ops;
  SmallVector<Type, 3> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, BlockwiseGemmOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(BlockwiseGemmOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// BlockwiseCopyOp
//===----------------------------------------------------------------------===//

static ParseResult parseBlockwiseCopyOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  SmallVector<Type, 2> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, BlockwiseCopyOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(BlockwiseCopyOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// ThreadwiseCopyOp
//===----------------------------------------------------------------------===//

static ParseResult parseThreadwiseCopyOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::OperandType, 2> ops;
  SmallVector<Type, 2> types;
  return failure(
      parser.parseOperandList(ops, OpAsmParser::Delimiter::Paren) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonTypeList(types) ||
      parser.resolveOperands(ops, types, parser.getNameLoc(), result.operands));
}

static void print(OpAsmPrinter &p, ThreadwiseCopyOp op) {
  p << op.getOperationName() << "(" << op.getOperands() << ")";
  p.printOptionalAttrDict(op.getAttrs());
  p << " : " << op.getOperandTypes();
}

static LogicalResult verify(ThreadwiseCopyOp op) {
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/MIOpenOps/MIOpenOps.cpp.inc"
