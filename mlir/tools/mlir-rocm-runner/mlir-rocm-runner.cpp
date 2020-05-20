//===- mlir-rocm-runner.cpp - MLIR ROCM Execution Driver-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that executes an MLIR file on the GPU by
// translating MLIR to ROCDL/LLVM IR before JIT-compiling and executing the
// latter.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/ROCDLIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

// MC headers.
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"

// lld headers.
#include "lld/Common/Driver.h"

using namespace mlir;
using namespace llvm;

using Blob = SmallVector<char, 0>;

static cl::opt<std::string> TripleName("triple", cl::desc("target triple"),
                                       cl::value_desc("triple string"),
                                       cl::init("amdgcn-amd-amdhsa"));

static cl::opt<std::string> TargetChip("target", cl::desc("target chip"),
                                       cl::value_desc("AMDGPU ISA version"),
                                       cl::init("gfx900"));

static cl::opt<std::string> Features("feature", cl::desc("target features"),
                                     cl::value_desc("AMDGPU target features"),
                                     cl::init("-code-object-v3"));

static LogicalResult assembleIsa(const std::string isa, StringRef name,
                                 Blob &result) {
  raw_svector_ostream OS(result);
  std::unique_ptr<buffer_ostream> BOS = std::make_unique<buffer_ostream>(OS);

  std::string Error;
  Triple TheTriple(Triple::normalize(TripleName));
  const Target *TheTarget =
      TargetRegistry::lookupTarget(TheTriple.normalize(), Error);
  if (!TheTarget) {
    WithColor::error(errs(), name) << Error;
    return failure();
  }

  SourceMgr SrcMgr;
  SrcMgr.AddNewSourceBuffer(MemoryBuffer::getMemBuffer(isa), SMLoc());

  const MCTargetOptions MCOptions;
  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TripleName));
  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
  MAI->setRelaxELFRelocations(true);

  MCObjectFileInfo MOFI;
  MCContext Ctx(MAI.get(), MRI.get(), &MOFI, &SrcMgr, &MCOptions);
  MOFI.InitMCObjectFileInfo(TheTriple, false, Ctx, false);

  SmallString<128> CWD;
  if (!sys::fs::current_path(CWD))
    Ctx.setCompilationDir(CWD);

  std::unique_ptr<MCStreamer> Str;
  std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
  std::unique_ptr<MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TripleName, TargetChip, Features));

  MCCodeEmitter *CE = TheTarget->createMCCodeEmitter(*MCII, *MRI, Ctx);
  MCAsmBackend *MAB = TheTarget->createMCAsmBackend(*STI, *MRI, MCOptions);
  Str.reset(TheTarget->createMCObjectStreamer(
      TheTriple, Ctx, std::unique_ptr<MCAsmBackend>(MAB),
      MAB->createObjectWriter(OS), std::unique_ptr<MCCodeEmitter>(CE), *STI,
      MCOptions.MCRelaxAll, MCOptions.MCIncrementalLinkerCompatible,
      /*DWARFMustBeAtTheEnd*/ false));
  Str->setUseAssemblerInfoForParsing(true);

  std::unique_ptr<MCAsmParser> Parser(
      createMCAsmParser(SrcMgr, Ctx, *Str, *MAI));
  std::unique_ptr<MCTargetAsmParser> TAP(
      TheTarget->createMCAsmParser(*STI, *Parser, *MCII, MCOptions));

  if (!TAP) {
    WithColor::error(errs(), name) << "assembler initialization error.\n";
    return failure();
  }

  Parser->setTargetParser(*TAP);
  Parser->Run(false);

  return success();
}

static LogicalResult createHsaco(Blob &isaBlob, StringRef name,
                                 Blob &hsacoBlob) {
  // Save the ISA binary to a temp file.
  int TempISABinaryFD = -1;
  SmallString<128> TempISABinaryFilename;
  std::error_code EC = sys::fs::createTemporaryFile(
      "kernel", "o", TempISABinaryFD, TempISABinaryFilename);
  if (EC) {
    WithColor::error(errs(), name)
        << "temporary file for ISA binary creation error.\n";
    return failure();
  }
  raw_fd_ostream TempISABinaryOS(TempISABinaryFD, true);
  TempISABinaryOS << isaBlob;
  TempISABinaryOS.close();

  // Create a temp file for HSA code object.
  int TempHsacoFD = -1;
  SmallString<128> TempHsacoFilename;
  EC = sys::fs::createTemporaryFile("kernel", "hsaco", TempHsacoFD,
                                    TempHsacoFilename);
  if (EC) {
    WithColor::error(errs(), name)
        << "temporary file for HSA code object creation error.\n";
    return failure();
  }

  // Invoke lld. Expect a true return value from lld.
  bool Ret = lld::elf::link({"ld.lld", "-shared", TempISABinaryFilename.c_str(),
                             "-o", TempHsacoFilename.c_str()},
                            /*canEarlyExit=*/false, llvm::outs(), llvm::errs());
  if (!Ret) {
    WithColor::error(errs(), name) << "lld invocation error.\n";
    return failure();
  }

  // Load the HSA code object.
  auto HsacoFile = mlir::openInputFile(TempHsacoFilename);
  if (!HsacoFile) {
    WithColor::error(errs(), name)
        << "read HSA code object from temp file error.\n";
    return failure();
  }
  hsacoBlob.assign(HsacoFile->getBuffer().begin(),
                   HsacoFile->getBuffer().end());

  // Remove temp files.
  EC = sys::fs::remove(TempISABinaryFilename);
  if (EC) {
    WithColor::error(errs(), name) << "remove ISA binary temp file error.\n";
    return failure();
  }
  EC = sys::fs::remove(TempHsacoFilename);
  if (EC) {
    WithColor::error(errs(), name)
        << "remove HSA code object temp file error.\n";
    return failure();
  }
  return success();
}

static std::unique_ptr<llvm::Module> compileModuleToROCDLIR(Operation *m) {
  auto llvmModule = translateModuleToROCDLIR(m);
  // TODO(whchung): Link with ROCm-Device-Libs in case needed (ex: the Module
  // depends on math functions).
  return llvmModule;
}

static OwnedBlob compileISAToHsaco(const std::string isa, Location loc,
                                   StringRef name) {
  // ISA -> ISA in binary form via MC.
  // Use lld to create HSA code object.
  Blob IsaBlob;
  Blob HsacoBlob;

  if (succeeded(assembleIsa(isa, name, IsaBlob)) &&
      succeeded(createHsaco(IsaBlob, name, HsacoBlob)))
    return std::make_unique<std::vector<char>>(HsacoBlob.begin(),
                                               HsacoBlob.end());

  WithColor::error(errs(), name) << "producing HSA code object error.\n";
  return {};
}

static LogicalResult runMLIRPasses(ModuleOp m) {
  PassManager pm(m.getContext());
  applyPassManagerCLOptions(pm);

  pm.addPass(createGpuKernelOutliningPass());
  auto &kernelPm = pm.nest<gpu::GPUModuleOp>();
  kernelPm.addPass(createStripDebugInfoPass());
  kernelPm.addPass(createLowerGpuOpsToROCDLOpsPass());
  kernelPm.addPass(createConvertGPUKernelToBlobPass(
      compileModuleToROCDLIR, compileISAToHsaco, TripleName, TargetChip,
      Features, /*gpuBinaryAnnotation=*/"rocdl.hsaco"));
  pm.addPass(createLowerToLLVMPass());
  pm.addPass(createConvertGpuLaunchFuncToGpuRuntimeCallsPass(
      /*gpuBinaryAnnotation=*/"rocdl.hsaco"));

  return pm.run(m);
}

int main(int argc, char **argv) {
  registerPassManagerCLOptions();
  mlir::registerAllDialects();
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();

  // Initialize LLVM AMDGPU backend.
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();

  mlir::initializeLLVMPasses();
  return mlir::JitRunnerMain(argc, argv, &runMLIRPasses);
}
