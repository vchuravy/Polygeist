//===- ParallelLower.cpp - Lower gpu code to triple nested loops ------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower gpu kernels in NVVM/gpu dialects into
// a generic parallel for representation
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#define DEBUG_TYPE "parallel-lower-opt"

using namespace mlir;

namespace {
// The store to load forwarding relies on three conditions:
//
// 1) they need to have mathematically equivalent affine access functions
// (checked after full composition of load/store operands); this implies that
// they access the same single memref element for all iterations of the common
// surrounding loop,
//
// 2) the store op should dominate the load op,
//
// 3) among all op's that satisfy both (1) and (2), the one that postdominates
// all store op's that have a dependence into the load, is provably the last
// writer to the particular memref location being loaded at the load op, and its
// store value can be forwarded to the load. Note that the only dependences
// that are to be considered are those that are satisfied at the block* of the
// innermost common surrounding loop of the <store, load> being considered.
//
// (* A dependence being satisfied at a block: a dependence that is satisfied by
// virtue of the destination operation appearing textually / lexically after
// the source operation within the body of a 'affine.for' operation; thus, a
// dependence is always either satisfied by a loop or by a block).
//
// The above conditions are simple to check, sufficient, and powerful for most
// cases in practice - they are sufficient, but not necessary --- since they
// don't reason about loops that are guaranteed to execute at least once or
// multiple sources to forward from.
//
// TODO: more forwarding can be done when support for
// loop/conditional live-out SSA values is available.
// TODO: do general dead store elimination for memref's. This pass
// currently only eliminates the stores only if no other loads/uses (other
// than dealloc) remain.
//
struct ParallelLower : public ParallelLowerBase<ParallelLower> {
  void runOnFunction() override;
};

} // end anonymous namespace

/// Creates a pass to perform optimizations relying on memref dataflow such as
/// store to load forwarding, elimination of dead stores, and dead allocs.
std::unique_ptr<OperationPass<FuncOp>> mlir::createParallelLowerPass() {
  return std::make_unique<ParallelLower>();
}

#include "mlir/Transforms/InliningUtils.h"

void ParallelLower::runOnFunction() {
  // Only supports single block functions at the moment.
  getFunction().walk([&](gpu::LaunchOp launchOp) {

  #if 0
  launchOp.walk([&](CallOp caller) {

    // Build the inliner interface.
    InlinerInterface interface(&getContext());

    Region *targetRegion = caller.getCallableForCallee().resolveCallable(&symbolTable).getCallableRegion();
    inlineCall(
         interface, caller, cast<CallableOpInterface>(targetRegion->getParentOp()),
         targetRegion, /*shouldCloneInlinedRegion=*/false);

    // If the inlining was successful then erase the call and callee if
    // possible.
    caller.erase();
  });
  #endif

  Block* nb = &launchOp.getRegion().front();
  mlir::OpBuilder builder(launchOp.getContext());
  auto loc = builder.getUnknownLoc();

  builder.setInsertionPoint(launchOp->getBlock(), launchOp->getIterator());
  auto zindex = builder.create<mlir::ConstantIndexOp>(loc, 0);

  auto oneindex = builder.create<mlir::ConstantIndexOp>(loc, 1);

  auto block = builder.create<mlir::scf::ParallelOp>(loc, ValueRange({zindex, zindex, zindex}), ValueRange({launchOp.gridSizeX(), launchOp.gridSizeY(), launchOp.gridSizeZ()}), ValueRange({oneindex, oneindex, oneindex}));
  Block* blockB;
  {
    auto iter = block.getRegion().getBlocks().begin();
    blockB = &*iter;
    blockB->begin()->erase();
  }
  builder.setInsertionPointToStart(blockB);

  auto threadr = builder.create<mlir::scf::ParallelOp>(loc, ValueRange({zindex, zindex, zindex}), ValueRange({launchOp.blockSizeX(), launchOp.blockSizeX(), launchOp.blockSizeX()}), ValueRange({oneindex, oneindex, oneindex}));
  builder.create<mlir::scf::YieldOp>(loc);
  Block* threadB;
    auto iter = threadr.getRegion().getBlocks().begin();
    threadB = &*iter;
    //threadB->begin()->erase();

  //threadr.getRegion().getBlocks().clear();
  //builder.create<mlir::scf::YieldOp>(loc);
  //builder.setInsertionPointToStart(threadB);

  auto container = threadr;//builder.create<mlir::scf::ContainerOp>(loc, std::vector<mlir::Type>());

  threadB->getOperations().clear();
  threadB->getOperations().splice(threadB->begin(), nb->getOperations());
  nb->erase();

  //mlir::OpBuilder builder2(f.getContext());
  //builder2.setInsertionPointToStart(threadB);
  //iter++;
  //builder2.create<mlir::BranchOp>(loc, &*iter);

  container.walk([&](mlir::gpu::BlockIdOp bidx) {
      mlir::OpBuilder bz(launchOp.getContext());
      bz.setInsertionPoint(bidx);
      int idx = -1;
      if (bidx.dimension() == "x") idx = 0;
      else if (bidx.dimension() == "y") idx = 1;
      else if (bidx.dimension() == "z") idx = 2;
      else assert(0 && "illegal dimension");
      bidx.replaceAllUsesWith((mlir::Value)blockB->getArgument(idx));
      bidx.erase();
  });

  container.walk([&](mlir::gpu::ThreadIdOp bidx) {
      mlir::OpBuilder bz(launchOp.getContext());
      bz.setInsertionPoint(bidx);
      int idx = -1;
      if (bidx.dimension() == "x") idx = 0;
      else if (bidx.dimension() == "y") idx = 1;
      else if (bidx.dimension() == "z") idx = 2;
      else assert(0 && "illegal dimension");
      bidx.replaceAllUsesWith((mlir::Value)threadB->getArgument(idx));
      bidx.erase();
  });

  container.walk([&](gpu::TerminatorOp op) {
      mlir::OpBuilder bz(launchOp.getContext());
      bz.setInsertionPoint(op);
      bz.create<mlir::scf::YieldOp>(loc);
      op.erase();
  });

  container.walk([&](mlir::NVVM::Barrier0Op op) {
      mlir::OpBuilder bz(launchOp.getContext());
      bz.setInsertionPoint(op);
      bz.create<mlir::scf::BarrierOp>(loc);
      op.erase();
  });

  container.walk([&](gpu::GridDimOp bidx) {
    Value val = nullptr;
    if (bidx.dimension() == "x")
      val = launchOp.gridSizeX();
    else if (bidx.dimension() == "y")
      val = launchOp.gridSizeY();
    else if (bidx.dimension() == "z")
      val = launchOp.gridSizeZ();
    else assert(0 && "illegal dimension");
    bidx.replaceAllUsesWith(val);
    bidx.erase();
  });

  container.walk([&](gpu::BlockDimOp bidx) {
    Value val = nullptr;
    if (bidx.dimension() == "x")
      val = launchOp.blockSizeX();
    else if (bidx.dimension() == "y")
      val = launchOp.blockSizeY();
    else if (bidx.dimension() == "z")
      val = launchOp.blockSizeZ();
    else assert(0 && "illegal dimension");
    bidx.replaceAllUsesWith(val);
    bidx.erase();
  });

  launchOp.erase();

  });
}
