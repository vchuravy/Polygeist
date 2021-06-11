//===- MemRefDataFlowOpt.cpp - MemRef DataFlow Optimization pass ------ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to forward memref stores to loads, thereby
// potentially getting rid of intermediate memref's entirely. It also removes
// redundant loads.
// TODO: In the future, similar techniques could be used to eliminate
// dead memref store's and perform more complex forwarding when support for
// SSA scalars live out of 'affine.for'/'affine.if' statements is available.
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>

#define DEBUG_TYPE "memref-dataflow-opt"

using namespace mlir;

namespace {
// The store to load forwarding and load CSE rely on three conditions:
//
// 1) store/load and load need to have mathematically equivalent affine access
// functions (checked after full composition of load/store operands); this
// implies that they access the same single memref element for all iterations of
// the common surrounding loop,
//
// 2) the store/load op should dominate the load op,
//
// 3) among all op's that satisfy both (1) and (2), for store to load
// forwarding, the one that postdominates all store op's that have a dependence
// into the load, is provably the last writer to the particular memref location
// being loaded at the load op, and its store value can be forwarded to the
// load; for load CSE, any op that postdominates all store op's that have a
// dependence into the load can be forwarded and the first one found is chosen.
// Note that the only dependences that are to be considered are those that are
// satisfied at the block* of the innermost common surrounding loop of the
// <store/load, load> being considered.
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
struct MemRefDataFlowOpt : public MemRefDataFlowOptBase<MemRefDataFlowOpt> {
  void runOnFunction() override;

  LogicalResult forwardStoreToLoad(AffineReadOpInterface loadOp,
                                   SmallVectorImpl<Operation *> &loadOpsToErase,
                                   SmallPtrSetImpl<Value> &memrefsToErase,
                                   DominanceInfo *domInfo,
                                   PostDominanceInfo *postDominanceInfo);
  void removeUnusedStore(AffineWriteOpInterface loadOp,
                         SmallVectorImpl<Operation *> &loadOpsToErase,
                         SmallPtrSetImpl<Value> &memrefsToErase,
                         DominanceInfo *domInfo,
                         PostDominanceInfo *postDominanceInfo);
  void loadCSE(AffineReadOpInterface loadOp,
               SmallVectorImpl<Operation *> &loadOpsToErase,
               DominanceInfo *domInfo);
};

} // end anonymous namespace

/// Creates a pass to perform optimizations relying on memref dataflow such as
/// store to load forwarding, elimination of dead stores, and dead allocs.
std::unique_ptr<OperationPass<FuncOp>> mlir::createMemRefDataFlowOptPass() {
  return std::make_unique<MemRefDataFlowOpt>();
}

/// Ensure that all operations between start (noninclusive) and memOp
/// do not have the potential memory effect EffectType on memOp
template <typename EffectType, typename T>
bool hasNoInterveningEffect(Operation *start, T memOp) {

  Value originalMemref = memOp.getMemRef();
  bool isOriginalAllocation =
      originalMemref.getDefiningOp<memref::AllocaOp>() ||
      originalMemref.getDefiningOp<memref::AllocOp>();
  bool legal = true;

  // Check whether the effect on memOp can be caused by
  // a given operation op.
  std::function<void(Operation *)> check = [&](Operation *op) {
    // If the effect has alreay been found, early exit
    if (!legal)
      return;

    if (auto memEffect = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance, 1> effects;
      memEffect.getEffects(effects);

      for (auto effect : effects) {
        // If op causes EffectType on a potentially aliasing
        // location for memOp, mark as illegal.
        if (isa<EffectType>(effect.getEffect())) {
          if (isOriginalAllocation && effect.getValue() &&
              (effect.getValue().getDefiningOp<memref::AllocaOp>() ||
               effect.getValue().getDefiningOp<memref::AllocOp>()))
            if (effect.getValue() != originalMemref)
              continue;
          legal = false;
          return;
        }
      }
    } else if (op->hasTrait<OpTrait::HasRecursiveSideEffects>()) {
      // Recurse into the regions for this op and check whether
      // the internal operations may have the effect
      for (auto &region : op->getRegions())
        for (auto &block : region)
          for (auto &op : block)
            check(&op);
    } else {
      // Otherwise, conservatively assume generic operations have
      // the effect on the operation
      legal = false;
      return;
    }
  };

  // Check all paths from ancestor op `parent` to the
  // operation `to` for the effect. It is known that
  // `to` must be contained within `parent`
  auto until = [&](Operation *parent, Operation *to) {
    // TODO check only the paths from `parent` to `to`
    // Currently we fallback an check the entire parent op.
    assert(parent->isAncestor(to));
    check(parent);
  };

  // Check for all paths from operation `from` to operation
  // `to` for the given memory effect.
  std::function<void(Operation *, Operation *)> recur = [&](Operation *from,
                                                            Operation *to) {
    assert(from->getParentRegion()->isAncestor(to->getParentRegion()));

    // If the operations are in different regions, recursively
    // consider all path from `from` to the parent of `to` and
    // all paths from the parent of `to` to `to`.
    if (from->getParentRegion() != to->getParentRegion()) {
      recur(from, to->getParentOp());
      until(to->getParentOp(), to);
      return;
    }

    // Now, assuming that from and to exist in the same region, perform
    // a CFG traversal to check all the relevant operations

    // Additional blocks to consider
    std::deque<Block *> todo;
    {
      // First consier the parent block of `from` an check all operations
      // after `from`.
      for (auto iter = ++from->getIterator(), end = from->getBlock()->end();
           iter != end && &*iter != to; iter++) {
        check(&*iter);
      }

      // If the parent of `from` doesn't contain `to`, add the successors
      // to the list of blocks to check.
      if (to->getBlock() != from->getBlock())
        for (auto succ : from->getBlock()->getSuccessors())
          todo.push_back(succ);
    }

    SmallPtrSet<Block *, 4> done;
    // Traverse the CFG until hitting `to`
    while (todo.size()) {
      auto blk = todo.front();
      todo.pop_front();
      if (done.count(blk))
        continue;
      done.insert(blk);
      for (auto &op : *blk) {
        if (&op == to)
          break;
        check(&op);
        if (&op == blk->getTerminator())
          for (auto succ : blk->getSuccessors())
            todo.push_back(succ);
      }
    }
  };
  recur(start, memOp.getOperation());
  return legal;
}

// This attempts to remove stores which have no impact on the final result.
// A writing op writeA will be eliminated if there exists an op writeB if
// 1) writeA and writeB have mathematically equivalent affine access functions.
// 2) writeB postdominates loadA.
// 3) There is no potential read between writeA and writeB
void MemRefDataFlowOpt::removeUnusedStore(
    AffineWriteOpInterface writeA, SmallVectorImpl<Operation *> &opsToErase,
    SmallPtrSetImpl<Value> &memrefsToErase, DominanceInfo *domInfo,
    PostDominanceInfo *postDominanceInfo) {

  for (auto *user : writeA.getMemRef().getUsers()) {
    // Only consider writing operations
    auto writeB = dyn_cast<AffineWriteOpInterface>(user);
    if (!writeB)
      continue;

    // The operations must be distinct
    if (writeB == writeA)
      continue;

    // Both operations must lie in the same region
    if (writeB->getParentRegion() != writeA->getParentRegion())
      continue;

    // Both operations must write to the same memory
    MemRefAccess srcAccess(writeB);
    MemRefAccess destAccess(writeA);

    if (srcAccess != destAccess)
      continue;

    // writeB must postdominate writeA
    if (!postDominanceInfo->postDominates(writeB, writeA))
      continue;

    // There cannot be an operation which reads from memory between
    // the two writes
    if (!hasNoInterveningEffect<MemoryEffects::Read>(writeA, writeB))
      continue;

    opsToErase.push_back(writeA);
    break;
  }
}

// This is a straightforward implementation not optimized for speed. Optimize
// if needed.
LogicalResult MemRefDataFlowOpt::forwardStoreToLoad(
    AffineReadOpInterface loadOp, SmallVectorImpl<Operation *> &loadOpsToErase,
    SmallPtrSetImpl<Value> &memrefsToErase, DominanceInfo *domInfo,
    PostDominanceInfo *postDominanceInfo) {
  // First pass over the use list to get the minimum number of surrounding
  // loops common between the load op and the store op, with min taken across
  // all store ops.
  SmallVector<Operation *, 8> storeOps;
  unsigned minSurroundingLoops = getNestingDepth(loadOp);
  for (auto *user : loadOp.getMemRef().getUsers()) {
    auto storeOp = dyn_cast<AffineWriteOpInterface>(user);
    if (!storeOp)
      continue;
    unsigned nsLoops = getNumCommonSurroundingLoops(*loadOp, *storeOp);
    minSurroundingLoops = std::min(nsLoops, minSurroundingLoops);
    storeOps.push_back(storeOp);
  }

  // The list of store op candidates for forwarding that satisfy conditions
  // (1) and (2) above - they will be filtered later when checking (3).
  SmallVector<Operation *, 8> fwdingCandidates;

  // Store ops that have a dependence into the load (even if they aren't
  // forwarding candidates). Each forwarding candidate will be checked for a
  // post-dominance on these. 'fwdingCandidates' are a subset of depSrcStores.
  SmallVector<Operation *, 8> depSrcStores;
  for (auto *storeOp : storeOps) {
    MemRefAccess srcAccess(storeOp);
    MemRefAccess destAccess(loadOp);

    // Stores that *may* be reaching the load.
    depSrcStores.push_back(storeOp);

    // 1. Check if the store and the load have mathematically equivalent
    // affine access functions; this implies that they statically refer to the
    // same single memref element. As an example this filters out cases like:
    //     store %A[%i0 + 1]
    //     load %A[%i0]
    //     store %A[%M]
    //     load %A[%N]
    // Use the AffineValueMap difference based memref access equality checking.
    if (srcAccess != destAccess)
      continue;

    if (!domInfo->dominates(storeOp, loadOp))
      continue;

    if (!hasNoInterveningEffect<MemoryEffects::Write>(storeOp, loadOp))
      continue;

    // We now have a candidate for forwarding.
    fwdingCandidates.push_back(storeOp);
  }

  // 3. Of all the store op's that meet the above criteria, the store that
  // postdominates all 'depSrcStores' (if one exists) is the unique store
  // providing the value to the load, i.e., provably the last writer to that
  // memref loc.
  // Note: this can be implemented in a cleaner way with postdominator tree
  // traversals. Consider this for the future if needed.
  Operation *lastWriteStoreOp = nullptr;
  for (auto *storeOp : fwdingCandidates) {
    assert(!lastWriteStoreOp);
    lastWriteStoreOp = storeOp;
  }

  if (!lastWriteStoreOp)
    return failure();

  // Perform the actual store to load forwarding.
  Value storeVal =
      cast<AffineWriteOpInterface>(lastWriteStoreOp).getValueToStore();
  // Check if 2 values have the same shape. This is needed for affine vector
  // loads and stores.
  if (storeVal.getType() != loadOp.getValue().getType())
    return failure();
  loadOp.getValue().replaceAllUsesWith(storeVal);
  // Record the memref for a later sweep to optimize away.
  memrefsToErase.insert(loadOp.getMemRef());
  // Record this to erase later.
  loadOpsToErase.push_back(loadOp);

  return success();
}

// The load to load forwarding / redundant load elimination is similar to the
// store to load forwarding.
// loadA will be be replaced with loadB if:
// 1) loadA and loadB have mathematically equivalent affine access functions.
// 2) loadB dominates loadA.
// 3) There is no write between loadA and loadB
void MemRefDataFlowOpt::loadCSE(AffineReadOpInterface loadA,
                                SmallVectorImpl<Operation *> &loadOpsToErase,
                                DominanceInfo *domInfo) {
  SmallVector<AffineReadOpInterface, 4> LoadOptions;
  for (auto *user : loadA.getMemRef().getUsers()) {
    auto loadB = dyn_cast<AffineReadOpInterface>(user);
    if (!loadB || loadB == loadA)
      continue;

    MemRefAccess srcAccess(loadB);
    MemRefAccess destAccess(loadA);

    if (srcAccess != destAccess) {
      continue;
    }

    // 2. The store has to dominate the load op to be candidate.
    if (!domInfo->dominates(loadB, loadA))
      continue;

    if (!hasNoInterveningEffect<MemoryEffects::Write>(loadB.getOperation(),
                                                      loadA))
      continue;

    // Check if 2 values have the same shape. This is needed for affine vector
    // loads.
    if (loadB.getValue().getType() != loadA.getValue().getType())
      continue;

    LoadOptions.push_back(loadB);
  }

  // Of the legal load candidates, use the one that dominates all others
  // to minimize the subsequent need to loadCSE
  Value loadB = nullptr;
  for (auto option : LoadOptions) {
    if (llvm::all_of(LoadOptions, [&](AffineReadOpInterface depStore) {
          return depStore == option ||
                 domInfo->dominates(option.getOperation(),
                                    depStore.getOperation());
        })) {
      loadB = option.getValue();
      break;
    }
  }

  if (loadB) {
    loadA.getValue().replaceAllUsesWith(loadB);
    // Record this to erase later.
    loadOpsToErase.push_back(loadA);
  }
}

void MemRefDataFlowOpt::runOnFunction() {
  // Only supports single block functions at the moment.
  FuncOp f = getFunction();

  // Load op's whose results were replaced by those forwarded from stores.
  SmallVector<Operation *, 8> opsToErase;

  // A list of memref's that are potentially dead / could be eliminated.
  SmallPtrSet<Value, 4> memrefsToErase;

  auto domInfo = &getAnalysis<DominanceInfo>();
  auto postDominanceInfo = &getAnalysis<PostDominanceInfo>();

  // Walk all load's and perform store to load forwarding.
  f.walk([&](AffineReadOpInterface loadOp) {
    if (failed(forwardStoreToLoad(loadOp, opsToErase, memrefsToErase, domInfo,
                                  postDominanceInfo))) {
      loadCSE(loadOp, opsToErase, domInfo);
    }
  });

  // Erase all load op's whose results were replaced with store fwd'ed ones.
  for (auto *op : opsToErase)
    op->erase();
  opsToErase.clear();

  f.walk([&](AffineWriteOpInterface loadOp) {
    removeUnusedStore(loadOp, opsToErase, memrefsToErase, domInfo,
                      postDominanceInfo);
  });

  // Erase all store op's which are unnecessary.
  for (auto *op : opsToErase)
    op->erase();
  opsToErase.clear();

  // Check if the store fwd'ed memrefs are now left with only stores and can
  // thus be completely deleted. Note: the canonicalize pass should be able
  // to do this as well, but we'll do it here since we collected these anyway.
  for (auto memref : memrefsToErase) {
    // If the memref hasn't been alloc'ed in this function, skip.
    Operation *defOp = memref.getDefiningOp();
    if (!defOp || !isa<memref::AllocOp>(defOp))
      // TODO: if the memref was returned by a 'call' operation, we
      // could still erase it if the call had no side-effects.
      continue;
    if (llvm::any_of(memref.getUsers(), [&](Operation *ownerOp) {
          return !isa<AffineWriteOpInterface, memref::DeallocOp>(ownerOp);
        }))
      continue;

    // Erase all stores, the dealloc, and the alloc on the memref.
    for (auto *user : llvm::make_early_inc_range(memref.getUsers()))
      user->erase();
    defOp->erase();
  }
}
