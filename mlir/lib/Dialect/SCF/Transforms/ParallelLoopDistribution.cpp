//===- ParallelLoopDistrbute.cpp - Distribute loops around barriers -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "cpuify"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;

/// Populates `crossing` with values (op results) that are defined in the same
/// block as `op` and above it, and used by at least one op in the same block
/// below `op`. Uses may be in nested regions.
static void findValuesUsedBelow(Operation *op,
                                llvm::SetVector<Value> &crossing) {
  for (Operation *it = op->getPrevNode(); it != nullptr;
       it = it->getPrevNode()) {
    for (Value value : it->getResults()) {
      for (Operation *user : value.getUsers()) {
        // If the user is nested in another op, find its ancestor op that lives
        // in the same block as the barrier.
        while (user->getBlock() != op->getBlock())
          user = user->getBlock()->getParentOp();

        if (op->isBeforeInBlock(user)) {
          crossing.insert(value);
          break;
        }
      }
    }
  }

  // No need to process block arguments, they are assumed to be induction
  // variables and will be replicated.
}

/// Returns `true` if the given operation has a BarrierOp transitively nested in
/// one of its regions.
static bool hasNestedBarrier(Operation *op) {
  auto result =
      op->walk([](scf::BarrierOp) { return WalkResult::interrupt(); });
  return result.wasInterrupted();
}

namespace {
/// Replaces an conditional with a loop that may iterate 0 or 1 time, that is:
///
/// scf.if %cond {
///   @then()
/// } else {
///   @else()
/// }
///
/// is replaced with
///
/// scf.for %i = 0 to %cond step 1 {
///   @then()
/// }
/// scf.for %i = 0 to %cond - 1 step 1 {
///   @else()
/// }
struct ReplaceIfWithFors : public OpRewritePattern<scf::IfOp> {
  ReplaceIfWithFors(MLIRContext *ctx) : OpRewritePattern<scf::IfOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    assert(op.condition().getType().isInteger(1));

    // TODO: we can do this by having "undef" values as inputs, or do reg2mem.
    if (op.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[if-to-for] 'if' with results, need reg2mem\n";
                 DBGS() << op);
      return failure();
    }

    if (!hasNestedBarrier(op)) {
      LLVM_DEBUG(DBGS() << "[if-to-for] no nested barrier\n");
      return failure();
    }

    Location loc = op.getLoc();
    auto zero = rewriter.create<ConstantIndexOp>(loc, 0);
    auto one = rewriter.create<ConstantIndexOp>(loc, 1);
    auto cond = rewriter.create<IndexCastOp>(loc, rewriter.getIndexType(),
                                             op.condition());
    auto thenLoop = rewriter.create<scf::ForOp>(loc, zero, cond, one);
    op->getParentOfType<FuncOp>()->dump();
    rewriter.mergeBlockBefore(op.getBody(0), &thenLoop.getBody()->back());
    rewriter.eraseOp(&thenLoop.getBody()->back());

    if (!op.elseRegion().empty()) {
      auto negCondition = rewriter.create<SubIOp>(loc, one, cond);
      auto elseLoop = rewriter.create<scf::ForOp>(loc, zero, negCondition, one);
      rewriter.mergeBlockBefore(op.getBody(1), &elseLoop.getBody()->back());
      rewriter.eraseOp(&elseLoop.getBody()->back());
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// Returns `true` if `value` is defined outside of the region that contains
/// `user`.
static bool isDefinedAbove(Value value, Operation *user) {
  return value.getParentRegion()->isProperAncestor(user->getParentRegion());
}

/// Returns `true` if the loop has a form expected by interchange patterns.
static bool isNormalized(scf::ForOp op) {
  return isDefinedAbove(op.lowerBound(), op) && isDefinedAbove(op.step(), op);
}

/// Transforms a loop to the normal form expected by interchange patterns, i.e.
/// with zero lower bound and unit step.
struct NormalizeLoop : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (isNormalized(op) || !isa<scf::ParallelOp>(op->getParentOp())) {
      LLVM_DEBUG(DBGS() << "[normalize-loop] loop already normalized\n");
      return failure();
    }
    if (op.getNumResults()) {
      LLVM_DEBUG(DBGS() << "[normalize-loop] not handling reduction loops\n");
      return failure();
    }

    OpBuilder::InsertPoint point = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op->getParentOp());
    Value zero = rewriter.create<ConstantIndexOp>(op.getLoc(), 0);
    Value one = rewriter.create<ConstantIndexOp>(op.getLoc(), 1);
    rewriter.restoreInsertionPoint(point);

    Value difference =
        rewriter.create<SubIOp>(op.getLoc(), op.upperBound(), op.lowerBound());
    Value tripCount =
        rewriter.create<SignedCeilDivIOp>(op.getLoc(), difference, op.step());
    auto newForOp =
        rewriter.create<scf::ForOp>(op.getLoc(), zero, tripCount, one);
    rewriter.setInsertionPointToStart(newForOp.getBody());
    Value scaled = rewriter.create<MulIOp>(
        op.getLoc(), newForOp.getInductionVar(), op.step());
    Value iv = rewriter.create<AddIOp>(op.getLoc(), op.lowerBound(), scaled);
    rewriter.mergeBlockBefore(op.getBody(), &newForOp.getBody()->back(), {iv});
    rewriter.eraseOp(&newForOp.getBody()->back());
    rewriter.eraseOp(op);
    return success();
  }
};

/// Emits the IR  computing the total number of iterations in the loop. We don't
/// need to linearize them since we can allocate an nD array instead.
static llvm::SmallVector<Value> emitIterationCounts(PatternRewriter &rewriter,
                                                    scf::ParallelOp op) {
  SmallVector<Value> iterationCounts;
  for (auto bounds : llvm::zip(op.lowerBound(), op.upperBound(), op.step())) {
    Value lowerBound = std::get<0>(bounds);
    Value upperBound = std::get<1>(bounds);
    Value step = std::get<2>(bounds);
    Value diff = rewriter.create<SubIOp>(op.getLoc(), upperBound, lowerBound);
    Value count = rewriter.create<SignedCeilDivIOp>(op.getLoc(), diff, step);
    iterationCounts.push_back(count);
  }
  return iterationCounts;
}

/// Returns `true` if the loop has a form expected by interchange patterns.
static bool isNormalized(scf::ParallelOp op) {
  auto isZero = [](Value v) {
    APInt value;
    return matchPattern(v, m_ConstantInt(&value)) && value.isNullValue();
  };
  auto isOne = [](Value v) {
    APInt value;
    return matchPattern(v, m_ConstantInt(&value)) && value.isOneValue();
  };
  return llvm::all_of(op.lowerBound(), isZero) &&
         llvm::all_of(op.step(), isOne);
}

/// Transforms a loop to the normal form expected by interchange patterns, i.e.
/// with zero lower bounds and unit steps.
struct NormalizeParallel : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (isNormalized(op)) {
      LLVM_DEBUG(DBGS() << "[normalize-parallel] loop already normalized\n");
      return failure();
    }
    if (op->getNumResults() != 0) {
      LLVM_DEBUG(
          DBGS() << "[normalize-parallel] not processing reduction loops\n");
      return failure();
    }
    if (!hasNestedBarrier(op)) {
      LLVM_DEBUG(DBGS() << "[normalize-parallel] no nested barrier\n");
      return failure();
    }

    Value zero = rewriter.create<ConstantIndexOp>(op.getLoc(), 0);
    Value one = rewriter.create<ConstantIndexOp>(op.getLoc(), 1);
    SmallVector<Value> iterationCounts = emitIterationCounts(rewriter, op);
    auto newOp = rewriter.create<scf::ParallelOp>(
        op.getLoc(), SmallVector<Value>(iterationCounts.size(), zero),
        iterationCounts, SmallVector<Value>(iterationCounts.size(), one));

    SmallVector<Value> inductionVars;
    inductionVars.reserve(iterationCounts.size());
    rewriter.setInsertionPointToStart(newOp.getBody());
    for (unsigned i = 0, e = iterationCounts.size(); i < e; ++i) {
      Value scaled = rewriter.create<MulIOp>(
          op.getLoc(), newOp.getInductionVars()[i], op.step()[i]);
      Value shifted =
          rewriter.create<AddIOp>(op.getLoc(), op.lowerBound()[i], scaled);
      inductionVars.push_back(shifted);
    }

    rewriter.mergeBlockBefore(op.getBody(), &newOp.getBody()->back(),
                              inductionVars);
    rewriter.eraseOp(&newOp.getBody()->back());
    rewriter.eraseOp(op);

    return success();
  }
};

/// Checks if `op` may need to be wrapped in a pair of barriers. This is a
/// necessary but insufficient condition.
static LogicalResult canWrapWithBarriers(Operation *op) {
  if (!isa<scf::ParallelOp>(op->getParentOp())) {
    LLVM_DEBUG(DBGS() << "[wrap] not nested in a pfor\n");
    return failure();
  }

  if (op->getNumResults() != 0) {
    LLVM_DEBUG(DBGS() << "[wrap] ignoring loop with reductions\n");
    return failure();
  }

  if (!hasNestedBarrier(op)) {
    LLVM_DEBUG(DBGS() << "[wrap] no nested barrier\n");
    return failure();
  }

  return success();
}

/// Puts a barrier before and/or after `op` if there isn't already one.
/// `extraPrevCheck` is called on the operation immediately preceding `op` and
/// can be used to look further upward if the immediately preceding operation is
/// not a barrier.
static LogicalResult wrapWithBarriers(
    Operation *op, PatternRewriter &rewriter,
    llvm::function_ref<bool(Operation *)> extraPrevCheck = nullptr) {
  Operation *prevOp = op->getPrevNode();
  Operation *nextOp = op->getNextNode();
  bool hasPrevBarrierLike = prevOp == nullptr || isa<scf::BarrierOp>(prevOp);
  if (extraPrevCheck && !hasPrevBarrierLike)
    hasPrevBarrierLike = extraPrevCheck(prevOp);
  bool hasNextBarrierLike =
      nextOp == &op->getBlock()->back() || isa<scf::BarrierOp>(nextOp);

  if (hasPrevBarrierLike && hasNextBarrierLike) {
    LLVM_DEBUG(DBGS() << "[wrap] already has sufficient barriers\n");
    return failure();
  }

  if (!hasPrevBarrierLike)
    rewriter.create<scf::BarrierOp>(op->getLoc());

  if (!hasNextBarrierLike) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);
    rewriter.create<scf::BarrierOp>(op->getLoc());
  }

  // We don't actually change the op, but the pattern infra wants us to. Just
  // pretend we changed it in-place.
  rewriter.updateRootInPlace(op, [] {});
  LLVM_DEBUG(DBGS() << "[wrap] wrapped '" << op->getName().getStringRef()
                    << "' with barriers\n");
  return success();
}

/// Puts a barrier before and/or after a "for" operation if there isn't already
/// one, potentially with a single load that supplies the upper bound of a
/// (normalized) loop.
struct WrapForWithBarrier : public OpRewritePattern<scf::ForOp> {
  WrapForWithBarrier(MLIRContext *ctx) : OpRewritePattern<scf::ForOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<scf::ParallelOp>(op->getParentOp())) {
      LLVM_DEBUG(DBGS() << "[wrap-for] not nested in a pfor\n");
      return failure();
    }

    if (!hasNestedBarrier(op)) {
      LLVM_DEBUG(DBGS() << "[wrap-for] no nested barrier\n");
      return failure();
    }

    if (op.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[wrap-for] ignoring loop with reductions\n");
      return failure();
    }

    if (!isNormalized(op)) {
      LLVM_DEBUG(DBGS() << "[wrap-for] non-normalized loop\n");
      return failure();
    }

    return wrapWithBarriers(op, rewriter, [&](Operation *prevOp) {
      if (auto loadOp = dyn_cast_or_null<memref::LoadOp>(prevOp)) {
        if (loadOp.result() == op.upperBound() &&
            loadOp.indices() ==
                cast<scf::ParallelOp>(op->getParentOp()).getInductionVars()) {
          prevOp = prevOp->getPrevNode();
          return prevOp == nullptr || isa<scf::BarrierOp>(prevOp);
        }
      }
      return false;
    });
  }
};

/// Moves the body from `forLoop` contained in `op` to a parallel op newly
/// created at the start of `newForLoop`.
static void moveBodies(PatternRewriter &rewriter, scf::ParallelOp op,
                       scf::ForOp forLoop, scf::ForOp newForLoop) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(newForLoop.getBody());
  auto newParallel = rewriter.create<scf::ParallelOp>(
      op.getLoc(), op.lowerBound(), op.upperBound(), op.step());
  // Merge in two stages so we can properly replace uses of two induction
  // varibales defined in different blocks.
  rewriter.mergeBlockBefore(op.getBody(), &newParallel.getBody()->back(),
                            newParallel.getInductionVars());
  rewriter.eraseOp(&newParallel.getBody()->back());
  rewriter.mergeBlockBefore(forLoop.getBody(), &newParallel.getBody()->back(),
                            newForLoop.getInductionVar());
  rewriter.eraseOp(&newParallel.getBody()->back());
  rewriter.eraseOp(op);
}

/// Interchanges a parallel for loop with a for loop perfectly nested within it.
struct InterchangeForPFor : public OpRewritePattern<scf::ParallelOp> {
  InterchangeForPFor(MLIRContext *ctx)
      : OpRewritePattern<scf::ParallelOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    // A perfect nest must have two operations in the outermost body: a "for"
    // loop, and a terminator.
    if (std::next(op.getBody()->begin(), 2) != op.getBody()->end() ||
        !isa<scf::ForOp>(op.getBody()->front())) {
      LLVM_DEBUG(DBGS() << "[interchange] not a perfect pfor(for) nest\n");
      return failure();
    }

    // We shouldn't have parallel reduction loops coming from GPU anyway, and
    // sequential reduction loops can be transformed by reg2mem.
    auto forLoop = cast<scf::ForOp>(op.getBody()->front());
    if (op.getNumResults() != 0 || forLoop.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[interchange] not matching reduction loops\n");
      return failure();
    }

    if (!isNormalized(op) || !isNormalized(forLoop)) {
      LLVM_DEBUG(DBGS() << "[interchange] non-normalized loop\n");
    }

    if (!hasNestedBarrier(forLoop)) {
      LLVM_DEBUG(DBGS() << "[interchange] no nested barrier\n";);
      return failure();
    }

    auto newForLoop =
        rewriter.create<scf::ForOp>(forLoop.getLoc(), forLoop.lowerBound(),
                                    forLoop.upperBound(), forLoop.step());
    moveBodies(rewriter, op, forLoop, newForLoop);
    return success();
  }
};

/// Interchanges a parallel for loop with a normalized (zero lower bound and
/// unit step) for loop nested within it. The for loop must have a barrier
/// inside and is preceeded by a load operation that supplies its upper bound.
/// The barrier semantics implies that all threads must executed the same number
/// of times, which means that the inner loop must have the same trip count in
/// all iterations of the outer loop. Therefore, the load of the upper bound can
/// be hoisted and read any value, because all values are identical in a
/// semantically valid program.
struct InterchangeForPForLoad : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (std::next(op.getBody()->begin(), 2) == op.getBody()->end() ||
        std::next(op.getBody()->begin(), 3) != op.getBody()->end()) {
      LLVM_DEBUG(DBGS() << "[interchange load] expected two nested ops\n");
      return failure();
    }
    auto loadOp = dyn_cast<memref::LoadOp>(op.getBody()->front());
    auto forOp = dyn_cast<scf::ForOp>(op.getBody()->front().getNextNode());
    if (!loadOp || !forOp || loadOp.result() != forOp.upperBound() ||
        loadOp.indices() != op.getInductionVars()) {
      LLVM_DEBUG(DBGS() << "[interchange-load] expected pfor(load, for)");
      return failure();
    }

    if (!isNormalized(op) || !isNormalized(forOp)) {
      LLVM_DEBUG(DBGS() << "[interchange-load] non-normalized loop\n");
      return failure();
    }

    if (!hasNestedBarrier(forOp)) {
      LLVM_DEBUG(DBGS() << "[interchange load] no nested barrier\n");
      return failure();
    }

    // In the GPU model, the trip count of the inner sequential containing a
    // barrier must be the same for all threads. So read the value written by
    // the first thread outside of the loop to enable interchange.
    Value zero = rewriter.create<ConstantIndexOp>(forOp.getLoc(), 0);
    Value tripCount = rewriter.create<memref::LoadOp>(
        loadOp.getLoc(), loadOp.getMemRef(),
        SmallVector<Value>(loadOp.getMemRefType().getRank(), zero));
    auto newForLoop = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.lowerBound(), tripCount, forOp.step());

    moveBodies(rewriter, op, forOp, newForLoop);
    return success();
  }
};

/// Returns the insertion point (as block pointer and itertor in it) immediately
/// after the definition of `v`.
static std::pair<Block *, Block::iterator> getInsertionPointAfterDef(Value v) {
  if (Operation *op = v.getDefiningOp())
    return {op->getBlock(), std::next(Block::iterator(op))};

  BlockArgument blockArg = v.cast<BlockArgument>();
  return {blockArg.getParentBlock(), blockArg.getParentBlock()->begin()};
}

/// Returns the insertion point that post-dominates `first` and `second`.
static std::pair<Block *, Block::iterator>
findNearestPostDominatingInsertionPoint(
    const std::pair<Block *, Block::iterator> &first,
    const std::pair<Block *, Block::iterator> &second,
    const PostDominanceInfo &postDominanceInfo) {
  // Same block, take the last op.
  if (first.first == second.first)
    return first.second->isBeforeInBlock(&*second.second) ? second : first;

  // Same region, use "normal" dominance analysis.
  if (first.first->getParent() == second.first->getParent()) {
    Block *block =
        postDominanceInfo.findNearestCommonDominator(first.first, second.first);
    assert(block);
    if (block == first.first)
      return first;
    if (block == second.first)
      return second;
    return {block, block->begin()};
  }

  if (first.first->getParent()->isAncestor(second.first->getParent()))
    return second;

  assert(second.first->getParent()->isAncestor(first.first->getParent()) &&
         "expected values to be defined in nested regions");
  return first;
}

/// Returns the insertion point that post-dominates all `values`.
static std::pair<Block *, Block::iterator>
findNesrestPostDominatingInsertionPoint(
    ArrayRef<Value> values, const PostDominanceInfo &postDominanceInfo) {
  assert(!values.empty());
  std::pair<Block *, Block::iterator> insertPoint =
      getInsertionPointAfterDef(values[0]);
  for (unsigned i = 1, e = values.size(); i < e; ++i)
    insertPoint = findNearestPostDominatingInsertionPoint(
        insertPoint, getInsertionPointAfterDef(values[i]), postDominanceInfo);
  return insertPoint;
}

/// Splits a parallel loop around the first barrier it immediately contains.
/// Values defined before the barrier are stored in newly allocated buffers and
/// loaded back when needed.
struct DistributeAroundBarrier : public OpRewritePattern<scf::ParallelOp> {
  DistributeAroundBarrier(MLIRContext *ctx)
      : OpRewritePattern<scf::ParallelOp>(ctx) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumResults() != 0) {
      LLVM_DEBUG(DBGS() << "[distribute] not matching reduction loops\n");
      return failure();
    }

    if (!isNormalized(op)) {
      LLVM_DEBUG(DBGS() << "[distribute] non-normalized loop\n");
      return failure();
    }

    auto it =
        llvm::find_if(op.getBody()->getOperations(), [](Operation &nested) {
          return isa<scf::BarrierOp>(nested);
        });
    if (it == op.getBody()->end()) {
      LLVM_DEBUG(DBGS() << "[distribute] no barrier in the loop\n");
      return failure();
    }

    llvm::SetVector<Value> crossing;
    findValuesUsedBelow(&*it, crossing);

    // Find the earliest insertion point where loop bounds are fully defined.
    PostDominanceInfo postDominanceInfo(op->getParentOfType<FuncOp>());
    SmallVector<Value> operands;
    llvm::append_range(operands, op.lowerBound());
    llvm::append_range(operands, op.upperBound());
    llvm::append_range(operands, op.step());
    std::pair<Block *, Block::iterator> insertPoint =
        findNesrestPostDominatingInsertionPoint(operands, postDominanceInfo);

    // Emit code computing the total number of iterations in the loop. We don't
    // need to linearize them since we can allocate an nD array instead.
    SmallVector<Value> iterationCounts;
    rewriter.setInsertionPoint(insertPoint.first, insertPoint.second);
    for (auto bounds : llvm::zip(op.lowerBound(), op.upperBound(), op.step())) {
      Value lowerBound = std::get<0>(bounds);
      Value upperBound = std::get<1>(bounds);
      Value step = std::get<2>(bounds);
      Value diff = rewriter.create<SubIOp>(op.getLoc(), upperBound, lowerBound);
      Value count = rewriter.create<SignedCeilDivIOp>(op.getLoc(), diff, step);
      iterationCounts.push_back(count);
    }

    // Allocate space for values crossing the barrier.
    SmallVector<Value> allocations;
    allocations.reserve(crossing.size());
    SmallVector<int64_t> bufferSize(iterationCounts.size(),
                                    ShapedType::kDynamicSize);
    for (Value v : crossing) {
      auto type = MemRefType::get(bufferSize, v.getType());
      Value alloc =
          rewriter.create<memref::AllocaOp>(op.getLoc(), type, iterationCounts);
      allocations.push_back(alloc);
    }

    // Store values crossing the barrier in caches immediately when ready.
    for (auto pair : llvm::zip(crossing, allocations)) {
      Value v = std::get<0>(pair);
      Value alloc = std::get<1>(pair);
      rewriter.setInsertionPointAfter(v.getDefiningOp());
      rewriter.create<memref::StoreOp>(v.getLoc(), v, alloc,
                                       op.getInductionVars());
    }

    // Insert the terminator for the new loop immediately before the barrier.
    rewriter.setInsertionPoint(&*it);
    rewriter.create<scf::YieldOp>(op.getBody()->back().getLoc());

    // Create the second loop.
    rewriter.setInsertionPointAfter(op);
    auto newLoop = rewriter.create<scf::ParallelOp>(
        op.getLoc(), op.lowerBound(), op.upperBound(), op.step());
    rewriter.eraseOp(&newLoop.getBody()->back());

    // Recreate the operations in the new loop with new values.
    rewriter.setInsertionPointToStart(newLoop.getBody());
    BlockAndValueMapping mapping;
    mapping.map(op.getInductionVars(), newLoop.getInductionVars());
    SmallVector<Operation *> toDelete;
    toDelete.push_back(&*it);
    for (Operation *o = it->getNextNode(); o != nullptr; o = o->getNextNode()) {
      rewriter.clone(*o, mapping);
      toDelete.push_back(o);
    }

    // Erase original operations and the barrier.
    for (Operation *o : llvm::reverse(toDelete))
      rewriter.eraseOp(o);

    // Replace uses of values defined above the barrier (now, in a different
    // loop) with fresh loads from scratchpad. This may not be the most
    // efficient IR, but this avoids creating new crossing values for the
    // following barriers as opposed to putting loads at the start of the new
    // loop. We expect mem2reg and repeated load elimitation to improve the IR.
    newLoop.getBody()->walk([&](Operation *nested) {
      for (OpOperand &operand : nested->getOpOperands()) {
        auto it = llvm::find(crossing, operand.get());
        if (it == crossing.end())
          continue;

        size_t pos = std::distance(crossing.begin(), it);
        rewriter.setInsertionPoint(nested);
        Value reloaded = rewriter.create<memref::LoadOp>(
            operand.getOwner()->getLoc(), allocations[pos],
            newLoop.getInductionVars());
        rewriter.startRootUpdate(nested);
        operand.set(reloaded);
        rewriter.finalizeRootUpdate(nested);
      }
    });

    LLVM_DEBUG(DBGS() << "[distribute] distributed arround a barrier\n");
    return success();
  }
};

struct CPUifyPass : public SCFCPUifyBase<CPUifyPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<ReplaceIfWithFors, WrapForWithBarrier, InterchangeForPFor,
                    InterchangeForPForLoad, NormalizeLoop, NormalizeParallel,
                    DistributeAroundBarrier>(&getContext());
    GreedyRewriteConfig config;
    config.maxIterations = 42;
    if (failed(applyPatternsAndFoldGreedily(getFunction(), std::move(patterns),
                                            config)))
      signalPassFailure();
  }
};

} // end namespace

namespace mlir {
std::unique_ptr<Pass> createCPUifyPass() {
  return std::make_unique<CPUifyPass>();
}
} // namespace mlir
