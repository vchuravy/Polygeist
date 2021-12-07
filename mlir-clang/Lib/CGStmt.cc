//===- CGStmt.cc - Emit MLIR IRs by walking stmt-like AST nodes-*- C++ --*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IfScope.h"
#include "clang-mlir.h"
#include "mlir/IR/Diagnostics.h"
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/SCF/SCF.h>

#define DEBUG_TYPE "CGStmt"

using namespace mlir;
using namespace mlir::arith;

static bool isTerminator(Operation *op) {
  return op->mightHaveTrait<OpTrait::IsTerminator>();
}

bool MLIRScanner::getLowerBound(clang::ForStmt *fors,
                                mlirclang::AffineLoopDescriptor &descr) {
  auto init = fors->getInit();
  if (auto declStmt = dyn_cast<DeclStmt>(init))
    if (declStmt->isSingleDecl()) {
      auto decl = declStmt->getSingleDecl();
      if (auto varDecl = dyn_cast<VarDecl>(decl)) {
        if (varDecl->hasInit()) {
          mlir::Value val = VisitVarDecl(varDecl).getValue(builder);
          descr.setName(varDecl);
          descr.setType(val.getType());
          LLVM_DEBUG(descr.getType().print(llvm::dbgs()));

          if (descr.getForwardMode())
            descr.setLowerBound(val);
          else {
            val = builder.create<AddIOp>(loc, val, getConstantIndex(1));
            descr.setUpperBound(val);
          }
          return true;
        }
      }
    }

  // BinaryOperator 0x7ff7aa17e938 'int' '='
  // |-DeclRefExpr 0x7ff7aa17e8f8 'int' lvalue Var 0x7ff7aa17e758 'i' 'int'
  // -IntegerLiteral 0x7ff7aa17e918 'int' 0
  if (auto binOp = dyn_cast<clang::BinaryOperator>(init))
    if (binOp->getOpcode() == clang::BinaryOperator::Opcode::BO_Assign)
      if (auto declRefStmt = dyn_cast<DeclRefExpr>(binOp->getLHS())) {
        mlir::Value val = Visit(binOp->getRHS()).getValue(builder);
        val = builder.create<IndexCastOp>(
            loc, val, mlir::IndexType::get(builder.getContext()));
        descr.setName(cast<VarDecl>(declRefStmt->getDecl()));
        descr.setType(getMLIRType(declRefStmt->getDecl()->getType()));
        if (descr.getForwardMode())
          descr.setLowerBound(val);
        else {
          val = builder.create<AddIOp>(loc, val, getConstantIndex(1));
          descr.setUpperBound(val);
        }
        return true;
      }
  return false;
}

// Make sure that the induction variable initialized in
// the for is the same as the one used in the condition.
bool matchIndvar(const Expr *expr, VarDecl *indVar) {
  while (auto IC = dyn_cast<ImplicitCastExpr>(expr)) {
    expr = IC->getSubExpr();
  }
  if (auto declRef = dyn_cast<DeclRefExpr>(expr)) {
    auto declRefName = declRef->getDecl();
    if (declRefName == indVar)
      return true;
  }
  return false;
}

bool MLIRScanner::getUpperBound(clang::ForStmt *fors,
                                mlirclang::AffineLoopDescriptor &descr) {
  auto cond = fors->getCond();
  if (auto binaryOp = dyn_cast<clang::BinaryOperator>(cond)) {
    auto lhs = binaryOp->getLHS();
    if (!matchIndvar(lhs, descr.getName()))
      return false;

    if (descr.getForwardMode()) {
      if (binaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_LT &&
          binaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_LE)
        return false;

      auto rhs = binaryOp->getRHS();
      mlir::Value val = Visit(rhs).getValue(builder);
      val = builder.create<IndexCastOp>(loc, val,
                                        mlir::IndexType::get(val.getContext()));
      if (binaryOp->getOpcode() == clang::BinaryOperator::Opcode::BO_LE)
        val = builder.create<AddIOp>(loc, val, getConstantIndex(1));
      descr.setUpperBound(val);
      return true;
    } else {
      if (binaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_GT &&
          binaryOp->getOpcode() != clang::BinaryOperator::Opcode::BO_GE)
        return false;

      auto rhs = binaryOp->getRHS();
      mlir::Value val = Visit(rhs).getValue(builder);
      val = builder.create<IndexCastOp>(loc, val,
                                        mlir::IndexType::get(val.getContext()));
      if (binaryOp->getOpcode() == clang::BinaryOperator::Opcode::BO_GT)
        val = builder.create<AddIOp>(loc, val, getConstantIndex(1));
      descr.setLowerBound(val);
      return true;
    }
  }
  return false;
}

bool MLIRScanner::getConstantStep(clang::ForStmt *fors,
                                  mlirclang::AffineLoopDescriptor &descr) {
  auto inc = fors->getInc();
  if (auto unaryOp = dyn_cast<clang::UnaryOperator>(inc))
    if (unaryOp->isPrefix() || unaryOp->isPostfix()) {
      bool forwardLoop =
          unaryOp->getOpcode() == clang::UnaryOperator::Opcode::UO_PostInc ||
          unaryOp->getOpcode() == clang::UnaryOperator::Opcode::UO_PreInc;
      descr.setStep(1);
      descr.setForwardMode(forwardLoop);
      return true;
    }
  return false;
}

bool MLIRScanner::isTrivialAffineLoop(clang::ForStmt *fors,
                                      mlirclang::AffineLoopDescriptor &descr) {
  if (!getConstantStep(fors, descr)) {
    LLVM_DEBUG(llvm::dbgs() << "getConstantStep -> false\n");
    return false;
  }
  if (!getLowerBound(fors, descr)) {
    LLVM_DEBUG(llvm::dbgs() << "getLowerBound -> false\n");
    return false;
  }
  if (!getUpperBound(fors, descr)) {
    LLVM_DEBUG(llvm::dbgs() << "getUpperBound -> false\n");
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "isTrivialAffineLoop -> true\n");
  return true;
}
void MLIRScanner::buildAffineLoopImpl(
    clang::ForStmt *fors, mlir::Location loc, mlir::Value lb, mlir::Value ub,
    const mlirclang::AffineLoopDescriptor &descr) {
  auto affineOp = builder.create<AffineForOp>(
      loc, lb, builder.getSymbolIdentityMap(), ub,
      builder.getSymbolIdentityMap(), descr.getStep(),
      /*iterArgs=*/llvm::None);

  auto &reg = affineOp.getLoopBody();

  auto val = (mlir::Value)affineOp.getInductionVar();

  reg.front().clear();

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();

  builder.setInsertionPointToEnd(&reg.front());

  auto er = builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  er.region().push_back(new Block());
  builder.setInsertionPointToStart(&er.region().back());
  builder.create<scf::YieldOp>(loc);
  builder.setInsertionPointToStart(&er.region().back());

  if (!descr.getForwardMode()) {
    val = builder.create<SubIOp>(loc, val, lb);
    val = builder.create<SubIOp>(
        loc, builder.create<SubIOp>(loc, ub, getConstantIndex(1)), val);
  }
  auto idx = builder.create<IndexCastOp>(loc, val, descr.getType());
  assert(params.find(descr.getName()) != params.end());
  params[descr.getName()].store(builder, idx);

  // TODO: set loop context.
  Visit(fors->getBody());

  builder.setInsertionPointToEnd(&reg.front());
  builder.create<AffineYieldOp>(loc);

  // TODO: set the value of the iteration value to the final bound at the
  // end of the loop.
  builder.setInsertionPoint(oldblock, oldpoint);
}

void MLIRScanner::buildAffineLoop(
    clang::ForStmt *fors, mlir::Location loc,
    const mlirclang::AffineLoopDescriptor &descr) {
  mlir::Value lb = descr.getLowerBound();
  mlir::Value ub = descr.getUpperBound();
  buildAffineLoopImpl(fors, loc, lb, ub, descr);
  return;
}

ValueCategory MLIRScanner::VisitForStmt(clang::ForStmt *fors) {
  IfScope scope(*this);

  auto loc = getMLIRLocation(fors->getForLoc());

  mlirclang::AffineLoopDescriptor affineLoopDescr;
  if (Glob.scopLocList.isInScop(fors->getForLoc()) &&
      isTrivialAffineLoop(fors, affineLoopDescr)) {
    buildAffineLoop(fors, loc, affineLoopDescr);
  } else {

    if (auto s = fors->getInit()) {
      Visit(s);
    }

    auto i1Ty = builder.getIntegerType(1);
    auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
    auto truev = builder.create<ConstantIntOp>(loc, true, 1);

    LoopContext lctx{builder.create<mlir::memref::AllocaOp>(loc, type),
                     builder.create<mlir::memref::AllocaOp>(loc, type)};
    builder.create<mlir::memref::StoreOp>(loc, truev, lctx.noBreak);

    auto toadd = builder.getInsertionBlock()->getParent();
    auto &condB = *(new Block());
    toadd->getBlocks().push_back(&condB);
    auto &bodyB = *(new Block());
    toadd->getBlocks().push_back(&bodyB);
    auto &exitB = *(new Block());
    toadd->getBlocks().push_back(&exitB);

    builder.create<mlir::BranchOp>(loc, &condB);

    builder.setInsertionPointToStart(&condB);

    if (auto s = fors->getCond()) {
      auto condRes = Visit(s);
      auto cond = condRes.getValue(builder);
      if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
        auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
        cond = builder.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
      }
      auto ty = cond.getType().cast<mlir::IntegerType>();
      if (ty.getWidth() != 1) {
        ty = builder.getIntegerType(1);
        cond = builder.create<arith::TruncIOp>(loc, cond, ty);
      }
      auto nb = builder.create<mlir::memref::LoadOp>(
          loc, lctx.noBreak, std::vector<mlir::Value>());
      cond = builder.create<AndIOp>(loc, cond, nb);
      builder.create<mlir::CondBranchOp>(loc, cond, &bodyB, &exitB);
    }

    builder.setInsertionPointToStart(&bodyB);
    builder.create<mlir::memref::StoreOp>(
        loc,
        builder.create<mlir::memref::LoadOp>(loc, lctx.noBreak,
                                             std::vector<mlir::Value>()),
        lctx.keepRunning, std::vector<mlir::Value>());

    loops.push_back(lctx);
    Visit(fors->getBody());
    if (auto s = fors->getInc()) {
      Visit(s);
    }
    loops.pop_back();
    if (builder.getInsertionBlock()->empty() ||
        !isTerminator(&builder.getInsertionBlock()->back())) {
      builder.create<mlir::BranchOp>(loc, &condB);
    }

    builder.setInsertionPointToStart(&exitB);
  }
  return nullptr;
}

ValueCategory MLIRScanner::VisitOMPParallelForDirective(
    clang::OMPParallelForDirective *fors) {
  IfScope scope(*this);

  Visit(fors->getPreInits());

  SmallVector<mlir::Value> inits;
  for (auto f : fors->inits()) {
    auto initV =
        cast<OMPCapturedExprDecl>(
            cast<DeclRefExpr>(
                cast<clang::CastExpr>(cast<clang::BinaryOperator>(f)->getRHS())
                    ->getSubExpr())
                ->getDecl())
            ->getInit();
    inits.push_back(builder.create<IndexCastOp>(
        loc, Visit(initV).getValue(builder), builder.getIndexType()));
  }

  SmallVector<mlir::Value> finals;
  for (auto f : fors->finals()) {
    auto bo = cast<clang::BinaryOperator>(
        cast<clang::BinaryOperator>(
            cast<clang::CastExpr>(cast<clang::BinaryOperator>(f)->getRHS())
                ->getSubExpr())
            ->getRHS());
    auto bo2 = cast<clang::BinaryOperator>(
        cast<clang::BinaryOperator>(
            cast<clang::BinaryOperator>(
                cast<ParenExpr>(cast<clang::BinaryOperator>(
                                    cast<ParenExpr>(bo->getLHS())->getSubExpr())
                                    ->getLHS())
                    ->getSubExpr())
                ->getLHS())
            ->getLHS());
    auto rhs = cast<OMPCapturedExprDecl>(
        cast<DeclRefExpr>(
            cast<clang::CastExpr>(
                cast<ParenExpr>(
                    cast<clang::CastExpr>(bo2->getLHS())->getSubExpr())
                    ->getSubExpr())
                ->getSubExpr())
            ->getDecl());
    finals.push_back(builder.create<IndexCastOp>(
        loc, Visit(rhs->getInit()).getValue(builder), builder.getIndexType()));
  }

  SmallVector<mlir::Value> incs;
  for (auto f : fors->updates()) {
    auto bo = cast<clang::BinaryOperator>(
        cast<clang::CastExpr>(cast<clang::BinaryOperator>(f)->getRHS())
            ->getSubExpr());
    auto rhs = cast<OMPCapturedExprDecl>(
        cast<DeclRefExpr>(
            cast<clang::CastExpr>(
                cast<clang::CastExpr>(
                    cast<clang::BinaryOperator>(bo->getRHS())->getRHS())
                    ->getSubExpr())
                ->getSubExpr())
            ->getDecl());

    incs.push_back(builder.create<IndexCastOp>(
        loc, Visit(rhs->getInit()).getValue(builder), builder.getIndexType()));
  }

  auto affineOp = builder.create<scf::ParallelOp>(loc, inits, finals, incs);

  auto inds = affineOp.getInductionVars();

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();

  builder.setInsertionPointToStart(&affineOp.region().front());

  auto executeRegion =
      builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  executeRegion.region().push_back(new Block());
  builder.setInsertionPointToStart(&executeRegion.region().back());

  auto oldScope = allocationScope;
  allocationScope = &executeRegion.region().back();

  for (auto zp : zip(inds, fors->counters())) {
    auto idx = builder.create<IndexCastOp>(
        loc, std::get<0>(zp),
        getMLIRType(fors->getIterationVariable()->getType()));
    VarDecl *name =
        cast<VarDecl>(cast<DeclRefExpr>(std::get<1>(zp))->getDecl());
    assert(params.find(name) == params.end() &&
           "OpenMP induction variable is dual initialized");

    bool LLVMABI = false;
    bool isArray = false;
    if (Glob.getMLIRType(
                Glob.CGM.getContext().getLValueReferenceType(name->getType()))
            .isa<mlir::LLVM::LLVMPointerType>())
      LLVMABI = true;
    else
      Glob.getMLIRType(name->getType(), &isArray);

    auto allocop = createAllocOp(idx.getType(), name, /*memtype*/ 0,
                                 /*isArray*/ isArray, /*LLVMABI*/ LLVMABI);
    params[name] = ValueCategory(allocop, true);
    params[name].store(builder, idx);
  }

  // TODO: set loop context.
  Visit(fors->getBody());

  builder.create<scf::YieldOp>(loc);

  allocationScope = oldScope;

  // TODO: set the value of the iteration value to the final bound at the
  // end of the loop.
  builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

ValueCategory MLIRScanner::VisitDoStmt(clang::DoStmt *fors) {
  IfScope scope(*this);

  auto loc = getMLIRLocation(fors->getDoLoc());

  auto i1Ty = builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = builder.create<ConstantIntOp>(loc, true, 1);
  loops.push_back(
      (LoopContext){builder.create<mlir::memref::AllocaOp>(loc, type),
                    builder.create<mlir::memref::AllocaOp>(loc, type)});
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);

  auto toadd = builder.getInsertionBlock()->getParent();
  auto &condB = *(new Block());
  toadd->getBlocks().push_back(&condB);
  auto &bodyB = *(new Block());
  toadd->getBlocks().push_back(&bodyB);
  auto &exitB = *(new Block());
  toadd->getBlocks().push_back(&exitB);

  builder.create<mlir::BranchOp>(loc, &bodyB);

  builder.setInsertionPointToStart(&condB);

  if (auto s = fors->getCond()) {
    auto condRes = Visit(s);
    auto cond = condRes.getValue(builder);
    if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      cond = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
    }
    auto ty = cond.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      ty = builder.getIntegerType(1);
      cond = builder.create<arith::TruncIOp>(loc, cond, ty);
    }
    auto nb = builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                                   std::vector<mlir::Value>());
    cond = builder.create<AndIOp>(loc, cond, nb);
    builder.create<mlir::CondBranchOp>(loc, cond, &bodyB, &exitB);
  }

  builder.setInsertionPointToStart(&bodyB);
  builder.create<mlir::memref::StoreOp>(
      loc,
      builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                           std::vector<mlir::Value>()),
      loops.back().keepRunning, std::vector<mlir::Value>());

  Visit(fors->getBody());
  loops.pop_back();

  builder.create<mlir::BranchOp>(loc, &condB);

  builder.setInsertionPointToStart(&exitB);

  return nullptr;
}

ValueCategory MLIRScanner::VisitWhileStmt(clang::WhileStmt *fors) {
  IfScope scope(*this);

  auto loc = getMLIRLocation(fors->getLParenLoc());

  auto i1Ty = builder.getIntegerType(1);
  auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
  auto truev = builder.create<ConstantIntOp>(loc, true, 1);
  loops.push_back(
      (LoopContext){builder.create<mlir::memref::AllocaOp>(loc, type),
                    builder.create<mlir::memref::AllocaOp>(loc, type)});
  builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);

  auto toadd = builder.getInsertionBlock()->getParent();
  auto &condB = *(new Block());
  toadd->getBlocks().push_back(&condB);
  auto &bodyB = *(new Block());
  toadd->getBlocks().push_back(&bodyB);
  auto &exitB = *(new Block());
  toadd->getBlocks().push_back(&exitB);

  builder.create<mlir::BranchOp>(loc, &condB);

  builder.setInsertionPointToStart(&condB);

  if (auto s = fors->getCond()) {
    auto condRes = Visit(s);
    auto cond = condRes.getValue(builder);
    if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
      auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
      cond = builder.create<mlir::LLVM::ICmpOp>(
          loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
    }
    auto ty = cond.getType().cast<mlir::IntegerType>();
    if (ty.getWidth() != 1) {
      ty = builder.getIntegerType(1);
      cond = builder.create<arith::TruncIOp>(loc, cond, ty);
    }
    auto nb = builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                                   std::vector<mlir::Value>());
    cond = builder.create<AndIOp>(loc, cond, nb);
    builder.create<mlir::CondBranchOp>(loc, cond, &bodyB, &exitB);
  }

  builder.setInsertionPointToStart(&bodyB);
  builder.create<mlir::memref::StoreOp>(
      loc,
      builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak,
                                           std::vector<mlir::Value>()),
      loops.back().keepRunning, std::vector<mlir::Value>());

  Visit(fors->getBody());
  loops.pop_back();

  builder.create<mlir::BranchOp>(loc, &condB);

  builder.setInsertionPointToStart(&exitB);

  return nullptr;
}

ValueCategory MLIRScanner::VisitIfStmt(clang::IfStmt *stmt) {
  IfScope scope(*this);
  auto loc = getMLIRLocation(stmt->getIfLoc());
  auto cond = Visit(stmt->getCond()).getValue(builder);
  assert(cond != nullptr && "must be a non-null");

  auto oldpoint = builder.getInsertionPoint();
  auto oldblock = builder.getInsertionBlock();
  if (auto LT = cond.getType().dyn_cast<mlir::LLVM::LLVMPointerType>()) {
    auto nullptr_llvm = builder.create<mlir::LLVM::NullOp>(loc, LT);
    cond = builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::ne, cond, nullptr_llvm);
  }
  auto prevTy = cond.getType().cast<mlir::IntegerType>();
  if (!prevTy.isInteger(1)) {
    auto postTy = builder.getI1Type();
    cond = builder.create<arith::TruncIOp>(loc, cond, postTy);
  }
  bool hasElseRegion = stmt->getElse();
  auto ifOp = builder.create<mlir::scf::IfOp>(loc, cond, hasElseRegion);

  ifOp.thenRegion().back().clear();
  builder.setInsertionPointToStart(&ifOp.thenRegion().back());
  Visit(stmt->getThen());
  builder.create<scf::YieldOp>(loc);
  if (hasElseRegion) {
    ifOp.elseRegion().back().clear();
    builder.setInsertionPointToStart(&ifOp.elseRegion().back());
    Visit(stmt->getElse());
    builder.create<scf::YieldOp>(loc);
  }

  builder.setInsertionPoint(oldblock, oldpoint);
  return nullptr;
}

ValueCategory MLIRScanner::VisitSwitchStmt(clang::SwitchStmt *stmt) {
  IfScope scope(*this);
  auto cond = Visit(stmt->getCond()).getValue(builder);
  assert(cond != nullptr);
  SmallVector<int64_t> caseVals;

  auto er = builder.create<scf::ExecuteRegionOp>(loc, ArrayRef<mlir::Type>());
  er.region().push_back(new Block());
  auto oldpoint2 = builder.getInsertionPoint();
  auto oldblock2 = builder.getInsertionBlock();

  auto &exitB = *(new Block());
  builder.setInsertionPointToStart(&exitB);
  builder.create<scf::YieldOp>(loc);
  builder.setInsertionPointToStart(&exitB);

  SmallVector<Block *> blocks;
  bool inCase = false;

  Block *defaultB = &exitB;

  for (auto cse : stmt->getBody()->children()) {
    if (auto cses = dyn_cast<CaseStmt>(cse)) {
      auto &condB = *(new Block());

      auto cval = Visit(cses->getLHS());
      if (!cval.val) {
          cses->getLHS()->dump();
      }
      assert(cval.val);
      auto cint = cval.getValue(builder).getDefiningOp<ConstantIntOp>();
      if (!cint) {
          cses->getLHS()->dump();
          llvm::errs() << "cval: " << cval.val << "\n";
      }
      assert(cint);
      caseVals.push_back(cint.value());

      if (inCase) {
        auto noBreak =
            builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak);
        builder.create<mlir::CondBranchOp>(loc, noBreak, &condB, &exitB);
        loops.pop_back();
      }

      inCase = true;
      er.region().getBlocks().push_back(&condB);
      blocks.push_back(&condB);
      builder.setInsertionPointToStart(&condB);

      auto i1Ty = builder.getIntegerType(1);
      auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
      auto truev = builder.create<ConstantIntOp>(loc, true, 1);
      loops.push_back(
          (LoopContext){builder.create<mlir::memref::AllocaOp>(loc, type),
                        builder.create<mlir::memref::AllocaOp>(loc, type)});
      builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);
      builder.create<mlir::memref::StoreOp>(loc, truev,
                                            loops.back().keepRunning);
      Visit(cses->getSubStmt());
    } else if (auto cses = dyn_cast<DefaultStmt>(cse)) {
      auto &condB = *(new Block());

      if (inCase) {
        auto noBreak =
            builder.create<mlir::memref::LoadOp>(loc, loops.back().noBreak);
        builder.create<mlir::CondBranchOp>(loc, noBreak, &condB, &exitB);
        loops.pop_back();
      }

      inCase = true;
      er.region().getBlocks().push_back(&condB);
      builder.setInsertionPointToStart(&condB);

      auto i1Ty = builder.getIntegerType(1);
      auto type = mlir::MemRefType::get({}, i1Ty, {}, 0);
      auto truev = builder.create<ConstantIntOp>(loc, true, 1);
      loops.push_back(
          (LoopContext){builder.create<mlir::memref::AllocaOp>(loc, type),
                        builder.create<mlir::memref::AllocaOp>(loc, type)});
      builder.create<mlir::memref::StoreOp>(loc, truev, loops.back().noBreak);
      builder.create<mlir::memref::StoreOp>(loc, truev,
                                            loops.back().keepRunning);
      defaultB = &condB;
      Visit(cses->getSubStmt());
    } else {
      Visit(cse);
    }
  }

  if (inCase)
    loops.pop_back();
  builder.create<mlir::BranchOp>(loc, &exitB);

  er.region().getBlocks().push_back(&exitB);

  DenseIntElementsAttr caseValuesAttr;
  ShapedType caseValueType = mlir::VectorType::get(
      static_cast<int64_t>(caseVals.size()), cond.getType());
  auto ity = cond.getType().cast<mlir::IntegerType>();
  if (ity.getWidth() == 64)
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseVals);
  else if (ity.getWidth() == 32) {
    SmallVector<int32_t> caseVals32;
    for (auto v : caseVals)
      caseVals32.push_back((int32_t)v);
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseVals32);
  } else if (ity.getWidth() == 16) {
    SmallVector<int16_t> caseVals16;
    for (auto v : caseVals)
      caseVals16.push_back((int16_t)v);
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseVals16);
  } else {
    assert(ity.getWidth() == 8);
    SmallVector<int8_t> caseVals8;
    for (auto v : caseVals)
      caseVals8.push_back((int8_t)v);
    caseValuesAttr = DenseIntElementsAttr::get(caseValueType, caseVals8);
  }

  builder.setInsertionPointToStart(&er.region().front());
  builder.create<mlir::SwitchOp>(
      loc, cond, defaultB, ArrayRef<mlir::Value>(), caseValuesAttr, blocks,
      SmallVector<mlir::ValueRange>(caseVals.size(), ArrayRef<mlir::Value>()));
  builder.setInsertionPoint(oldblock2, oldpoint2);
  return nullptr;
}

ValueCategory MLIRScanner::VisitDeclStmt(clang::DeclStmt *decl) {
  IfScope scope(*this);
  for (auto sub : decl->decls()) {
    if (auto vd = dyn_cast<VarDecl>(sub)) {
      VisitVarDecl(vd);
    } else if (isa<TypeAliasDecl, RecordDecl, StaticAssertDecl, TypedefDecl,
                   UsingDecl>(sub)) {
    } else {
      emitError(getMLIRLocation(decl->getBeginLoc()))
          << " + visiting unknonwn sub decl stmt\n";
      sub->dump();
      assert(0 && "unknown sub decl");
    }
  }
  return nullptr;
}

ValueCategory MLIRScanner::VisitAttributedStmt(AttributedStmt *AS) {
  emitWarning(getMLIRLocation(AS->getAttrLoc())) << "ignoring attributes\n";
  return Visit(AS->getSubStmt());
}

ValueCategory MLIRScanner::VisitCompoundStmt(clang::CompoundStmt *stmt) {
  for (auto a : stmt->children()) {
    IfScope scope(*this);
    Visit(a);
  }
  return nullptr;
}

ValueCategory MLIRScanner::VisitBreakStmt(clang::BreakStmt *stmt) {
  IfScope scope(*this);
  assert(loops.size() && "must be non-empty");
  assert(loops.back().keepRunning && "keep running false");
  assert(loops.back().noBreak && "no break false");
  auto vfalse =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), false, 1);
  builder.create<mlir::memref::StoreOp>(loc, vfalse, loops.back().keepRunning);
  builder.create<mlir::memref::StoreOp>(loc, vfalse, loops.back().noBreak);

  return nullptr;
}

ValueCategory MLIRScanner::VisitContinueStmt(clang::ContinueStmt *stmt) {
  IfScope scope(*this);
  assert(loops.size() && "must be non-empty");
  assert(loops.back().keepRunning && "keep running false");
  auto vfalse =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), false, 1);
  builder.create<mlir::memref::StoreOp>(loc, vfalse, loops.back().keepRunning);
  return nullptr;
}

ValueCategory MLIRScanner::VisitLabelStmt(clang::LabelStmt *stmt) {
  auto toadd = builder.getInsertionBlock()->getParent();
  Block *labelB;
  auto found = labels.find(stmt);
  if (found != labels.end()) {
    labelB = found->second;
  } else {
    labelB = new Block();
    labels[stmt] = labelB;
  }
  toadd->getBlocks().push_back(labelB);
  builder.create<mlir::BranchOp>(loc, labelB);
  builder.setInsertionPointToStart(labelB);
  Visit(stmt->getSubStmt());
  return nullptr;
}

ValueCategory MLIRScanner::VisitGotoStmt(clang::GotoStmt *stmt) {
  auto labelstmt = stmt->getLabel()->getStmt();
  Block *labelB;
  auto found = labels.find(labelstmt);
  if (found != labels.end()) {
    labelB = found->second;
  } else {
    labelB = new Block();
    labels[labelstmt] = labelB;
  }
  builder.create<mlir::BranchOp>(loc, labelB);
  return nullptr;
}

ValueCategory MLIRScanner::VisitReturnStmt(clang::ReturnStmt *stmt) {
  IfScope scope(*this);
  bool isArrayReturn = false;
  Glob.getMLIRType(EmittingFunctionDecl->getReturnType(), &isArrayReturn);

  if (isArrayReturn) {
    auto rv = Visit(stmt->getRetValue());
    assert(rv.val && "expect right value to be valid");
    assert(rv.isReference && "right value must be a reference");
    auto op = function.getArgument(function.getNumArguments() - 1);
    assert(rv.val.getType().cast<MemRefType>().getElementType() ==
               op.getType().cast<MemRefType>().getElementType() &&
           "type mismatch");
    assert(op.getType().cast<MemRefType>().getShape().size() == 2 &&
           "expect 2d memref");
    assert(rv.val.getType().cast<MemRefType>().getShape().size() == 2 &&
           "expect 2d memref");
    assert(rv.val.getType().cast<MemRefType>().getShape()[1] ==
           op.getType().cast<MemRefType>().getShape()[1]);

    for (int i = 0; i < op.getType().cast<MemRefType>().getShape()[1]; i++) {
      std::vector<mlir::Value> idx = {getConstantIndex(0), getConstantIndex(i)};
      assert(rv.val.getType().cast<MemRefType>().getShape().size() == 2);
      builder.create<mlir::memref::StoreOp>(
          loc, builder.create<mlir::memref::LoadOp>(loc, rv.val, idx), op, idx);
    }
  } else if (stmt->getRetValue()) {
    auto rv = Visit(stmt->getRetValue());
    if (!stmt->getRetValue()->getType()->isVoidType()) {
      if (!rv.val) {
        stmt->dump();
      }
      assert(rv.val && "expect right value to be valid");

      mlir::Value val;
      if (stmt->getRetValue()->isLValue() || stmt->getRetValue()->isXValue()) {
        assert(rv.isReference);
        val = rv.val;
      } else {
        val = rv.getValue(builder);
      }

      auto postTy = returnVal.getType().cast<MemRefType>().getElementType();
      if (auto prevTy = val.getType().dyn_cast<mlir::IntegerType>()) {
        auto ipostTy = postTy.cast<mlir::IntegerType>();
        if (prevTy != ipostTy) {
          val = builder.create<arith::TruncIOp>(loc, val, ipostTy);
        }
      } else if (val.getType().isa<MemRefType>() &&
                 postTy.isa<LLVM::LLVMPointerType>())
        val = builder.create<polygeist::Memref2PointerOp>(loc, postTy, val);
      else if (val.getType().isa<LLVM::LLVMPointerType>() &&
               postTy.isa<MemRefType>())
        val = builder.create<polygeist::Pointer2MemrefOp>(loc, postTy, val);
      builder.create<mlir::memref::StoreOp>(loc, val, returnVal);
    }
  }

  assert(loops.size() && "must be non-empty");
  auto vfalse =
      builder.create<ConstantIntOp>(builder.getUnknownLoc(), false, 1);
  for (auto l : loops) {
    builder.create<mlir::memref::StoreOp>(loc, vfalse, l.keepRunning);
    builder.create<mlir::memref::StoreOp>(loc, vfalse, l.noBreak);
  }

  return nullptr;
}
