// RUN: mlir-clang++ %s --function=fn | FileCheck %s

#include <functional>

template<typename T>
long call(T f) {
  return f();
}

long fn(int a, long b) {
  return call([=]() {
    return a * b;
  });
}

// CHECK:   func @_Z2fnil(%arg0: i32, %arg1: i64) -> i64 {
// CHECK-NEXT:     %c1_i64 = constant 1 : i64
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(i32, memref<4xi8>, i64)> : (i64) -> !llvm.ptr<struct<(i32, memref<4xi8>, i64)>>
// CHECK-NEXT:     %1 = llvm.alloca %c1_i64 x !llvm.struct<(i32, memref<4xi8>, i64)> : (i64) -> !llvm.ptr<struct<(i32, memref<4xi8>, i64)>>
// CHECK-NEXT:     %2 = llvm.getelementptr %1[%c0_i32, %c2_i32] : (!llvm.ptr<struct<(i32, memref<4xi8>, i64)>>, i32, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:     llvm.store %arg1, %2 : !llvm.ptr<i64>
// CHECK-NEXT:     %3 = llvm.getelementptr %1[%c0_i32, %c0_i32] : (!llvm.ptr<struct<(i32, memref<4xi8>, i64)>>, i32, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %arg0, %3 : !llvm.ptr<i32>
// CHECK-NEXT:     call @_ZZ2fnilEN3$_0C1EOS_(%0, %1) : (!llvm.ptr<struct<(i32, memref<4xi8>, i64)>>, !llvm.ptr<struct<(i32, memref<4xi8>, i64)>>) -> ()
// CHECK-NEXT:     %4 = llvm.load %0 : !llvm.ptr<struct<(i32, memref<4xi8>, i64)>>
// CHECK-NEXT:     %5 = call @_Z4callIZ2fnilE3$_0ElT_(%4) : (!llvm.struct<(i32, memref<4xi8>, i64)>) -> i64
// CHECK-NEXT:     return %5 : i64
// CHECK-NEXT:   }

// CHECK-NEXT:  func private @_Z4callIZ2fnilE3$_0ElT_(%arg0: !llvm.struct<(i32, memref<4xi8>, i64)>) -> i64 {
// CHECK-NEXT:    %c1_i64 = constant 1 : i64
// CHECK-NEXT:    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i32, memref<4xi8>, i64)> : (i64) -> !llvm.ptr<struct<(i32, memref<4xi8>, i64)>>
// CHECK-NEXT:    llvm.store %arg0, %0 : !llvm.ptr<struct<(i32, memref<4xi8>, i64)>>
// CHECK-NEXT:    %1 = call @_ZZ2fnilENK3$_0clEv(%0) : (!llvm.ptr<struct<(i32, memref<4xi8>, i64)>>) -> i64
// CHECK-NEXT:    return %1 : i64
// CHECK-NEXT:  }
// CHECK-NEXT:  func private @_ZZ2fnilEN3$_0C1EOS_(%arg0: !llvm.ptr<struct<(i32, memref<4xi8>, i64)>>, %arg1: !llvm.ptr<struct<(i32, memref<4xi8>, i64)>>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c2_i32 = constant 2 : i32
// CHECK-NEXT:     %0 = llvm.getelementptr %arg1[%c0_i32, %c0_i32] : (!llvm.ptr<struct<(i32, memref<4xi8>, i64)>>, i32, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:     %1 = llvm.load %0 : !llvm.ptr<i32>
// CHECK-NEXT:     %2 = llvm.getelementptr %arg0[%c0_i32, %c0_i32] : (!llvm.ptr<struct<(i32, memref<4xi8>, i64)>>, i32, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:     llvm.store %1, %2 : !llvm.ptr<i32>
// CHECK-NEXT:     %3 = llvm.getelementptr %arg1[%c0_i32, %c2_i32] : (!llvm.ptr<struct<(i32, memref<4xi8>, i64)>>, i32, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:     %4 = llvm.load %3 : !llvm.ptr<i64>
// CHECK-NEXT:     %5 = llvm.getelementptr %arg0[%c0_i32, %c2_i32] : (!llvm.ptr<struct<(i32, memref<4xi8>, i64)>>, i32, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:     llvm.store %4, %5 : !llvm.ptr<i64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:  func private @_ZZ2fnilENK3$_0clEv(%arg0: !llvm.ptr<struct<(i32, memref<4xi8>, i64)>>) -> i64 {
// CHECK-NEXT:    %c0_i32 = constant 0 : i32
// CHECK-NEXT:    %c2_i32 = constant 2 : i32
// CHECK-NEXT:    %[[i0:.+]] = llvm.getelementptr %arg0[%c0_i32, %c0_i32] : (!llvm.ptr<struct<(i32, memref<4xi8>, i64)>>, i32, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:    %1 = llvm.load %[[i0]] : !llvm.ptr<i32>
// CHECK-NEXT:    %2 = sexti %1 : i32 to i64
// CHECK-NEXT:    %3 = llvm.getelementptr %arg0[%c0_i32, %c2_i32] : (!llvm.ptr<struct<(i32, memref<4xi8>, i64)>>, i32, i32) -> !llvm.ptr<i64>
// CHECK-NEXT:    %4 = llvm.load %3 : !llvm.ptr<i64>
// CHECK-NEXT:    %5 = muli %2, %4 : i64
// CHECK-NEXT:    return %5 : i64
// CHECK-NEXT:  }
