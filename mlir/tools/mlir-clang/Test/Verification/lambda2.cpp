// RUN: mlir-clang++ %s --function=fn | FileCheck %s

#include <functional>

template<typename T>
long call(T f) {
  return f();
}

long fn(int a, long b) {
  return call([=]() {
    return 42;
  });
}


// CHECK:   func @_Z2fnil(%arg0: i32, %arg1: i64) -> i64 {
// CHECK-NEXT:     %c1_i64 = constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<()> : (i64) -> !llvm.ptr<struct<()>>
// CHECK-NEXT:     %1 = llvm.alloca %c1_i64 x !llvm.struct<()> : (i64) -> !llvm.ptr<struct<()>>
// CHECK-NEXT:     %2 = llvm.alloca %c1_i64 x !llvm.struct<()> : (i64) -> !llvm.ptr<struct<()>>
// CHECK-NEXT:     %3 = llvm.load %1 : !llvm.ptr<struct<()>>
// CHECK-NEXT:     llvm.store %3, %2 : !llvm.ptr<struct<()>>
// CHECK-NEXT:     call @_ZZ2fnilEN3$_0C1EOS_(%0, %2) : (!llvm.ptr<struct<()>>, !llvm.ptr<struct<()>>) -> ()
// CHECK-NEXT:     %4 = llvm.load %0 : !llvm.ptr<struct<()>>
// CHECK-NEXT:     %5 = call @_Z4callIZ2fnilE3$_0ElT_(%4) : (!llvm.struct<()>) -> i64
// CHECK-NEXT:     return %5 : i64
// CHECK-NEXT:   }

// CHECK:   func private @_Z4callIZ2fnilE3$_0ElT_(%arg0: !llvm.struct<()>) -> i64 {
// CHECK-NEXT:     %c1_i64 = constant 1 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<()> : (i64) -> !llvm.ptr<struct<()>>
// CHECK-NEXT:     llvm.store %arg0, %0 : !llvm.ptr<struct<()>>
// CHECK-NEXT:     %1 = call @_ZZ2fnilENK3$_0clEv(%0) : (!llvm.ptr<struct<()>>) -> i32
// CHECK-NEXT:     %2 = sexti %1 : i32 to i64
// CHECK-NEXT:     return %2 : i64
// CHECK-NEXT:   }

// CHECK:   func private @_ZZ2fnilEN3$_0C1EOS_(%arg0: !llvm.ptr<struct<()>>, %arg1: !llvm.ptr<struct<()>>) {
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK:   func private @_ZZ2fnilENK3$_0clEv(%arg0: !llvm.ptr<struct<()>>) -> i32 {
// CHECK-NEXT:     %c42_i32 = constant 42 : i32
// CHECK-NEXT:     return %c42_i32 : i32
// CHECK-NEXT:   }


