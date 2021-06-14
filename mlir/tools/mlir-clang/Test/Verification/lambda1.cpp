// RUN: mlir-clang++ %s --function=fn | FileCheck %s

#include <functional>

template<typename T>
long call(T f) {
  return f();
}

long fn(long a, long b) {
  return call([=]() {
    return a * b;
  });
}

// CHECK:   func @_Z2fnll(%arg0: i64, %arg1: i64) -> i64 {
// CHECK-NEXT:     %0 = memref.alloca() : memref<1x2xi64>
// CHECK-NEXT:     %1 = memref.cast %0 : memref<1x2xi64> to memref<?x2xi64>
// CHECK-NEXT:     %2 = memref.alloca() : memref<1x2xi64>
// CHECK-NEXT:     %3 = memref.cast %2 : memref<1x2xi64> to memref<?x2xi64>
// CHECK-DAG:     affine.store %arg0, %2[0, 0] : memref<1x2xi64>
// CHECK-DAG:     affine.store %arg1, %2[0, 1] : memref<1x2xi64>
// CHECK-NEXT:     call @_ZZ2fnllEN3$_0C1EOS_(%1, %3) : (memref<?x2xi64>, memref<?x2xi64>) -> ()
// CHECK-NEXT:     %4 = memref.alloca() : memref<1x2xi64>
// CHECK-NEXT:     %5 = affine.load %0[0, 0] : memref<1x2xi64>
// CHECK-NEXT:     affine.store %5, %4[0, 0] : memref<1x2xi64>
// CHECK-NEXT:     %6 = affine.load %0[0, 1] : memref<1x2xi64>
// CHECK-NEXT:     affine.store %6, %4[0, 1] : memref<1x2xi64>
// CHECK-NEXT:     %7 = memref.cast %4 : memref<1x2xi64> to memref<?x2xi64>
// CHECK-NEXT:     %8 = call @_Z4callIZ2fnllE3$_0ElT_(%7) : (memref<?x2xi64>) -> i64
// CHECK-NEXT:     return %8 : i64
// CHECK-NEXT:   }
// CHECK-NEXT:   func private @_Z4callIZ2fnllE3$_0ElT_(%arg0: memref<?x2xi64>) -> i64 {
// CHECK-NEXT:     %0 = call @_ZZ2fnllENK3$_0clEv(%arg0) : (memref<?x2xi64>) -> i64
// CHECK-NEXT:     return %0 : i64
// CHECK-NEXT:   }
//
// CHECK:   func private @_ZZ2fnllEN3$_0C1EOS_(%arg0: memref<?x2xi64>, %arg1: memref<?x2xi64>) {
// CHECK-NEXT:     %0 = affine.load %arg1[0, 1] : memref<?x2xi64>
// CHECK-NEXT:     affine.store %0, %arg0[0, 0] : memref<?x2xi64>
// CHECK-NEXT:     %1 = affine.load %arg1[0, 1] : memref<?x2xi64>
// CHECK-NEXT:     affine.store %1, %arg0[0, 1] : memref<?x2xi64>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
//
// CHECK:   func private @_ZZ2fnllENK3$_0clEv(%arg0: memref<?x2xi64>) -> i64 {
// CHECK-NEXT:     %0 = affine.load %arg0[0, 0] : memref<?x2xi64>
// CHECK-NEXT:     %1 = affine.load %arg0[0, 1] : memref<?x2xi64>
// CHECK-NEXT:     %2 = muli %0, %1 : i64
// CHECK-NEXT:     return %2 : i64
// CHECK-NEXT:   }
