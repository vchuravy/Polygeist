// RUN: mlir-clang++ %s | FileCheck %s
#include <stdio.h>

class Vec {
    public:
    double x;
};

class MetaVec {
    public:
        Vec v;
        int n;
};

void func(Vec& v) {

}
int main() {
    MetaVec mv;
    Vec v(mv.v);
    func(mv.v);
    printf("%f\n", v.x);
}

// CHECK:   func @main() -> i32 {
// CHECK-NEXT:     %c1_i64 = constant 1 : i64
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %c1_i32 = constant 1 : i32
// CHECK-NEXT:     %c0_i64 = constant 0 : i64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(struct<(f64, f64)>, i32)> : (i64) -> !llvm.ptr<struct<(struct<(f64, f64)>, i32)>>
// CHECK-NEXT:     call @_ZN7MetaVecC1Ev(%0) : (!llvm.ptr<struct<(struct<(f64, f64)>, i32)>>) -> ()
// CHECK-NEXT:     %1 = memref.alloca() : memref<1x2xf64>
// CHECK-NEXT:     %2 = memref.cast %1 : memref<1x2xf64> to memref<?x2xf64>
// CHECK-NEXT:     %3 = llvm.getelementptr %0[%c0_i32, %c0_i32] : (!llvm.ptr<struct<(struct<(f64, f64)>, i32)>>, i32, i32) -> !llvm.ptr<struct<(f64, f64)>>
// CHECK-NEXT:     %4 = memref.alloca() : memref<1x2xf64>
// CHECK-NEXT:     %5 = llvm.getelementptr %3[%c0_i32, %c0_i32] : (!llvm.ptr<struct<(f64, f64)>>, i32, i32) -> !llvm.ptr<f64>
// CHECK-NEXT:     %6 = llvm.load %5 : !llvm.ptr<f64>
// CHECK-NEXT:     affine.store %6, %4[0, 0] : memref<1x2xf64>
// CHECK-NEXT:     %7 = llvm.getelementptr %3[%c0_i32, %c1_i32] : (!llvm.ptr<struct<(f64, f64)>>, i32, i32) -> !llvm.ptr<f64>
// CHECK-NEXT:     %8 = llvm.load %7 : !llvm.ptr<f64>
// CHECK-NEXT:     affine.store %8, %4[0, 1] : memref<1x2xf64>
// CHECK-NEXT:     %9 = memref.cast %4 : memref<1x2xf64> to memref<?x2xf64>
// CHECK-NEXT:     call @_ZN3VecC1ERKS_(%2, %9) : (memref<?x2xf64>, memref<?x2xf64>) -> ()
// CHECK-NEXT:     %10 = affine.load %4[0, 0] : memref<1x2xf64>
// CHECK-NEXT:     llvm.store %10, %5 : !llvm.ptr<f64>
// CHECK-NEXT:     %11 = affine.load %4[0, 1] : memref<1x2xf64>
// CHECK-NEXT:     llvm.store %11, %7 : !llvm.ptr<f64>
// CHECK-NEXT:     %12 = memref.alloca() : memref<1x2xf64>
// CHECK-NEXT:     %13 = llvm.load %5 : !llvm.ptr<f64>
// CHECK-NEXT:     affine.store %13, %12[0, 0] : memref<1x2xf64>
// CHECK-NEXT:     %14 = llvm.load %7 : !llvm.ptr<f64>
// CHECK-NEXT:     affine.store %14, %12[0, 1] : memref<1x2xf64>
// CHECK-NEXT:     %15 = memref.cast %12 : memref<1x2xf64> to memref<?x2xf64>
// CHECK-NEXT:     call @_Z4funcR3Vec(%15) : (memref<?x2xf64>) -> ()
// CHECK-NEXT:     %16 = affine.load %12[0, 0] : memref<1x2xf64>
// CHECK-NEXT:     llvm.store %16, %5 : !llvm.ptr<f64>
// CHECK-NEXT:     %17 = affine.load %12[0, 1] : memref<1x2xf64>
// CHECK-NEXT:     llvm.store %17, %7 : !llvm.ptr<f64>
// CHECK-NEXT:     %18 = llvm.mlir.addressof @str0 : !llvm.ptr<array<4 x i8>>
// CHECK-NEXT:     %19 = llvm.getelementptr %18[%c0_i64, %c0_i64] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK-NEXT:     %20 = affine.load %1[0, 0] : memref<1x2xf64>
// CHECK-NEXT:     %21 = llvm.call @printf(%19, %20) : (!llvm.ptr<i8>, f64) -> i32
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }
