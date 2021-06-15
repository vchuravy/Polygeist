// RUN: mlir-clang++ %s | FileCheck %s

#include <iostream>

template <typename T, int size>
struct alignas(16) Array {
  T data[size];

  Array(const Array&) = default;

  // Fill the array with x.
  Array(T x) {
    for (int i = 0; i < size; i++) {
      data[i] = x;
    }
  }
};

int main() {
    Array<int, 2> A(3);
    Array<int, 2> B(A);
    std::cout << "x: " << B.data[0] << " y: " << B.data[1] << "\n";
}

// CHECK:   func @_ZN5ArrayIiLi2EEC1ERKS0_(%arg0: !llvm.ptr<struct<(array<2 x i32>)>>, %arg1: !llvm.ptr<struct<(array<2 x i32>)>>) {
// CHECK-NEXT:     %c0_i32 = constant 0 : i32
// CHECK-NEXT:     %0 = llvm.getelementptr %arg0[%c0_i32, %c0_i32] : (!llvm.ptr<struct<(array<2 x i32>)>>, i32, i32) -> !llvm.ptr<array<2 x i32>>
// CHECK-NEXT:     affine.for %arg2 = 0 to 2 {
// CHECK-NEXT:       %1 = llvm.getelementptr %arg1[%c0_i32, %c0_i32] : (!llvm.ptr<struct<(array<2 x i32>)>>, i32, i32) -> !llvm.ptr<array<2 x i32>>
// CHECK-NEXT:       %2 = llvm.getelementptr %1[%c0_i32, %c0_i32] : (!llvm.ptr<array<2 x i32>>, i32, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:       %3 = index_cast %arg2 : index to i64
// CHECK-NEXT:       %4 = llvm.getelementptr %2[%3] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:       %5 = llvm.load %4 : !llvm.ptr<i32>
// CHECK-NEXT:       %6 = llvm.getelementptr %0[%c0_i32, %c0_i32] : (!llvm.ptr<array<2 x i32>>, i32, i32) -> !llvm.ptr<i32>
// CHECK-NEXT:       %7 = llvm.getelementptr %6[%3] : (!llvm.ptr<i32>, i64) -> !llvm.ptr<i32>
// CHECK-NEXT:       llvm.store %5, %7 : !llvm.ptr<i32>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

