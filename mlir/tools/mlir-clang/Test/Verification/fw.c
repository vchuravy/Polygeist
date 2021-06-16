// RUN: mlir-clang %s %stdinclude | FileCheck %s

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

#   define N 2800

/* Array initialization. */
static
void init_array (int path[N])
{
  //path[0][1] = 2;
}

int main(int argc, char** argv)
{
  /* Retrieve problem size. */

  /* Variable declaration/allocation. */
  //POLYBENCH_1D_ARRAY_DECL(path, int, N, n);
  int (*path)[N];
  //int path[POLYBENCH_C99_SELECT(N,n) + POLYBENCH_PADDING_FACTOR];
  path = (int(*)[N])polybench_alloc_data (N, sizeof(int)) ;

  /* Initialize array(s). */
  init_array (*path);


  return 0;
}

// CHECK:     func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
// CHECK:     %c0_i32 = constant 0 : i32
// CHECK:     %[[alloc:.+]] = memref.alloc() : memref<2800xi32>
// CHECK-DAG:     %[[cst:.+]] = memref.cast %[[alloc]] : memref<2800xi32> to memref<?xi32>
// CHECK-DAG:     call @init_array(%[[cst]]) : (memref<?xi32>) -> ()
// CHECK-DAG:     return %c0_i32 : i32
// CHECK-NEXT:   }

// CHECK:   func private @init_array(%arg0: memref<?xi32>) {
// CHECK-NEXT:     return
// CHECK-NEXT:   }
