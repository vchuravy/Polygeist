// RUN: mlir-clang++ %s --function=fn | FileCheck %s

// XFAIL: *
// Currently fails as allowing a store of memref into ptr doesn't succeed

#include <functional>

template<typename T>
long call(T f) {
  return f();
}

long fn(int a, long b) {
  return call([&]() {
    return a * b;
  });
}
