#pragma once

#define RPA_SA_PRIVATE_OPTION(NAME, VAL, ...)         \
  if (!VAL) {                                      \
    static const bool NAME = false;                \
    return __VA_ARGS__();                          \
  } else {                                         \
    static const bool NAME = true;                 \
    return __VA_ARGS__();                          \
  }

#define RPA_SA_DISPATCH_CONDITION(ARG1, ...)                 \
  [&] {                                                   \
    RPA_SA_PRIVATE_OPTION(ARG1, ARG1, __VA_ARGS__)           \
  }()
