#pragma once

namespace rpa_sa {
    namespace ops {
        namespace {
            enum RotativePadInterpolationMethod {
                nearest, lerp
            };

            inline RotativePadInterpolationMethod
            get_interpolation_method(const std::string &interpolation) {
                if (interpolation == "lerp")
                    return RotativePadInterpolationMethod::lerp;
                return RotativePadInterpolationMethod::nearest;
            }
        }
    }
}

#define RPA_SA_ROTATIVE_PAD_INTERP_METHOD_OPTION(VAL, ...)  \
  if (VAL == rpa_sa::ops::RotativePadInterpolationMethod::lerp) {  \
    static const rpa_sa::ops::RotativePadInterpolationMethod interp_method = rpa_sa::ops::RotativePadInterpolationMethod::lerp;  \
    return __VA_ARGS__();  \
  } else {  \
    static const rpa_sa::ops::RotativePadInterpolationMethod interp_method = rpa_sa::ops::RotativePadInterpolationMethod::nearest;  \
    return __VA_ARGS__();  \
  }
