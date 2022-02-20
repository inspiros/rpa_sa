#pragma once

namespace rpa_sa {
    namespace ops {
        namespace {
            enum SphericalPadInterpolationMethod {
                nearest, alerp, slerp
            };

            inline SphericalPadInterpolationMethod
            get_interpolation_method(const std::string &interpolation) {
                if (interpolation == "alerp")
                    return SphericalPadInterpolationMethod::alerp;
                else if (interpolation == "slerp")
                    return SphericalPadInterpolationMethod::slerp;
                return SphericalPadInterpolationMethod::nearest;
            }
        }
    }
}

#define RPA_SA_SPHERICAL_PAD_INTERP_METHOD_OPTION(VAL, ...)  \
  if (VAL == rpa_sa::ops::SphericalPadInterpolationMethod::alerp) {  \
    static const rpa_sa::ops::SphericalPadInterpolationMethod interp_method = rpa_sa::ops::SphericalPadInterpolationMethod::alerp;  \
    return __VA_ARGS__();  \
  } else if (VAL == rpa_sa::ops::SphericalPadInterpolationMethod::slerp) {  \
    static const rpa_sa::ops::SphericalPadInterpolationMethod interp_method = rpa_sa::ops::SphericalPadInterpolationMethod::slerp;  \
    return __VA_ARGS__();  \
  } else {  \
    static const rpa_sa::ops::SphericalPadInterpolationMethod interp_method = rpa_sa::ops::SphericalPadInterpolationMethod::nearest;  \
    return __VA_ARGS__();  \
  }
