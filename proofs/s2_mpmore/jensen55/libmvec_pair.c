/* Test-only ABI adapter for the exact glibc intrinsic in original aggregation.
 * Original gfortran -O2 object calls _ZGVbN2v_log on habits 3 and 4 together.
 */
#include <stddef.h>
typedef double v2df __attribute__((vector_size(16)));
extern v2df _ZGVbN2v_log(v2df);
void original_pair_log(size_t n, const double *left, const double *right,
                       double *out_left, double *out_right) {
    for (size_t i = 0; i < n; ++i) {
        v2df value = {left[i], right[i]};
        value = _ZGVbN2v_log(value);
        out_left[i] = value[0];
        out_right[i] = value[1];
    }
}
