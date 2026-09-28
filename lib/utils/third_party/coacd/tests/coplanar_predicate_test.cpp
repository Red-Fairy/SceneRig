#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include "include/predicates.h"
#include "intersection.h"

struct Vec {
    float x, y, z;
    float& operator[](size_t i) { return i == 0 ? x : i == 1 ? y : z; }
    const float& operator[](size_t i) const { return i == 0 ? x : i == 1 ? y : z; }
};
using Tri = std::array<Vec, 3>;
using Predicate = threeyd::moeller::TriangleIntersects<Vec>;

struct Case {
    const char* name;
    Tri a, b;
    bool expected;
    bool nondegenerate = true;
};

int failures = 0, checks = 0;
void check(bool actual, bool expected, const char* name) {
    ++checks;
    if (actual != expected) {
        ++failures;
        if (failures <= 12)
            std::fprintf(stderr, "FAIL %s: got %d expected %d\n", name, actual, expected);
    }
}

Tri transformed(const Tri& source, int axis, float scale) {
    Tri result{};
    for (size_t i = 0; i < 3; ++i)
        for (size_t k = 0; k < 3; ++k)
            result[i][(k + axis) % 3] = source[i][k] * scale;
    return result;
}

void run_case(const Case& item) {
    const int permutations[6][3] = {{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}};
    for (int axis = 0; axis < 3; ++axis) {
        Vec normal{};
        normal[(2 + axis) % 3] = 1;
        for (float scale : {0x1p-10f, 1.0f, 0x1p10f}) {
            const Tri first = transformed(item.a, axis, scale);
            const Tri second = transformed(item.b, axis, scale);
            for (const auto& p : permutations) {
                for (const auto& q : permutations) {
                    const Tri a{first[p[0]], first[p[1]], first[p[2]]};
                    const Tri b{second[q[0]], second[q[1]], second[q[2]]};
                    for (bool reverse : {false, true}) {
                        const Tri& x = reverse ? b : a;
                        const Tri& y = reverse ? a : b;
                        check(Predicate::coplanar_tri_tri(normal, normal, x[0], x[1], x[2], y[0], y[1], y[2]),
                              item.expected, item.name);
                        if (item.nondegenerate)
                            check(Predicate::triangle(x[0], x[1], x[2], y[0], y[1], y[2]), item.expected, item.name);
                    }
                }
            }
        }
    }
    std::printf("case %s complete\n", item.name);
}

int main() {
    const std::vector<Case> cases{
        {"captured_rack_false_positive",
         {{{0.5223366618156433f, 0.43321502208709717f, 0.10208412259817123f},
           {0.6939776539802551f, 0.3258453905582428f, 0.10208412259817123f},
           {0.7002800107002258f, 0.32486552000045776f, 0.10208412259817123f}}},
         {{{0.9582437872886658f, 0.32389816641807556f, 0.10208412259817123f},
           {0.33727559447288513f, 0.5458984971046448f, 0.10208412259817123f},
           {0.5152189135551453f, 0.43754899501800537f, 0.10208412259817123f}}}, false},
        {"crossing", {{{0,0,0},{4,0,0},{2,4,0}}}, {{{0,3,0},{4,3,0},{2,-1,0}}}, true},
        {"containment", {{{0,0,0},{4,0,0},{0,4,0}}}, {{{0.5f,0.5f,0},{1,0.5f,0},{0.5f,1,0}}}, true},
        {"shared_edge", {{{0,0,0},{2,0,0},{0,2,0}}}, {{{2,0,0},{0,0,0},{2,-2,0}}}, true},
        {"point_contact", {{{0,0,0},{2,0,0},{0,2,0}}}, {{{2,0,0},{3,0,0},{3,1,0}}}, true},
        {"collinear_overlap", {{{0,0,0},{3,0,0},{0,1,0}}}, {{{1,0,0},{4,0,0},{4,-1,0}}}, true},
        {"collinear_disjoint", {{{0,0,0},{1,0,0},{0,1,0}}}, {{{2,0,0},{3,0,0},{3,-1,0}}}, false},
        {"disjoint", {{{0,0,0},{1,0,0},{0,1,0}}}, {{{2,2,0},{3,2,0},{2,3,0}}}, false},
        {"parallel_small_gap", {{{0,0,0},{4,0,0},{0,1,0}}}, {{{0,-0x1p-22f,0},{4,-0x1p-22f,0},{4,-1,0}}}, false},
        {"identical", {{{0,0,0},{1,0,0},{0,1,0}}}, {{{0,0,0},{1,0,0},{0,1,0}}}, true},
        {"degenerate_crossing_segments", {{{0,0,0},{2,0,0},{1,0,0}}}, {{{1,-1,0},{1,1,0},{1,0,0}}}, true, false},
        {"degenerate_overlapping_segments", {{{0,0,0},{2,0,0},{1,0,0}}}, {{{1,0,0},{3,0,0},{2,0,0}}}, true, false},
        {"degenerate_disjoint_segments", {{{0,0,0},{1,0,0},{0.5f,0,0}}}, {{{2,0,0},{3,0,0},{2.5f,0,0}}}, false, false},
        {"point_inside", {{{0.25f,0.25f,0},{0.25f,0.25f,0},{0.25f,0.25f,0}}}, {{{0,0,0},{2,0,0},{0,2,0}}}, true, false},
        {"point_outside", {{{3,3,0},{3,3,0},{3,3,0}}}, {{{0,0,0},{2,0,0},{0,2,0}}}, false, false},
        {"identical_points", {{{1,1,0},{1,1,0},{1,1,0}}}, {{{1,1,0},{1,1,0},{1,1,0}}}, true, false},
        {"distinct_points", {{{1,1,0},{1,1,0},{1,1,0}}}, {{{2,1,0},{2,1,0},{2,1,0}}}, false, false},
    };
    for (const auto& item : cases) run_case(item);

    // The normal entry point retains ordinary noncoplanar intersections/rejections.
    const Tri base{{{0,0,0},{2,0,0},{0,2,0}}};
    check(Predicate::triangle(base[0], base[1], base[2], Vec{0.5f,0.5f,-1}, Vec{0.5f,0.5f,1}, Vec{1,0.5f,0}), true, "noncoplanar_crossing");
    check(Predicate::triangle(base[0], base[1], base[2], Vec{3,0,-1}, Vec{3,0,1}, Vec{3,1,0}), false, "noncoplanar_disjoint");

    // Positive evidence for the adaptive path: ordinary double determinants lose
    // this unit signed area. CDT's expansion fallback retains the exact sign.
    const double n = 0x1p27;
    volatile double left = n * n, right = (n - 1) * (n + 1);
    check(left - right == 0, true, "double_cancellation_fixture");
    check(predicates::adaptive::orient2d<double>(0,0,n,n-1,n+1,n) > 0, true, "adaptive_orientation_positive");
    check(predicates::adaptive::orient2d<double>(0,0,n+1,n,n,n-1) < 0, true, "adaptive_orientation_negative");

    // Exercise the production projected predicate itself, not only CDT: a point
    // lies just outside this thin triangle. Plain double orientation wrongly
    // rounds its signed distance to zero and accepts a false edge contact.
    struct Vec64 {
        double x, y, z;
        double& operator[](size_t i) { return i == 0 ? x : i == 1 ? y : z; }
        const double& operator[](size_t i) const { return i == 0 ? x : i == 1 ? y : z; }
    };
    using Precise = threeyd::moeller::TriangleIntersects<Vec64>;
    const Vec64 normal{0,0,1}, a{0,0,0}, b{n,n-1,0}, c{n+1,n,0}, outside{n-1,n-2,0};
    check(Precise::coplanar_tri_tri(normal,normal,a,b,c,outside,outside,outside), false, "adaptive_false_endpoint_rejected");
    check(Precise::coplanar_tri_tri(normal,normal,a,b,c,b,b,b), true, "adaptive_real_endpoint_kept");

    std::printf("fixture_groups=%zu assertions=%d failures=%d\n", cases.size() + 7, checks, failures);
    return failures ? 1 : 0;
}
