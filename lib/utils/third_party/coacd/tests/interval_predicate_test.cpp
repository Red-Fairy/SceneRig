// Captured IMG/Bridge pairs and thin-plane interval regressions.
// Every literal is an exact binary float; input order must remain reproducible.
#include <array>
#include <cstdio>

#include "intersection.h"

struct Vec {
    float x, y, z;

    float& operator[](size_t i) {
        return i == 0 ? x : i == 1 ? y : z;
    }

    const float& operator[](size_t i) const {
        return i == 0 ? x : i == 1 ? y : z;
    }
};

using Predicate = threeyd::moeller::TriangleIntersects<Vec>;

struct Case {
    const char* name;
    std::array<Vec, 6> points;
    bool expected;
};

int main() {
    const Case cases[] = {
        {"captured_img_disjoint",
         {{
             {0x1.9b23c60000000p-2f, 0x1.ca75280000000p-2f, -0x1.cd1bdc0000000p-5f},
             {0x1.758d9e0000000p-2f, 0x1.b369cc0000000p-2f, 0x1.53c3880000000p-6f},
             {0x1.7cde3e0000000p-2f, 0x1.b7e5e80000000p-2f, 0x1.8ca1060000000p-6f},
             {0x1.8f6a740000000p-2f, 0x1.c345080000000p-2f, -0x1.17175c0000000p-4f},
             {0x1.8f6a5a0000000p-2f, 0x1.c3451c0000000p-2f, -0x1.17175c0000000p-4f},
             {0x1.8f6a5a0000000p-2f, 0x1.c3451c0000000p-2f, -0x1.a8b4400000000p-4f},
         }},
         false},
        {"captured_bridge_disjoint",
         {{
             {0x1.324d900000000p-2f, 0x1.fe227e0000000p-2f, 0x1.88c8160000000p-3f},
             {0x1.324d900000000p-2f, 0x1.fe22800000000p-2f, 0x1.88c8160000000p-3f},
             {0x1.32e1120000000p-2f, 0x1.0292c00000000p-1f, 0x1.88c7640000000p-3f},
             {0x1.3b30300000000p-2f, 0x1.fe22800000000p-2f, 0x1.aa07300000000p-3f},
             {0x1.352ace0000000p-2f, 0x1.11f89a0000000p-1f, 0x1.87dbd00000000p-3f},
             {0x1.34137a0000000p-2f, 0x1.0a64620000000p-1f, 0x1.88715a0000000p-3f},
         }},
         false},
        {"thin_true_crossing",
         {{
             {-0x1.0000000000000p+0f, -0x1.0000000000000p+0f, 0x0.0p+0f},
             {0x1.0000000000000p+0f, -0x1.0000000000000p+0f, 0x0.0p+0f},
             {0x0.0p+0f, 0x1.0000000000000p+0f, 0x0.0p+0f},
             {0x0.0p+0f, -0x1.0000000000000p-1f, -0x1.0c6f7a0b5ed8dp-21f},
             {0x0.0p+0f, 0x1.0000000000000p-1f, 0x1.0c6f7a0b5ed8dp-21f},
             {0x0.0p+0f, -0x1.0000000000000p-1f, 0x1.0c6f7a0b5ed8dp-21f},
         }},
         true},
        {"thin_same_side",
         {{
             {-0x1.0000000000000p+0f, -0x1.0000000000000p+0f, 0x0.0p+0f},
             {0x1.0000000000000p+0f, -0x1.0000000000000p+0f, 0x0.0p+0f},
             {0x0.0p+0f, 0x1.0000000000000p+0f, 0x0.0p+0f},
             {0x0.0p+0f, -0x1.0000000000000p-1f, 0x1.0c6f7a0b5ed8dp-22f},
             {0x0.0p+0f, 0x1.0000000000000p-1f, 0x1.0c6f7a0b5ed8dp-21f},
             {0x0.0p+0f, -0x1.0000000000000p-1f, 0x1.0c6f7a0b5ed8dp-21f},
         }},
         false},
        {"ordinary_true_crossing",
         {{
             {-0x1.0000000000000p+0f, -0x1.0000000000000p+0f, 0x0.0p+0f},
             {0x1.0000000000000p+0f, -0x1.0000000000000p+0f, 0x0.0p+0f},
             {0x0.0p+0f, 0x1.0000000000000p+0f, 0x0.0p+0f},
             {0x0.0p+0f, -0x1.0000000000000p-1f, -0x1.0000000000000p-1f},
             {0x0.0p+0f, 0x1.0000000000000p-1f, 0x1.0000000000000p-1f},
             {0x0.0p+0f, -0x1.0000000000000p-1f, 0x1.0000000000000p-1f},
         }},
         true},
        {"ordinary_separated",
         {{
             {-0x1.0000000000000p+0f, -0x1.0000000000000p+0f, 0x0.0p+0f},
             {0x1.0000000000000p+0f, -0x1.0000000000000p+0f, 0x0.0p+0f},
             {0x0.0p+0f, 0x1.0000000000000p+0f, 0x0.0p+0f},
             {0x1.0000000000000p+1f, -0x1.0000000000000p-1f, -0x1.0000000000000p-1f},
             {0x1.0000000000000p+1f, 0x1.0000000000000p-1f, 0x1.0000000000000p-1f},
             {0x1.0000000000000p+1f, -0x1.0000000000000p-1f, 0x1.0000000000000p-1f},
         }},
         false},
    };
    const int permutations[6][3] = {
        {0, 1, 2}, {0, 2, 1}, {1, 0, 2},
        {1, 2, 0}, {2, 0, 1}, {2, 1, 0},
    };
    int checks = 0;
    int failures = 0;

    for (const auto& item : cases) {
        for (int axis = 0; axis < 3; ++axis) {
            std::array<Vec, 6> points{};
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 3; ++j) {
                    points[i][(j + axis) % 3] = item.points[i][j];
                }
            }
            for (const auto& a : permutations) {
                for (const auto& b : permutations) {
                    const Vec& a0 = points[a[0]];
                    const Vec& a1 = points[a[1]];
                    const Vec& a2 = points[a[2]];
                    const Vec& b0 = points[3 + b[0]];
                    const Vec& b1 = points[3 + b[1]];
                    const Vec& b2 = points[3 + b[2]];
                    for (bool reverse : {false, true}) {
                        bool actual = reverse
                            ? Predicate::triangle(b0, b1, b2, a0, a1, a2)
                            : Predicate::triangle(a0, a1, a2, b0, b1, b2);
                        ++checks;
                        if (actual != item.expected) {
                            ++failures;
                            if (failures <= 12) {
                                std::fprintf(stderr,
                                    "FAIL %s axis=%d order=%d%d%d/%d%d%d "
                                    "reverse=%d actual=%d\n",
                                    item.name, axis, a[0], a[1], a[2],
                                    b[0], b[1], b[2], reverse, actual);
                            }
                        }
                    }
                }
            }
        }
        std::printf("case %s complete\n", item.name);
    }
    std::printf("fixture_groups=%zu assertions=%d failures=%d\n",
                std::size(cases), checks, failures);
    return failures ? 1 : 0;
}
