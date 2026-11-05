// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"

namespace {
using Shader::Ampere::Instruction;
using Shader::Ampere::LowerToMaxwell;
constexpr Instruction Exit{{0x794d, 0, 0x03800000, 0}};

// TLD4.PTP.F16 gathers rejected as "native texture gather mode" in Ocarina of Time's
// (0400486024DEC000) handheld frames: per-tap offsets (bits 76-77 = 2) with half results (bit 79).
// R23/R24, R10 (x, y), R2 (offsets in R2, R3) and R0/R2, R8 (x, y), R34 (offsets in R34, R35).
constexpr std::array<Instruction, 2> LoggedHalfPerTapGathers{{
    {{0x0a177b63, 0x30003002, 0x001eaf18, 0x000fc200}},
    {{0x08007b63, 0x30003022, 0x001eaf02, 0x000fc400}},
}};

u64 Copy(u64 predicate, u32 target, u32 source) {
    return 0x5c98000000000000ULL | predicate | target | (15ULL << 39) | (u64{source} << 20);
}

u64 MoveImmediate(u64 predicate, u32 target, u32 value) {
    return 0x0100000000000000ULL | predicate | target | (15ULL << 12) | (u64{value} << 20);
}
} // namespace

TEST_CASE("Ampere half per-tap gathers lower to TLD4.PTP and pack F16 pairs",
          "[video_core][ampere][gather]") {
    for (const auto& gather : LoggedHalfPerTapGathers) {
        const u32 offsets = gather.Bits(32, 8);
        const u32 x = gather.Bits(24, 8);
        const u32 dest = gather.Bits(16, 8);
        const u32 second = gather.Bits(64, 8);
        const std::array<Instruction, 4> code{{
            {{0x00007802 | (offsets << 16), 0xff00ff00, 0x00000f00, 0x000fe400}},
            {{0x00007802 | ((offsets + 1) << 16), 0x01000001, 0x00000f00, 0x000fe400}},
            gather,
            Exit,
        }};
        const auto lowered = LowerToMaxwell(code);
        INFO(lowered.error);
        REQUIRE(lowered);
        const u64 pred = u64{7} << 16;
        const std::array<u64, 11> expected{
            MoveImmediate(pred, 248, 0xff00ff00),
            MoveImmediate(pred, 249, 0x01000001),
            Copy(pred, 250, x),
            Copy(pred, 251, x + 1),
            0xc800000000000000ULL | pred | 250 | (250ULL << 8) | (248ULL << 20) | (2ULL << 28) |
                (15ULL << 31) | (u64{gather.Bits(40, 14)} << 36) | (7ULL << 51) | (2ULL << 54),
            0x5ca8000000000000ULL | pred | 246 | (1ULL << 8) | (2ULL << 10) | (250ULL << 20),
            0x5ca8000000000000ULL | pred | 247 | (1ULL << 8) | (2ULL << 10) | (251ULL << 20),
            0x5c18000000000000ULL | pred | dest | (247ULL << 8) | (246ULL << 20) | (16ULL << 39),
            0x5ca8000000000000ULL | pred | 246 | (1ULL << 8) | (2ULL << 10) | (252ULL << 20),
            0x5ca8000000000000ULL | pred | 247 | (1ULL << 8) | (2ULL << 10) | (253ULL << 20),
            0x5c18000000000000ULL | pred | second | (247ULL << 8) | (246ULL << 20) | (16ULL << 39),
        };
        const auto begin = lowered.instructions.begin() + lowered.source_instruction_offsets[2];
        REQUIRE(std::equal(expected.begin(), expected.end(), begin));
    }
}

TEST_CASE("Ampere half per-tap gathers still need proven constant offsets",
          "[video_core][ampere][gather]") {
    for (const auto& gather : LoggedHalfPerTapGathers) {
        CHECK(LowerToMaxwell(std::array{gather, Exit}).error ==
              "native per-tap gather missing offsets");
        auto bit80{gather};
        bit80.words[2] ^= 1U << 16;
        CHECK_FALSE(LowerToMaxwell(std::array{bit80, Exit}));
    }
}
