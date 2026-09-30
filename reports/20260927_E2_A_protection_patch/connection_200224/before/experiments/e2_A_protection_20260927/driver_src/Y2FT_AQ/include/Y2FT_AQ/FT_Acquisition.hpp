#pragma once
// Pure packet provenance; no sockets, ROS, limits, or robot commands.
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>

struct FTAcquisition
{
    using Clock = std::chrono::steady_clock;
    std::uint64_t sequence = 0;
    bool valid = false;
    bool invalid_packet_seen = false;  // latched across later valid packets
    std::array<double, 6> raw{};  // sensor axes, before software zero/filter/gravity
    Clock::time_point received_at{};

    void received(const char* bytes, std::size_t size, Clock::time_point now)
    {
        ++sequence;
        received_at = now;
        valid = size >= 24;
        if (!valid) { invalid_packet_seen = true; return; }
        for (std::size_t i = 0; i < raw.size(); ++i) {
            const auto* p = reinterpret_cast<const unsigned char*>(bytes + 4*i);
            const std::uint32_t bits = (std::uint32_t(p[0]) << 24) |
                (std::uint32_t(p[1]) << 16) | (std::uint32_t(p[2]) << 8) | p[3];
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            raw[i] = value;
            valid = valid && std::isfinite(value);
        }
        invalid_packet_seen = invalid_packet_seen || !valid;
    }

    double age(Clock::time_point now) const
    {
        return std::chrono::duration<double>(now - received_at).count();
    }
};
