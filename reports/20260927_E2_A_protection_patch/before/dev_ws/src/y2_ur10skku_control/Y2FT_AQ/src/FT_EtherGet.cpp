#include "Y2FT_AQ/FT_EtherGet.hpp"


FT_EtherGet::FT_EtherGet(const std::string& IP, const int PORT):
IP_(IP), PORT_(PORT), init_force(3,0.0), init_moment(3,0.0)
{
    start_time_ = std::chrono::steady_clock::now();
    last_wait_msg_time_ = start_time_;

    // 1️⃣ 소켓 생성
    s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s < 0) {
        std::cerr << "Failed to create socket" << std::endl;
        std::exit(1);
    }

    const int flags = fcntl(s, F_GETFL, 0);
    if (flags != -1 && fcntl(s, F_SETFL, flags | O_NONBLOCK) == -1) {
        perror("[ERROR] Failed to set O_NONBLOCK");
    }

    // 2️⃣ 주소 설정
    struct sockaddr_in sensorAddr;
    sensorAddr.sin_family = AF_INET;
    sensorAddr.sin_port = htons(PORT_);
    inet_pton(AF_INET, IP_.c_str(), &sensorAddr.sin_addr);

    // 3️⃣ 짧은 타임아웃 설정
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 5000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    // 4️⃣ 데이터 전송
    std::string sendData = "000302";
    std::vector<unsigned char> sendDataBytes(sendData.size() / 2);
    for (size_t i = 0; i < sendData.size(); i += 2) {
        sendDataBytes[i / 2] = std::stoi(sendData.substr(i, 2), nullptr, 16);
    }

    int sentBytes = sendto(s, sendDataBytes.data(), sendDataBytes.size(), 0,
                           (struct sockaddr *)&sensorAddr, sizeof(sensorAddr));
    if (sentBytes < 0) {
        perror("[ERROR] Send failed");
        close(s);
        std::exit(1);
    }

    std::cout << "[INFO] Sent " << sentBytes << " bytes to sensor.\n";
}

FT_EtherGet::~FT_EtherGet()
{
    close(s);
    std::cout << "[INFO] Socket closed.\n";
}

/* Data initialization */
bool FT_EtherGet::FT_init(const unsigned int init_count_num)
{
    auto now = std::chrono::steady_clock::now();
    const double elapsed_sec =
        std::chrono::duration<double>(now - start_time_).count();

    if (!wait_5sec_passed_) {
        const double dt_msg =
            std::chrono::duration<double>(now - last_wait_msg_time_).count();

        if (dt_msg > 1.0) {
            std::printf("\033[32m[FT_EtherGet] Waiting 5 sec before initialization... (%.2f sec) \033[0m\n",
                        elapsed_sec);
            last_wait_msg_time_ = now;
        }

        if (elapsed_sec < 5.0) {
            return false;
        }

        wait_5sec_passed_ = true;
        std::printf("\033[33m[FT_EtherGet] 5 sec passed. Starting initialization.\033[0m\n");
        init_flag = false;
        init_count = 0;
        std::fill(init_force.begin(), init_force.end(), 0.0);
        std::fill(init_moment.begin(), init_moment.end(), 0.0);
    }

    if (init_flag) {
        return true;
    }

    if (init_count < init_count_num) {
        const FTData ftdata = FTGet();
        init_force[0] += ftdata.Fx / static_cast<double>(init_count_num);
        init_force[1] += ftdata.Fy / static_cast<double>(init_count_num);
        init_force[2] += ftdata.Fz / static_cast<double>(init_count_num);

        init_moment[0] += ftdata.Mx / static_cast<double>(init_count_num);
        init_moment[1] += ftdata.My / static_cast<double>(init_count_num);
        init_moment[2] += ftdata.Mz / static_cast<double>(init_count_num);
        ++init_count;
        return false;
    }

    std::printf("\033[31m[FT_EtherGet] Initialization complete.\033[0m\n");
    std::cout << "  Force offsets : "
              << init_force[0] << ", "
              << init_force[1] << ", "
              << init_force[2] << "\n"
              << "  Moment offsets: "
              << init_moment[0] << ", "
              << init_moment[1] << ", "
              << init_moment[2] << std::endl;

    init_flag = true;
    return true;

}

/* Data Aquisition */
FTData FT_EtherGet::FTGet()
{
    char recvData[RECV_SIZE] = {0};
    const ssize_t bytesReceived = recvMsg(recvData, sizeof(recvData));
    if (bytesReceived < 24) {
        return last_ftdata_;
    }

    const auto now = std::chrono::steady_clock::now();
    constexpr unsigned int expression_count = 3000;
    if (have_last_stamp_) {
        const auto dt_us = std::chrono::duration_cast<std::chrono::microseconds>(
                               now - last_stamp_).count();
        sum_dt_us_ += dt_us;
        ++sample_count_;

        if (sample_count_ == expression_count) {
            const double mean_dt = sum_dt_us_ / static_cast<double>(expression_count);
            const double mean_freq = 1e6 / mean_dt;
            std::printf("[FT_EtherGet] mean dt = %.2f us, mean freq ≒ %.3f Hz (sample_count: %u)\n",
                        mean_dt, mean_freq, expression_count);
            sample_count_ = 0;
            sum_dt_us_ = 0;
        }
    } else {
        have_last_stamp_ = true;
    }
    last_stamp_ = now;

    /* Get the raw data */
    last_ftdata_.Fx = unpackFloat(recvData) - static_cast<double>(init_flag)*init_force[0];
    last_ftdata_.Fy = unpackFloat(recvData+4) - static_cast<double>(init_flag)*init_force[1];
    last_ftdata_.Fz = unpackFloat(recvData+8) - static_cast<double>(init_flag)*init_force[2];
    last_ftdata_.Mx = unpackFloat(recvData+12) - static_cast<double>(init_flag)*init_moment[0];
    last_ftdata_.My = unpackFloat(recvData+16) - static_cast<double>(init_flag)*init_moment[1];
    last_ftdata_.Mz = unpackFloat(recvData+20) - static_cast<double>(init_flag)*init_moment[2];

    if (bytesReceived >= 48) {
        last_ftdata_.LAx = unpackFloat(recvData + 24);
        last_ftdata_.LAy = unpackFloat(recvData + 28);
        last_ftdata_.LAz = unpackFloat(recvData + 32);
        last_ftdata_.AAx = unpackFloat(recvData + 36);
        last_ftdata_.AAy = unpackFloat(recvData + 40);
        last_ftdata_.AAz = unpackFloat(recvData + 44);
    }

    return last_ftdata_;
}



/*** Base functions ***/

// Function to unpack a float from a byte array
float FT_EtherGet::unpackFloat(const char *bytes) {
    uint32_t asInt = 0;
    std::memcpy(&asInt, bytes, sizeof(asInt));
    asInt = ntohl(asInt); // Convert from network byte order to host byte order
    float result;
    std::memcpy(&result, &asInt, sizeof(result));
    return result;
}

// Function to receive message with retry logic
ssize_t FT_EtherGet::recvMsg(char *recvData, std::size_t recvSize) {
    struct sockaddr_in from;
    socklen_t fromLen = sizeof(from);

    int attempt = 0;
    while (attempt < MAX_RETRY) {
        ssize_t bytesReceived = recvfrom(s, recvData, recvSize, 0,
                                         (struct sockaddr *)&from, &fromLen);

        if (bytesReceived >= 24) {
            return bytesReceived;
        } else if (bytesReceived < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                return -1;
            }
            perror("[ERROR] Failed to receive data");
        } else {
            std::cout << "[WARNING] Received " << bytesReceived 
                      << " bytes, expected at least 24. Retrying...\n";
        }

        attempt++;
        usleep(10000); // 10 ms 대기
    }

    std::cerr << "[ERROR] Failed to receive proper data after " 
              << MAX_RETRY << " attempts.\n";
    return -1;
}
