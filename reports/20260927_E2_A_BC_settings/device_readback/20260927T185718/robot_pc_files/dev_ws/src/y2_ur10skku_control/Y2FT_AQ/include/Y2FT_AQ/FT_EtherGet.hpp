#pragma once

#include <iostream>
#include <iomanip>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cerrno>
#include <fcntl.h>
#include "Y2Matrix/YMatrix.hpp" // YMatrix library for matrix operations

#define MAX_RETRY 5
#define RECV_SIZE 50

/* Basic ethernet communication with AFT200-D80-EN version*/
struct FTData
{
    FTData(double fx = 0.0, double fy = 0.0, double fz = 0.0,
           double mx = 0.0, double my = 0.0, double mz = 0.0,
           double lax = 0.0, double lay = 0.0, double laz = 0.0,
           double aax = 0.0, double aay = 0.0, double aaz = 0.0)
        : Fx(fx)
        , Fy(fy)
        , Fz(fz)
        , Mx(mx)
        , My(my)
        , Mz(mz)
        , LAx(lax)
        , LAy(lay)
        , LAz(laz)
        , AAx(aax)
        , AAy(aay)
        , AAz(aaz)
    {
    }

    double Fx, Fy, Fz;
    double Mx, My, Mz;
    double LAx, LAy, LAz;
    double AAx, AAy, AAz;
};


class FT_EtherGet
{
    public:
        FT_EtherGet(const std::string& IP, const int PORT);
        ~FT_EtherGet();
        FTData FTGet();
        bool FT_init(const unsigned int init_count_num);

    private:
        std::string IP_;
        unsigned int PORT_;
        std::vector<double> init_force, init_moment;
        int s; // socket file descriptor
        FTData last_ftdata_;

        bool init_flag = false;
        unsigned int init_count = 0;
        std::chrono::steady_clock::time_point start_time_;
        std::chrono::steady_clock::time_point last_wait_msg_time_;
        std::chrono::steady_clock::time_point last_stamp_;
        bool wait_5sec_passed_ = false;
        bool have_last_stamp_ = false;
        unsigned int sample_count_ = 0;
        long long sum_dt_us_ = 0;

        /* Base functions */
        float unpackFloat(const char *bytes);
        ssize_t recvMsg(char *recvData, std::size_t recvSize);
};
