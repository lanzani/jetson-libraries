/*
* Copyright 2017-2023 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

//---------------------------------------------------------------------------
//! \file AppDecPerf.cpp
//! \brief Source file for AppDecPerf sample
//!
//!  This sample application measures decoding performance in FPS.
//!  The application creates multiple host threads and runs a different decoding session on each thread.
//!  The number of threads can be controlled by the CLI option "-thread".
//!  The application creates 2 host threads, each with a separate decode session, by default.
//!  The application supports measuring the decode performance only (keeping decoded
//!  frames in device memory as well as measuring the decode performance including transfer
//!  of frames to the host memory.
//---------------------------------------------------------------------------

#include <cuda.h>
#include <cudaProfiler.h>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <string.h>
#include <memory>
#include "NvDecoder/NvDecoder.h"
#include "../Utils/NvCodecUtils.h"
#include "../Utils/FFmpegDemuxer.h"
#include <chrono>

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

/**
*   @brief  Function to decode media file using NvDecoder interface
*   @param  pDec    - Handle to NvDecoder
*   @param  demuxer - Pointer to an FFmpegDemuxer instance
*   @param  pnFrame - Variable to record the number of frames decoded
*   @param  ex      - Stores current exception in case of failure
*/
void DecProc(NvDecoder *pDec, FFmpegDemuxer *demuxer, int *pnFrame, std::exception_ptr &ex)
{
    try
    {
        int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
        uint8_t *pVideo = NULL, *pFrame = NULL;

        do {
            demuxer->Demux(&pVideo, &nVideoBytes);
            nFrameReturned = pDec->Decode(pVideo, nVideoBytes);
            if (!nFrame && nFrameReturned)
                LOG(INFO) << pDec->GetVideoInfo();

            nFrame += nFrameReturned;
        } while (nVideoBytes);
        *pnFrame = nFrame;
    }
    catch (std::exception&)
    {
        ex = std::current_exception();
    }
}

void ShowHelpAndExit(const char *szBadOption = NULL)
{
    std::ostringstream oss;
    bool bThrowError = false;
    if (szBadOption)
    {
        bThrowError = true;
        oss << "Error parsing \"" << szBadOption << "\"" << std::endl;
    }
    oss << "Options:" << std::endl
        << "-i           Input file path" << std::endl
        << "-gpu         Ordinal of GPU to use" << std::endl
        << "-thread      Number of decoding thread" << std::endl
        << "-single      (No value) Use single context (this may result in suboptimal performance; default is multiple contexts)" << std::endl
        << "-host        (No value) Copy frame to host memory (this may result in suboptimal performance; default is device memory)" << std::endl
        ;
    if (bThrowError)
    {
        throw std::invalid_argument(oss.str());
    }
    else
    {
        std::cout << oss.str();
        exit(0);
    }
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, int &iGpu, int &nThread, bool &bSingle, bool &bHost) 
{
    for (int i = 1; i < argc; i++) {
        if (!_stricmp(argv[i], "-h")) {
            ShowHelpAndExit();
        }
        if (!_stricmp(argv[i], "-i")) {
            if (++i == argc) {
                ShowHelpAndExit("-i");
            }
            sprintf(szInputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-gpu")) {
            if (++i == argc) {
                ShowHelpAndExit("-gpu");
            }
            iGpu = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-thread")) {
            if (++i == argc) {
                ShowHelpAndExit("-thread");
            }
            nThread = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-single")) {
            bSingle = true;
            continue;
        }
        if (!_stricmp(argv[i], "-host")) {
            bHost = true;
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }
}

struct NvDecPerfData
{
    uint8_t *pBuf;
    std::vector<uint8_t *> *pvpPacketData; 
    std::vector<int> *pvpPacketDataSize;
};

int CUDAAPI HandleVideoData(void *pUserData, CUVIDSOURCEDATAPACKET *pPacket) {
    NvDecPerfData *p = (NvDecPerfData *)pUserData;
    memcpy(p->pBuf, pPacket->payload, pPacket->payload_size);
    p->pvpPacketData->push_back(p->pBuf);
    p->pvpPacketDataSize->push_back(pPacket->payload_size);
    p->pBuf += pPacket->payload_size;
    return 1;
}

int main(int argc, char **argv)
{
    char szInFilePath[256] = "";
    int iGpu = 0;
    int nThread = 2; 
    bool bSingle = false;
    bool bHost = false;
    std::vector<std::exception_ptr> vExceptionPtrs;
    try {
        ParseCommandLine(argc, argv, szInFilePath, iGpu, nThread, bSingle, bHost);
        CheckInputFile(szInFilePath);

        struct stat st;
        if (stat(szInFilePath, &st) != 0) {
            return 1;
        }
        int nBufSize = st.st_size;

        uint8_t *pBuf = NULL;
        try {
            pBuf = new uint8_t[nBufSize];
        }
        catch (std::bad_alloc) {
            std::cout << "Failed to allocate memory in BufferedReader" << std::endl;
            return 1;
        }
        std::vector<uint8_t *> vpPacketData;
        std::vector<int> vnPacketData;

        NvDecPerfData userData = { pBuf, &vpPacketData, &vnPacketData };

        ck(cuInit(0));
        int nGpu = 0;
        ck(cuDeviceGetCount(&nGpu));
        if (iGpu < 0 || iGpu >= nGpu) {
            std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
            return 1;
        }
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, iGpu));
        char szDeviceName[80];
        ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        std::cout << "GPU in use: " << szDeviceName << std::endl;

        std::vector<std::unique_ptr<FFmpegDemuxer>> vDemuxer;
        std::vector<std::unique_ptr<NvDecoder>> vDec;
        CUcontext cuContext = NULL;
        ck(cuCtxCreate(&cuContext, 0, cuDevice));
        vExceptionPtrs.resize(nThread);
        for (int i = 0; i < nThread; i++)
        {
            if (!bSingle)
            {
                ck(cuCtxCreate(&cuContext, 0, cuDevice));
            }
            std::unique_ptr<FFmpegDemuxer> demuxer(new FFmpegDemuxer(szInFilePath));

            NvDecoder* sessionObject = new NvDecoder(cuContext, !bHost, FFmpeg2NvCodecId(demuxer->GetVideoCodec()));
            sessionObject->setDecoderSessionID(i);
            std::unique_ptr<NvDecoder> dec(sessionObject);
            vDemuxer.push_back(std::move(demuxer));
            vDec.push_back(std::move(dec));
        }

        std::vector<std::chrono::high_resolution_clock::time_point> vnDecSessionStartTime;
        std::vector<int64_t> vnDecSessionDuration;
        float totalFPS = 0;
        std::vector<NvThread> vThread;
        std::vector<int> vnFrame;
        vnFrame.resize(nThread, 0);

        for (int i = 0; i < nThread; i++)
        {
            vThread.push_back(NvThread(std::thread(DecProc, vDec[i].get(), vDemuxer[i].get(), &vnFrame[i], std::ref(vExceptionPtrs[i]))));
            vnDecSessionStartTime.push_back(std::chrono::high_resolution_clock::now());
        }
        for (int i = 0; i < nThread; i++)
        {
            vThread[i].join();
        }

        int nTotal = 0;
        for (int i = 0; i < nThread; i++)
        {
            nTotal += vnFrame[i];
            vDec[i].reset(nullptr);

            int64_t elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - vnDecSessionStartTime[i]).count();
            vnDecSessionDuration.push_back(elapsedTime);
        }

        for (int i = 0; i < nThread; i++)
        {
            float decodeDuration = (vnDecSessionDuration[i] - NvDecoder::getDecoderSessionOverHead(i)) / 1000.0f;

            // System FPS is sum of individual session's FPS
            totalFPS += vnFrame[i] / decodeDuration;
        }

        std::cout << "Total Frames Decoded=" << nTotal << " FPS = " << totalFPS << std::endl;

        ck(cuProfilerStop());

        for (int i = 0; i < nThread; i++)
        {
            if (vExceptionPtrs[i])
            {
                std::rethrow_exception(vExceptionPtrs[i]);
            }
        }
    }
    catch (const std::exception& ex)
    {
        std::cout << ex.what();
        exit(1);
    }
    return 0;
}
