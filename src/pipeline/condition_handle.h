#ifndef __SD_PIPELINE_CONDITION_HANDLE_H__
#define __SD_PIPELINE_CONDITION_HANDLE_H__

#include "generation.h"
#include "stable-diffusion.h"

struct sd_condition_t {
    sd_ctx_t* ctx = nullptr;
    sd::pipeline::ImageGenerationEmbeds embeds;
    // Dimensions used when this condition was encoded. Only meaningful for
    // model families whose conditioning embeds encode size (SDXL/ADM-vector).
    int width  = 0;
    int height = 0;
    // True when the conditioner's embeddings depend on width/height, so reuse
    // must check dimensions match (see detailer.cpp reuse rule).
    bool size_dependent = false;
};

#endif  // __SD_PIPELINE_CONDITION_HANDLE_H__
