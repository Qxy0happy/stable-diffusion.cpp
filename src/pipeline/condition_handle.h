#ifndef __SD_PIPELINE_CONDITION_HANDLE_H__
#define __SD_PIPELINE_CONDITION_HANDLE_H__

#include "generation.h"
#include "stable-diffusion.h"

struct sd_condition_t {
    sd_ctx_t* ctx = nullptr;
    sd::pipeline::ImageGenerationEmbeds embeds;
};

#endif  // __SD_PIPELINE_CONDITION_HANDLE_H__