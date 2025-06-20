#include "logger.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "symbolic_utils.h"
#include <fmt/ranges.h>

std::shared_ptr<spdlog::logger> logger = spdlog::stdout_color_mt("log");