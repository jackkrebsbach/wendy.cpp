#include "logger.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <fmt/ranges.h>
#include <symengine/expression.h>
#include <xtensor/containers/xarray.hpp>


std::shared_ptr<spdlog::logger> console = spdlog::stdout_color_mt("console");
