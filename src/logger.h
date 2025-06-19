#ifndef LOGGER_H
#define LOGGER_H
#pragma once
#include <memory>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <fmt/ranges.h>

extern std::shared_ptr<spdlog::logger> console;


#endif //LOGGER_H
