#include <iostream>
#include <chrono>
#include <unordered_map>
#include <string>

class ScopedProfiler {
public:
    ScopedProfiler(const std::string& section) : section_name(section), start_time(std::chrono::high_resolution_clock::now()) {}

    ~ScopedProfiler() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        total_times[section_name] += elapsed.count();
        std::cout << section_name << " measured " << elapsed.count() << "s\n";
    }

    static void report() {
        for (const auto& [section, time] : total_times) {
            std::cout << section << " total time: " << time << " seconds.\n";
        }
    }

private:
    std::string section_name;
    std::chrono::high_resolution_clock::time_point start_time;
    static inline std::unordered_map<std::string, double> total_times;
};
