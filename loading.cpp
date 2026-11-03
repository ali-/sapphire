#include <chrono>
#include <iostream>
#include <string>
#include <string_view>

namespace Sapphire::Loading {
	class Timer {
		std::string name;
		std::chrono::steady_clock::time_point start;
		bool enabled;

		public:
			explicit Timer(std::string_view name, bool enabled = true) : name(enabled ? name : std::string_view{}), enabled(enabled) {
				if (!enabled) return;
				start = std::chrono::steady_clock::now();
				std::cout << "[loading] " << this->name << "..." << std::endl;
			}

			void Stop() {
				if (!enabled) return;
				const double milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
				enabled = false;
				std::cout << "[loading] " << name << " finished in " << milliseconds << " ms" << std::endl;
			}

			~Timer() { Stop(); }
	};
}
