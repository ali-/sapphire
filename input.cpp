#include <array>
#include <SDL3/SDL.h>

namespace Sapphire::Input {
	static std::array<bool, SDL_SCANCODE_COUNT> pressed{};
	static std::array<bool, SDL_SCANCODE_COUNT> released{};

	void BeginFrame() {
		pressed.fill(false);
		released.fill(false);
	}

	void ProcessEvent(const SDL_Event& event) {
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) pressed[event.key.scancode] = true;
		else if (event.type == SDL_EVENT_KEY_UP) released[event.key.scancode] = true; 
	}

	static SDL_Scancode ToScancode(const char* key) {
		return key ? SDL_GetScancodeFromName(key) : SDL_SCANCODE_UNKNOWN;
	}

	bool GetKey(const char* key) {
		const auto code = ToScancode(key);
		if (code == SDL_SCANCODE_UNKNOWN) return false;
		const bool* keyboard = SDL_GetKeyboardState(nullptr);
		return keyboard[code];
	}

	bool GetKeyDown(const char* key) {
		const auto code = ToScancode(key);
		return code != SDL_SCANCODE_UNKNOWN && pressed[code];
	}

	bool GetKeyUp(const char* key) {
		const auto code = ToScancode(key);
		return code != SDL_SCANCODE_UNKNOWN && released[code];
	}
}