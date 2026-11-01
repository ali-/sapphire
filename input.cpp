#include <array>
#include <cmath>
#include <cstring>
#include <SDL3/SDL.h>
#include <SDL3/SDL_gamepad.h>

namespace Sapphire::Input {
	static std::array<bool, SDL_SCANCODE_COUNT> pressed{};
	static std::array<bool, SDL_SCANCODE_COUNT> released{};
	static SDL_Gamepad* gamepad = nullptr;
	static float mouseDeltaX = 0.0f;
	static float mouseDeltaY = 0.0f;

	static void OpenFirstGamepad() {
		int count = 0;
		SDL_JoystickID* gamepads = SDL_GetGamepads(&count);
		for (int index = 0; index < count && !gamepad; ++index) gamepad = SDL_OpenGamepad(gamepads[index]);
		SDL_free(gamepads);
	}

	void Initialize() {
		OpenFirstGamepad();
	}

	void Shutdown() {
		if (gamepad) SDL_CloseGamepad(gamepad);
		gamepad = nullptr;
	}

	void BeginFrame() {
		pressed.fill(false);
		released.fill(false);
		mouseDeltaX = 0.0f;
		mouseDeltaY = 0.0f;
	}

	void ProcessEvent(const SDL_Event& event) {
		if (event.type == SDL_EVENT_KEY_DOWN && !event.key.repeat) pressed[event.key.scancode] = true;
		else if (event.type == SDL_EVENT_KEY_UP) released[event.key.scancode] = true;
		else if (event.type == SDL_EVENT_MOUSE_MOTION) {
			mouseDeltaX += event.motion.xrel;
			mouseDeltaY += event.motion.yrel;
		}
		else if (event.type == SDL_EVENT_GAMEPAD_ADDED && !gamepad) gamepad = SDL_OpenGamepad(event.gdevice.which);
		else if (event.type == SDL_EVENT_GAMEPAD_REMOVED && gamepad && SDL_GetGamepadID(gamepad) == event.gdevice.which) {
			SDL_CloseGamepad(gamepad);
			gamepad = nullptr;
			OpenFirstGamepad();
		}
		else if (event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN && gamepad && SDL_GetGamepadID(gamepad) == event.gbutton.which && event.gbutton.button == SDL_GAMEPAD_BUTTON_START) pressed[SDL_SCANCODE_ESCAPE] = true;
		else if (event.type == SDL_EVENT_GAMEPAD_BUTTON_UP && gamepad && SDL_GetGamepadID(gamepad) == event.gbutton.which && event.gbutton.button == SDL_GAMEPAD_BUTTON_START) released[SDL_SCANCODE_ESCAPE] = true;
		else if (event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN && gamepad && SDL_GetGamepadID(gamepad) == event.gbutton.which && event.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH) pressed[SDL_SCANCODE_SPACE] = true;
		else if (event.type == SDL_EVENT_GAMEPAD_BUTTON_UP && gamepad && SDL_GetGamepadID(gamepad) == event.gbutton.which && event.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH) released[SDL_SCANCODE_SPACE] = true;
	}

	static SDL_Scancode ToScancode(const char* key) {
		if (!key) return SDL_SCANCODE_UNKNOWN;
		if (std::strcmp(key, "Cross") == 0) return SDL_SCANCODE_SPACE;
		return SDL_GetScancodeFromName(key);
	}

	bool GetKey(const char* key) {
		const auto code = ToScancode(key);
		if (code == SDL_SCANCODE_UNKNOWN) return false;
		const bool* keyboard = SDL_GetKeyboardState(nullptr);
		return keyboard[code] || (code == SDL_SCANCODE_ESCAPE && gamepad && SDL_GetGamepadButton(gamepad, SDL_GAMEPAD_BUTTON_START)) || (code == SDL_SCANCODE_SPACE && gamepad && SDL_GetGamepadButton(gamepad, SDL_GAMEPAD_BUTTON_SOUTH));
	}

	bool GetKeyDown(const char* key) {
		const auto code = ToScancode(key);
		return code != SDL_SCANCODE_UNKNOWN && pressed[code];
	}

	bool GetKeyUp(const char* key) {
		const auto code = ToScancode(key);
		return code != SDL_SCANCODE_UNKNOWN && released[code];
	}

	float GetAxis(const char* axis) {
		if (!axis) return 0.0f;
		if (std::strcmp(axis, "Mouse X") == 0) return mouseDeltaX;
		if (std::strcmp(axis, "Mouse Y") == 0) return -mouseDeltaY;
		if (!gamepad) return 0.0f;
		SDL_GamepadAxis gamepadAxis;
		float direction = 1.0f;
		if (std::strcmp(axis, "Horizontal") == 0) gamepadAxis = SDL_GAMEPAD_AXIS_LEFTX;
		else if (std::strcmp(axis, "Vertical") == 0) {
			gamepadAxis = SDL_GAMEPAD_AXIS_LEFTY;
			direction = -1.0f;
		}
		else if (std::strcmp(axis, "LookHorizontal") == 0) gamepadAxis = SDL_GAMEPAD_AXIS_RIGHTX;
		else if (std::strcmp(axis, "LookVertical") == 0) {
			gamepadAxis = SDL_GAMEPAD_AXIS_RIGHTY;
			direction = -1.0f;
		}
		else return 0.0f;
		const Sint16 rawValue = SDL_GetGamepadAxis(gamepad, gamepadAxis);
		const float value = direction * (rawValue < 0 ? static_cast<float>(rawValue) / 32768.0f : static_cast<float>(rawValue) / 32767.0f);
		return std::abs(value) < 0.25f ? 0.0f : value;
	}
}
