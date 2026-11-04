#include <array>
#include <cmath>
#include <cstring>
#include <SDL3/SDL.h>
#include <SDL3/SDL_gamepad.h>

namespace Sapphire::Input {
	static std::array<bool, SDL_SCANCODE_COUNT> pressed{};
	static std::array<bool, SDL_SCANCODE_COUNT> released{};
	static SDL_Gamepad* gamepad = nullptr;
	static bool previousButtons[2]{};
	static float mouseDeltaX = 0.0f;
	static float mouseDeltaY = 0.0f;

	static void OpenFirstGamepad() {
		int count = 0;
		SDL_JoystickID* gamepads = SDL_GetGamepads(&count);
		for (int index = 0; index < count && !gamepad; ++index) gamepad = SDL_OpenGamepad(gamepads[index]);
		SDL_free(gamepads);
	}

	void Initialize() {
		previousButtons[0] = previousButtons[1] = false;
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

	void EndEvents() {
		const SDL_Scancode keys[2] = {SDL_SCANCODE_SPACE, SDL_SCANCODE_ESCAPE};
		const SDL_GamepadButton buttons[2] = {SDL_GAMEPAD_BUTTON_SOUTH, SDL_GAMEPAD_BUTTON_START};
		const bool* keyboard = SDL_GetKeyboardState(nullptr);
		for (int index = 0; index < 2; ++index) {
			const bool held = keyboard[keys[index]] || (gamepad && SDL_GetGamepadButton(gamepad, buttons[index]));
			pressed[keys[index]] = !previousButtons[index] && (pressed[keys[index]] || held);
			released[keys[index]] = !held && (released[keys[index]] || previousButtons[index]);
			previousButtons[index] = held;
		}
	}

	static SDL_Scancode ToScancode(const char* key) {
		if (!key) return SDL_SCANCODE_UNKNOWN;
		if (std::strcmp(key, "Cross") == 0) return SDL_SCANCODE_SPACE;
		return SDL_GetScancodeFromName(key);
	}


	bool GetKey(SDL_Scancode code) {
		if (code == SDL_SCANCODE_UNKNOWN) return false;
		return SDL_GetKeyboardState(nullptr)[code] || (code == SDL_SCANCODE_ESCAPE && gamepad && SDL_GetGamepadButton(gamepad, SDL_GAMEPAD_BUTTON_START)) || (code == SDL_SCANCODE_SPACE && gamepad && SDL_GetGamepadButton(gamepad, SDL_GAMEPAD_BUTTON_SOUTH));
	}

	bool GetKey(const char* key) { return GetKey(ToScancode(key)); }

	bool GetKeyDown(SDL_Scancode code) { return code != SDL_SCANCODE_UNKNOWN && pressed[code]; }

	bool GetKeyUp(SDL_Scancode code) { return code != SDL_SCANCODE_UNKNOWN && released[code]; }

	bool GetKeyDown(const char* key) {
		return GetKeyDown(ToScancode(key));
	}

	bool GetKeyUp(const char* key) {
		return GetKeyUp(ToScancode(key));
	}

	enum class Axis { GamepadLeftX, GamepadLeftY, GamepadRightX, GamepadRightY, MouseX, MouseY };

	float GetAxis(Axis axis) {
		if (axis == Axis::MouseX) return mouseDeltaX;
		if (axis == Axis::MouseY) return -mouseDeltaY;
		if (!gamepad) return 0.0f;
		constexpr SDL_GamepadAxis axes[] = {SDL_GAMEPAD_AXIS_LEFTX, SDL_GAMEPAD_AXIS_LEFTY, SDL_GAMEPAD_AXIS_RIGHTX, SDL_GAMEPAD_AXIS_RIGHTY};
		const auto index = static_cast<unsigned>(axis);
		if (index >= 4) return 0.0f;
		const Sint16 rawValue = SDL_GetGamepadAxis(gamepad, axes[index]);
		const float value = rawValue < 0 ? static_cast<float>(rawValue) / 32768.0f : static_cast<float>(rawValue) / 32767.0f;
		return std::abs(value) < 0.25f ? 0.0f : value;
	}

	float GetAxis(const char* axis) {
		if (!axis) return 0.0f;
		constexpr const char* names[] = {"Gamepad Left X", "Gamepad Left Y", "Gamepad Right X", "Gamepad Right Y", "Mouse X", "Mouse Y"};
		for (unsigned index = 0; index < 6; ++index) if (std::strcmp(axis, names[index]) == 0) return GetAxis(static_cast<Axis>(index));
		return 0.0f;
	}
}
