#include <array>
#include <cmath>
#include <iterator>
#include <SDL3/SDL.h>
#include <SDL3/SDL_gamepad.h>

namespace Sapphire::Input {
	static std::array<bool, SDL_SCANCODE_COUNT> pressed{};
	static std::array<bool, SDL_SCANCODE_COUNT> released{};
	static std::array<bool, SDL_GAMEPAD_BUTTON_COUNT> gamepadPressed{};
	static std::array<bool, SDL_GAMEPAD_BUTTON_COUNT> gamepadReleased{};
	static SDL_Gamepad* gamepad = nullptr;
	static std::array<bool, SDL_GAMEPAD_BUTTON_COUNT> previousGamepadButtons{};
	static float mouseDeltaX = 0.0f;
	static float mouseDeltaY = 0.0f;

	static bool ValidScancode(SDL_Scancode code) { return code > SDL_SCANCODE_UNKNOWN && code < SDL_SCANCODE_COUNT; }
	static bool ValidGamepadButton(int button) { return button >= 0 && button < SDL_GAMEPAD_BUTTON_COUNT; }

	static void OpenFirstGamepad() {
		int count = 0;
		SDL_JoystickID* gamepads = SDL_GetGamepads(&count);
		for (int index = 0; index < count && !gamepad; ++index) gamepad = SDL_OpenGamepad(gamepads[index]);
		SDL_free(gamepads);
	}

	void Initialize() {
		previousGamepadButtons.fill(false);
		OpenFirstGamepad();
	}

	void Shutdown() {
		if (gamepad) SDL_CloseGamepad(gamepad);
		gamepad = nullptr;
	}

	void BeginFrame() {
		pressed.fill(false);
		released.fill(false);
		gamepadPressed.fill(false);
		gamepadReleased.fill(false);
		mouseDeltaX = 0.0f;
		mouseDeltaY = 0.0f;
	}

	void ProcessEvent(const SDL_Event& event) {
		switch (event.type) {
		case SDL_EVENT_KEY_DOWN:
		case SDL_EVENT_KEY_UP:
			if (ValidScancode(event.key.scancode) && (event.type != SDL_EVENT_KEY_DOWN || !event.key.repeat))
				(event.type == SDL_EVENT_KEY_DOWN ? pressed : released)[event.key.scancode] = true;
			break;
		case SDL_EVENT_MOUSE_MOTION:
			mouseDeltaX += event.motion.xrel;
			mouseDeltaY += event.motion.yrel;
			break;
		case SDL_EVENT_GAMEPAD_ADDED:
			if (!gamepad) gamepad = SDL_OpenGamepad(event.gdevice.which);
			break;
		case SDL_EVENT_GAMEPAD_REMOVED:
			if (gamepad && SDL_GetGamepadID(gamepad) == event.gdevice.which) {
				SDL_CloseGamepad(gamepad);
				gamepad = nullptr;
				OpenFirstGamepad();
			}
			break;
		case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
		case SDL_EVENT_GAMEPAD_BUTTON_UP:
			if (gamepad && SDL_GetGamepadID(gamepad) == event.gbutton.which && ValidGamepadButton(event.gbutton.button))
				(event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN ? gamepadPressed : gamepadReleased)[event.gbutton.button] = true;
			break;
		default:
			break;
		}
	}

	void EndEvents() {
		for (int button = 0; button < SDL_GAMEPAD_BUTTON_COUNT; ++button) {
			const bool held = gamepad && SDL_GetGamepadButton(gamepad, static_cast<SDL_GamepadButton>(button));
			gamepadPressed[button] = !previousGamepadButtons[button] && (gamepadPressed[button] || held);
			gamepadReleased[button] = !held && (gamepadReleased[button] || previousGamepadButtons[button]);
			previousGamepadButtons[button] = held;
		}
	}

	static SDL_Scancode ToScancode(const char* key) {
		if (!key) return SDL_SCANCODE_UNKNOWN;
		return SDL_GetScancodeFromName(key);
	}


	bool GetKey(SDL_Scancode code) {
		if (!ValidScancode(code)) return false;
		return SDL_GetKeyboardState(nullptr)[code];
	}

	bool GetKey(const char* key) {
		return GetKey(ToScancode(key));
	}

	bool GetKeyDown(SDL_Scancode code) {
		return ValidScancode(code) && pressed[code];
	}

	bool GetKeyUp(SDL_Scancode code) {
		return ValidScancode(code) && released[code];
	}

	bool GetKeyDown(const char* key) {
		return GetKeyDown(ToScancode(key));
	}

	bool GetKeyUp(const char* key) {
		return GetKeyUp(ToScancode(key));
	}

	bool GetButton(SDL_GamepadButton button) {
		return ValidGamepadButton(button) && gamepad && SDL_GetGamepadButton(gamepad, button);
	}

	bool GetButtonDown(SDL_GamepadButton button) {
		return ValidGamepadButton(button) && gamepadPressed[button];
	}

	bool GetButtonUp(SDL_GamepadButton button) {
		return ValidGamepadButton(button) && gamepadReleased[button];
	}

	enum class Axis {
		GamepadLeftX,
		GamepadLeftY,
		GamepadRightX,
		GamepadRightY,
		MouseX,
		MouseY
	};

	float GetAxis(Axis axis) {
		if (axis == Axis::MouseX) {
			return mouseDeltaX;
		}
		if (axis == Axis::MouseY) {
			return -mouseDeltaY;
		}
		if (!gamepad) {
			return 0.0f;
		}
		constexpr SDL_GamepadAxis axes[] = {
			SDL_GAMEPAD_AXIS_LEFTX,
			SDL_GAMEPAD_AXIS_LEFTY,
			SDL_GAMEPAD_AXIS_RIGHTX,
			SDL_GAMEPAD_AXIS_RIGHTY
		};
		const auto index = static_cast<unsigned>(axis);
		if (index >= std::size(axes)) {
			return 0.0f;
		}
		const Sint16 rawValue = SDL_GetGamepadAxis(gamepad, axes[index]);
		const float value = rawValue < 0 ? static_cast<float>(rawValue) / 32768.0f : static_cast<float>(rawValue) / 32767.0f;
		if (std::abs(value) < 0.25f) {
			return 0.0f;
		}
		return value;
	}

}
