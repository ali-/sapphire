#include <SDL3/SDL.h>
#include "core.cpp"
#include "physics.cpp"
#include "input.cpp"
#include "gui.cpp"

namespace Sapphire {
	namespace {
		bool running = false;
	}

	void Exit() {
		running = false;
	}

	int App(const char* windowTitle, Scene& scene) {
		running = true;
		SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD);
		Input::Initialize();
		SDL_Window *window = SDL_CreateWindow(windowTitle, Screen::width, Screen::height, SDL_WINDOW_RESIZABLE);
		SDL_GetWindowSize(window, &Screen::width, &Screen::height);
		SDL_GPUDevice *gpuDevice = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_MSL, true, "metal");
		SDL_ClaimWindowForGPUDevice(gpuDevice, window);
		SDL_SetWindowAspectRatio(window, 16.0f / 9.0f, 16.0f / 9.0f);
		Graphics::Initialize(gpuDevice, window);
		GUI::Initialize(gpuDevice, window);

		const Uint64 fixed_step_ns = static_cast<Uint64>(Time::fixedDeltaTime * 1'000'000'000.0f);
		Uint64 accumulator = 0;
		for (auto& object : scene.Objects()) object->Initialize();
		for (auto& object : scene.Objects()) {
			if (auto* body = object->GetComponent<PhysicsBody>()) body->previousPosition = object->transform.position;
		}
		scene.SelectFirstCamera();
		Physics::Initialize(scene);
		Graphics::PrepareScene(scene);
		Uint64 previous_time = SDL_GetTicksNS();
		while (running) {
			const Uint64 frameStart = SDL_GetTicksNS();
			Input::BeginFrame();
			GUI::BeginFrame();
			SDL_Event event;
			while (SDL_PollEvent(&event)) {
				Input::ProcessEvent(event);
				if (event.type == SDL_EVENT_QUIT) running = false;
				if (event.type == SDL_EVENT_WINDOW_RESIZED && event.window.windowID == SDL_GetWindowID(window)) {
					Screen::width = event.window.data1;
					Screen::height = event.window.data2;
				}
			}

			const Uint64 current_time = frameStart;
			const Uint64 frame_time_ns = current_time - previous_time;
			Time::deltaTime = static_cast<float>(frame_time_ns) / 1'000'000'000.0f;
			accumulator += frame_time_ns;
			previous_time = current_time;
			while (accumulator >= fixed_step_ns) {
				for (auto& object : scene.Objects()) object->FixedUpdate();
				Physics::Simulate(scene, Time::fixedDeltaTime);
				accumulator -= fixed_step_ns;
			}
			Time::interpolationAlpha = static_cast<float>(accumulator) / fixed_step_ns;

			for (auto& object : scene.Objects()) {
				object->Update();
				object->GUI();
			}

			SDL_GPUCommandBuffer *commandBuffer = SDL_AcquireGPUCommandBuffer(gpuDevice);
			SDL_GPUTexture *swapchainTexture = nullptr;
			Uint32 swapchainWidth = 0;
			Uint32 swapchainHeight = 0;
			SDL_WaitAndAcquireGPUSwapchainTexture(commandBuffer, window, &swapchainTexture, &swapchainWidth, &swapchainHeight);
			if (swapchainTexture) {
				Graphics::Render(scene, commandBuffer, swapchainTexture, swapchainWidth, swapchainHeight);
				GUI::Render(commandBuffer, swapchainTexture, swapchainWidth, swapchainHeight);
			}
			Graphics::Submit(commandBuffer);
			SDL_Delay(1);
		}

		GUI::Shutdown();
		Graphics::Shutdown();
		Physics::Shutdown();
		Input::Shutdown();
		SDL_ReleaseWindowFromGPUDevice(gpuDevice, window);
		SDL_DestroyGPUDevice(gpuDevice);
		SDL_DestroyWindow(window);
		SDL_Quit();

		return 0;
	}
}
