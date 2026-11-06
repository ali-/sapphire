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

	static void Shutdown(SDL_GPUDevice* gpuDevice, SDL_Window* window, bool claimed = false) {
		GUI::Shutdown();
		if (gpuDevice) Graphics::Shutdown();
		Physics::Shutdown();
		Input::Shutdown();
		if (gpuDevice && window && claimed) SDL_ReleaseWindowFromGPUDevice(gpuDevice, window);
		if (gpuDevice) SDL_DestroyGPUDevice(gpuDevice);
		if (window) SDL_DestroyWindow(window);
		SDL_Quit();
	}

	int App(const char* windowTitle, Scene& scene) {
		Loading::Timer loading("Engine initialization");
		Loading::Timer windowLoading("SDL, window and GPU device");
		running = true;
		if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) return 1;
		Input::Initialize();
		SDL_Window *window = SDL_CreateWindow(windowTitle, Screen::width, Screen::height, SDL_WINDOW_RESIZABLE | SDL_WINDOW_FULLSCREEN);
		if (!window) {
			Shutdown(nullptr, nullptr);
			return 1;
		}
		SDL_SetWindowRelativeMouseMode(window, true);
		SDL_GetWindowSize(window, &Screen::width, &Screen::height);
		SDL_GPUDevice *gpuDevice = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_MSL, SDL_GetHintBoolean("SAPPHIRE_GPU_DEBUG", false), "metal");
		const bool claimed = gpuDevice && SDL_ClaimWindowForGPUDevice(gpuDevice, window);
		if (!claimed) {
			Shutdown(gpuDevice, window, claimed);
			return 1;
		}
		try {
			SDL_SetWindowAspectRatio(window, 16.0f / 9.0f, 16.0f / 9.0f);
			windowLoading.Stop();
			if (!Graphics::Initialize(gpuDevice, window) || !GUI::Initialize(gpuDevice, window) || !std::isfinite(Time::fixedDeltaTime) || Time::fixedDeltaTime <= 0.0f) {
				SDL_Log("Engine initialization failed: %s", SDL_GetError());
				Shutdown(gpuDevice, window, claimed);
				return 1;
			}

			const float fixedStep = Time::fixedDeltaTime;
			const Uint64 fixedStepNs = std::max<Uint64>(1, static_cast<Uint64>(fixedStep * 1'000'000'000.0));
			Uint64 accumulator = 0;
			Loading::Timer sceneLoading("Scene object initialization");
			scene.SetAppActive(true);
			auto initializeObjects = [&] {
				for (std::size_t index = 0; index < scene.Objects().size(); ++index) {
					Object* object = scene.Objects()[index].get();
					if (object->initializedForApp) continue;
					object->UpdateWorldTransform();
					object->Initialize();
					object->initializedForApp = true;
					if (auto* body = object->GetComponent<PhysicsBody>()) body->previousPosition = object->transform.position;
				}
				scene.UpdateWorldPartition();
			};
			for (auto& object : scene.Objects()) object->initializedForApp = false;
			initializeObjects();
			scene.SelectFirstCamera();
			sceneLoading.Stop();
			Physics::Initialize(scene);
			Graphics::PrepareScene(scene);
			loading.Stop();
			std::cout.flush();
			bool firstFrame = true;
			Uint64 previous_time = SDL_GetTicksNS();
			while (running) {
				Loading::Timer frameLoading("First frame CPU preparation, swapchain acquire and submission", firstFrame);
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

				Input::EndEvents();

				const Uint64 current_time = frameStart;
				const Uint64 frame_time_ns = current_time - previous_time;
				Time::deltaTime = static_cast<float>(frame_time_ns) / 1'000'000'000.0f;
				accumulator += std::min(frame_time_ns, fixedStepNs * 8);
				previous_time = current_time;
				initializeObjects();
				while (accumulator >= fixedStepNs) {
					const std::size_t phaseCount = scene.Objects().size();
					for (std::size_t index = 0; index < phaseCount; ++index) {
						Object* object = scene.Objects()[index].get();
						if (object->initializedForApp) {
							object->UpdateWorldTransform();
							object->FixedUpdate();
						}
					}
					Physics::Simulate(scene, fixedStep);
					scene.UpdateTransforms();
					accumulator -= fixedStepNs;
				}
				Time::interpolationAlpha = static_cast<float>(accumulator) / fixedStepNs;

				const std::size_t updateCount = scene.Objects().size();
				for (std::size_t index = 0; index < updateCount; ++index) {
					Object* object = scene.Objects()[index].get();
					if (!object->initializedForApp) continue;
					object->UpdateWorldTransform();
					object->Update();
					if (object->initializedForApp) object->GUI();
					if (object->HasChildren()) object->UpdateWorldTransformSubtree();
				}
				scene.UpdateTransforms();

				SDL_GPUCommandBuffer *commandBuffer = SDL_AcquireGPUCommandBuffer(gpuDevice);
				if (!commandBuffer) throw std::runtime_error(std::string("Acquire frame command buffer: ") + SDL_GetError());
				SDL_GPUTexture *swapchainTexture = nullptr;
				Uint32 swapchainWidth = 0;
				Uint32 swapchainHeight = 0;
				bool acquired = false;
				bool submitted = false;
				try {
					if (!SDL_WaitAndAcquireGPUSwapchainTexture(commandBuffer, window, &swapchainTexture, &swapchainWidth, &swapchainHeight)) throw std::runtime_error(std::string("Acquire swapchain texture: ") + SDL_GetError());
					acquired = true;
					if (swapchainTexture) {
						Graphics::Render(scene, commandBuffer, swapchainTexture, swapchainWidth, swapchainHeight);
						GUI::Render(commandBuffer, swapchainTexture, swapchainWidth, swapchainHeight);
					}
					Graphics::FlushUploads();
					submitted = true;
					Graphics::Submit(commandBuffer);
				}
				catch (...) {
					if (!submitted) {
						if (acquired) SDL_SubmitGPUCommandBuffer(commandBuffer);
						else SDL_CancelGPUCommandBuffer(commandBuffer);
					}
					throw;
				}
				frameLoading.Stop();
				if (firstFrame) std::cout.flush();
				firstFrame = false;
				if (!swapchainTexture) SDL_Delay(10);
			}

			scene.SetAppActive(false);
			Shutdown(gpuDevice, window, claimed);

			return 0;
		}
		catch (const std::exception& error) {
			SDL_Log("Engine failure: %s", error.what());
			scene.SetAppActive(false);
			Shutdown(gpuDevice, window, claimed);
			return 1;
		}
	}
}
