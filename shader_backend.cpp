#include <SDL3/SDL_gpu.h>
#include <cstring>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxcapi.h>
#include <wrl/client.h>
#undef near
#undef far
#endif

namespace Sapphire::Graphics {
	struct GPUBackend {
		const char* driver;
		SDL_GPUShaderFormat shaderFormat;
	};

#if defined(_WIN32)
	inline constexpr GPUBackend gpuBackend{"direct3d12", SDL_GPU_SHADERFORMAT_DXIL};
#else
	inline constexpr GPUBackend gpuBackend{"metal", SDL_GPU_SHADERFORMAT_MSL};
#endif

	struct ShaderCode {
		const Uint8* bytes = nullptr;
		std::size_t size = 0;
#if defined(_WIN32)
		Microsoft::WRL::ComPtr<IDxcBlob> bytecode;
#endif
	};

#if defined(_WIN32)
	static Microsoft::WRL::ComPtr<IDxcBlob> CompileShader(const char* source, const wchar_t* profile) {
		using Microsoft::WRL::ComPtr;
		static thread_local ComPtr<IDxcCompiler3> compiler;
		if (!compiler && FAILED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(compiler.GetAddressOf())))) {
			SDL_SetError("Could not initialize DirectX Shader Compiler");
			return {};
		}
		const DxcBuffer input{source, std::strlen(source), DXC_CP_UTF8};
		const wchar_t* arguments[] = {L"-E", L"main0", L"-T", profile, L"-HV", L"2021", L"-Zpc"};
		ComPtr<IDxcResult> result;
		if (FAILED(compiler->Compile(&input, arguments, SDL_arraysize(arguments), nullptr, IID_PPV_ARGS(result.GetAddressOf())))) {
			SDL_SetError("DirectX Shader Compiler could not process shader source");
			return {};
		}
		ComPtr<IDxcBlob> bytecode;
		HRESULT status;
		if (FAILED(result->GetStatus(&status)) || FAILED(status) || FAILED(result->GetResult(bytecode.GetAddressOf()))) {
			ComPtr<IDxcBlobUtf8> errors;
			result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(errors.GetAddressOf()), nullptr);
			SDL_SetError("HLSL compilation failed: %s", errors && errors->GetStringLength() ? errors->GetStringPointer() : "no compiler diagnostics");
		}
		return bytecode;
	}
#endif

	static ShaderCode PrepareShaderCode(const char* source, const wchar_t* profile) {
		ShaderCode shader;
#if defined(_WIN32)
		shader.bytecode = CompileShader(source, profile);
		if (shader.bytecode) {
			shader.bytes = static_cast<const Uint8*>(shader.bytecode->GetBufferPointer());
			shader.size = shader.bytecode->GetBufferSize();
		}
#else
		(void)profile;
		shader.bytes = reinterpret_cast<const Uint8*>(source);
		shader.size = std::strlen(source) + 1;
#endif
		return shader;
	}
}
