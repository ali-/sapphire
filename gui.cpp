#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>
#include <SDL3/SDL.h>
#include <SDL3/SDL_gpu.h>
#include "graphics.cpp"

namespace Sapphire::GUI {
	namespace {
		struct Vertex {
			float position[2];
			float uv[2];
		};

		struct DrawRange {
			SDL_GPUTexture* texture;
			Uint32 firstVertex;
			Uint32 vertexCount;
			bool depthTexture;
		};

		struct ImageTexture {
			std::string path;
			SDL_GPUTexture* texture;
		};

		struct FontAtlas {
			SDL_GPUTexture* texture = nullptr;
			int cellWidth = 0;
			int cellHeight = 0;
			int columns = 0;
			int firstCharacter = 0;
			int glyphCount = 0;
			int advance = 0;
		};

		SDL_GPUDevice* gpuDevice = nullptr;
		SDL_GPUGraphicsPipeline* pipeline = nullptr;
		SDL_GPUGraphicsPipeline* depthPipeline = nullptr;
		SDL_GPUBuffer* vertexBuffer = nullptr;
		SDL_GPUSampler* sampler = nullptr;
		SDL_GPUTransferBuffer* vertexTransferBuffer = nullptr;
		FontAtlas fontAtlas;
		std::vector<Vertex> vertices;
		std::vector<DrawRange> drawRanges;
		std::vector<ImageTexture> images;
		constexpr std::size_t maxVertices = 16384;
		constexpr std::size_t headerSize = 20;

		constexpr const char* vertexShaderSource = R"(
			#include <metal_stdlib>
			using namespace metal;
			struct VertexInput { float2 position [[attribute(0)]]; float2 uv [[attribute(1)]]; };
			struct VertexOutput { float4 position [[position]]; float2 uv; };
			vertex VertexOutput main0(VertexInput input [[stage_in]]) {
				VertexOutput output;
				output.position = float4(input.position, 0.0, 1.0);
				output.uv = input.uv;
				return output;
			}
		)";

		constexpr const char* fragmentShaderSource = R"(
			#include <metal_stdlib>
			using namespace metal;
			struct FragmentInput { float4 position [[position]]; float2 uv; };
			fragment float4 main0(FragmentInput input [[stage_in]],
				texture2d<float> image [[texture(0)]], sampler imageSampler [[sampler(0)]]) {
				return image.sample(imageSampler, input.uv);
			}
		)";

		constexpr const char* depthFragmentShaderSource = R"(
			#include <metal_stdlib>
			using namespace metal;
			struct FragmentInput { float4 position [[position]]; float2 uv; };
			fragment float4 main0(FragmentInput input [[stage_in]], depth2d<float> image [[texture(0)]], sampler imageSampler [[sampler(0)]]) {
				const float depth = image.sample(imageSampler, input.uv);
				return float4(depth, depth, depth, 1.0);
			}
		)";

		std::uint16_t ReadU16(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
			return static_cast<std::uint16_t>(bytes[offset] | (bytes[offset + 1] << 8));
		}

		std::filesystem::path ResourcePath(std::string_view fileName) {
			return std::filesystem::path(SDL_GetBasePath()) / ".." / "resources" / std::string(fileName);
		}

		bool LoadFont() {
			std::ifstream file(ResourcePath("iosevka.sbf"), std::ios::binary);
			std::vector<std::uint8_t> bytes(std::istreambuf_iterator<char>(file), {});
			if (bytes.size() < headerSize || std::memcmp(bytes.data(), "SBF1", 4) != 0) return false;

			const int width = ReadU16(bytes, 4);
			const int height = ReadU16(bytes, 6);
			fontAtlas.cellWidth = ReadU16(bytes, 8);
			fontAtlas.cellHeight = ReadU16(bytes, 10);
			fontAtlas.columns = ReadU16(bytes, 12);
			fontAtlas.firstCharacter = ReadU16(bytes, 14);
			fontAtlas.glyphCount = ReadU16(bytes, 16);
			fontAtlas.advance = ReadU16(bytes, 18);
			const int rows = fontAtlas.columns > 0 ? (fontAtlas.glyphCount + fontAtlas.columns - 1) / fontAtlas.columns : 0;
			const std::size_t pixelBytes = static_cast<std::size_t>(width) * height * 4;
			if (width <= 0 || height <= 0 || fontAtlas.cellWidth <= 0 || fontAtlas.cellHeight <= 0 ||
				fontAtlas.columns <= 0 || fontAtlas.glyphCount <= 0 || fontAtlas.advance <= 0 ||
				width != fontAtlas.columns * fontAtlas.cellWidth || height != rows * fontAtlas.cellHeight ||
				pixelBytes != bytes.size() - headerSize) return false;

			fontAtlas.texture = Graphics::CreateTexture(gpuDevice, width, height, bytes.data() + headerSize);
			return fontAtlas.texture != nullptr;
		}

		void QueueQuad(SDL_GPUTexture* texture, float x, float y, float width, float height, float u0, float v0, float u1, float v1, bool depthTexture = false) {
			if (vertices.size() + 6 > maxVertices) return;
			const Uint32 firstVertex = static_cast<Uint32>(vertices.size());
			vertices.push_back({{x, y}, {u0, v0}});
			vertices.push_back({{x + width, y}, {u1, v0}});
			vertices.push_back({{x + width, y + height}, {u1, v1}});
			vertices.push_back({{x, y}, {u0, v0}});
			vertices.push_back({{x + width, y + height}, {u1, v1}});
			vertices.push_back({{x, y + height}, {u0, v1}});
			if (!drawRanges.empty() && drawRanges.back().texture == texture && drawRanges.back().depthTexture == depthTexture) drawRanges.back().vertexCount += 6;
			else drawRanges.push_back({texture, firstVertex, 6, depthTexture});
		}

		SDL_GPUTexture* LoadImage(std::string_view imagePath) {
			const std::string path(imagePath);
			for (const auto& image : images) if (image.path == path) return image.texture;
			SDL_GPUTexture* texture = Graphics::LoadTexture(imagePath);
			if (texture) images.push_back({path, texture});
			return texture;
		}
	}

	bool Initialize(SDL_GPUDevice* device, SDL_Window* window) {
		gpuDevice = device;
		SDL_GPUShader* vertexShader = Graphics::CreateShader(gpuDevice, vertexShaderSource, SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL);
		SDL_GPUShader* fragmentShader = Graphics::CreateShader(gpuDevice, fragmentShaderSource, SDL_GPU_SHADERSTAGE_FRAGMENT, SDL_GPU_SHADERFORMAT_MSL, 1);
		SDL_GPUShader* depthFragmentShader = Graphics::CreateShader(gpuDevice, depthFragmentShaderSource, SDL_GPU_SHADERSTAGE_FRAGMENT, SDL_GPU_SHADERFORMAT_MSL, 1);

		SDL_GPUVertexBufferDescription vertexBufferDescription{};
		vertexBufferDescription.slot = 0;
		vertexBufferDescription.pitch = sizeof(Vertex);
		vertexBufferDescription.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;
		SDL_GPUVertexAttribute vertexAttributes[2]{};
		vertexAttributes[0].location = 0;
		vertexAttributes[0].buffer_slot = 0;
		vertexAttributes[0].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2;
		vertexAttributes[0].offset = offsetof(Vertex, position);
		vertexAttributes[1].location = 1;
		vertexAttributes[1].buffer_slot = 0;
		vertexAttributes[1].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2;
		vertexAttributes[1].offset = offsetof(Vertex, uv);
		SDL_GPUColorTargetDescription colorTarget{};
		colorTarget.format = SDL_GetGPUSwapchainTextureFormat(gpuDevice, window);
		colorTarget.blend_state.enable_blend = true;
		colorTarget.blend_state.src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA;
		colorTarget.blend_state.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
		colorTarget.blend_state.color_blend_op = SDL_GPU_BLENDOP_ADD;
		colorTarget.blend_state.src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
		colorTarget.blend_state.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
		colorTarget.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_ADD;
		SDL_GPUGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.vertex_shader = vertexShader;
		pipelineInfo.fragment_shader = fragmentShader;
		pipelineInfo.vertex_input_state.num_vertex_buffers = 1;
		pipelineInfo.vertex_input_state.vertex_buffer_descriptions = &vertexBufferDescription;
		pipelineInfo.vertex_input_state.num_vertex_attributes = 2;
		pipelineInfo.vertex_input_state.vertex_attributes = vertexAttributes;
		pipelineInfo.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
		pipelineInfo.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
		pipelineInfo.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
		pipelineInfo.multisample_state.sample_count = SDL_GPU_SAMPLECOUNT_1;
		pipelineInfo.target_info.num_color_targets = 1;
		pipelineInfo.target_info.color_target_descriptions = &colorTarget;
		pipeline = Graphics::CreatePipeline(gpuDevice, pipelineInfo);
		pipelineInfo.fragment_shader = depthFragmentShader;
		depthPipeline = Graphics::CreatePipeline(gpuDevice, pipelineInfo);
		SDL_ReleaseGPUShader(gpuDevice, vertexShader);
		SDL_ReleaseGPUShader(gpuDevice, fragmentShader);
		SDL_ReleaseGPUShader(gpuDevice, depthFragmentShader);

		SDL_GPUSamplerCreateInfo samplerInfo{};
		samplerInfo.min_filter = SDL_GPU_FILTER_NEAREST;
		samplerInfo.mag_filter = SDL_GPU_FILTER_NEAREST;
		samplerInfo.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
		samplerInfo.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
		samplerInfo.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
		samplerInfo.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
		sampler = Graphics::CreateSampler(gpuDevice, samplerInfo);
		vertexBuffer = Graphics::CreateBuffer(gpuDevice, SDL_GPU_BUFFERUSAGE_VERTEX, static_cast<Uint32>(sizeof(Vertex) * maxVertices));
		return pipeline && depthPipeline && sampler && vertexBuffer && LoadFont();
	}

	void BeginFrame() {
		if (vertexTransferBuffer) SDL_ReleaseGPUTransferBuffer(gpuDevice, vertexTransferBuffer);
		vertexTransferBuffer = nullptr;
		vertices.clear();
		drawRanges.clear();
	}

	void Label(int x, int y, std::string_view text) {
		const int startX = x;
		for (const unsigned char character : text) {
			if (character == '\n') {
				x = startX;
				y += fontAtlas.cellHeight;
				continue;
			}
			const int endCharacter = fontAtlas.firstCharacter + fontAtlas.glyphCount;
			const int safeCharacter = character >= fontAtlas.firstCharacter && character < endCharacter ? character : '?';
			const int index = safeCharacter - fontAtlas.firstCharacter;
			const int column = index % fontAtlas.columns;
			const int row = index / fontAtlas.columns;
			const int rows = (fontAtlas.glyphCount + fontAtlas.columns - 1) / fontAtlas.columns;
			QueueQuad(fontAtlas.texture, static_cast<float>(x), static_cast<float>(y), static_cast<float>(fontAtlas.cellWidth), static_cast<float>(fontAtlas.cellHeight), static_cast<float>(column) / fontAtlas.columns, static_cast<float>(row) / rows, static_cast<float>(column + 1) / fontAtlas.columns, static_cast<float>(row + 1) / rows);
			x += fontAtlas.advance;
		}
	}

	void Image(int x, int y, int width, int height, std::string_view imagePath) {
		if (SDL_GPUTexture* texture = LoadImage(imagePath)) {
			QueueQuad(texture, static_cast<float>(x), static_cast<float>(y), static_cast<float>(width), static_cast<float>(height), 0.0f, 0.0f, 1.0f, 1.0f);
		}
	}

	void DebugDepthImage(int x, int y, int width, int height, SDL_GPUTexture* texture) {
		if (texture) QueueQuad(texture, static_cast<float>(x), static_cast<float>(y), static_cast<float>(width), static_cast<float>(height), 0.0f, 0.0f, 1.0f, 1.0f, true);
	}

	void Render(SDL_GPUCommandBuffer* commandBuffer, SDL_GPUTexture* colorTarget, Uint32 width, Uint32 height) {
		if (!colorTarget) return;
		if (!vertices.empty()) {
			for (auto& vertex : vertices) {
				vertex.position[0] = vertex.position[0] / static_cast<float>(width) * 2.0f - 1.0f;
				vertex.position[1] = 1.0f - vertex.position[1] / static_cast<float>(height) * 2.0f;
			}
			SDL_GPUTransferBufferCreateInfo transferInfo{};
			transferInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
			transferInfo.size = static_cast<Uint32>(vertices.size() * sizeof(Vertex));
			vertexTransferBuffer = SDL_CreateGPUTransferBuffer(gpuDevice, &transferInfo);
			void* mappedVertices = SDL_MapGPUTransferBuffer(gpuDevice, vertexTransferBuffer, false);
			std::memcpy(mappedVertices, vertices.data(), transferInfo.size);
			SDL_UnmapGPUTransferBuffer(gpuDevice, vertexTransferBuffer);
			SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(commandBuffer);
			SDL_GPUTransferBufferLocation source{};
			source.transfer_buffer = vertexTransferBuffer;
			SDL_GPUBufferRegion destination{};
			destination.buffer = vertexBuffer;
			destination.size = transferInfo.size;
			SDL_UploadToGPUBuffer(copyPass, &source, &destination, true);
			SDL_EndGPUCopyPass(copyPass);
		}

		SDL_GPUColorTargetInfo colorInfo{};
		colorInfo.texture = colorTarget;
		colorInfo.load_op = SDL_GPU_LOADOP_LOAD;
		colorInfo.store_op = SDL_GPU_STOREOP_STORE;
		SDL_GPURenderPass* renderPass = SDL_BeginGPURenderPass(commandBuffer, &colorInfo, 1, nullptr);
		if (!vertices.empty()) {
			SDL_GPUBufferBinding vertexBinding{};
			vertexBinding.buffer = vertexBuffer;
			SDL_BindGPUVertexBuffers(renderPass, 0, &vertexBinding, 1);
			SDL_GPUTextureSamplerBinding samplerBinding{};
			samplerBinding.sampler = sampler;
			SDL_GPUGraphicsPipeline* activePipeline = nullptr;
			for (const auto& range : drawRanges) {
				SDL_GPUGraphicsPipeline* rangePipeline = range.depthTexture ? depthPipeline : pipeline;
				if (rangePipeline != activePipeline) {
					SDL_BindGPUGraphicsPipeline(renderPass, rangePipeline);
					activePipeline = rangePipeline;
				}
				samplerBinding.texture = range.texture;
				SDL_BindGPUFragmentSamplers(renderPass, 0, &samplerBinding, 1);
				SDL_DrawGPUPrimitives(renderPass, range.vertexCount, 1, range.firstVertex, 0);
			}
		}
		SDL_EndGPURenderPass(renderPass);
	}

	void Shutdown() {
		if (!gpuDevice) return;
		if (vertexTransferBuffer) SDL_ReleaseGPUTransferBuffer(gpuDevice, vertexTransferBuffer);
		if (vertexBuffer) SDL_ReleaseGPUBuffer(gpuDevice, vertexBuffer);
		if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
		if (depthPipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, depthPipeline);
		if (sampler) SDL_ReleaseGPUSampler(gpuDevice, sampler);
		if (fontAtlas.texture) SDL_ReleaseGPUTexture(gpuDevice, fontAtlas.texture);
		for (const auto& image : images) SDL_ReleaseGPUTexture(gpuDevice, image.texture);
		gpuDevice = nullptr;
		pipeline = nullptr;
		depthPipeline = nullptr;
		vertexBuffer = nullptr;
		sampler = nullptr;
		fontAtlas = FontAtlas{};
		images.clear();
		vertices.clear();
		drawRanges.clear();
	}
}
