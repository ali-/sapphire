#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <map>
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
			Uint32 firstIndex;
			Uint32 indexCount;
		};

		struct FontAtlas {
			SDL_GPUTexture* texture = nullptr;
			int cellWidth = 0;
			int cellHeight = 0;
			int columns = 0;
			int firstCharacter = 0;
			int glyphCount = 0;
			int advance = 0;
			std::vector<std::array<float, 4>> glyphs;
		};

		SDL_GPUDevice* gpuDevice = nullptr;
		SDL_GPUGraphicsPipeline* pipeline = nullptr;
		SDL_GPUBuffer* vertexBuffer = nullptr;
		SDL_GPUBuffer* indexBuffer = nullptr;
		SDL_GPUSampler* sampler = nullptr;
		SDL_GPUTransferBuffer* vertexTransferBuffer = nullptr;
		FontAtlas fontAtlas;
		std::vector<Vertex> vertices;
		std::vector<DrawRange> drawRanges;
		std::map<std::string, SDL_GPUTexture*, std::less<>> images;
		bool overflowReported = false;
		constexpr std::size_t maxVertices = 16384;
		constexpr std::size_t headerSize = 20;


		std::uint16_t ReadU16(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
			return static_cast<std::uint16_t>(bytes[offset] | (bytes[offset + 1] << 8));
		}

		bool LoadFont() {
			std::ifstream file(Graphics::ResourcePath("iosevka.sbf"), std::ios::binary);
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

			fontAtlas.glyphs.reserve(fontAtlas.glyphCount);
			for (int index = 0; index < fontAtlas.glyphCount; ++index) {
				const int column = index % fontAtlas.columns;
				const int row = index / fontAtlas.columns;
				fontAtlas.glyphs.push_back({static_cast<float>(column) / fontAtlas.columns, static_cast<float>(row) / rows, static_cast<float>(column + 1) / fontAtlas.columns, static_cast<float>(row + 1) / rows});
			}
			fontAtlas.texture = Graphics::CreateTexture(gpuDevice, width, height, bytes.data() + headerSize);
			return fontAtlas.texture != nullptr;
		}

		void QueueQuad(SDL_GPUTexture* texture, float x, float y, float width, float height, float u0, float v0, float u1, float v1) {
			if (vertices.size() + 4 > maxVertices) {
				if (!overflowReported) SDL_Log("GUI vertex capacity exceeded");
				overflowReported = true;
				return;
			}
			const Uint32 firstIndex = static_cast<Uint32>(vertices.size() / 4 * 6);
			vertices.push_back({{x, y}, {u0, v0}});
			vertices.push_back({{x + width, y}, {u1, v0}});
			vertices.push_back({{x + width, y + height}, {u1, v1}});
			vertices.push_back({{x, y + height}, {u0, v1}});
			if (!drawRanges.empty() && drawRanges.back().texture == texture) drawRanges.back().indexCount += 6;
			else drawRanges.push_back({texture, firstIndex, 6});
		}

		SDL_GPUTexture* LoadImage(std::string_view imagePath) {
			if (auto image = images.find(imagePath); image != images.end()) return image->second;
			SDL_GPUTexture* texture = Graphics::LoadTexture(imagePath, Graphics::linearOutput, false);
			images.emplace(std::string(imagePath), texture);
			return texture;
		}
	}

	bool Initialize(SDL_GPUDevice* device, SDL_Window* window) {
		gpuDevice = device;
		SDL_GPUShader* vertexShader = Graphics::CreateShader(gpuDevice, Graphics::Shaders::guiVertexShaderSource, SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 0, 1);
		SDL_GPUShader* fragmentShader = Graphics::CreateShader(gpuDevice, Graphics::Shaders::guiFragmentShaderSource, SDL_GPU_SHADERSTAGE_FRAGMENT, SDL_GPU_SHADERFORMAT_MSL, 1);

		if (!vertexShader || !fragmentShader) {
			if (vertexShader) SDL_ReleaseGPUShader(gpuDevice, vertexShader);
			if (fragmentShader) SDL_ReleaseGPUShader(gpuDevice, fragmentShader);
			return false;
		}

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
		try { pipeline = Graphics::CreatePipeline(gpuDevice, pipelineInfo); }
		catch (...) {
			SDL_ReleaseGPUShader(gpuDevice, vertexShader);
			SDL_ReleaseGPUShader(gpuDevice, fragmentShader);
			throw;
		}
		SDL_ReleaseGPUShader(gpuDevice, vertexShader);
		SDL_ReleaseGPUShader(gpuDevice, fragmentShader);

		SDL_GPUSamplerCreateInfo samplerInfo{};
		samplerInfo.min_filter = SDL_GPU_FILTER_NEAREST;
		samplerInfo.mag_filter = SDL_GPU_FILTER_NEAREST;
		samplerInfo.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
		samplerInfo.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
		samplerInfo.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
		samplerInfo.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
		sampler = Graphics::CreateSampler(gpuDevice, samplerInfo);
		vertexBuffer = Graphics::CreateBuffer(gpuDevice, SDL_GPU_BUFFERUSAGE_VERTEX, static_cast<Uint32>(sizeof(Vertex) * maxVertices));
		std::vector<std::uint16_t> indices;
		indices.reserve(maxVertices / 4 * 6);
		for (std::size_t vertex = 0; vertex < maxVertices; vertex += 4) {
			const auto first = static_cast<std::uint16_t>(vertex);
			indices.insert(indices.end(), {first, static_cast<std::uint16_t>(first + 1), static_cast<std::uint16_t>(first + 2), first, static_cast<std::uint16_t>(first + 2), static_cast<std::uint16_t>(first + 3)});
		}
		indexBuffer = Graphics::UploadBuffer(SDL_GPU_BUFFERUSAGE_INDEX, indices.data(), static_cast<Uint32>(indices.size() * sizeof(std::uint16_t)));
		return pipeline && sampler && vertexBuffer && indexBuffer && LoadFont();
	}

	void RetryImages() {
		for (auto image = images.begin(); image != images.end();) {
			if (!image->second) image = images.erase(image);
			else ++image;
		}
	}

	void BeginFrame() {
		overflowReported = false;
		vertices.clear();
		drawRanges.clear();
	}

	void Label(int x, int y, std::string_view text) {
		const int startX = x;
		const int endCharacter = fontAtlas.firstCharacter + fontAtlas.glyphCount;
		const int fallback = '?' >= fontAtlas.firstCharacter && '?' < endCharacter ? '?' - fontAtlas.firstCharacter : 0;
		for (const unsigned char character : text) {
			if (character == '\n') {
				x = startX;
				y += fontAtlas.cellHeight;
				continue;
			}
			if (character == ' ') {
				x += fontAtlas.advance;
				continue;
			}
			const int index = character >= fontAtlas.firstCharacter && character < endCharacter ? character - fontAtlas.firstCharacter : fallback;
			const auto& uv = fontAtlas.glyphs[index];
			QueueQuad(fontAtlas.texture, static_cast<float>(x), static_cast<float>(y), static_cast<float>(fontAtlas.cellWidth), static_cast<float>(fontAtlas.cellHeight), uv[0], uv[1], uv[2], uv[3]);
			x += fontAtlas.advance;
		}
	}

	void Image(int x, int y, int width, int height, std::string_view imagePath) {
		if (SDL_GPUTexture* texture = LoadImage(imagePath)) {
			QueueQuad(texture, static_cast<float>(x), static_cast<float>(y), static_cast<float>(width), static_cast<float>(height), 0.0f, 0.0f, 1.0f, 1.0f);
		}
	}

	void Render(SDL_GPUCommandBuffer* commandBuffer, SDL_GPUTexture* colorTarget, Uint32 width, Uint32 height) {
		if (!colorTarget || !width || !height) return;
		Graphics::FlushUploads();
		if (vertices.empty()) return;
		const float scale[2] = {2.0f / width, -2.0f / height};
		SDL_PushGPUVertexUniformData(commandBuffer, 0, scale, sizeof(scale));
		const Uint32 vertexBytes = static_cast<Uint32>(vertices.size() * sizeof(Vertex));
		if (!vertexTransferBuffer) vertexTransferBuffer = Graphics::CreateTransferBuffer(gpuDevice, static_cast<Uint32>(maxVertices * sizeof(Vertex)));
		void* mappedVertices = SDL_MapGPUTransferBuffer(gpuDevice, vertexTransferBuffer, true);
		if (!mappedVertices) throw std::runtime_error(std::string("Map GUI vertex transfer buffer: ") + SDL_GetError());
		std::memcpy(mappedVertices, vertices.data(), vertexBytes);
		SDL_UnmapGPUTransferBuffer(gpuDevice, vertexTransferBuffer);
		SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(commandBuffer);
		if (!copyPass) throw std::runtime_error(std::string("Begin GUI copy pass: ") + SDL_GetError());
		SDL_GPUTransferBufferLocation source{};
		source.transfer_buffer = vertexTransferBuffer;
		SDL_GPUBufferRegion destination{};
		destination.buffer = vertexBuffer;
		destination.size = vertexBytes;
		SDL_UploadToGPUBuffer(copyPass, &source, &destination, true);
		Graphics::RecordUpload(vertexBytes);
		SDL_EndGPUCopyPass(copyPass);

		SDL_GPUColorTargetInfo colorInfo{};
		colorInfo.texture = colorTarget;
		colorInfo.load_op = SDL_GPU_LOADOP_LOAD;
		colorInfo.store_op = SDL_GPU_STOREOP_STORE;
		SDL_GPURenderPass* renderPass = SDL_BeginGPURenderPass(commandBuffer, &colorInfo, 1, nullptr);
		if (!renderPass) throw std::runtime_error(std::string("Begin GUI render pass: ") + SDL_GetError());
		SDL_GPUBufferBinding vertexBinding{};
		vertexBinding.buffer = vertexBuffer;
		SDL_BindGPUVertexBuffers(renderPass, 0, &vertexBinding, 1);
		SDL_GPUBufferBinding indexBinding{indexBuffer, 0};
		SDL_BindGPUIndexBuffer(renderPass, &indexBinding, SDL_GPU_INDEXELEMENTSIZE_16BIT);
		SDL_GPUTextureSamplerBinding samplerBinding{};
		samplerBinding.sampler = sampler;
		SDL_BindGPUGraphicsPipeline(renderPass, pipeline);
		for (const auto& range : drawRanges) {
			samplerBinding.texture = range.texture;
			SDL_BindGPUFragmentSamplers(renderPass, 0, &samplerBinding, 1);
			SDL_DrawGPUIndexedPrimitives(renderPass, range.indexCount, 1, range.firstIndex, 0, 0);
		}
		SDL_EndGPURenderPass(renderPass);
	}

	void Shutdown() {
		if (!gpuDevice) return;
		if (vertexTransferBuffer) Graphics::ReleaseTransferBuffer(gpuDevice, vertexTransferBuffer);
		if (vertexBuffer) Graphics::ReleaseBuffer(gpuDevice, vertexBuffer);
		if (indexBuffer) Graphics::ReleaseBuffer(gpuDevice, indexBuffer);
		if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
		if (sampler) SDL_ReleaseGPUSampler(gpuDevice, sampler);
		if (fontAtlas.texture) Graphics::ReleaseTexture(gpuDevice, fontAtlas.texture);
		for (const auto& image : images) Graphics::ReleaseTexture(gpuDevice, image.second);
		gpuDevice = nullptr;
		pipeline = nullptr;
		vertexBuffer = nullptr;
		indexBuffer = nullptr;
		vertexTransferBuffer = nullptr;
		sampler = nullptr;
		fontAtlas = FontAtlas{};
		images.clear();
		vertices.clear();
		drawRanges.clear();
	}
}
