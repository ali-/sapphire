#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <SDL3/SDL.h>
#include <SDL3/SDL_gpu.h>

namespace Sapphire::Graphics {
	SDL_GPUBuffer* CreateBuffer(SDL_GPUDevice* gpuDevice, SDL_GPUBufferUsageFlags usage, Uint32 size);

	struct MeshVertex {
		Vector3 position;
		Vector3 normal;
		float uv[2];
	};

	struct MeshData {
		std::vector<MeshVertex> vertices;
		std::vector<std::uint32_t> indices;
	};

	namespace {
		struct Matrix4 {
			float values[16]{};
		};

		struct ObjectUniforms {
			float mvp[16];
			float model[16];
			float normal[16];
		};

		struct GpuMesh {
			SDL_GPUBuffer* vertexBuffer = nullptr;
			SDL_GPUBuffer* indexBuffer = nullptr;
			Uint32 indexCount = 0;
		};

		SDL_GPUDevice* gpuDevice = nullptr;
		SDL_GPUGraphicsPipeline* pipeline = nullptr;
		SDL_GPUTexture* depthTexture = nullptr;
		Uint32 depthWidth = 0;
		Uint32 depthHeight = 0;
		GpuMesh cubeMesh;
		GpuMesh planeMesh;

		constexpr const char* vertexShaderSource = R"(
			#include <metal_stdlib>
			using namespace metal;
			struct VertexInput { float3 position [[attribute(0)]]; float3 normal [[attribute(1)]]; float2 uv [[attribute(2)]]; };
			struct VertexOutput { float4 position [[position]]; float3 normal; float2 uv; };
			struct ObjectUniforms { float4x4 mvp; float4x4 model; float4x4 normal; };
			vertex VertexOutput main0(VertexInput input [[stage_in]], constant ObjectUniforms& uniforms [[buffer(0)]]) {
				VertexOutput output;
				output.position = uniforms.mvp * float4(input.position, 1.0);
				output.normal = (uniforms.normal * float4(input.normal, 0.0)).xyz;
				output.uv = input.uv;
				return output;
			}
		)";

		constexpr const char* fragmentShaderSource = R"(
			#include <metal_stdlib>
			using namespace metal;
			struct FragmentInput { float4 position [[position]]; float3 normal; float2 uv; };
			fragment float4 main0(FragmentInput input [[stage_in]]) {
				float light = max(dot(normalize(input.normal), normalize(float3(-0.4, 0.8, -0.5))), 0.2);
				return float4(float3(0.65, 0.72, 0.82) * light, 1.0);
			}
		)";

		Matrix4 Identity() {
			Matrix4 matrix{};
			matrix.values[0] = 1.0f;
			matrix.values[5] = 1.0f;
			matrix.values[10] = 1.0f;
			matrix.values[15] = 1.0f;
			return matrix;
		}

		Matrix4 Multiply(const Matrix4& left, const Matrix4& right) {
			Matrix4 result{};
			for (int column = 0; column < 4; ++column) {
				for (int row = 0; row < 4; ++row) {
					for (int index = 0; index < 4; ++index) result.values[column * 4 + row] += left.values[index * 4 + row] * right.values[column * 4 + index];
				}
			}
			return result;
		}

		Matrix4 Model(const Transform& transform) {
			const float x = transform.rotation.x * Math::degreesToRadians;
			const float y = transform.rotation.y * Math::degreesToRadians;
			const float z = transform.rotation.z * Math::degreesToRadians;
			const float cx = std::cos(x);
			const float sx = std::sin(x);
			const float cy = std::cos(y);
			const float sy = std::sin(y);
			const float cz = std::cos(z);
			const float sz = std::sin(z);
			Matrix4 matrix = Identity();
			matrix.values[0] = cz * cy * transform.scale.x;
			matrix.values[1] = sz * cy * transform.scale.x;
			matrix.values[2] = -sy * transform.scale.x;
			matrix.values[4] = (cz * sy * sx - sz * cx) * transform.scale.y;
			matrix.values[5] = (sz * sy * sx + cz * cx) * transform.scale.y;
			matrix.values[6] = cy * sx * transform.scale.y;
			matrix.values[8] = (cz * sy * cx + sz * sx) * transform.scale.z;
			matrix.values[9] = (sz * sy * cx - cz * sx) * transform.scale.z;
			matrix.values[10] = cy * cx * transform.scale.z;
			matrix.values[12] = transform.position.x;
			matrix.values[13] = transform.position.y;
			matrix.values[14] = transform.position.z;
			return matrix;
		}

		Matrix4 Normal(const Transform& transform) {
			Transform rotation = transform;
			rotation.scale = {1.0f, 1.0f, 1.0f};
			Matrix4 matrix = Model(rotation);
			matrix.values[0] /= transform.scale.x;
			matrix.values[1] /= transform.scale.x;
			matrix.values[2] /= transform.scale.x;
			matrix.values[4] /= transform.scale.y;
			matrix.values[5] /= transform.scale.y;
			matrix.values[6] /= transform.scale.y;
			matrix.values[8] /= transform.scale.z;
			matrix.values[9] /= transform.scale.z;
			matrix.values[10] /= transform.scale.z;
			matrix.values[12] = 0.0f;
			matrix.values[13] = 0.0f;
			matrix.values[14] = 0.0f;
			return matrix;
		}

		Matrix4 View(const Transform& transform) {
			const float pitch = transform.rotation.x * Math::degreesToRadians;
			const float yaw = transform.rotation.y * Math::degreesToRadians;
			const float cosPitch = std::cos(pitch);
			const float sinPitch = std::sin(pitch);
			const float cosYaw = std::cos(yaw);
			const float sinYaw = std::sin(yaw);
			const Vector3 right{cosYaw, 0.0f, -sinYaw};
			const Vector3 up{sinYaw * sinPitch, cosPitch, cosYaw * sinPitch};
			const Vector3 forward{sinYaw * cosPitch, -sinPitch, cosYaw * cosPitch};
			const Vector3 position = transform.position;
			Matrix4 matrix = Identity();
			matrix.values[0] = right.x;
			matrix.values[4] = right.y;
			matrix.values[8] = right.z;
			matrix.values[12] = -(right.x * position.x + right.y * position.y + right.z * position.z);
			matrix.values[1] = up.x;
			matrix.values[5] = up.y;
			matrix.values[9] = up.z;
			matrix.values[13] = -(up.x * position.x + up.y * position.y + up.z * position.z);
			matrix.values[2] = forward.x;
			matrix.values[6] = forward.y;
			matrix.values[10] = forward.z;
			matrix.values[14] = -(forward.x * position.x + forward.y * position.y + forward.z * position.z);
			return matrix;
		}

		Matrix4 Perspective(float fieldOfView, float aspectRatio, float nearClip, float farClip) {
			const float scale = 1.0f / std::tan(fieldOfView * Math::degreesToRadians * 0.5f);
			Matrix4 matrix{};
			matrix.values[0] = scale / aspectRatio;
			matrix.values[5] = scale;
			matrix.values[10] = farClip / (farClip - nearClip);
			matrix.values[11] = 1.0f;
			matrix.values[14] = -(nearClip * farClip) / (farClip - nearClip);
			return matrix;
		}

		MeshVertex MakeVertex(Vector3 position, Vector3 normal, float u, float v) {
			return {position, normal, {u, v}};
		}

		void AddFace(MeshData& mesh, const Vector3 (&positions)[4], Vector3 normal) {
			const std::uint32_t first = static_cast<std::uint32_t>(mesh.vertices.size());
			mesh.vertices.push_back(MakeVertex(positions[0], normal, 0.0f, 0.0f));
			mesh.vertices.push_back(MakeVertex(positions[1], normal, 1.0f, 0.0f));
			mesh.vertices.push_back(MakeVertex(positions[2], normal, 1.0f, 1.0f));
			mesh.vertices.push_back(MakeVertex(positions[3], normal, 0.0f, 1.0f));
			mesh.indices.insert(mesh.indices.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
		}

		MeshData CreatePlane() {
			MeshData mesh;
			const Vector3 positions[4] = {{-0.5f, 0.0f, -0.5f}, {-0.5f, 0.0f, 0.5f}, {0.5f, 0.0f, 0.5f}, {0.5f, 0.0f, -0.5f}};
			AddFace(mesh, positions, {0.0f, 1.0f, 0.0f});
			return mesh;
		}

		MeshData CreateCube() {
			MeshData mesh;
			const Vector3 faces[6][4] = {
				{{-0.5f,-0.5f,0.5f}, {0.5f,-0.5f,0.5f}, {0.5f,0.5f,0.5f}, {-0.5f,0.5f,0.5f}},
				{{0.5f,-0.5f,-0.5f}, {-0.5f,-0.5f,-0.5f}, {-0.5f,0.5f,-0.5f}, {0.5f,0.5f,-0.5f}},
				{{0.5f,-0.5f,0.5f}, {0.5f,-0.5f,-0.5f}, {0.5f,0.5f,-0.5f}, {0.5f,0.5f,0.5f}},
				{{-0.5f,-0.5f,-0.5f}, {-0.5f,-0.5f,0.5f}, {-0.5f,0.5f,0.5f}, {-0.5f,0.5f,-0.5f}},
				{{-0.5f,0.5f,0.5f}, {0.5f,0.5f,0.5f}, {0.5f,0.5f,-0.5f}, {-0.5f,0.5f,-0.5f}},
				{{-0.5f,-0.5f,-0.5f}, {0.5f,-0.5f,-0.5f}, {0.5f,-0.5f,0.5f}, {-0.5f,-0.5f,0.5f}}
			};
			const Vector3 normals[6] = {{0,0,1}, {0,0,-1}, {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}};
			for (int face = 0; face < 6; ++face) AddFace(mesh, faces[face], normals[face]);
			return mesh;
		}

		SDL_GPUBuffer* UploadBuffer(SDL_GPUBufferUsageFlags usage, const void* data, Uint32 size) {
			SDL_GPUBuffer* buffer = CreateBuffer(gpuDevice, usage, size);
			SDL_GPUTransferBufferCreateInfo transferInfo{};
			transferInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
			transferInfo.size = size;
			SDL_GPUTransferBuffer* transferBuffer = SDL_CreateGPUTransferBuffer(gpuDevice, &transferInfo);
			void* mappedData = SDL_MapGPUTransferBuffer(gpuDevice, transferBuffer, false);
			std::memcpy(mappedData, data, size);
			SDL_UnmapGPUTransferBuffer(gpuDevice, transferBuffer);
			SDL_GPUCommandBuffer* commandBuffer = SDL_AcquireGPUCommandBuffer(gpuDevice);
			SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(commandBuffer);
			SDL_GPUTransferBufferLocation source{};
			source.transfer_buffer = transferBuffer;
			SDL_GPUBufferRegion destination{};
			destination.buffer = buffer;
			destination.size = size;
			SDL_UploadToGPUBuffer(copyPass, &source, &destination, false);
			SDL_EndGPUCopyPass(copyPass);
			SDL_SubmitGPUCommandBuffer(commandBuffer);
			SDL_ReleaseGPUTransferBuffer(gpuDevice, transferBuffer);
			return buffer;
		}

		GpuMesh UploadMesh(const MeshData& mesh) {
			GpuMesh gpuMesh;
			gpuMesh.vertexBuffer = UploadBuffer(SDL_GPU_BUFFERUSAGE_VERTEX, mesh.vertices.data(), static_cast<Uint32>(mesh.vertices.size() * sizeof(MeshVertex)));
			gpuMesh.indexBuffer = UploadBuffer(SDL_GPU_BUFFERUSAGE_INDEX, mesh.indices.data(), static_cast<Uint32>(mesh.indices.size() * sizeof(std::uint32_t)));
			gpuMesh.indexCount = static_cast<Uint32>(mesh.indices.size());
			return gpuMesh;
		}

		SDL_GPUTexture* CreateDepthTexture(Uint32 width, Uint32 height) {
			SDL_GPUTextureCreateInfo textureInfo{};
			textureInfo.type = SDL_GPU_TEXTURETYPE_2D;
			textureInfo.format = SDL_GPU_TEXTUREFORMAT_D16_UNORM;
			textureInfo.usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET;
			textureInfo.width = width;
			textureInfo.height = height;
			textureInfo.layer_count_or_depth = 1;
			textureInfo.num_levels = 1;
			textureInfo.sample_count = SDL_GPU_SAMPLECOUNT_1;
			return SDL_CreateGPUTexture(gpuDevice, &textureInfo);
		}

		GpuMesh& GetMesh(Primitive primitive) {
			return primitive == Primitive::Plane ? planeMesh : cubeMesh;
		}
	}

	SDL_GPUShader* CreateShader(SDL_GPUDevice* gpuDevice, const char* source, SDL_GPUShaderStage stage, SDL_GPUShaderFormat format, Uint32 samplerCount = 0, Uint32 uniformBufferCount = 0) {
		SDL_GPUShaderCreateInfo shaderInfo{};
		shaderInfo.code_size = std::strlen(source) + 1;
		shaderInfo.code = reinterpret_cast<const Uint8*>(source);
		shaderInfo.entrypoint = "main0";
		shaderInfo.format = format;
		shaderInfo.stage = stage;
		shaderInfo.num_samplers = samplerCount;
		shaderInfo.num_uniform_buffers = uniformBufferCount;
		return SDL_CreateGPUShader(gpuDevice, &shaderInfo);
	}

	SDL_GPUTexture* CreateTexture(SDL_GPUDevice* gpuDevice, int width, int height, const std::uint8_t* pixels) {
		SDL_GPUTextureCreateInfo textureInfo{};
		textureInfo.type = SDL_GPU_TEXTURETYPE_2D;
		textureInfo.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
		textureInfo.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER;
		textureInfo.width = width;
		textureInfo.height = height;
		textureInfo.layer_count_or_depth = 1;
		textureInfo.num_levels = 1;
		textureInfo.sample_count = SDL_GPU_SAMPLECOUNT_1;
		SDL_GPUTexture* texture = SDL_CreateGPUTexture(gpuDevice, &textureInfo);
		const Uint32 pixelBytes = static_cast<Uint32>(width * height * 4);
		SDL_GPUTransferBufferCreateInfo transferInfo{};
		transferInfo.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
		transferInfo.size = pixelBytes;
		SDL_GPUTransferBuffer* transferBuffer = SDL_CreateGPUTransferBuffer(gpuDevice, &transferInfo);
		void* mappedPixels = SDL_MapGPUTransferBuffer(gpuDevice, transferBuffer, false);
		std::memcpy(mappedPixels, pixels, pixelBytes);
		SDL_UnmapGPUTransferBuffer(gpuDevice, transferBuffer);
		SDL_GPUCommandBuffer* commandBuffer = SDL_AcquireGPUCommandBuffer(gpuDevice);
		SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(commandBuffer);
		SDL_GPUTextureTransferInfo source{};
		source.transfer_buffer = transferBuffer;
		source.pixels_per_row = width;
		source.rows_per_layer = height;
		SDL_GPUTextureRegion destination{};
		destination.texture = texture;
		destination.w = width;
		destination.h = height;
		destination.d = 1;
		SDL_UploadToGPUTexture(copyPass, &source, &destination, false);
		SDL_EndGPUCopyPass(copyPass);
		SDL_SubmitGPUCommandBuffer(commandBuffer);
		SDL_ReleaseGPUTransferBuffer(gpuDevice, transferBuffer);
		return texture;
	}

	SDL_GPUBuffer* CreateBuffer(SDL_GPUDevice* gpuDevice, SDL_GPUBufferUsageFlags usage, Uint32 size) {
		SDL_GPUBufferCreateInfo bufferInfo{};
		bufferInfo.usage = usage;
		bufferInfo.size = size;
		return SDL_CreateGPUBuffer(gpuDevice, &bufferInfo);
	}

	SDL_GPUSampler* CreateSampler(SDL_GPUDevice* gpuDevice, const SDL_GPUSamplerCreateInfo& samplerInfo) {
		return SDL_CreateGPUSampler(gpuDevice, &samplerInfo);
	}

	SDL_GPUGraphicsPipeline* CreatePipeline(SDL_GPUDevice* gpuDevice, const SDL_GPUGraphicsPipelineCreateInfo& pipelineInfo) {
		return SDL_CreateGPUGraphicsPipeline(gpuDevice, &pipelineInfo);
	}

	void Initialize(SDL_GPUDevice* device, SDL_Window* window) {
		gpuDevice = device;
		SDL_GPUShader* vertexShader = CreateShader(gpuDevice, vertexShaderSource, SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 0, 1);
		SDL_GPUShader* fragmentShader = CreateShader(gpuDevice, fragmentShaderSource, SDL_GPU_SHADERSTAGE_FRAGMENT, SDL_GPU_SHADERFORMAT_MSL);
		SDL_GPUVertexBufferDescription vertexBufferDescription{};
		vertexBufferDescription.slot = 0;
		vertexBufferDescription.pitch = sizeof(MeshVertex);
		vertexBufferDescription.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;
		SDL_GPUVertexAttribute vertexAttributes[3]{};
		vertexAttributes[0].location = 0;
		vertexAttributes[0].buffer_slot = 0;
		vertexAttributes[0].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3;
		vertexAttributes[0].offset = offsetof(MeshVertex, position);
		vertexAttributes[1].location = 1;
		vertexAttributes[1].buffer_slot = 0;
		vertexAttributes[1].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3;
		vertexAttributes[1].offset = offsetof(MeshVertex, normal);
		vertexAttributes[2].location = 2;
		vertexAttributes[2].buffer_slot = 0;
		vertexAttributes[2].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2;
		vertexAttributes[2].offset = offsetof(MeshVertex, uv);
		SDL_GPUColorTargetDescription colorTarget{};
		colorTarget.format = SDL_GetGPUSwapchainTextureFormat(gpuDevice, window);
		SDL_GPUGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.vertex_shader = vertexShader;
		pipelineInfo.fragment_shader = fragmentShader;
		pipelineInfo.vertex_input_state.num_vertex_buffers = 1;
		pipelineInfo.vertex_input_state.vertex_buffer_descriptions = &vertexBufferDescription;
		pipelineInfo.vertex_input_state.num_vertex_attributes = 3;
		pipelineInfo.vertex_input_state.vertex_attributes = vertexAttributes;
		pipelineInfo.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
		pipelineInfo.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
		pipelineInfo.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
		pipelineInfo.multisample_state.sample_count = SDL_GPU_SAMPLECOUNT_1;
		pipelineInfo.depth_stencil_state.enable_depth_test = true;
		pipelineInfo.depth_stencil_state.enable_depth_write = true;
		pipelineInfo.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_LESS;
		pipelineInfo.target_info.num_color_targets = 1;
		pipelineInfo.target_info.color_target_descriptions = &colorTarget;
		pipelineInfo.target_info.has_depth_stencil_target = true;
		pipelineInfo.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D16_UNORM;
		pipeline = CreatePipeline(gpuDevice, pipelineInfo);
		SDL_ReleaseGPUShader(gpuDevice, vertexShader);
		SDL_ReleaseGPUShader(gpuDevice, fragmentShader);
		cubeMesh = UploadMesh(CreateCube());
		planeMesh = UploadMesh(CreatePlane());
	}

	void Render(Scene& scene, SDL_GPUCommandBuffer* commandBuffer, SDL_GPUTexture* colorTarget, Uint32 width, Uint32 height) {
		if (!colorTarget) return;
		if (depthWidth != width || depthHeight != height) {
			if (depthTexture) SDL_ReleaseGPUTexture(gpuDevice, depthTexture);
			depthTexture = CreateDepthTexture(width, height);
			depthWidth = width;
			depthHeight = height;
		}
		SDL_GPUColorTargetInfo colorInfo{};
		colorInfo.texture = colorTarget;
		colorInfo.clear_color = {0.0f, 0.0f, 0.0f, 1.0f};
		colorInfo.load_op = SDL_GPU_LOADOP_CLEAR;
		colorInfo.store_op = SDL_GPU_STOREOP_STORE;
		SDL_GPUDepthStencilTargetInfo depthInfo{};
		depthInfo.texture = depthTexture;
		depthInfo.clear_depth = 1.0f;
		depthInfo.load_op = SDL_GPU_LOADOP_CLEAR;
		depthInfo.store_op = SDL_GPU_STOREOP_DONT_CARE;
		depthInfo.stencil_load_op = SDL_GPU_LOADOP_DONT_CARE;
		depthInfo.stencil_store_op = SDL_GPU_STOREOP_DONT_CARE;
		SDL_GPURenderPass* renderPass = SDL_BeginGPURenderPass(commandBuffer, &colorInfo, 1, &depthInfo);
		if (scene.activeCamera && scene.activeCamera->owner && width > 0 && height > 0) {
			SDL_BindGPUGraphicsPipeline(renderPass, pipeline);
			const Transform& cameraTransform = scene.activeCamera->owner->transform;
			const Matrix4 view = View(cameraTransform);
			const Matrix4 projection = Perspective(scene.activeCamera->fieldOfView, static_cast<float>(width) / height, scene.activeCamera->nearClip, scene.activeCamera->farClip);
			for (auto& object : scene.Objects()) {
				Mesh* mesh = object->GetComponent<Mesh>();
				if (!mesh) continue;
				GpuMesh& gpuMesh = GetMesh(mesh->primitive);
				const Matrix4 model = Model(object->transform);
				const Matrix4 normal = Normal(object->transform);
				const Matrix4 mvp = Multiply(projection, Multiply(view, model));
				ObjectUniforms uniforms{};
				std::memcpy(uniforms.mvp, mvp.values, sizeof(uniforms.mvp));
				std::memcpy(uniforms.model, model.values, sizeof(uniforms.model));
				std::memcpy(uniforms.normal, normal.values, sizeof(uniforms.normal));
				SDL_PushGPUVertexUniformData(commandBuffer, 0, &uniforms, sizeof(uniforms));
				SDL_GPUBufferBinding vertexBinding{};
				vertexBinding.buffer = gpuMesh.vertexBuffer;
				SDL_BindGPUVertexBuffers(renderPass, 0, &vertexBinding, 1);
				SDL_GPUBufferBinding indexBinding{};
				indexBinding.buffer = gpuMesh.indexBuffer;
				SDL_BindGPUIndexBuffer(renderPass, &indexBinding, SDL_GPU_INDEXELEMENTSIZE_32BIT);
				SDL_DrawGPUIndexedPrimitives(renderPass, gpuMesh.indexCount, 1, 0, 0, 0);
			}
		}
		SDL_EndGPURenderPass(renderPass);
	}

	void Shutdown() {
		if (cubeMesh.vertexBuffer) SDL_ReleaseGPUBuffer(gpuDevice, cubeMesh.vertexBuffer);
		if (cubeMesh.indexBuffer) SDL_ReleaseGPUBuffer(gpuDevice, cubeMesh.indexBuffer);
		if (planeMesh.vertexBuffer) SDL_ReleaseGPUBuffer(gpuDevice, planeMesh.vertexBuffer);
		if (planeMesh.indexBuffer) SDL_ReleaseGPUBuffer(gpuDevice, planeMesh.indexBuffer);
		if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
		if (depthTexture) SDL_ReleaseGPUTexture(gpuDevice, depthTexture);
		cubeMesh = GpuMesh{};
		planeMesh = GpuMesh{};
		pipeline = nullptr;
		depthTexture = nullptr;
		gpuDevice = nullptr;
	}
}
