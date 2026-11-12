#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <SDL3/SDL.h>
#include <SDL3/SDL_gpu.h>
#include SAPPHIRE_SHADER_SOURCE

namespace Sapphire::Graphics {
	SDL_GPUBuffer* CreateBuffer(SDL_GPUDevice* gpuDevice, SDL_GPUBufferUsageFlags usage, Uint32 size);
	SDL_GPUTexture* CreateTexture(SDL_GPUDevice* gpuDevice, int width, int height, const std::uint8_t* pixels);
	SDL_GPUTexture* CreateTexture(SDL_GPUDevice* gpuDevice, int width, int height, const std::uint8_t* pixels, bool srgb);
	SDL_GPUTexture* UploadTexture(SDL_GPUDevice* gpuDevice, int width, int height, const void* pixels, SDL_GPUTextureFormat format, Uint32 stride, Uint32 rowPitch = 0, bool mipmaps = false, bool masked = false, bool blended = false);
	SDL_GPUTexture* LoadTexture(std::string_view fileName, bool srgb, bool mipmaps, bool blended);
	inline bool linearOutput = false;
	SDL_GPUTexture* LoadTexture(std::string_view fileName);
	SDL_GPUTexture* LoadTexture(std::string_view fileName, bool srgb);
	SDL_GPUTexture* LoadTexture(std::string_view fileName, bool srgb, bool mipmaps);

	SDL_GPUTransferBuffer* CreateTransferBuffer(SDL_GPUDevice* device, Uint32 size);
	void ReleaseTransferBuffer(SDL_GPUDevice* device, SDL_GPUTransferBuffer* buffer);
	void ReleaseBuffer(SDL_GPUDevice* device, SDL_GPUBuffer* buffer);
	void ReleaseTexture(SDL_GPUDevice* device, SDL_GPUTexture* texture);
	void FlushUploads();
	void RecordUpload(Uint32 size);

	namespace {
		using namespace Math;
		using namespace Shaders;
		constexpr int shadowMapCount = 3;
		constexpr int maxPointLights = 16;
		constexpr float shadowSlopeFactor = 3.0f;
		constexpr Uint32 fogSliceCount = 64;
		constexpr Uint32 fogMaxGridWidth = 240;
		constexpr Uint32 fogMaxGridHeight = 135;
		constexpr float volumetricFogEndDistanceScale = 0.9f;
		Uint32 shadowMapResolution = 2048;
		using CellKey = std::tuple<int, int, int>;

		struct GpuMesh;

		struct RenderItem {
			Mesh* mesh;
			Skeleton* skeleton = nullptr;
			Matrix4 model;
			Matrix4 normal;
			int cellX = 0;
			int cellY = 0;
			int cellZ = 0;
			Uint32 firstIndex = 0;
			Uint32 indexCount = 0;
			Uint32 materialIndex = 0;
			Bounds bounds;
			mutable GpuMesh* gpu = nullptr;
			mutable Uint32 lightMask = 0;
			mutable Vector3 shadowDirection{};
			mutable Bounds lightBounds{};
			mutable bool lightBoundsReady = false;
			mutable float viewDepth = 0.0f;
			mutable float depthBucket = 0.0f;
		};

		struct ObjectUniforms {
			float mvp[16];
			float model[16];
			float normal[16];
			float displacementParameters[4];
			float uvTransform[4];
			float modifierTime[4];
			float shadowMvp[shadowMapCount][16];
			float shadowState[4];
		};

		struct FragmentUniforms {
			float shadowParameters[shadowMapCount][4];
			float lightingParameters[4];
			float baseColor[4];
			float surfaceParameters[4];
			Uint32 pointLightIndices[4][4];
		};

		struct FrameUniforms {
			float cameraPosition[4];
			float fogColor[4];
			float fogParameters[4];
			float fogHeightParameters[4];
			float fogVolumeParameters[4];
			float fogScreenParameters[4];
			float fogViewParameters[4];
			float cameraRight[4];
			float cameraUp[4];
			float cameraForward[4];
			float ambientLight[4];
			float directionalDirection[4];
			float directionalColor[4];
			float pointLightPositionRange[maxPointLights][4];
			float pointLightColorIntensity[maxPointLights][4];
			float pointLightShadowInfo[maxPointLights][4];
			float pointLightShadowBias[maxPointLights][4];
		};

		struct LightData {
			Vector3 direction;
			Vector3 color;
			float intensity;
			bool castShadows;
			float shadowBias;
			bool renderStatic;
			const Light* source = nullptr;
		};

		struct PointLightData {
			Vector3 position;
			Vector3 color;
			float intensity;
			float range;
			float shadowBias;
			bool castShadows;
			bool renderStatic;
			int staticShadowIndex = -1;
			int dynamicShadowIndex = -1;
			bool staticShadowValid = false;
			bool dynamicShadowValid = false;
			const Light* source = nullptr;
		};

		struct PointShadowState {
			const Light* source = nullptr;
			Vector3 position;
			float range = 0.0f;
			bool renderStatic = false;
			bool staticReady = false;
			bool staticValid = false;
			bool dynamicValid = false;
			bool relevant = false;
			std::vector<const RenderItem*> staticCasters;
			std::vector<const RenderItem*> dynamicCasters;
		};

		struct ShadowMapData {
			Matrix4 mvp;
			float depthRange = 1.0f;
			bool valid = false;
		};

		struct StaticShadowMap {
			SDL_GPUTexture* texture = nullptr;
			ShadowMapData data;
			bool ready = false;
		};

		using MaterialTextures = std::array<std::string, 7>;

		struct GpuMaterial {
			const MaterialTextures* source = nullptr;
			std::array<SDL_GPUTexture*, 6> textures{};
			std::size_t references = 0;
			Uint32 flags = 0;
			bool masked = false;
		};

		struct GpuMeshMaterial {
			GpuMaterial* material = nullptr;
			Material cachedMaterial;
			FragmentUniforms fragment{};
			float uv[4]{};
			float displacementScale = 0.0f;
			bool uniformsReady = false;
		};

		struct GpuMesh {
			SDL_GPUBuffer* vertexBuffer = nullptr;
			SDL_GPUBuffer* indexBuffer = nullptr;
			SDL_GPUBuffer* instanceBuffer = nullptr;
			SDL_GPUBuffer* skinBuffer = nullptr;
			std::vector<GpuMeshMaterial> materials;
			Uint32 indexCount = 0;
			Uint32 vertexCapacity = 0;
			Uint32 indexCapacity = 0;
			Uint32 instanceCapacity = 0;
			Uint32 skinCapacity = 0;
			Uint32 uploadedInstanceCount = 0;
			const MeshInstance* instanceSource = nullptr;
			std::size_t instanceSourceCount = 0;
			std::uint64_t instanceSourceRevision = 0;
			std::uint64_t instanceGeometryRevision = 0;
			bool instanceSourceInstanced = false;
			bool geometryDirty = false;
		};

		struct GpuSkeleton {
			SDL_GPUBuffer* poseBuffer = nullptr;
			Uint32 poseCapacity = 0;
			std::uint64_t poseRevision = 0;
		};

		struct SkinBoundsCache {
			std::uint64_t meshRevision = 0;
			const MeshVertex* vertices = nullptr;
			const SkinInfluence* weights = nullptr;
			const Skeleton* skeleton = nullptr;
			std::uint64_t poseRevision = 0;
			std::vector<Bounds> boneBounds;
			Bounds posedBounds = EmptyBounds();
		};

		struct BufferBindings {
			SDL_GPUBuffer* vertices = nullptr;
			SDL_GPUBuffer* indices = nullptr;

			void Bind(SDL_GPURenderPass* pass, const GpuMesh& mesh) {
				if (vertices != mesh.vertexBuffer) {
					SDL_GPUBufferBinding binding{mesh.vertexBuffer, 0};
					SDL_BindGPUVertexBuffers(pass, 0, &binding, 1);
					vertices = mesh.vertexBuffer;
				}
				if (indices != mesh.indexBuffer) {
					SDL_GPUBufferBinding binding{mesh.indexBuffer, 0};
					SDL_BindGPUIndexBuffer(pass, &binding, SDL_GPU_INDEXELEMENTSIZE_32BIT);
					indices = mesh.indexBuffer;
				}
			}
		};

		struct SkeletonBindings {
			SDL_GPUBuffer* buffer = nullptr;

			void Bind(SDL_GPURenderPass* pass, const GpuSkeleton& skeleton) {
				if (buffer == skeleton.poseBuffer) return;
				SDL_BindGPUVertexStorageBuffers(pass, 0, &skeleton.poseBuffer, 1);
				buffer = skeleton.poseBuffer;
			}
		};

		struct PendingGpuSubmission {
			SDL_GPUFence* fence;
			Uint64 submittedAt;
		};

		SDL_GPUDevice* gpuDevice = nullptr;
		SDL_GPUTextureFormat pipelineColorFormat = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
		SDL_GPUComputePipeline* fogComputePipeline = nullptr;
		SDL_GPUGraphicsPipeline* fogBackgroundPipeline = nullptr;
		SDL_GPUSampler* fogSampler = nullptr;
		SDL_GPUTexture* fogVolumeTexture = nullptr;
		SDL_GPUTexture* emptyFogVolumeTexture = nullptr;
		Uint32 fogGridWidth = 0;
		Uint32 fogGridHeight = 0;
		bool volumetricFogSupported = false;
		SDL_GPUGraphicsPipeline* colliderDebugPipeline = nullptr;
		SDL_GPUBuffer* colliderDebugBuffer = nullptr;
		Uint32 colliderDebugCapacity = 0;
		SDL_GPUGraphicsPipeline* pipelines[2][2]{};
		SDL_GPUGraphicsPipeline* instancePipelines[2][2]{};
		SDL_GPUGraphicsPipeline* transparentPipelines[2]{};
		SDL_GPUGraphicsPipeline* instanceTransparentPipelines[2]{};
		SDL_GPUGraphicsPipeline* shadowPipelines[2][2]{};
		SDL_GPUGraphicsPipeline* skinnedPipelines[2][2]{};
		SDL_GPUGraphicsPipeline* skinnedTransparentPipelines[2]{};
		SDL_GPUGraphicsPipeline* skinnedShadowPipelines[2][2]{};
		struct ModifierPipelines {
			SDL_GPUGraphicsPipeline* main[2][2][2]{};
			SDL_GPUGraphicsPipeline* transparent[2][2]{};
			SDL_GPUGraphicsPipeline* shadow[2][2][2]{};
			SDL_GPUGraphicsPipeline* skinnedMain[2][2]{};
			SDL_GPUGraphicsPipeline* skinnedTransparent[2]{};
			SDL_GPUGraphicsPipeline* skinnedShadow[2][2]{};
		};
		std::map<std::pair<std::string, bool>, ModifierPipelines> modifierPipelines;
		ModifierPipelines& GetModifierPipelines(const std::string& source, bool skinned = false);
		SDL_GPUGraphicsPipeline* instanceShadowPipelines[2][2]{};
		SDL_GPUTexture* depthTexture = nullptr;
		SDL_GPUTexture* emptyShadowTexture = nullptr;
		SDL_GPUTexture* dynamicShadowTexture = nullptr;
		SDL_GPUTexture* fallbackShadowTexture = nullptr;
		SDL_GPUTexture* pointShadowTexture = nullptr;
		Uint32 pointShadowResolution = 0;
		std::vector<std::pair<const Light*, bool>> pointShadowSlots;
		std::vector<PointShadowState> pointShadowStates;
		ShadowMapData fallbackShadowMap;
		bool fallbackShadowReady = false;
		int batchCellSize = -1;
		const Light* shadowLight = nullptr;
		std::map<CellKey, StaticShadowMap> staticShadowMaps;
		std::unordered_map<Mesh*, CellKey> staticBatchCells;
		std::unordered_map<Mesh*, const Object*> staticBatchOwners;
		Vector3 staticShadowDirection{};
		SDL_GPUTexture* whiteTexture = nullptr;
		SDL_GPUSampler* sampler = nullptr;
		SDL_GPUSampler* shadowSampler = nullptr;
		Uint32 depthWidth = 0;
		Uint32 depthHeight = 0;
		float fpsElapsed = 0.0f;
		std::uint32_t fpsFrames = 0;
		std::uint64_t cullingTimeNanoseconds = 0;
		std::uint64_t renderTimeNanoseconds = 0;
		std::uint64_t drawCallCount = 0;
		std::uint64_t sceneTriangleCount = 0;
		std::uint64_t shadowDrawCount = 0;
		std::uint64_t shadowTriangleCount = 0;
		std::uint64_t meshRebuildCount = 0;
		std::uint64_t uploadedBytes = 0;
		std::uint64_t previousUploadedBytes = 0;
		std::uint64_t frameUploadedBytes = 0;
		std::size_t peakUploadBytes = 0;
		struct Allocation {
			std::uint64_t bytes;
			bool texture;
		};
		std::unordered_map<const void*, Allocation> allocations;
		std::uint64_t textureBytes = 0;
		std::uint64_t bufferBytes = 0;
		std::uint64_t peakResourceBytes = 0;
		std::array<float, 120> frameSamples{};
		std::size_t frameSampleCount = 0;
		std::size_t frameSampleIndex = 0;
		float medianFrameTime = 0.0f;
		float slowFrameTime = 0.0f;
		std::uint64_t submitTimeNanoseconds = 0;
		std::uint64_t gpuCompletionLatencyNanoseconds = 0;
		bool hasGpuCompletionLatency = false;
		std::string fpsString = "0";
		std::string triangleCountString = "0/0";
		std::unordered_map<Mesh*, GpuMesh> gpuMeshes;
		std::unordered_map<Skeleton*, GpuSkeleton> gpuSkeletons;
		std::map<MaterialTextures, GpuMaterial> gpuMaterials;
		std::unordered_map<Mesh*, Bounds> meshBounds;
		std::unordered_map<Mesh*, SkinBoundsCache> skinBounds;
				std::unordered_map<std::string, SDL_GPUTexture*> textures;
		std::deque<PendingGpuSubmission> pendingGpuSubmissions;
		std::vector<std::unique_ptr<Mesh>> staticBatches;
		std::vector<RenderItem> staticDraws;
		std::vector<const RenderItem*> renderItems;
		std::vector<const RenderItem*> staticItems;
		std::vector<const RenderItem*> dynamicItems;
		std::vector<const RenderItem*> visibleItems;
		std::vector<const RenderItem*> shadowCasters;
		std::vector<const RenderItem*> faceCasters;
		std::vector<PointLightData> pointLights;
		std::vector<CellKey> visibleCells;
		struct MeshSource {
			Mesh* mesh = nullptr;
			PhysicsBody* body = nullptr;
			std::uint64_t signature = 0;
			std::uint64_t geometrySignature = 0;
			std::uint64_t spatialSignature = 0;
			std::uint64_t boundsSignature = 0;
			std::uint64_t shadowSignature = 0;
			bool batched = false;
			bool staticCaster = false;
			bool seen = false;
			const MeshInstance* instanceData = nullptr;
			std::size_t instanceCount = 0;
			std::uint64_t instanceRevision = 0;
			std::uint64_t instanceGeometryRevision = 0;
			std::set<CellKey> cells;
			std::vector<CellKey> triangleCells;
			std::vector<Vector3> preparedPositions;
			std::vector<RenderItem> draws;
			bool drawReady = false;
			Bounds shadowBounds{};
		};
		std::unordered_map<const Object*, MeshSource> meshSources;
		struct MeshEntry {
			const Object* object;
			MeshSource* source;
		};
		struct LightEntry {
			const Object* object;
			Light* light;
		};
		std::vector<MeshEntry> meshEntries;
		std::vector<LightEntry> lightEntries;
		std::set<CellKey> dirtyCells;
		std::vector<Bounds> shadowChanges;

		std::unordered_map<SDL_GPUTexture*, bool> maskedTextures;
		std::unordered_map<SDL_GPUTexture*, std::size_t> textureReferences;
		SDL_GPUCommandBuffer* uploadCommandBuffer = nullptr;
		SDL_GPUCopyPass* uploadPass = nullptr;
		std::vector<SDL_GPUTransferBuffer*> uploadTransfers;
		std::size_t uploadBytes = 0;


		void TrackAllocation(const void* resource, std::uint64_t bytes, bool texture) {
			if (!resource) return;
			allocations.emplace(resource, Allocation{bytes, texture});
			(texture ? textureBytes : bufferBytes) += bytes;
			peakResourceBytes = std::max(peakResourceBytes, textureBytes + bufferBytes);
		}

		void ReleaseAllocation(const void* resource) {
			const auto allocation = allocations.find(resource);
			if (allocation == allocations.end()) return;
			(allocation->second.texture ? textureBytes : bufferBytes) -= allocation->second.bytes;
			allocations.erase(allocation);
		}

		SDL_GPUTexture* CreateTextureResource(SDL_GPUDevice* device, const SDL_GPUTextureCreateInfo& info) {
			SDL_GPUTexture* texture = SDL_CreateGPUTexture(device, &info);
			if (!texture) throw std::runtime_error(std::string("Create GPU texture: ") + SDL_GetError());
			std::uint64_t bytes = 0;
			for (Uint32 level = 0; level < info.num_levels; ++level) {
				const Uint32 layers = info.type == SDL_GPU_TEXTURETYPE_3D ? std::max(1u, info.layer_count_or_depth >> level) : info.layer_count_or_depth;
				bytes += static_cast<std::uint64_t>(SDL_CalculateGPUTextureFormatSize(info.format, std::max(1u, info.width >> level), std::max(1u, info.height >> level), 1)) * layers;
			}
			TrackAllocation(texture, bytes * (1u << info.sample_count), true);
			return texture;
		}

		SDL_GPUCopyPass* BeginUploads() {
			if (!uploadCommandBuffer) uploadCommandBuffer = SDL_AcquireGPUCommandBuffer(gpuDevice);
			if (!uploadCommandBuffer) throw std::runtime_error(std::string("Acquire upload command buffer: ") + SDL_GetError());
			if (!uploadPass) uploadPass = SDL_BeginGPUCopyPass(uploadCommandBuffer);
			if (!uploadPass) throw std::runtime_error(std::string("Begin upload copy pass: ") + SDL_GetError());
			return uploadPass;
		}

		std::uint64_t Hash(std::uint64_t hash, const void* data, std::size_t size) {
			const auto* bytes = static_cast<const unsigned char*>(data);
			for (std::size_t index = 0; index < size; ++index) hash = (hash ^ bytes[index]) * 1099511628211ull;
			return hash;
		}

		template <typename T>
		void AddHash(std::uint64_t& hash, const T& value) { hash = Hash(hash, &value, sizeof(value)); }

		void AddHash(std::uint64_t& hash, const std::string& value) { hash = Hash(hash, value.c_str(), value.size() + 1); }

		void AddMaterialHash(std::uint64_t& hash, const Material& material) {
			std::apply([&](const auto&... values) { (AddHash(hash, values), ...); }, material.Key());
		}

		void AddMeshMaterialsHash(std::uint64_t& hash, const Mesh& mesh) {
			AddMaterialHash(hash, mesh.material);
			AddHash(hash, mesh.materials.size());
			for (const auto& material : mesh.materials) AddMaterialHash(hash, material);
			AddHash(hash, mesh.sections.size());
			for (const auto& section : mesh.sections) {
				AddHash(hash, section.firstIndex);
				AddHash(hash, section.indexCount);
				AddHash(hash, section.materialIndex);
			}
		}

		bool IsBatchable(const Material& material) { return !material.transparent && material.displacementTexturePath.empty() && material.vertexModifierSource.empty(); }

		bool IsBatchable(const Mesh& mesh) {
			if (mesh.IsSkinned()) return false;
			if (mesh.materials.empty()) return IsBatchable(mesh.material);
			for (const auto& material : mesh.materials) if (!IsBatchable(material)) return false;
			return true;
		}

		std::uint64_t GeometrySignature(const Mesh& mesh) {
			MeshRevision revision;
			revision.Set(mesh);
			std::uint64_t hash = 14695981039346656037ull;
			AddHash(hash, revision.revision);
			AddHash(hash, revision.vertexCount);
			AddHash(hash, revision.indexCount);
			AddHash(hash, revision.vertices);
			AddHash(hash, revision.indices);
			AddHash(hash, revision.skinWeights);
			AddHash(hash, revision.skinWeightCount);
			return hash;
		}

		std::uint64_t MeshSignature(const Object& object, const Mesh& mesh) {
			std::uint64_t hash = 14695981039346656037ull;
			AddHash(hash, object.renderStatic);
			if (object.renderStatic) {
				AddHash(hash, object.transform.position);
				AddHash(hash, object.transform.rotation);
				AddHash(hash, object.transform.scale);
			}
			AddHash(hash, &mesh);
			AddHash(hash, mesh.revision);
			AddHash(hash, mesh.instanced);
			AddHash(hash, mesh.vertexAnimated);
			AddHash(hash, mesh.vertexBoundsPadding);
			AddHash(hash, mesh.castShadows);
			AddHash(hash, mesh.cullBackfaces);
			AddMeshMaterialsHash(hash, mesh);
			AddHash(hash, mesh.data.vertices.size());
			AddHash(hash, mesh.data.indices.size());
			AddHash(hash, mesh.data.vertices.data());
			AddHash(hash, mesh.data.indices.data());
			AddHash(hash, mesh.data.skinWeights.data());
			AddHash(hash, mesh.data.skinWeights.size());
			return hash;
		}

		std::uint64_t BoundsSignature(const Mesh& mesh) {
			std::uint64_t hash = GeometrySignature(mesh);
			AddHash(hash, mesh.instanced);
			AddHash(hash, mesh.instances.data());
			AddHash(hash, mesh.instances.size());
			AddHash(hash, mesh.instanceRevision);
			AddHash(hash, mesh.vertexAnimated);
			AddHash(hash, mesh.vertexBoundsPadding);
			AddMeshMaterialsHash(hash, mesh);
			return hash;
		}

		std::uint64_t SpatialSignature(const Object& object, const Mesh& mesh) {
			std::uint64_t hash = GeometrySignature(mesh);
			AddHash(hash, object.transform.position);
			AddHash(hash, object.transform.rotation);
			AddHash(hash, object.transform.scale);
			return hash;
		}

		std::uint64_t ShadowSignature(const Object& object, const Mesh& mesh) {
			std::uint64_t hash = SpatialSignature(object, mesh);
			AddHash(hash, object.renderStatic);
			AddHash(hash, mesh.castShadows);
			AddHash(hash, mesh.vertexAnimated);
			AddHash(hash, mesh.vertexBoundsPadding);
			AddHash(hash, mesh.cullBackfaces);
			AddMeshMaterialsHash(hash, mesh);
			return hash;
		}

		std::string Milliseconds(double value) {
			std::ostringstream stream;
			stream << std::fixed << std::setprecision(2) << value << " ms";
			return stream.str();
		}

		void UpdateGpuCompletionLatency() {
			while (!pendingGpuSubmissions.empty()) {
				auto& submission = pendingGpuSubmissions.front();
				if (!SDL_QueryGPUFence(gpuDevice, submission.fence)) return;
				gpuCompletionLatencyNanoseconds = SDL_GetTicksNS() - submission.submittedAt;
				hasGpuCompletionLatency = true;
				SDL_ReleaseGPUFence(gpuDevice, submission.fence);
				pendingGpuSubmissions.pop_front();
			}
		}


		Transform CameraWorldTransform(const Camera& camera) {
			Transform transform = camera.owner->transform;
			transform.scale = {1.0f, 1.0f, 1.0f};
			return transform;
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

		Matrix4 ViewDirection(const Vector3& position, const Vector3& direction, const Vector3& upDirection) {
			const Vector3 forward = Normalize(direction);
			const Vector3 right = Normalize(Cross(upDirection, forward));
			const Vector3 up = Cross(forward, right);
			Matrix4 matrix = Identity();
			matrix.values[0] = right.x;
			matrix.values[4] = right.y;
			matrix.values[8] = right.z;
			matrix.values[12] = -Dot(right, position);
			matrix.values[1] = up.x;
			matrix.values[5] = up.y;
			matrix.values[9] = up.z;
			matrix.values[13] = -Dot(up, position);
			matrix.values[2] = forward.x;
			matrix.values[6] = forward.y;
			matrix.values[10] = forward.z;
			matrix.values[14] = -Dot(forward, position);
			return matrix;
		}

		LightData CollectLights() {
			LightData directional{};
			pointLights.clear();
			for (const auto& entry : lightEntries) {
				const Object* object = entry.object;
				Light* light = entry.light;
				if (!light || !light->enabled || light->intensity <= 0.0f) continue;
				if (light->type == Light::Type::Directional && !directional.source) {
					Vector3 axes[3];
					GetAxes(object->transform, axes);
					directional = {Scale(axes[2], -1.0f), light->color, light->intensity, light->castShadows, light->shadowBias, object->renderStatic, light};
				}
				else if (light->type == Light::Type::Point && light->range > 0.0f && pointLights.size() < maxPointLights) {
					PointLightData data{object->transform.position, light->color, light->intensity, light->range, light->shadowBias, light->castShadows, object->renderStatic};
					data.source = light;
					pointLights.push_back(data);
				}
			}
			return directional;
		}

		namespace Culling {
			struct Frustum {
				Vector3 position;
				Vector3 right;
				Vector3 up;
				Vector3 forward;
				float tanHalfFov;
				float aspectRatio;
				float farClip;
				Vector3 planes[6]{};
				float offsets[6]{};
			};

			struct Basic {
				float behindCameraDistance = 1.0f;
				float fovPadding = 1.05f;
			};
			constexpr Basic basic{};

			void PrepareFrustum(Frustum& frustum) {
				const float height = frustum.tanHalfFov * basic.fovPadding;
				const float width = height * frustum.aspectRatio;
				frustum.planes[0] = Add(frustum.right, Scale(frustum.forward, width));
				frustum.planes[1] = Add(Scale(frustum.right, -1.0f), Scale(frustum.forward, width));
				frustum.planes[2] = Add(frustum.up, Scale(frustum.forward, height));
				frustum.planes[3] = Add(Scale(frustum.up, -1.0f), Scale(frustum.forward, height));
				frustum.planes[4] = Scale(frustum.forward, -1.0f);
				frustum.planes[5] = frustum.forward;
				frustum.offsets[4] = frustum.farClip;
				frustum.offsets[5] = basic.behindCameraDistance;
			}

			Frustum FromCamera(const Camera& camera, const Transform& transform, float aspectRatio) {
				Vector3 axes[3];
				GetAxes(transform, axes);
				Frustum frustum{
					transform.position,
				Scale(axes[0], -1.0f),
					axes[2],
					Scale(axes[1], -1.0f),
					std::tan(camera.fieldOfView * Math::degreesToRadians * 0.5f),
					aspectRatio,
					camera.far
				};
				PrepareFrustum(frustum);
				return frustum;
			}

			const Bounds& GetBounds(Mesh* mesh) {
				auto [entry, inserted] = meshBounds.try_emplace(mesh);
				if (inserted) {
					if (mesh->data.vertices.empty()) return entry->second;
										entry->second.minimum = mesh->data.vertices.front().position;
					entry->second.maximum = mesh->data.vertices.front().position;
					for (const auto& vertex : mesh->data.vertices) {
						Include(entry->second, vertex.position);
					}
				if (mesh->instanced) {
					float radius = 0.0f;
					for (const auto& vertex : mesh->data.vertices) radius = std::max(radius, std::sqrt(Dot(vertex.position, vertex.position)));
					bool first = true;
					for (const auto& instance : mesh->instances) {
						const float extent = radius * std::max({std::abs(instance.scale.x), std::abs(instance.scale.y), std::abs(instance.scale.z)});
						const Vector3 minimum = Subtract(instance.position, {extent, extent, extent});
						const Vector3 maximum = Add(instance.position, {extent, extent, extent});
						if (first) {
							entry->second = {minimum, maximum};
							first = false;
						}
						else {
							Include(entry->second, minimum);
							Include(entry->second, maximum);
						}
					}
				}
					entry->second.minimum = Subtract(entry->second.minimum, mesh->vertexBoundsPadding);
					entry->second.maximum = Add(entry->second.maximum, mesh->vertexBoundsPadding);
					float displacementPadding = 0.0f;
					for (std::size_t index = 0; index < mesh->MaterialCount(); ++index) {
						const auto& material = mesh->GetMaterial(index);
						if (!material.displacementTexturePath.empty()) displacementPadding = std::max(displacementPadding, std::abs(material.displacementScale));
					}
					if (displacementPadding > 0.0f) {
						entry->second.minimum.x -= displacementPadding;
						entry->second.minimum.y -= displacementPadding;
						entry->second.minimum.z -= displacementPadding;
						entry->second.maximum.x += displacementPadding;
						entry->second.maximum.y += displacementPadding;
						entry->second.maximum.z += displacementPadding;
					}
				}
				return entry->second;
			}

			bool IsVisible(const Bounds& bounds, const Frustum& frustum) {
				const Vector3 center = Subtract(Scale(Add(bounds.minimum, bounds.maximum), 0.5f), frustum.position);
				const Vector3 extent = Scale(Subtract(bounds.maximum, bounds.minimum), 0.5f);
				for (int index = 0; index < 6; ++index) {
					const Vector3& normal = frustum.planes[index];
					if (Dot(normal, center) + frustum.offsets[index] + std::abs(normal.x) * extent.x + std::abs(normal.y) * extent.y + std::abs(normal.z) * extent.z < 0.0f) return false;
				}
				return true;
			}
		}

		void PointShadowCasters(const std::vector<const RenderItem*>& items, const Vector3& position, float range, std::vector<const RenderItem*>& casters) {
			casters.clear();
			for (const auto* item : items) if (item->mesh->castShadows && IntersectsSphere(item->bounds, position, range)) casters.push_back(item);
		}

		const Vector3 pointDirections[6] = {{1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}};
		const Vector3 pointUps[6] = {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};

		ShadowMapData BuildPointShadowMap(const Vector3& position, float range, int face) {
			const float nearClip = std::min(0.1f, range * 0.01f);
			const Matrix4 projection = Perspective(90.0f, 1.0f, nearClip, range);
			ShadowMapData map{};
			map.mvp = Multiply(projection, ViewDirection(position, pointDirections[face], pointUps[face]));
			map.depthRange = range - nearClip;
			map.valid = true;
			return map;
		}

		Matrix4 DirectionalShadowMatrix(const Vector3& right, const Vector3& up, const Vector3& direction, const Vector3& center, float radius, float minimumDepth, float maximumDepth) {
			Matrix4 matrix{};
			const float depthRange = std::max(maximumDepth - minimumDepth, 0.01f);
			matrix.values[0] = right.x / radius;
			matrix.values[4] = right.y / radius;
			matrix.values[8] = right.z / radius;
			matrix.values[12] = -Dot(right, center) / radius;
			matrix.values[1] = up.x / radius;
			matrix.values[5] = up.y / radius;
			matrix.values[9] = up.z / radius;
			matrix.values[13] = -Dot(up, center) / radius;
			matrix.values[2] = direction.x / depthRange;
			matrix.values[6] = direction.y / depthRange;
			matrix.values[10] = direction.z / depthRange;
			matrix.values[14] = -minimumDepth / depthRange;
			matrix.values[15] = 1.0f;
			return matrix;
		}

		std::array<Vector3, 8> FrustumCorners(const Culling::Frustum& frustum, float nearDepth, float farDepth) {
			std::array<Vector3, 8> corners{};
			int index = 0;
			for (float depth : {nearDepth, farDepth}) {
				const float halfHeight = depth * frustum.tanHalfFov;
				const float halfWidth = halfHeight * frustum.aspectRatio;
				for (int vertical : {-1, 1}) {
					for (int horizontal : {-1, 1}) {
						corners[index++] = Add(Add(frustum.position, Scale(frustum.forward, depth)), Add(Scale(frustum.right, halfWidth * horizontal), Scale(frustum.up, halfHeight * vertical)));
					}
				}
			}
			return corners;
		}

		struct ShadowBasis {
			Vector3 source;
			Vector3 direction;
			Vector3 right;
			Vector3 up;
			bool ready = false;
		};

		const ShadowBasis& GetShadowBasis(const Vector3& direction) {
			static ShadowBasis basis;
			if (!basis.ready || basis.source.x != direction.x || basis.source.y != direction.y || basis.source.z != direction.z) {
				basis.source = direction;
				basis.direction = Normalize(direction);
				const Vector3 referenceUp = std::abs(basis.direction.z) > 0.99f ? Vector3{1.0f, 0.0f, 0.0f} : Vector3{0.0f, 0.0f, 1.0f};
				basis.right = Normalize(Cross(referenceUp, basis.direction));
				basis.up = Normalize(Cross(basis.direction, basis.right));
				basis.ready = true;
			}
			return basis;
		}

		const Bounds& LightBounds(const RenderItem& item, const Vector3& direction, const Vector3& right, const Vector3& up) {
			if (!item.lightBoundsReady || item.shadowDirection.x != direction.x || item.shadowDirection.y != direction.y || item.shadowDirection.z != direction.z) {
				const auto x = ProjectBounds(item.bounds, right);
				const auto y = ProjectBounds(item.bounds, up);
				const auto z = ProjectBounds(item.bounds, direction);
				item.lightBounds = {{x.first, y.first, z.first}, {x.second, y.second, z.second}};
				item.shadowDirection = direction;
				item.lightBoundsReady = true;
			}
			return item.lightBounds;
		}

		ShadowMapData BuildStaticShadowMap(const Vector3& lightDirection, int cellX, int cellY, int cellZ, int cellSize, const std::vector<const RenderItem*>& items, std::vector<const RenderItem*>& casters) {
			ShadowMapData shadowMap{};
			casters.clear();
			if (items.empty()) return shadowMap;
			const auto& basis = GetShadowBasis(lightDirection);
			const Vector3& direction = basis.direction;
			const Vector3& right = basis.right;
			const Vector3& up = basis.up;
			float minimumX = std::numeric_limits<float>::max();
			float maximumX = std::numeric_limits<float>::lowest();
			float minimumY = std::numeric_limits<float>::max();
			float maximumY = std::numeric_limits<float>::lowest();
			float minimumDepth = std::numeric_limits<float>::max();
			float maximumDepth = std::numeric_limits<float>::lowest();
			if (cellSize > 0) {
				for (int corner = 0; corner < 8; ++corner) {
					const Vector3 point{
						(corner & 1 ? cellX + 1 : cellX) * static_cast<float>(cellSize),
						(corner & 2 ? cellY + 1 : cellY) * static_cast<float>(cellSize),
						(corner & 4 ? cellZ + 1 : cellZ) * static_cast<float>(cellSize)
					};
					const float x = Dot(right, point);
					const float y = Dot(up, point);
					const float depth = Dot(direction, point);
					minimumX = std::min(minimumX, x);
					maximumX = std::max(maximumX, x);
					minimumY = std::min(minimumY, y);
					maximumY = std::max(maximumY, y);
					minimumDepth = std::min(minimumDepth, depth);
					maximumDepth = std::max(maximumDepth, depth);
				}
			}
			else {
				for (const auto* item : items) {
					const Bounds& projected = LightBounds(*item, direction, right, up);
					const float minX = projected.minimum.x, maxX = projected.maximum.x;
					const float minY = projected.minimum.y, maxY = projected.maximum.y;
					const float minDepth = projected.minimum.z, maxDepth = projected.maximum.z;
					minimumX = std::min(minimumX, minX);
					maximumX = std::max(maximumX, maxX);
					minimumY = std::min(minimumY, minY);
					maximumY = std::max(maximumY, maxY);
					minimumDepth = std::min(minimumDepth, minDepth);
					maximumDepth = std::max(maximumDepth, maxDepth);
				}
			}
			const float extent = std::max(maximumX - minimumX, maximumY - minimumY);
			const float radius = extent * 0.5f + std::max(1.0f, extent * 4.0f / shadowMapResolution);
			const float texel = radius * 2.0f / shadowMapResolution;
			const float centerX = std::round((minimumX + maximumX) * 0.5f / texel) * texel;
			const float centerY = std::round((minimumY + maximumY) * 0.5f / texel) * texel;
			const Vector3 center = Add(Scale(right, centerX), Scale(up, centerY));
			bool hasCaster = false;
			for (const RenderItem* itemPointer : items) {
				const RenderItem& item = *itemPointer;
				if (!item.mesh->castShadows) continue;
				const Bounds& projected = LightBounds(item, direction, right, up);
				const float itemMinimumX = projected.minimum.x, itemMaximumX = projected.maximum.x;
				const float itemMinimumY = projected.minimum.y, itemMaximumY = projected.maximum.y;
				const float itemMinimumDepth = projected.minimum.z;
				const bool overlaps = !(itemMaximumX < centerX - radius || itemMinimumX > centerX + radius || itemMaximumY < centerY - radius || itemMinimumY > centerY + radius);
				if (!overlaps || itemMinimumDepth > maximumDepth) continue;
				minimumDepth = std::min(minimumDepth, itemMinimumDepth);
				casters.push_back(&item);
				hasCaster = true;
			}
			if (!hasCaster) return shadowMap;
			minimumDepth -= 1.0f;
			maximumDepth += 1.0f;
			shadowMap.depthRange = std::max(maximumDepth - minimumDepth, 0.01f);
			shadowMap.mvp = DirectionalShadowMatrix(right, up, direction, center, radius, minimumDepth, maximumDepth);
			shadowMap.valid = true;
			return shadowMap;
		}

		ShadowMapData BuildDynamicShadowMap(const Camera& camera, const Culling::Frustum& frustum, float shadowFar, const LightData& light, const std::vector<const RenderItem*>& items, std::vector<const RenderItem*>& casters) {
			ShadowMapData map{};
			casters.clear();
			const float near = std::max(camera.near, 0.01f);
			const auto corners = FrustumCorners(frustum, near, std::max(near + 0.1f, std::min(camera.far, shadowFar)));
			const auto& basis = GetShadowBasis(light.direction);
			const Vector3& direction = basis.direction;
			const Vector3& right = basis.right;
			const Vector3& up = basis.up;
			float receiverMinX = std::numeric_limits<float>::max();
			float receiverMaxX = std::numeric_limits<float>::lowest();
			float receiverMinY = receiverMinX;
			float receiverMaxY = receiverMaxX;
			float maximumDepth = receiverMaxX;
			for (const auto& point : corners) {
				receiverMinX = std::min(receiverMinX, Dot(right, point));
				receiverMaxX = std::max(receiverMaxX, Dot(right, point));
				receiverMinY = std::min(receiverMinY, Dot(up, point));
				receiverMaxY = std::max(receiverMaxY, Dot(up, point));
				maximumDepth = std::max(maximumDepth, Dot(direction, point));
			}
			float minimumX = std::numeric_limits<float>::max();
			float maximumX = std::numeric_limits<float>::lowest();
			float minimumY = minimumX;
			float maximumY = maximumX;
			float minimumDepth = minimumX;
			bool hasCaster = false;
			for (const RenderItem* itemPointer : items) {
				const RenderItem& item = *itemPointer;
				if (!item.mesh->castShadows) continue;
				const Bounds& projected = LightBounds(item, direction, right, up);
				const float minX = projected.minimum.x, maxX = projected.maximum.x;
				const float minY = projected.minimum.y, maxY = projected.maximum.y;
				const float depth = projected.minimum.z;
				if (maxX < receiverMinX || minX > receiverMaxX || maxY < receiverMinY || minY > receiverMaxY || depth > maximumDepth) continue;
				casters.push_back(&item);
				minimumX = std::min(minimumX, std::max(minX, receiverMinX));
				maximumX = std::max(maximumX, std::min(maxX, receiverMaxX));
				minimumY = std::min(minimumY, std::max(minY, receiverMinY));
				maximumY = std::max(maximumY, std::min(maxY, receiverMaxY));
				minimumDepth = std::min(minimumDepth, depth);
				hasCaster = true;
			}
			if (!hasCaster) return map;
			const float radius = std::ceil((std::max(maximumX - minimumX, maximumY - minimumY) * 0.5f + 0.25f) * 16.0f) / 16.0f;
			const float texel = radius * 2.0f / shadowMapResolution;
			const float centerX = std::round((minimumX + maximumX) * 0.5f / texel) * texel;
			const float centerY = std::round((minimumY + maximumY) * 0.5f / texel) * texel;
			const Vector3 center = Add(Scale(right, centerX), Scale(up, centerY));
			minimumDepth -= 1.0f;
			maximumDepth += 1.0f;
			map.depthRange = std::max(maximumDepth - minimumDepth, 0.01f);
			map.mvp = DirectionalShadowMatrix(right, up, direction, center, radius, minimumDepth, maximumDepth);
			map.valid = true;
			return map;
		}

		std::array<const std::string*, 6> MaterialPaths(const Material& material) {
			return {&material.baseTexturePath, &material.normalTexturePath, &material.roughnessTexturePath, &material.aoTexturePath, &material.displacementTexturePath, &material.displacementMaskTexturePath};
		}

		GpuMaterial* AcquireMaterial(const Material& material) {
			const auto paths = MaterialPaths(material);
			MaterialTextures key;
			for (std::size_t index = 0; index < paths.size(); ++index) key[index] = *paths[index];
			key[6] = material.transparent ? "blended" : "masked";
			auto [entry, inserted] = gpuMaterials.try_emplace(std::move(key));
			auto& gpu = entry->second;
			if (inserted) {
				gpu.source = &entry->first;
				for (std::size_t index = 0; index < gpu.textures.size(); ++index) {
					SDL_GPUTexture* texture = entry->first[index].empty() ? nullptr : LoadTexture(entry->first[index], index == 0, index < 4, index == 0 && material.transparent);
					gpu.textures[index] = texture ? texture : whiteTexture;
					if (texture && index > 0 && index < 5) gpu.flags |= 1u << (index - 1);
				}
				const auto opacity = maskedTextures.find(gpu.textures[0]);
				gpu.masked = opacity != maskedTextures.end() && opacity->second;
			}
			++gpu.references;
			return &gpu;
		}

		void ReleaseMaterial(GpuMaterial* material) {
			if (!material || --material->references) return;
			for (auto* texture : material->textures) if (texture != whiteTexture) ReleaseTexture(gpuDevice, texture);
			gpuMaterials.erase(*material->source);
		}

		GpuMesh& GetMesh(Mesh* mesh);

		const Bounds& GetSkinnedBounds(Mesh* mesh, Skeleton* skeleton) {
			SkinBoundsCache& cache = skinBounds[mesh];
			if (cache.meshRevision != mesh->revision || cache.vertices != mesh->data.vertices.data() || cache.weights != mesh->data.skinWeights.data() || cache.skeleton != skeleton || cache.boneBounds.size() != skeleton->bones.size()) {
				mesh->ValidateSkeleton(*skeleton);
				cache.boneBounds.assign(skeleton->bones.size(), EmptyBounds());
				const auto& inverseBind = skeleton->InverseBindMatrices();
				for (std::size_t vertexIndex = 0; vertexIndex < mesh->data.vertices.size(); ++vertexIndex) {
					const auto& vertex = mesh->data.vertices[vertexIndex];
					const auto& skin = mesh->data.skinWeights[vertexIndex];
					for (int influence = 0; influence < 4; ++influence) {
						if (!skin.weights[influence]) continue;
						const std::size_t boneIndex = skin.boneIndices[influence];
						Include(cache.boneBounds[boneIndex], TransformPoint(inverseBind[boneIndex], vertex.position));
					}
				}
				cache.meshRevision = mesh->revision;
				cache.vertices = mesh->data.vertices.data();
				cache.weights = mesh->data.skinWeights.data();
				cache.skeleton = skeleton;
				cache.poseRevision = 0;
			}
			if (cache.poseRevision != skeleton->poseRevision) {
				cache.posedBounds = EmptyBounds();
				const auto& poseGlobals = skeleton->PoseGlobalMatrices();
				for (std::size_t index = 0; index < cache.boneBounds.size(); ++index) {
					const Bounds& boneBounds = cache.boneBounds[index];
					if (boneBounds.minimum.x > boneBounds.maximum.x) continue;
					const Bounds posedBoneBounds = TransformBounds(boneBounds, poseGlobals[index]);
					Include(cache.posedBounds, posedBoneBounds.minimum);
					Include(cache.posedBounds, posedBoneBounds.maximum);
				}
				cache.poseRevision = skeleton->poseRevision;
			}
			return cache.posedBounds;
		}

		RenderItem PrepareDraw(Mesh* mesh, const Transform& transform, const CellKey& cell, std::size_t sectionIndex = 0) {
			Vector3 axes[3];
			GetAxes(transform, axes);
			RenderItem item{};
			item.mesh = mesh;
			item.skeleton = mesh->IsSkinned() && mesh->owner ? mesh->owner->GetComponent<Skeleton>() : nullptr;
			item.model = Model(transform, axes);
			item.normal = NormalMatrix(transform, axes);
			item.cellX = std::get<0>(cell);
			item.cellY = std::get<1>(cell);
			item.cellZ = std::get<2>(cell);
			const MeshSection section = mesh->GetSection(sectionIndex);
			item.firstIndex = section.firstIndex;
			item.indexCount = section.indexCount;
			item.materialIndex = section.materialIndex;
			Bounds localBounds = item.skeleton ? GetSkinnedBounds(mesh, item.skeleton) : Culling::GetBounds(mesh);
			if (item.skeleton) {
				localBounds.minimum = Subtract(localBounds.minimum, mesh->vertexBoundsPadding);
				localBounds.maximum = Add(localBounds.maximum, mesh->vertexBoundsPadding);
				float displacementPadding = 0.0f;
				for (std::size_t index = 0; index < mesh->MaterialCount(); ++index) if (!mesh->GetMaterial(index).displacementTexturePath.empty()) displacementPadding = std::max(displacementPadding, std::abs(mesh->GetMaterial(index).displacementScale));
				localBounds.minimum = Subtract(localBounds.minimum, {displacementPadding, displacementPadding, displacementPadding});
				localBounds.maximum = Add(localBounds.maximum, {displacementPadding, displacementPadding, displacementPadding});
			}
			item.bounds = TransformBounds(localBounds, item.model);
			const auto gpu = gpuMeshes.find(mesh);
			item.gpu = gpu == gpuMeshes.end() ? nullptr : &gpu->second;
			return item;
		}

		void ReleaseMesh(Mesh* mesh, bool geometryOnly = false) {
			meshBounds.erase(mesh);
			skinBounds.erase(mesh);
			const auto entry = gpuMeshes.find(mesh);
			if (entry == gpuMeshes.end()) return;
			auto& gpuMesh = entry->second;
			if (geometryOnly) {
				gpuMesh.geometryDirty = true;
				return;
			}
			ReleaseBuffer(gpuDevice, gpuMesh.vertexBuffer);
			ReleaseBuffer(gpuDevice, gpuMesh.indexBuffer);
			ReleaseBuffer(gpuDevice, gpuMesh.instanceBuffer);
			ReleaseBuffer(gpuDevice, gpuMesh.skinBuffer);
			gpuMesh.vertexBuffer = nullptr;
			gpuMesh.indexBuffer = nullptr;
			gpuMesh.skinBuffer = nullptr;
			gpuMesh.indexCount = 0;
			for (auto& material : gpuMesh.materials) ReleaseMaterial(material.material);
			gpuMeshes.erase(entry);
		}

		CellKey TriangleCell(const Vector3& a, const Vector3& b, const Vector3& c, int cellSize) {
			if (cellSize <= 0) return {};
			const Vector3 center = Scale(Add(Add(a, b), c), 1.0f / 3.0f);
			return {static_cast<int>(std::floor(center.x / cellSize)), static_cast<int>(std::floor(center.y / cellSize)), static_cast<int>(std::floor(center.z / cellSize))};
		}

		void PrepareSourceCells(MeshSource& source, const Object& object, int cellSize) {
			const auto& data = source.mesh->data;
			source.preparedPositions.clear();
			source.triangleCells.clear();
			source.cells.clear();
			const Matrix4 model = Model(object.transform);
			source.preparedPositions.reserve(data.vertices.size());
			for (const auto& vertex : data.vertices) source.preparedPositions.push_back(TransformPoint(model, vertex.position));
			source.triangleCells.reserve(data.indices.size() / 3);
			for (std::size_t index = 0; index + 2 < data.indices.size(); index += 3) {
				const auto& positions = source.preparedPositions;
				const CellKey cell = TriangleCell(positions[data.indices[index]], positions[data.indices[index + 1]], positions[data.indices[index + 2]], cellSize);
				source.triangleCells.push_back(cell);
				source.cells.insert(cell);
			}
		}

		void BuildStaticBatches(Scene& scene, bool rebuildAll) {
			++meshRebuildCount;
			std::vector<GpuMaterial*> retainedMaterials;
			for (auto batch = staticBatches.begin(); batch != staticBatches.end();) {
				Mesh* mesh = batch->get();
				if (rebuildAll || dirtyCells.count(staticBatchCells.at(mesh))) {
					const auto gpu = gpuMeshes.find(mesh);
					if (gpu != gpuMeshes.end()) {
						for (auto& material : gpu->second.materials) if (material.material) {
							++material.material->references;
							retainedMaterials.push_back(material.material);
						}
					}
					ReleaseMesh(mesh);
					staticBatchCells.erase(mesh);
					staticBatchOwners.erase(mesh);
					batch = staticBatches.erase(batch);
				}
				else ++batch;
			}
			using GroupKey = std::tuple<Material, bool, bool>;
			using BatchKey = std::tuple<int, int, int, std::size_t>;
			std::map<GroupKey, std::size_t> groups;
			std::vector<GroupKey> groupData;
			using CellKey = std::tuple<int, int, int>;
			std::map<BatchKey, MeshData> batchData;
			std::map<BatchKey, const Object*> batchOwners;
			std::vector<Vector3> positions;
			for (auto* object : scene.ResidentRenderObjects()) {
				if (!object->renderStatic) continue;
				Mesh* mesh = object->GetComponent<Mesh>();
				if (mesh && mesh->instanced) continue;
				if (mesh && !IsBatchable(*mesh)) continue;
				if (!mesh || mesh->data.vertices.empty() || mesh->data.indices.empty()) continue;
				std::map<BatchKey, std::unordered_map<std::uint32_t, std::uint32_t>> vertexMaps;
				if (!rebuildAll) {
					const auto& cells = meshSources.at(object).cells;
					if (std::none_of(cells.begin(), cells.end(), [&](const CellKey& cell) { return dirtyCells.count(cell); })) continue;
				}
				const Matrix4 model = Model(object->transform);
				const Matrix4 normal = NormalMatrix(object->transform);
				auto& source = meshSources.at(object);
				positions.clear();
				if (!source.preparedPositions.empty()) positions.swap(source.preparedPositions);
				else {
					positions.reserve(mesh->data.vertices.size());
					for (const auto& vertex : mesh->data.vertices) positions.push_back(TransformPoint(model, vertex.position));
				}
				std::vector<Vector3>().swap(source.preparedPositions);
				const bool cacheCells = source.triangleCells.size() != mesh->data.indices.size() / 3;
				if (cacheCells) source.triangleCells.reserve(mesh->data.indices.size() / 3);
				for (std::size_t index = 0; index + 2 < mesh->data.indices.size(); index += 3) {
					const std::uint32_t sourceIndices[3] = {mesh->data.indices[index], mesh->data.indices[index + 1], mesh->data.indices[index + 2]};
					const Vector3 worldPositions[3] = {
						positions[sourceIndices[0]],
						positions[sourceIndices[1]],
						positions[sourceIndices[2]]
					};
					const CellKey cellKey = cacheCells ? TriangleCell(worldPositions[0], worldPositions[1], worldPositions[2], scene.cellSize) : source.triangleCells[index / 3];
					if (cacheCells) {
						source.triangleCells.push_back(cellKey);
						source.cells.insert(cellKey);
					}
					const auto [cellX, cellY, cellZ] = cellKey;
					if (!rebuildAll && !dirtyCells.count(cellKey)) continue;
					const std::uint32_t materialIndex = mesh->MaterialIndexAt(index);
					const GroupKey group{mesh->GetMaterial(materialIndex), mesh->cullBackfaces, mesh->castShadows};
					const auto [groupEntry, groupInserted] = groups.try_emplace(group, groupData.size());
					if (groupInserted) groupData.push_back(group);
					const BatchKey key{cellX, cellY, cellZ, groupEntry->second};
					auto [owner, insertedOwner] = batchOwners.try_emplace(key, object);
					if (!insertedOwner && owner->second != object) owner->second = nullptr;
					auto& output = batchData[key];
					auto& vertexMap = vertexMaps[key];
					for (int corner = 0; corner < 3; ++corner) {
						auto [entry, inserted] = vertexMap.try_emplace(sourceIndices[corner], static_cast<std::uint32_t>(output.vertices.size()));
						if (inserted) {
							MeshVertex vertex = mesh->data.vertices[sourceIndices[corner]];
							vertex.position = worldPositions[corner];
							vertex.normal = TransformNormal(normal, vertex.normal);
							output.vertices.push_back(vertex);
						}
						output.indices.push_back(entry->second);
					}
				}
			}
			staticBatches.reserve(staticBatches.size() + batchData.size());
			for (auto& [key, data] : batchData) {
				auto batch = std::make_unique<Mesh>();
				batch->data = std::move(data);
				batch->sections.push_back({0, static_cast<std::uint32_t>(batch->data.indices.size()), 0});
				const auto& group = groupData[std::get<3>(key)];
				batch->material = std::get<0>(group);
				batch->cullBackfaces = std::get<1>(group);
				batch->castShadows = std::get<2>(group);
								const CellKey cellKey{std::get<0>(key), std::get<1>(key), std::get<2>(key)};
				staticBatchCells.emplace(batch.get(), cellKey);
				staticBatchOwners.emplace(batch.get(), batchOwners.at(key));
				Culling::GetBounds(batch.get());
				GetMesh(batch.get());
				batch->data = {};
				staticBatches.push_back(std::move(batch));
			}
			staticDraws.clear();
			staticDraws.reserve(staticBatches.size());
			for (auto& batch : staticBatches) staticDraws.push_back(PrepareDraw(batch.get(), {}, staticBatchCells.at(batch.get())));
			for (auto* material : retainedMaterials) ReleaseMaterial(material);
		}

		SDL_GPUBuffer* UploadBuffer(SDL_GPUBufferUsageFlags usage, const void* data, Uint32 size, SDL_GPUBuffer* existing = nullptr) {
			SDL_GPUBuffer* buffer = existing ? existing : CreateBuffer(gpuDevice, usage, size);
			try {
				if (uploadBytes + size > 32 * 1024 * 1024) FlushUploads();
				uploadBytes += size;
				RecordUpload(size);
				peakUploadBytes = std::max(peakUploadBytes, uploadBytes);
				SDL_GPUTransferBuffer* transferBuffer = CreateTransferBuffer(gpuDevice, size);
				void* mappedData = SDL_MapGPUTransferBuffer(gpuDevice, transferBuffer, false);
				if (!mappedData) {
					ReleaseTransferBuffer(gpuDevice, transferBuffer);
					throw std::runtime_error(std::string("Map buffer transfer: ") + SDL_GetError());
				}
				std::memcpy(mappedData, data, size);
				SDL_UnmapGPUTransferBuffer(gpuDevice, transferBuffer);
				SDL_GPUTransferBufferLocation source{transferBuffer, 0};
				SDL_GPUBufferRegion destination{buffer, 0, size};
				try {
					SDL_UploadToGPUBuffer(BeginUploads(), &source, &destination, existing != nullptr);
				}
				catch (...) {
					ReleaseTransferBuffer(gpuDevice, transferBuffer);
					throw;
				}
				try {
					uploadTransfers.push_back(transferBuffer);
				}
				catch (...) {
					ReleaseTransferBuffer(gpuDevice, transferBuffer);
					throw;
				}
				return buffer;
			}
			catch (...) {
				if (!existing) ReleaseBuffer(gpuDevice, buffer);
				throw;
			}
		}

		GpuMesh UploadMesh(const MeshData& mesh, GpuMesh previous = {}) {
			const auto vertexBytes = mesh.vertices.size() * sizeof(MeshVertex);
			const auto indexBytes = mesh.indices.size() * sizeof(std::uint32_t);
			const auto skinBytes = mesh.skinWeights.size() * sizeof(SkinInfluence);
			mesh.Validate();
			SDL_GPUBuffer* nextVertices = previous.vertexCapacity < vertexBytes ? nullptr : previous.vertexBuffer;
			SDL_GPUBuffer* nextIndices = previous.indexCapacity < indexBytes ? nullptr : previous.indexBuffer;
			SDL_GPUBuffer* nextSkin = skinBytes && previous.skinCapacity >= skinBytes ? previous.skinBuffer : nullptr;
			const bool replaceVertices = !nextVertices;
			const bool replaceIndices = !nextIndices;
			const bool replaceSkin = !nextSkin;
			try {
				nextVertices = UploadBuffer(SDL_GPU_BUFFERUSAGE_VERTEX, mesh.vertices.data(), static_cast<Uint32>(vertexBytes), nextVertices);
				nextIndices = UploadBuffer(SDL_GPU_BUFFERUSAGE_INDEX, mesh.indices.data(), static_cast<Uint32>(indexBytes), nextIndices);
				if (skinBytes) nextSkin = UploadBuffer(SDL_GPU_BUFFERUSAGE_VERTEX, mesh.skinWeights.data(), static_cast<Uint32>(skinBytes), nextSkin);
			}
			catch (...) {
				if (replaceVertices && nextVertices) ReleaseBuffer(gpuDevice, nextVertices);
				if (replaceIndices && nextIndices) ReleaseBuffer(gpuDevice, nextIndices);
				if (replaceSkin && nextSkin) ReleaseBuffer(gpuDevice, nextSkin);
				throw;
			}
			if (replaceVertices) ReleaseBuffer(gpuDevice, previous.vertexBuffer);
			if (replaceIndices) ReleaseBuffer(gpuDevice, previous.indexBuffer);
			if (replaceSkin) ReleaseBuffer(gpuDevice, previous.skinBuffer);
			previous.vertexBuffer = nextVertices;
			previous.indexBuffer = nextIndices;
			previous.skinBuffer = skinBytes ? nextSkin : nullptr;
			previous.vertexCapacity = std::max(previous.vertexCapacity, static_cast<Uint32>(vertexBytes));
			previous.indexCapacity = std::max(previous.indexCapacity, static_cast<Uint32>(indexBytes));
			previous.skinCapacity = skinBytes ? std::max(previous.skinCapacity, static_cast<Uint32>(skinBytes)) : 0;
			previous.indexCount = static_cast<Uint32>(mesh.indices.size());
			previous.geometryDirty = false;
			return previous;
		}

		SDL_GPUTexture* CreateDepthTexture(Uint32 width, Uint32 height) {
			SDL_GPUTextureCreateInfo textureInfo{};
			textureInfo.type = SDL_GPU_TEXTURETYPE_2D;
			textureInfo.format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
			textureInfo.usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET;
			textureInfo.width = width;
			textureInfo.height = height;
			textureInfo.layer_count_or_depth = 1;
			textureInfo.num_levels = 1;
			textureInfo.sample_count = SDL_GPU_SAMPLECOUNT_1;
			return CreateTextureResource(gpuDevice, textureInfo);
		}

		bool EnsureFogVolumeTexture(Uint32 width, Uint32 height) {
			if (fogVolumeTexture && width == fogGridWidth && height == fogGridHeight) return true;
			SDL_GPUTextureCreateInfo textureInfo{};
			textureInfo.type = SDL_GPU_TEXTURETYPE_3D;
			textureInfo.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;
			textureInfo.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;
			textureInfo.width = width;
			textureInfo.height = height;
			textureInfo.layer_count_or_depth = fogSliceCount;
			textureInfo.num_levels = 1;
			textureInfo.sample_count = SDL_GPU_SAMPLECOUNT_1;
			try {
				SDL_GPUTexture* nextTexture = CreateTextureResource(gpuDevice, textureInfo);
				if (fogVolumeTexture) ReleaseTexture(gpuDevice, fogVolumeTexture);
				fogVolumeTexture = nextTexture;
				fogGridWidth = width;
				fogGridHeight = height;
				return true;
			}
			catch (const std::exception& error) {
				SDL_Log("Volumetric fog texture unavailable: %s", error.what());
				volumetricFogSupported = false;
				return false;
			}
		}

		SDL_GPUTexture* CreateShadowTexture(Uint32 resolution) {
			SDL_GPUTextureCreateInfo textureInfo{};
			textureInfo.type = SDL_GPU_TEXTURETYPE_2D;
			textureInfo.format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
			textureInfo.usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;
			textureInfo.width = resolution;
			textureInfo.height = resolution;
			textureInfo.layer_count_or_depth = 1;
			textureInfo.num_levels = 1;
			textureInfo.sample_count = SDL_GPU_SAMPLECOUNT_1;
			return CreateTextureResource(gpuDevice, textureInfo);
		}

		void EnsurePointShadowTexture(bool layoutChanged, SDL_GPUCommandBuffer* commandBuffer, const std::vector<std::pair<const Light*, bool>>& previousSlots) {
			const Uint32 resolution = pointShadowSlots.empty() ? 1 : shadowMapResolution;
			if (!layoutChanged && resolution == pointShadowResolution && pointShadowTexture) return;
			SDL_GPUTexture* previousTexture = pointShadowTexture;
			const bool preserve = previousTexture && pointShadowResolution == resolution;
			pointShadowResolution = resolution;
			for (auto& state : pointShadowStates) {
				if (!preserve || std::find(previousSlots.begin(), previousSlots.end(), std::pair<const Light*, bool>{state.source, false}) == previousSlots.end()) state.staticReady = false;
			}
			SDL_GPUTextureCreateInfo textureInfo{};
			textureInfo.type = SDL_GPU_TEXTURETYPE_2D_ARRAY;
			textureInfo.format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
			textureInfo.usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;
			textureInfo.width = resolution;
			textureInfo.height = resolution;
			textureInfo.layer_count_or_depth = static_cast<Uint32>(std::max<std::size_t>(pointShadowSlots.size(), 1) * 6);
			textureInfo.num_levels = 1;
			textureInfo.sample_count = SDL_GPU_SAMPLECOUNT_1;
			SDL_GPUTexture* nextTexture = CreateTextureResource(gpuDevice, textureInfo);
			if (preserve && layoutChanged) {
				SDL_GPUCopyPass* copy = SDL_BeginGPUCopyPass(commandBuffer);
				if (!copy) {
					ReleaseTexture(gpuDevice, nextTexture);
					throw std::runtime_error(std::string("Begin point shadow copy pass: ") + SDL_GetError());
				}
				for (std::size_t index = 0; index < pointShadowSlots.size(); ++index) {
					if (pointShadowSlots[index].second) continue;
					const auto previous = std::find(previousSlots.begin(), previousSlots.end(), pointShadowSlots[index]);
					if (previous == previousSlots.end()) continue;
					for (Uint32 face = 0; face < 6; ++face) {
						SDL_GPUTextureLocation source{};
						source.texture = previousTexture;
						source.layer = static_cast<Uint32>(previous - previousSlots.begin()) * 6 + face;
						SDL_GPUTextureLocation destination{};
						destination.texture = nextTexture;
						destination.layer = static_cast<Uint32>(index) * 6 + face;
						SDL_CopyGPUTextureToTexture(copy, &source, &destination, resolution, resolution, 1, false);
					}
				}
				SDL_EndGPUCopyPass(copy);
			}
			pointShadowTexture = nextTexture;
			if (previousTexture) ReleaseTexture(gpuDevice, previousTexture);
		}

		void EnsureShadowResolution(int requestedResolution) {
			const Uint32 resolution = static_cast<Uint32>(std::max(requestedResolution, 1));
			if (resolution == shadowMapResolution) return;
			if (dynamicShadowTexture) ReleaseTexture(gpuDevice, dynamicShadowTexture);
			if (fallbackShadowTexture) ReleaseTexture(gpuDevice, fallbackShadowTexture);
			if (pointShadowTexture) ReleaseTexture(gpuDevice, pointShadowTexture);
			pointShadowTexture = nullptr;
			pointShadowResolution = 0;
			pointShadowSlots.clear();
			pointShadowStates.clear();
			for (auto& [key, map] : staticShadowMaps) if (map.texture) ReleaseTexture(gpuDevice, map.texture);
			staticShadowMaps.clear();
			shadowMapResolution = resolution;
			dynamicShadowTexture = nullptr;
			fallbackShadowTexture = nullptr;
			fallbackShadowMap = {};
			fallbackShadowReady = false;
		}

		void EnsureDirectionalShadowTextures(bool enabled, bool staticEnabled) {
			if (enabled) {
				if (!dynamicShadowTexture) dynamicShadowTexture = CreateShadowTexture(shadowMapResolution);
			}
			else if (dynamicShadowTexture) {
				ReleaseTexture(gpuDevice, dynamicShadowTexture);
				dynamicShadowTexture = nullptr;
			}
			if (staticEnabled) {
				if (!fallbackShadowTexture) fallbackShadowTexture = CreateShadowTexture(shadowMapResolution);
			}
			else {
				if (fallbackShadowTexture) ReleaseTexture(gpuDevice, fallbackShadowTexture);
				fallbackShadowTexture = nullptr;
				for (auto& [key, map] : staticShadowMaps) if (map.texture) ReleaseTexture(gpuDevice, map.texture);
				staticShadowMaps.clear();
				fallbackShadowMap = {};
				fallbackShadowReady = false;
				shadowLight = nullptr;
			}
		}

		bool MatchesMaterial(const GpuMaterial* gpu, const Material& material) {
			if (!gpu || !gpu->source) return false;
			const auto paths = MaterialPaths(material);
			for (std::size_t index = 0; index < paths.size(); ++index) if (*paths[index] != (*gpu->source)[index]) return false;
			return (*gpu->source)[6] == (material.transparent ? "blended" : "masked");
		}

		void PrepareMaterial(GpuMeshMaterial& gpu, const Material& material) {
			if (gpu.uniformsReady && gpu.cachedMaterial.Key() == material.Key()) return;
			gpu.cachedMaterial = material;
			gpu.uniformsReady = true;
			Uint32 materialFlags = gpu.material->flags;
			if (material.normalScale <= 0.0f) materialFlags &= ~1u;
			if (material.aoStrength <= 0.0f) materialFlags &= ~4u;
			if (material.displacementScale == 0.0f) materialFlags &= ~8u;
			if (material.unlit) materialFlags |= 16u;
			gpu.fragment.lightingParameters[0] = static_cast<float>(materialFlags);
			gpu.fragment.lightingParameters[1] = std::clamp(material.opacity, 0.0f, 1.0f);
			gpu.fragment.baseColor[0] = material.baseColor.x;
			gpu.fragment.baseColor[1] = material.baseColor.y;
			gpu.fragment.baseColor[2] = material.baseColor.z;
			gpu.fragment.baseColor[3] = std::clamp(material.metallic, 0.0f, 1.0f);
			gpu.fragment.surfaceParameters[0] = std::clamp(material.roughness, 0.0f, 1.0f);
			gpu.fragment.surfaceParameters[1] = std::max(material.normalScale, 0.0f);
			gpu.fragment.surfaceParameters[2] = std::clamp(material.aoStrength, 0.0f, 1.0f);
			gpu.fragment.surfaceParameters[3] = material.flipNormalY ? -1.0f : 1.0f;
			gpu.uv[0] = material.uvScale[0];
			gpu.uv[1] = material.uvScale[1];
			gpu.uv[2] = material.uvOffset[0];
			gpu.uv[3] = material.uvOffset[1];
			gpu.displacementScale = (gpu.material->flags & 8u) ? material.displacementScale : 0.0f;
		}

		GpuMesh& GetMesh(Mesh* mesh) {
			GpuMesh& gpuMesh = gpuMeshes.try_emplace(mesh).first->second;
			const bool uploadGeometry = !gpuMesh.vertexBuffer || gpuMesh.geometryDirty;
			if (uploadGeometry && !mesh->data.vertices.empty() && !mesh->data.indices.empty()) {
				mesh->Validate();
				if (mesh->IsSkinned()) {
					if (!mesh->owner || !mesh->owner->GetComponent<Skeleton>()) throw std::invalid_argument("Skinned meshes require a skeleton on the same object");
					mesh->ValidateSkeleton(*mesh->owner->GetComponent<Skeleton>());
				}
				auto uploaded = UploadMesh(mesh->data, gpuMesh);
				gpuMesh.vertexCapacity = uploaded.vertexCapacity;
				gpuMesh.indexCapacity = uploaded.indexCapacity;
				gpuMesh.skinCapacity = uploaded.skinCapacity;
				gpuMesh.geometryDirty = false;
				gpuMesh.vertexBuffer = uploaded.vertexBuffer;
				gpuMesh.indexBuffer = uploaded.indexBuffer;
				gpuMesh.indexCount = uploaded.indexCount;
			}
			const bool uploadInstances = gpuMesh.instanceSource != mesh->instances.data() || gpuMesh.instanceSourceCount != mesh->instances.size() || gpuMesh.instanceSourceRevision != mesh->instanceRevision || gpuMesh.instanceGeometryRevision != mesh->revision || gpuMesh.instanceSourceInstanced != mesh->instanced || !gpuMesh.instanceBuffer;
			if (!mesh->instanced || mesh->instances.empty()) {
				gpuMesh.uploadedInstanceCount = 0;
				gpuMesh.instanceSourceInstanced = mesh->instanced;
			}
			if (mesh->instanced && !mesh->instances.empty() && uploadInstances) {
				if (mesh->instances.size() > std::numeric_limits<Uint32>::max() / sizeof(MeshInstance)) throw std::length_error("Instances exceed GPU buffer size");
				const Uint32 size = static_cast<Uint32>(mesh->instances.size() * sizeof(MeshInstance));
				SDL_GPUBuffer* previous = gpuMesh.instanceBuffer;
				SDL_GPUBuffer* reusable = gpuMesh.instanceCapacity >= size ? previous : nullptr;
				SDL_GPUBuffer* next = UploadBuffer(SDL_GPU_BUFFERUSAGE_VERTEX, mesh->instances.data(), size, reusable);
				if (!reusable) ReleaseBuffer(gpuDevice, previous);
				gpuMesh.instanceBuffer = next;
				gpuMesh.instanceCapacity = std::max(gpuMesh.instanceCapacity, size);
				gpuMesh.instanceSource = mesh->instances.data();
				gpuMesh.instanceSourceCount = mesh->instances.size();
				gpuMesh.instanceSourceRevision = mesh->instanceRevision;
				gpuMesh.instanceGeometryRevision = mesh->revision;
				gpuMesh.instanceSourceInstanced = true;
				gpuMesh.uploadedInstanceCount = static_cast<Uint32>(mesh->instances.size());
			}
			const std::size_t materialCount = mesh->MaterialCount();
			while (gpuMesh.materials.size() > materialCount) {
				ReleaseMaterial(gpuMesh.materials.back().material);
				gpuMesh.materials.pop_back();
			}
			gpuMesh.materials.resize(materialCount);
			for (std::size_t index = 0; index < materialCount; ++index) {
				auto& gpuMaterial = gpuMesh.materials[index];
				const auto& material = mesh->GetMaterial(index);
				if (!MatchesMaterial(gpuMaterial.material, material)) {
					GpuMaterial* next = AcquireMaterial(material);
					ReleaseMaterial(gpuMaterial.material);
					gpuMaterial.material = next;
					gpuMaterial.uniformsReady = false;
				}
				PrepareMaterial(gpuMaterial, material);
			}
			return gpuMesh;
		}

		GpuSkeleton& GetSkeleton(Skeleton* skeleton) {
			GpuSkeleton& gpuSkeleton = gpuSkeletons[skeleton];
			if (gpuSkeleton.poseRevision == skeleton->poseRevision && gpuSkeleton.poseBuffer) return gpuSkeleton;
			const auto& matrices = skeleton->PoseMatrices();
			if (matrices.empty()) throw std::invalid_argument("Skinned meshes require a skeleton with bones");
			const Uint64 bytes = matrices.size() * sizeof(Math::Matrix4);
			if (bytes > std::numeric_limits<Uint32>::max()) throw std::length_error("Skeleton pose exceeds GPU buffer size");
			const Uint32 size = static_cast<Uint32>(bytes);
			SDL_GPUBuffer* reusable = gpuSkeleton.poseCapacity >= size ? gpuSkeleton.poseBuffer : nullptr;
			SDL_GPUBuffer* next = UploadBuffer(SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ, matrices.data(), size, reusable);
			if (!reusable) ReleaseBuffer(gpuDevice, gpuSkeleton.poseBuffer);
			gpuSkeleton.poseBuffer = next;
			gpuSkeleton.poseCapacity = std::max(gpuSkeleton.poseCapacity, size);
			gpuSkeleton.poseRevision = skeleton->poseRevision;
			return gpuSkeleton;
		}

		bool UpdateBatchMaterial(const Object& object, const Mesh& mesh, const MeshSource& source) {
			if (!mesh.materials.empty() || !mesh.sections.empty()) return false;
			bool found = false;
			for (const auto& batch : staticBatches) {
				if (!source.cells.count(staticBatchCells.at(batch.get()))) continue;
				if (staticBatchOwners.at(batch.get()) != &object) return false;
				found = true;
			}
			if (!found) return false;
			for (auto& batch : staticBatches) {
				if (staticBatchOwners.at(batch.get()) != &object) continue;
				batch->material = mesh.material;
				GetMesh(batch.get());
			}
			return true;
		}

		MeshSource* UpdateMeshSource(Object& object, int cellSize, bool rebuildAll) {
			Mesh* mesh = object.GetComponent<Mesh>();
			if (!mesh) return nullptr;
			if (mesh->IsSkinned()) {
				if (mesh->instanced) throw std::invalid_argument("Skinned meshes cannot be instanced");
				Skeleton* skeleton = object.GetComponent<Skeleton>();
				if (!skeleton) throw std::invalid_argument("Skinned meshes require a skeleton on the same object");
				mesh->ValidateSkeleton(*skeleton);
			}
			sceneTriangleCount += mesh->data.indices.size() / 3 * (mesh->instanced ? mesh->instances.size() : 1);
			auto [entry, inserted] = meshSources.try_emplace(&object);
			auto& source = entry->second;
			source.body = object.GetComponent<PhysicsBody>();
			source.seen = true;
			const bool instancesChanged = inserted || source.instanceData != mesh->instances.data() || source.instanceCount != mesh->instances.size() || source.instanceRevision != mesh->instanceRevision || source.instanceGeometryRevision != mesh->revision;
			if (instancesChanged) {
				if (source.staticCaster) shadowChanges.push_back(source.shadowBounds);
				meshBounds.erase(mesh);
				source.drawReady = false;
				source.instanceData = mesh->instances.data();
				source.instanceCount = mesh->instances.size();
				source.instanceRevision = mesh->instanceRevision;
				source.instanceGeometryRevision = mesh->revision;
			}
			const auto signature = MeshSignature(object, *mesh);
			if (!inserted && source.signature == signature && !instancesChanged && (!rebuildAll || !source.batched)) return &source;
			if (mesh->materials.empty()) {
				if (!mesh->material.vertexModifierSource.empty()) GetModifierPipelines(mesh->material.vertexModifierSource, mesh->IsSkinned());
			}
			else for (const auto& material : mesh->materials) if (!material.vertexModifierSource.empty()) GetModifierPipelines(material.vertexModifierSource, mesh->IsSkinned());
			const auto spatialSignature = SpatialSignature(object, *mesh);
			const auto shadowSignature = ShadowSignature(object, *mesh);
			const bool shadowChanged = inserted || source.shadowSignature != shadowSignature || instancesChanged;
			const bool spatialChanged = inserted || source.spatialSignature != spatialSignature;
			if (!rebuildAll && !shadowChanged && !spatialChanged && source.batched && IsBatchable(*mesh) && UpdateBatchMaterial(object, *mesh, source)) {
				source.signature = signature;
				return &source;
			}
			dirtyCells.insert(source.cells.begin(), source.cells.end());
			if (shadowChanged && source.staticCaster) shadowChanges.push_back(source.shadowBounds);
			const bool batched = object.renderStatic && !mesh->instanced && IsBatchable(*mesh);
			const auto geometrySignature = GeometrySignature(*mesh);
			if (inserted || source.geometrySignature != geometrySignature) mesh->Validate();
			else if (source.signature != signature) mesh->ValidateSections();
			if (source.mesh && (source.mesh != mesh || source.batched != batched)) ReleaseMesh(source.mesh);
			else if (!batched && source.geometrySignature != geometrySignature) ReleaseMesh(mesh, true);
			const auto boundsSignature = BoundsSignature(*mesh);
			if (source.boundsSignature != boundsSignature) {
				meshBounds.erase(mesh);
					source.boundsSignature = boundsSignature;
			}
			source.geometrySignature = geometrySignature;
			source.spatialSignature = spatialSignature;
			source.shadowSignature = shadowSignature;
			source.mesh = mesh;
			source.signature = signature;
			source.drawReady = false;
			source.batched = batched;
			source.staticCaster = object.renderStatic && mesh->castShadows && !mesh->vertexAnimated && !mesh->IsSkinned() && !mesh->data.vertices.empty() && !mesh->data.indices.empty();
			if (source.staticCaster) {
				source.shadowBounds = TransformBounds(Culling::GetBounds(mesh), Model(object.transform));
				if (shadowChanged) shadowChanges.push_back(source.shadowBounds);
			}
			if (spatialChanged || rebuildAll || !batched) {
				source.cells.clear();
				source.triangleCells.clear();
				source.preparedPositions.clear();
			}
			if (batched && !rebuildAll && !mesh->data.indices.empty() && source.triangleCells.empty()) {
				PrepareSourceCells(source, object, cellSize);
				dirtyCells.insert(source.cells.begin(), source.cells.end());
			}
			return &source;
		}

		void InvalidateShadows(bool rebuildAll) {
			if (!rebuildAll && shadowChanges.empty()) return;
			fallbackShadowReady = false;
			for (auto& [key, map] : staticShadowMaps) {
				if (rebuildAll || !map.data.valid) map.ready = false;
				else {
					for (const auto& bounds : shadowChanges) {
						const Bounds projected = TransformBounds(bounds, map.data.mvp);
						if (projected.maximum.x < -1.0f || projected.minimum.x > 1.0f || projected.maximum.y < -1.0f || projected.minimum.y > 1.0f) continue;
						map.ready = false;
						break;
					}
				}
			}
			for (auto& state : pointShadowStates) {
				if (rebuildAll || std::any_of(shadowChanges.begin(), shadowChanges.end(), [&](const Bounds& bounds) { return IntersectsSphere(bounds, state.position, state.range); })) state.staticReady = false;
			}
		}

		void UpdateSceneResources(Scene& scene) {
			scene.UpdateWorldPartition();
			const bool rebuildAll = batchCellSize != scene.cellSize;
			shadowChanges.clear();
			dirtyCells.clear();
			sceneTriangleCount = 0;
			for (auto& [object, source] : meshSources) source.seen = false;
			meshEntries.clear();
			lightEntries.clear();
			for (auto* object : scene.ResidentRenderObjects()) {
				if (auto* source = UpdateMeshSource(*object, scene.cellSize, rebuildAll)) meshEntries.push_back({object, source});
			}
			for (auto* object : scene.ResidentObjects()) if (auto* light = object->GetComponent<Light>()) lightEntries.push_back({object, light});
			for (auto entry = meshSources.begin(); entry != meshSources.end();) {
				if (!entry->second.seen) {
					dirtyCells.insert(entry->second.cells.begin(), entry->second.cells.end());
					if (entry->second.staticCaster) shadowChanges.push_back(entry->second.shadowBounds);
					ReleaseMesh(entry->second.mesh);
					entry = meshSources.erase(entry);
				}
				else ++entry;
			}
			if (rebuildAll || !dirtyCells.empty()) BuildStaticBatches(scene, rebuildAll);
			batchCellSize = scene.cellSize;
			InvalidateShadows(rebuildAll);
		}

		void RenderShadowMap(SDL_GPUCommandBuffer* commandBuffer, SDL_GPUTexture* texture, const ShadowMapData& shadowMap, const std::vector<const RenderItem*>& items, float time, Uint8 layer = 0, bool cullBackfaces = true) {
			for (const auto* item : items) {
				item->gpu = &GetMesh(item->mesh);
				if (item->skeleton) GetSkeleton(item->skeleton);
			}
			FlushUploads();
			SDL_GPUDepthStencilTargetInfo depthInfo{};
			depthInfo.texture = texture;
			depthInfo.layer = layer;
			depthInfo.clear_depth = 1.0f;
			depthInfo.load_op = SDL_GPU_LOADOP_CLEAR;
			depthInfo.store_op = SDL_GPU_STOREOP_STORE;
			depthInfo.stencil_load_op = SDL_GPU_LOADOP_DONT_CARE;
			depthInfo.stencil_store_op = SDL_GPU_STOREOP_DONT_CARE;
			SDL_GPURenderPass* shadowPass = SDL_BeginGPURenderPass(commandBuffer, nullptr, 0, &depthInfo);
			if (!shadowPass) throw std::runtime_error(std::string("Begin shadow render pass: ") + SDL_GetError());
			if (!shadowMap.valid) {
				SDL_EndGPURenderPass(shadowPass);
				return;
			}
			SDL_GPUGraphicsPipeline* activePipeline = nullptr;
			GpuMaterial* activeMaterial = nullptr;
			BufferBindings buffers;
			SkeletonBindings skeletonBindings;
			for (const RenderItem* itemPointer : items) {
				const RenderItem& item = *itemPointer;
				Mesh* mesh = item.mesh;
				if (!mesh->castShadows || !item.gpu || !item.gpu->indexCount || (mesh->instanced && !item.gpu->uploadedInstanceCount)) continue;
				GpuMesh& gpuMesh = *item.gpu;
				const Material& material = mesh->GetMaterial(item.materialIndex);
				GpuMeshMaterial& gpuMeshMaterial = gpuMesh.materials.at(item.materialIndex);
				GpuMaterial* activeGpuMaterial = gpuMeshMaterial.material;
				SDL_GPUGraphicsPipeline* meshPipeline = nullptr;
				if (item.skeleton) meshPipeline = material.vertexModifierSource.empty() ? skinnedShadowPipelines[activeGpuMaterial->masked ? 1 : 0][cullBackfaces && mesh->cullBackfaces ? 1 : 0] : GetModifierPipelines(material.vertexModifierSource, true).skinnedShadow[activeGpuMaterial->masked ? 1 : 0][cullBackfaces && mesh->cullBackfaces ? 1 : 0];
				else meshPipeline = material.vertexModifierSource.empty() ? (mesh->instanced ? instanceShadowPipelines : shadowPipelines)[activeGpuMaterial->masked ? 1 : 0][cullBackfaces && mesh->cullBackfaces ? 1 : 0] : GetModifierPipelines(material.vertexModifierSource).shadow[mesh->instanced ? 1 : 0][activeGpuMaterial->masked ? 1 : 0][cullBackfaces && mesh->cullBackfaces ? 1 : 0];
				if (meshPipeline != activePipeline) {
					SDL_BindGPUGraphicsPipeline(shadowPass, meshPipeline);
					activePipeline = meshPipeline;
				}
				struct ShadowUniforms {
					float mvp[16];
					float model[16];
					float displacementParameters[4];
					float uvTransform[4];
					float modifierTime[4];
				} uniforms{};
				const Matrix4 mvp = Multiply(shadowMap.mvp, item.model);
				std::memcpy(uniforms.mvp, mvp.values, sizeof(uniforms.mvp));
				std::memcpy(uniforms.model, item.model.values, sizeof(uniforms.model));
				uniforms.displacementParameters[0] = gpuMeshMaterial.displacementScale;
				std::memcpy(uniforms.uvTransform, gpuMeshMaterial.uv, sizeof(uniforms.uvTransform));
				uniforms.modifierTime[0] = time;
				SDL_PushGPUVertexUniformData(commandBuffer, 0, &uniforms, sizeof(uniforms));
				buffers.Bind(shadowPass, gpuMesh);
				if (item.skeleton) {
					SDL_GPUBufferBinding skinBinding{gpuMesh.skinBuffer, 0};
					SDL_BindGPUVertexBuffers(shadowPass, 1, &skinBinding, 1);
					skeletonBindings.Bind(shadowPass, GetSkeleton(item.skeleton));
				}
				else if (mesh->instanced) {
					SDL_GPUBufferBinding instanceBinding{gpuMesh.instanceBuffer, 0};
					SDL_BindGPUVertexBuffers(shadowPass, 1, &instanceBinding, 1);
				}
				SDL_GPUTextureSamplerBinding textureBinding{};
				textureBinding.texture = activeGpuMaterial->textures[0];
				textureBinding.sampler = sampler;
				if (activeMaterial != activeGpuMaterial && activeGpuMaterial->masked) SDL_BindGPUFragmentSamplers(shadowPass, 0, &textureBinding, 1);
				SDL_GPUTextureSamplerBinding vertexTextureBindings[2]{};
				vertexTextureBindings[0].texture = activeGpuMaterial->textures[4];
				vertexTextureBindings[0].sampler = sampler;
				vertexTextureBindings[1].texture = activeGpuMaterial->textures[5];
				vertexTextureBindings[1].sampler = sampler;
				if (activeMaterial != activeGpuMaterial) SDL_BindGPUVertexSamplers(shadowPass, 0, vertexTextureBindings, 2);
				activeMaterial = activeGpuMaterial;
				SDL_DrawGPUIndexedPrimitives(shadowPass, item.indexCount, mesh->instanced ? gpuMesh.uploadedInstanceCount : 1, item.firstIndex, 0, 0);
				++drawCallCount;
				++shadowDrawCount;
				shadowTriangleCount += item.indexCount / 3 * (mesh->instanced ? mesh->instances.size() : 1);
			}
			SDL_EndGPURenderPass(shadowPass);
		}

		void RenderPointShadowMap(SDL_GPUCommandBuffer* commandBuffer, int slot, const PointLightData& light, const std::vector<const RenderItem*>& casters, float time) {
			for (int face = 0; face < 6; ++face) {
				faceCasters.clear();
				Culling::Frustum frustum{light.position, Cross(pointUps[face], pointDirections[face]), pointUps[face], pointDirections[face], 1.0f, 1.0f, light.range};
				Culling::PrepareFrustum(frustum);
				for (const auto* item : casters) if (Culling::IsVisible(item->bounds, frustum)) faceCasters.push_back(item);
				RenderShadowMap(commandBuffer, pointShadowTexture, BuildPointShadowMap(light.position, light.range, face), faceCasters, time, static_cast<Uint8>(slot * 6 + face), false);
			}
		}

		void RenderPointShadows(Scene& scene, SDL_GPUCommandBuffer* commandBuffer, float time) {
			auto activeLight = [&](const Light* source) {
				return std::find_if(pointLights.begin(), pointLights.end(), [&](const PointLightData& light) { return scene.shadowsEnabled && light.castShadows && light.source == source; });
			};
			const auto previousSlots = pointShadowSlots;
			pointShadowSlots.erase(std::remove_if(pointShadowSlots.begin(), pointShadowSlots.end(), [&](const auto& slot) {
				const auto light = activeLight(slot.first);
				return light == pointLights.end() || (!slot.second && !light->renderStatic);
			}), pointShadowSlots.end());
			pointShadowStates.erase(std::remove_if(pointShadowStates.begin(), pointShadowStates.end(), [&](const PointShadowState& state) { return activeLight(state.source) == pointLights.end(); }), pointShadowStates.end());
			for (auto& light : pointLights) {
				if (!scene.shadowsEnabled || !light.castShadows) continue;
				auto state = std::find_if(pointShadowStates.begin(), pointShadowStates.end(), [&](const PointShadowState& value) { return value.source == light.source; });
				if (state == pointShadowStates.end()) {
					pointShadowStates.emplace_back();
					state = pointShadowStates.end() - 1;
					state->source = light.source;
				}
				if (state->position.x != light.position.x || state->position.y != light.position.y || state->position.z != light.position.z || state->range != light.range || state->renderStatic != light.renderStatic) {
					state->position = light.position;
					state->range = light.range;
					state->renderStatic = light.renderStatic;
					state->staticReady = false;
					state->staticValid = false;
				}
				state->relevant = std::any_of(visibleItems.begin(), visibleItems.end(), [&](const RenderItem* item) { return !item->mesh->GetMaterial(item->materialIndex).unlit && (item->lightMask & (1u << static_cast<Uint32>(&light - pointLights.data()))) != 0; });
				state->dynamicValid = false;
				state->dynamicCasters.clear();
				if (!state->relevant) continue;
				if (light.renderStatic && !state->staticReady) {
					PointShadowCasters(staticItems, light.position, light.range, state->staticCasters);
					state->staticValid = !state->staticCasters.empty();
				}
				PointShadowCasters(light.renderStatic ? dynamicItems : renderItems, light.position, light.range, state->dynamicCasters);
				state->dynamicValid = !state->dynamicCasters.empty();
				for (bool dynamic : {false, true}) {
					if (dynamic ? !state->dynamicValid : !light.renderStatic || !state->staticValid) continue;
					const std::pair<const Light*, bool> key{light.source, dynamic};
					if (std::find(pointShadowSlots.begin(), pointShadowSlots.end(), key) == pointShadowSlots.end()) {
						pointShadowSlots.push_back(key);
					}
				}
			}
			pointShadowSlots.erase(std::remove_if(pointShadowSlots.begin(), pointShadowSlots.end(), [&](const auto& slot) {
				const auto state = std::find_if(pointShadowStates.begin(), pointShadowStates.end(), [&](const PointShadowState& value) { return value.source == slot.first; });
				return state == pointShadowStates.end() || !state->relevant || (slot.second ? !state->dynamicValid : !state->staticValid);
			}), pointShadowSlots.end());
			const bool layoutChanged = previousSlots != pointShadowSlots;
			EnsurePointShadowTexture(layoutChanged, commandBuffer, previousSlots);
			for (auto& light : pointLights) {
				if (!scene.shadowsEnabled || !light.castShadows) continue;
				auto state = std::find_if(pointShadowStates.begin(), pointShadowStates.end(), [&](const PointShadowState& value) { return value.source == light.source; });
				if (state == pointShadowStates.end() || !state->relevant) continue;
				auto slotIndex = [&](bool dynamic) {
					const auto slot = std::find(pointShadowSlots.begin(), pointShadowSlots.end(), std::pair<const Light*, bool>{light.source, dynamic});
					return slot == pointShadowSlots.end() ? -1 : static_cast<int>(slot - pointShadowSlots.begin());
				};
				light.staticShadowIndex = slotIndex(false);
				light.dynamicShadowIndex = slotIndex(true);
				if (light.renderStatic && light.staticShadowIndex >= 0 && !state->staticReady) {
					if (state->staticCasters.empty()) PointShadowCasters(staticItems, light.position, light.range, state->staticCasters);
					state->staticValid = !state->staticCasters.empty();
					if (state->staticValid) RenderPointShadowMap(commandBuffer, light.staticShadowIndex, light, state->staticCasters, time);
					state->staticReady = true;
					state->staticCasters.clear();
				}
				if (state->dynamicValid && light.dynamicShadowIndex >= 0) RenderPointShadowMap(commandBuffer, light.dynamicShadowIndex, light, state->dynamicCasters, time);
				light.staticShadowValid = light.renderStatic && state->staticReady && state->staticValid;
				light.dynamicShadowValid = state->dynamicValid && light.dynamicShadowIndex >= 0;
			}
		}

	}

	SDL_GPUShader* CreateShader(SDL_GPUDevice* gpuDevice, const char* source, SDL_GPUShaderStage stage, SDL_GPUShaderFormat format, Uint32 samplerCount = 0, Uint32 uniformBufferCount = 0, Uint32 storageBufferCount = 0) {
		SDL_GPUShaderCreateInfo shaderInfo{};
		shaderInfo.code_size = std::strlen(source) + 1;
		shaderInfo.code = reinterpret_cast<const Uint8*>(source);
		shaderInfo.entrypoint = "main0";
		shaderInfo.format = format;
		shaderInfo.stage = stage;
		shaderInfo.num_samplers = samplerCount;
		shaderInfo.num_uniform_buffers = uniformBufferCount;
		shaderInfo.num_storage_buffers = storageBufferCount;
		return SDL_CreateGPUShader(gpuDevice, &shaderInfo);
	}

	SDL_GPUTexture* UploadTexture(SDL_GPUDevice* gpuDevice, int width, int height, const void* pixels, SDL_GPUTextureFormat format, Uint32 stride, Uint32 rowPitch, bool mipmaps, bool masked, bool blended) {
		if (width <= 0 || height <= 0 || !pixels || !stride) throw std::invalid_argument("Invalid texture upload");
		const std::uint64_t checkedRowBytes = static_cast<std::uint64_t>(width) * stride;
		if (SDL_CalculateGPUTextureFormatSize(format, static_cast<Uint32>(width), 1, 1) != checkedRowBytes) throw std::invalid_argument("Texture stride does not match format");
		if (checkedRowBytes > std::numeric_limits<Uint32>::max() || (rowPitch && rowPitch < checkedRowBytes)) throw std::length_error("Texture row exceeds GPU upload size");
		if (mipmaps && (masked || blended) && (stride != 4 || (format != SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM && format != SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB))) throw std::invalid_argument("CPU alpha mipmaps require RGBA pixels");
		const bool generateOnGpu = mipmaps && !masked && !blended;
		SDL_GPUTextureCreateInfo textureInfo{};
		textureInfo.type = SDL_GPU_TEXTURETYPE_2D;
		textureInfo.format = format;
		textureInfo.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER | (generateOnGpu ? SDL_GPU_TEXTUREUSAGE_COLOR_TARGET : 0);
		textureInfo.width = width;
		textureInfo.height = height;
		textureInfo.layer_count_or_depth = 1;
		textureInfo.num_levels = 1;
		if (mipmaps) for (int size = std::max(width, height); size > 1; size /= 2) ++textureInfo.num_levels;
		textureInfo.sample_count = SDL_GPU_SAMPLECOUNT_1;
		const Uint32 uploadLevels = generateOnGpu ? 1 : textureInfo.num_levels;
		const Uint32 rowBytes = static_cast<Uint32>(checkedRowBytes);
		if (!rowPitch) rowPitch = rowBytes;
		std::uint64_t checkedTotalBytes = 0;
		for (Uint32 level = 0; level < uploadLevels; ++level) {
			checkedTotalBytes = (checkedTotalBytes + 255u) & ~std::uint64_t{255};
			checkedTotalBytes += static_cast<std::uint64_t>(std::max(1, width >> level)) * std::max(1, height >> level) * stride;
			if (checkedTotalBytes > std::numeric_limits<Uint32>::max()) throw std::length_error("Texture exceeds GPU upload size");
		}
		const Uint32 totalBytes = static_cast<Uint32>(checkedTotalBytes);
		SDL_GPUTexture* texture = CreateTextureResource(gpuDevice, textureInfo);
		struct UploadResources {
			SDL_GPUDevice* device;
			SDL_GPUTexture* texture;
			SDL_GPUTransferBuffer* transfer = nullptr;
			~UploadResources() {
				if (transfer) ReleaseTransferBuffer(device, transfer);
				if (texture) ReleaseTexture(device, texture);
			}
		};
		UploadResources resources{gpuDevice, texture};
		if (uploadBytes + totalBytes > 32 * 1024 * 1024) FlushUploads();
		uploadBytes += totalBytes;
		RecordUpload(totalBytes);
		peakUploadBytes = std::max(peakUploadBytes, uploadBytes);
		SDL_GPUTransferBuffer* transferBuffer = CreateTransferBuffer(gpuDevice, totalBytes);
		resources.transfer = transferBuffer;
		auto* mappedPixels = static_cast<std::uint8_t*>(SDL_MapGPUTransferBuffer(gpuDevice, transferBuffer, false));
		if (!mappedPixels) throw std::runtime_error(std::string("Map texture transfer buffer: ") + SDL_GetError());
		for (int row = 0; row < height; ++row) std::memcpy(mappedPixels + static_cast<std::size_t>(row) * rowBytes, static_cast<const std::uint8_t*>(pixels) + static_cast<std::size_t>(row) * rowPitch, rowBytes);
		static const auto linearColors = [] {
			std::array<float, 256> colors{};
			for (int value = 0; value < 256; ++value) {
				const float color = value / 255.0f;
				colors[value] = color <= 0.04045f ? color / 12.92f : std::pow((color + 0.055f) / 1.055f, 2.4f);
			}
			return colors;
		}();
		static const auto encodedColors = [] {
			std::array<std::uint8_t, 4096> colors{};
			for (int value = 0; value < 4096; ++value) {
				const float color = value / 4095.0f;
				colors[value] = static_cast<std::uint8_t>(std::round(255.0f * (color <= 0.0031308f ? color * 12.92f : 1.055f * std::pow(color, 1.0f / 2.4f) - 0.055f)));
			}
			return colors;
		}();
		float targetCoverage = 0.0f;
		if (masked && mipmaps) {
			for (std::size_t pixel = 0; pixel < static_cast<std::size_t>(width) * height; ++pixel) if (mappedPixels[pixel * 4 + 3] >= 128) targetCoverage += 1.0f;
			targetCoverage /= static_cast<float>(static_cast<std::uint64_t>(width) * height);
		}
		std::uint64_t offset = 0;
		std::uint64_t previousOffset = 0;
		for (Uint32 level = 0; level < uploadLevels; ++level) {
			offset = (offset + 255u) & ~std::uint64_t{255};
			const int levelWidth = std::max(1, width >> level);
			const int levelHeight = std::max(1, height >> level);
			auto* destinationPixels = mappedPixels + offset;
			if (level > 0) {
				const int previousWidth = std::max(1, width >> (level - 1));
				const int previousHeight = std::max(1, height >> (level - 1));
				const auto* sourcePixels = mappedPixels + previousOffset;
				std::array<Uint32, 256> histogram{};
				for (int y = 0; y < levelHeight; ++y) {
					for (int x = 0; x < levelWidth; ++x) {
						float colors[3]{};
						float alpha = 0.0f;
						int samples = 0;
						for (int sampleY = static_cast<std::int64_t>(y) * previousHeight / levelHeight; sampleY < (static_cast<std::int64_t>(y) + 1) * previousHeight / levelHeight; ++sampleY) {
							for (int sampleX = static_cast<std::int64_t>(x) * previousWidth / levelWidth; sampleX < (static_cast<std::int64_t>(x) + 1) * previousWidth / levelWidth; ++sampleX) {
								const auto* pixel = sourcePixels + (static_cast<std::size_t>(sampleY) * previousWidth + sampleX) * 4;
								for (int channel = 0; channel < 3; ++channel) colors[channel] += linearColors[pixel[channel]] * pixel[3];
								alpha += pixel[3];
								++samples;
							}
						}
						auto* pixel = destinationPixels + (static_cast<std::size_t>(y) * levelWidth + x) * 4;
						for (int channel = 0; channel < 3; ++channel) pixel[channel] = encodedColors[std::clamp(static_cast<int>(std::round(alpha > 0.0f ? colors[channel] / alpha * 4095.0f : 0.0f)), 0, 4095)];
						pixel[3] = static_cast<std::uint8_t>(std::round(alpha / samples));
						++histogram[pixel[3]];
					}
				}
				if (masked) {
					float low = 0.0f, high = 256.0f, bestScale = 1.0f, bestError = std::numeric_limits<float>::max();
					for (int iteration = 0; iteration < 14; ++iteration) {
						const float scale = iteration == 0 ? 1.0f : (low + high) * 0.5f;
						Uint32 covered = 0;
						for (int alpha = 1; alpha < 256; ++alpha) if (alpha * scale >= 127.5f) covered += histogram[alpha];
						const float coverage = static_cast<float>(covered) / (static_cast<std::uint64_t>(levelWidth) * levelHeight);
						const float error = std::abs(coverage - targetCoverage);
						if (error < bestError) {
							bestError = error;
							bestScale = scale;
						}
						if (coverage < targetCoverage) low = scale;
						else high = scale;
					}
					for (std::size_t pixel = 0; pixel < static_cast<std::size_t>(levelWidth) * levelHeight; ++pixel) destinationPixels[pixel * 4 + 3] = static_cast<std::uint8_t>(std::min(255.0f, std::round(destinationPixels[pixel * 4 + 3] * bestScale)));
				}
			}
			previousOffset = offset;
			offset += static_cast<std::uint64_t>(levelWidth) * levelHeight * stride;
		}
		SDL_UnmapGPUTransferBuffer(gpuDevice, transferBuffer);
		offset = 0;
		for (Uint32 level = 0; level < uploadLevels; ++level) {
			offset = (offset + 255u) & ~std::uint64_t{255};
			SDL_GPUTextureTransferInfo source{};
			source.transfer_buffer = transferBuffer;
			source.offset = static_cast<Uint32>(offset);
			source.pixels_per_row = std::max(1, width >> level);
			source.rows_per_layer = std::max(1, height >> level);
			SDL_GPUTextureRegion destination{};
			destination.texture = texture;
			destination.mip_level = level;
			destination.w = source.pixels_per_row;
			destination.h = source.rows_per_layer;
			destination.d = 1;
			SDL_UploadToGPUTexture(BeginUploads(), &source, &destination, false);
			offset += static_cast<std::uint64_t>(destination.w) * destination.h * stride;
		}
		uploadTransfers.push_back(transferBuffer);
		resources.transfer = nullptr;
		if (generateOnGpu && textureInfo.num_levels > 1) {
			SDL_EndGPUCopyPass(uploadPass);
			uploadPass = nullptr;
			SDL_GenerateMipmapsForGPUTexture(uploadCommandBuffer, texture);
		}
		resources.texture = nullptr;
		return texture;
	}

	SDL_GPUTexture* CreateTexture(SDL_GPUDevice* gpuDevice, int width, int height, const std::uint8_t* pixels) {
		return CreateTexture(gpuDevice, width, height, pixels, false);
	}

	SDL_GPUTexture* CreateTexture(SDL_GPUDevice* gpuDevice, int width, int height, const std::uint8_t* pixels, bool srgb) {
		return UploadTexture(gpuDevice, width, height, pixels, srgb ? SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB : SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, 4);
	}

	std::filesystem::path ResourcePath(std::string_view fileName) {
		return (std::filesystem::path(SDL_GetBasePath()) / ".." / "resources" / std::string(fileName)).lexically_normal();
	}

	SDL_GPUTexture* LoadTexture(std::string_view fileName) { return LoadTexture(fileName, false); }

	SDL_GPUTexture* LoadTexture(std::string_view fileName, bool srgb) { return LoadTexture(fileName, srgb, srgb); }

	SDL_GPUTexture* LoadTexture(std::string_view fileName, bool srgb, bool mipmaps) { return LoadTexture(fileName, srgb, mipmaps, false); }

	SDL_GPUTexture* LoadTexture(std::string_view fileName, bool srgb, bool mipmaps, bool blended) {
		const std::string path = ResourcePath(fileName).string();
		const std::string key = std::string(srgb ? "color:" : "data:") + (mipmaps ? "mips:" : "single:") + (blended ? "blended:" : "masked:") + path;
		if (auto found = textures.find(key); found != textures.end()) {
			++textureReferences[found->second];
			return found->second;
		}
		SDL_Surface* loadedSurface = SDL_LoadSurface(path.c_str());
		if (!loadedSurface) return nullptr;
		SDL_Surface* rgbaSurface = SDL_ConvertSurface(loadedSurface, SDL_PIXELFORMAT_RGBA32);
		SDL_DestroySurface(loadedSurface);
		if (!rgbaSurface) return nullptr;
		if (rgbaSurface->pitch <= 0) {
			SDL_DestroySurface(rgbaSurface);
			throw std::invalid_argument("Invalid texture surface pitch");
		}
		bool masked = false;
		for (int row = 0; row < rgbaSurface->h && !masked; ++row) {
			const auto* pixels = static_cast<const std::uint8_t*>(rgbaSurface->pixels) + row * rgbaSurface->pitch;
			for (int column = 0; column < rgbaSurface->w; ++column) if (pixels[column * 4 + 3] < 128) {
				masked = true;
				break;
			}
		}
		SDL_GPUTexture* texture = nullptr;
		try { texture = UploadTexture(gpuDevice, rgbaSurface->w, rgbaSurface->h, rgbaSurface->pixels, srgb ? SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB : SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, 4, static_cast<Uint32>(rgbaSurface->pitch), mipmaps, srgb && masked && !blended, srgb && blended); }
		catch (...) { SDL_DestroySurface(rgbaSurface); throw; }
		SDL_DestroySurface(rgbaSurface);
		if (texture) {
			textures[key] = texture;
			textureReferences[texture] = 1;
			maskedTextures[texture] = masked;
		}
		return texture;
	}

	void ReleaseTexture(SDL_GPUDevice* device, SDL_GPUTexture* texture) {
		if (!texture) return;
		if (auto found = textureReferences.find(texture); found != textureReferences.end()) {
			if (--found->second) return;
			textureReferences.erase(found);
			for (auto entry = textures.begin(); entry != textures.end(); ++entry) {
				if (entry->second == texture) {
					textures.erase(entry);
					break;
				}
			}
		}
		maskedTextures.erase(texture);
		ReleaseAllocation(texture);
		SDL_ReleaseGPUTexture(device, texture);
	}

	void ReleaseBuffer(SDL_GPUDevice* device, SDL_GPUBuffer* buffer) {
		if (!buffer) return;
		ReleaseAllocation(buffer);
		SDL_ReleaseGPUBuffer(device, buffer);
	}

	SDL_GPUTransferBuffer* CreateTransferBuffer(SDL_GPUDevice* device, Uint32 size) {
		SDL_GPUTransferBufferCreateInfo info{};
		info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
		info.size = size;
		SDL_GPUTransferBuffer* buffer = SDL_CreateGPUTransferBuffer(device, &info);
		if (!buffer) throw std::runtime_error(std::string("Create GPU transfer buffer: ") + SDL_GetError());
		return buffer;
	}

	void ReleaseTransferBuffer(SDL_GPUDevice* device, SDL_GPUTransferBuffer* buffer) {
		if (!buffer) return;
		SDL_ReleaseGPUTransferBuffer(device, buffer);
	}

	void RecordUpload(Uint32 size) { uploadedBytes += size; }

	void FlushUploads() {
		if (!uploadCommandBuffer) return;
		if (uploadPass) SDL_EndGPUCopyPass(uploadPass);
		const bool submitted = SDL_SubmitGPUCommandBuffer(uploadCommandBuffer);
		uploadCommandBuffer = nullptr;
		uploadPass = nullptr;
		for (auto* buffer : uploadTransfers) ReleaseTransferBuffer(gpuDevice, buffer);
		uploadTransfers.clear();
		uploadBytes = 0;
		if (!submitted) throw std::runtime_error(std::string("Submit upload command buffer: ") + SDL_GetError());
	}

	SDL_GPUBuffer* CreateBuffer(SDL_GPUDevice* gpuDevice, SDL_GPUBufferUsageFlags usage, Uint32 size) {
		SDL_GPUBufferCreateInfo bufferInfo{};
		bufferInfo.usage = usage;
		bufferInfo.size = size;
		SDL_GPUBuffer* buffer = SDL_CreateGPUBuffer(gpuDevice, &bufferInfo);
		if (!buffer) throw std::runtime_error(std::string("Create GPU buffer: ") + SDL_GetError());
		TrackAllocation(buffer, size, false);
		return buffer;
	}

	SDL_GPUSampler* CreateSampler(SDL_GPUDevice* gpuDevice, const SDL_GPUSamplerCreateInfo& samplerInfo) {
		auto* sampler = SDL_CreateGPUSampler(gpuDevice, &samplerInfo);
		if (!sampler) throw std::runtime_error(std::string("Create GPU sampler: ") + SDL_GetError());
		return sampler;
	}

	SDL_GPUGraphicsPipeline* CreatePipeline(SDL_GPUDevice* gpuDevice, const SDL_GPUGraphicsPipelineCreateInfo& pipelineInfo) {
		auto* pipeline = SDL_CreateGPUGraphicsPipeline(gpuDevice, &pipelineInfo);
		if (!pipeline) throw std::runtime_error(std::string("Create GPU pipeline: ") + SDL_GetError());
		return pipeline;
	}

	void PrepareFogBackgroundPipeline() {
		if (fogBackgroundPipeline) return;
		SDL_GPUShader* vertex = CreateShader(gpuDevice, fogBackgroundVertexShaderSource.c_str(), SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL);
		SDL_GPUShader* fragment = CreateShader(gpuDevice, fogBackgroundFragmentShaderSource.c_str(), SDL_GPU_SHADERSTAGE_FRAGMENT, SDL_GPU_SHADERFORMAT_MSL, 1, 2);
		if (!vertex || !fragment) {
			if (vertex) SDL_ReleaseGPUShader(gpuDevice, vertex);
			if (fragment) SDL_ReleaseGPUShader(gpuDevice, fragment);
			throw std::runtime_error(std::string("Create volumetric fog background shaders: ") + SDL_GetError());
		}
		SDL_GPUColorTargetDescription colorTarget{};
		colorTarget.format = pipelineColorFormat;
		SDL_GPUGraphicsPipelineCreateInfo info{};
		info.vertex_shader = vertex;
		info.fragment_shader = fragment;
		info.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
		info.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
		info.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
		info.multisample_state.sample_count = SDL_GPU_SAMPLECOUNT_1;
		info.target_info.num_color_targets = 1;
		info.target_info.color_target_descriptions = &colorTarget;
		info.target_info.has_depth_stencil_target = true;
		info.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
		try { fogBackgroundPipeline = CreatePipeline(gpuDevice, info); }
		catch (...) {
			SDL_ReleaseGPUShader(gpuDevice, vertex);
			SDL_ReleaseGPUShader(gpuDevice, fragment);
			throw;
		}
		SDL_ReleaseGPUShader(gpuDevice, vertex);
		SDL_ReleaseGPUShader(gpuDevice, fragment);
	}

	void PrepareColliderDebugPipeline() {
		if (colliderDebugPipeline) return;
		SDL_GPUShader* vertex = CreateShader(gpuDevice, vertexSource, SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 0, 1);
		SDL_GPUShader* fragment = CreateShader(gpuDevice, fragmentSource, SDL_GPU_SHADERSTAGE_FRAGMENT, SDL_GPU_SHADERFORMAT_MSL);
		if (!vertex || !fragment) throw std::runtime_error(std::string("Create collider debug shader: ") + SDL_GetError());
		SDL_GPUVertexBufferDescription bufferDescription{0, sizeof(Vector3), SDL_GPU_VERTEXINPUTRATE_VERTEX, 0};
		SDL_GPUVertexAttribute attribute{0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, 0};
		SDL_GPUColorTargetDescription colorTarget{};
		colorTarget.format = pipelineColorFormat;
		SDL_GPUGraphicsPipelineCreateInfo info{};
		info.vertex_shader = vertex;
		info.fragment_shader = fragment;
		info.vertex_input_state.vertex_buffer_descriptions = &bufferDescription;
		info.vertex_input_state.num_vertex_buffers = 1;
		info.vertex_input_state.vertex_attributes = &attribute;
		info.vertex_input_state.num_vertex_attributes = 1;
		info.primitive_type = SDL_GPU_PRIMITIVETYPE_LINELIST;
		info.target_info.color_target_descriptions = &colorTarget;
		info.target_info.num_color_targets = 1;
		info.target_info.has_depth_stencil_target = true;
		info.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
		colliderDebugPipeline = CreatePipeline(gpuDevice, info);
		SDL_ReleaseGPUShader(gpuDevice, vertex);
		SDL_ReleaseGPUShader(gpuDevice, fragment);
	}

	namespace {
	ModifierPipelines& GetModifierPipelines(const std::string& source, bool skinned) {
		auto [entry, inserted] = modifierPipelines.try_emplace(std::make_pair(source, skinned));
		if (!inserted) return entry->second;
		auto& family = entry->second;
		SDL_GPUShader* vertex[2]{};
		SDL_GPUShader* shadowVertex[2]{};
		SDL_GPUShader* skinnedVertex = nullptr;
		SDL_GPUShader* skinnedShadowVertex = nullptr;
		SDL_GPUShader* fragment[2]{};
		SDL_GPUShader* shadowFragment[2]{};
		auto releaseShaders = [&] {
			for (auto* shader : vertex) if (shader) SDL_ReleaseGPUShader(gpuDevice, shader);
			for (auto* shader : shadowVertex) if (shader) SDL_ReleaseGPUShader(gpuDevice, shader);
			if (skinnedVertex) SDL_ReleaseGPUShader(gpuDevice, skinnedVertex);
			if (skinnedShadowVertex) SDL_ReleaseGPUShader(gpuDevice, skinnedShadowVertex);
			for (auto* shader : fragment) if (shader) SDL_ReleaseGPUShader(gpuDevice, shader);
			for (auto* shader : shadowFragment) if (shader) SDL_ReleaseGPUShader(gpuDevice, shader);
		};
		try {
			for (int instanced = 0; instanced < 2; ++instanced) {
				const std::string prefix = std::string(instanced ? "#define INSTANCED 1\n" : "") + "#define CUSTOM_MODIFIER 1\n";
				const std::string mainSource = prefix + deformationShaderSource + source + vertexShaderSource.substr(deformationShaderSource.size());
				const std::string shadowSource = prefix + deformationShaderSource + source + shadowVertexShaderSource.substr(deformationShaderSource.size());
				vertex[instanced] = CreateShader(gpuDevice, mainSource.c_str(), SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 2, 1);
				shadowVertex[instanced] = CreateShader(gpuDevice, shadowSource.c_str(), SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 2, 1);
				if (!vertex[instanced] || !shadowVertex[instanced]) throw std::runtime_error(std::string("Compile vertex modifier: ") + SDL_GetError());
			}
			if (skinned) {
				const std::string skinnedPrefix = "#define SKINNED 1\n#define CUSTOM_MODIFIER 1\n";
				const std::string skinnedMainSource = skinnedPrefix + deformationShaderSource + source + vertexShaderSource.substr(deformationShaderSource.size());
				const std::string skinnedShadowSource = skinnedPrefix + deformationShaderSource + source + shadowVertexShaderSource.substr(deformationShaderSource.size());
				skinnedVertex = CreateShader(gpuDevice, skinnedMainSource.c_str(), SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 2, 1, 1);
				skinnedShadowVertex = CreateShader(gpuDevice, skinnedShadowSource.c_str(), SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 2, 1, 1);
				if (!skinnedVertex || !skinnedShadowVertex) throw std::runtime_error(std::string("Compile skinned vertex modifier: ") + SDL_GetError());
			}
			for (int masked = 0; masked < 2; ++masked) {
				const std::string prefix = masked ? "#define ALPHA_MASKED 1\n" : "#define ALPHA_MASKED 0\n";
				fragment[masked] = CreateShader(gpuDevice, (prefix + fragmentShaderSource).c_str(), SDL_GPU_SHADERSTAGE_FRAGMENT, SDL_GPU_SHADERFORMAT_MSL, 9, 2);
				shadowFragment[masked] = CreateShader(gpuDevice, (prefix + shadowFragmentShaderSource).c_str(), SDL_GPU_SHADERSTAGE_FRAGMENT, SDL_GPU_SHADERFORMAT_MSL, masked ? 1 : 0);
				if (!fragment[masked] || !shadowFragment[masked]) throw std::runtime_error(std::string("Compile modifier pipeline fragment shader: ") + SDL_GetError());
			}
			SDL_GPUVertexBufferDescription descriptions[2]{{0, sizeof(MeshVertex), SDL_GPU_VERTEXINPUTRATE_VERTEX, 0}, {1, sizeof(MeshInstance), SDL_GPU_VERTEXINPUTRATE_INSTANCE, 0}};
			SDL_GPUVertexAttribute attributes[8]{{0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(MeshVertex, position)}, {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(MeshVertex, normal)}, {2, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, offsetof(MeshVertex, uv)}, {3, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, offsetof(MeshVertex, color)}, {4, 1, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(MeshInstance, position)}, {5, 1, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(MeshInstance, rotation)}, {6, 1, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(MeshInstance, scale)}, {7, 1, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT, offsetof(MeshInstance, customData)}};
			SDL_GPUColorTargetDescription color{};
			color.format = pipelineColorFormat;
			SDL_GPUGraphicsPipelineCreateInfo info{};
			info.vertex_input_state.vertex_buffer_descriptions = descriptions;
			info.vertex_input_state.vertex_attributes = attributes;
			info.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
			info.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
			info.multisample_state.sample_count = SDL_GPU_SAMPLECOUNT_1;
			info.depth_stencil_state.enable_depth_test = true;
			info.depth_stencil_state.enable_depth_write = true;
			info.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_LESS;
			info.target_info.has_depth_stencil_target = true;
			info.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
			for (int instanced = 0; instanced < 2; ++instanced) {
				info.rasterizer_state.front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE;
				info.vertex_input_state.num_vertex_buffers = instanced ? 2 : 1;
				info.vertex_input_state.num_vertex_attributes = instanced ? 8 : 4;
				info.vertex_shader = vertex[instanced];
				info.rasterizer_state.enable_depth_bias = false;
				info.target_info.num_color_targets = 1;
				info.target_info.color_target_descriptions = &color;
				for (int masked = 0; masked < 2; ++masked) {
					info.fragment_shader = fragment[masked];
					for (int cull = 0; cull < 2; ++cull) {
						info.rasterizer_state.cull_mode = cull ? SDL_GPU_CULLMODE_BACK : SDL_GPU_CULLMODE_NONE;
						family.main[instanced][masked][cull] = CreatePipeline(gpuDevice, info);
					}
				}
				color.blend_state.enable_blend = true;
				color.blend_state.src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA;
				color.blend_state.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
				color.blend_state.color_blend_op = SDL_GPU_BLENDOP_ADD;
				color.blend_state.src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
				color.blend_state.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
				color.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_ADD;
				info.depth_stencil_state.enable_depth_write = false;
				info.fragment_shader = fragment[0];
				for (int cull = 0; cull < 2; ++cull) {
					info.rasterizer_state.cull_mode = cull ? SDL_GPU_CULLMODE_BACK : SDL_GPU_CULLMODE_NONE;
					family.transparent[instanced][cull] = CreatePipeline(gpuDevice, info);
				}
				color.blend_state.enable_blend = false;
				info.depth_stencil_state.enable_depth_write = true;
				info.vertex_shader = shadowVertex[instanced];
				info.rasterizer_state.front_face = SDL_GPU_FRONTFACE_CLOCKWISE;
				info.target_info.num_color_targets = 0;
				info.target_info.color_target_descriptions = nullptr;
				info.rasterizer_state.enable_depth_bias = true;
				info.rasterizer_state.depth_bias_constant_factor = 2.0f;
				info.rasterizer_state.depth_bias_slope_factor = shadowSlopeFactor;
				for (int masked = 0; masked < 2; ++masked) {
					info.fragment_shader = shadowFragment[masked];
					for (int cull = 0; cull < 2; ++cull) {
						info.rasterizer_state.cull_mode = cull ? SDL_GPU_CULLMODE_BACK : SDL_GPU_CULLMODE_NONE;
						family.shadow[instanced][masked][cull] = CreatePipeline(gpuDevice, info);
					}
				}
			}
			if (skinned) {
				SDL_GPUVertexBufferDescription skinnedDescriptions[2]{{0, sizeof(MeshVertex), SDL_GPU_VERTEXINPUTRATE_VERTEX, 0}, {1, sizeof(SkinInfluence), SDL_GPU_VERTEXINPUTRATE_VERTEX, 0}};
				SDL_GPUVertexAttribute skinnedAttributes[6]{{0, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(MeshVertex, position)}, {1, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(MeshVertex, normal)}, {2, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, offsetof(MeshVertex, uv)}, {3, 0, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4, offsetof(MeshVertex, color)}, {4, 1, SDL_GPU_VERTEXELEMENTFORMAT_USHORT4, offsetof(SkinInfluence, boneIndices)}, {5, 1, SDL_GPU_VERTEXELEMENTFORMAT_USHORT4_NORM, offsetof(SkinInfluence, weights)}};
				info.vertex_input_state.num_vertex_buffers = 2;
				info.vertex_input_state.vertex_buffer_descriptions = skinnedDescriptions;
				info.vertex_input_state.num_vertex_attributes = 6;
				info.vertex_input_state.vertex_attributes = skinnedAttributes;
				info.vertex_shader = skinnedVertex;
				info.rasterizer_state.front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE;
				info.depth_stencil_state.enable_depth_write = true;
				info.rasterizer_state.enable_depth_bias = false;
				info.target_info.num_color_targets = 1;
				info.target_info.color_target_descriptions = &color;
				for (int masked = 0; masked < 2; ++masked) {
					info.fragment_shader = fragment[masked];
					for (int cull = 0; cull < 2; ++cull) {
						info.rasterizer_state.cull_mode = cull ? SDL_GPU_CULLMODE_BACK : SDL_GPU_CULLMODE_NONE;
						family.skinnedMain[masked][cull] = CreatePipeline(gpuDevice, info);
					}
				}
				color.blend_state.enable_blend = true;
				color.blend_state.src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA;
				color.blend_state.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
				color.blend_state.color_blend_op = SDL_GPU_BLENDOP_ADD;
				color.blend_state.src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
				color.blend_state.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
				color.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_ADD;
				info.depth_stencil_state.enable_depth_write = false;
				info.fragment_shader = fragment[0];
				for (int cull = 0; cull < 2; ++cull) {
					info.rasterizer_state.cull_mode = cull ? SDL_GPU_CULLMODE_BACK : SDL_GPU_CULLMODE_NONE;
					family.skinnedTransparent[cull] = CreatePipeline(gpuDevice, info);
				}
				color.blend_state.enable_blend = false;
				info.depth_stencil_state.enable_depth_write = true;
				info.vertex_shader = skinnedShadowVertex;
				info.rasterizer_state.front_face = SDL_GPU_FRONTFACE_CLOCKWISE;
				info.target_info.num_color_targets = 0;
				info.target_info.color_target_descriptions = nullptr;
				info.rasterizer_state.enable_depth_bias = true;
				info.rasterizer_state.depth_bias_constant_factor = 2.0f;
				info.rasterizer_state.depth_bias_slope_factor = shadowSlopeFactor;
				for (int masked = 0; masked < 2; ++masked) {
					info.fragment_shader = shadowFragment[masked];
					for (int cull = 0; cull < 2; ++cull) {
						info.rasterizer_state.cull_mode = cull ? SDL_GPU_CULLMODE_BACK : SDL_GPU_CULLMODE_NONE;
						family.skinnedShadow[masked][cull] = CreatePipeline(gpuDevice, info);
					}
				}
			}
		}
		catch (...) {
			releaseShaders();
			for (auto& instanced : family.main) for (auto& masked : instanced) for (auto* pipeline : masked) if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
			for (auto& instanced : family.transparent) for (auto* pipeline : instanced) if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
			for (auto& instanced : family.shadow) for (auto& masked : instanced) for (auto* pipeline : masked) if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
			for (auto& masked : family.skinnedMain) for (auto* pipeline : masked) if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
			for (auto* pipeline : family.skinnedTransparent) if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
			for (auto& masked : family.skinnedShadow) for (auto* pipeline : masked) if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
			modifierPipelines.erase(entry);
			throw;
		}
		releaseShaders();
		return family;
	}
	}

	bool Initialize(SDL_GPUDevice* device, SDL_Window* window) {
		gpuDevice = device;
		linearOutput = window && SDL_WindowSupportsGPUSwapchainComposition(device, window, SDL_GPU_SWAPCHAINCOMPOSITION_SDR_LINEAR) && SDL_SetGPUSwapchainParameters(device, window, SDL_GPU_SWAPCHAINCOMPOSITION_SDR_LINEAR, SDL_GPU_PRESENTMODE_VSYNC);
		SDL_GPUShader* vertexShader = CreateShader(gpuDevice, vertexShaderSource.c_str(), SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 2, 1);
		const std::string instanceVertexSource = "#define INSTANCED 1\n" + vertexShaderSource;
		SDL_GPUShader* instanceVertexShader = CreateShader(gpuDevice, instanceVertexSource.c_str(), SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 2, 1);
		SDL_GPUShader* fragmentShaders[2]{};
		SDL_GPUShader* shadowVertexShader = CreateShader(gpuDevice, shadowVertexShaderSource.c_str(), SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 2, 1);
		const std::string instanceShadowSource = "#define INSTANCED 1\n" + shadowVertexShaderSource;
		SDL_GPUShader* instanceShadowShader = CreateShader(gpuDevice, instanceShadowSource.c_str(), SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 2, 1);
		const std::string skinnedVertexSource = "#define SKINNED 1\n" + vertexShaderSource;
		SDL_GPUShader* skinnedVertexShader = CreateShader(gpuDevice, skinnedVertexSource.c_str(), SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 2, 1, 1);
		const std::string skinnedShadowSource = "#define SKINNED 1\n" + shadowVertexShaderSource;
		SDL_GPUShader* skinnedShadowVertexShader = CreateShader(gpuDevice, skinnedShadowSource.c_str(), SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 2, 1, 1);
		SDL_GPUShader* shadowFragmentShaders[2]{};
		for (int masked = 0; masked < 2; ++masked) {
			const std::string define = masked ? "#define ALPHA_MASKED 1\n" : "#define ALPHA_MASKED 0\n";
			const std::string source = define + fragmentShaderSource;
			const std::string shadowSource = define + shadowFragmentShaderSource;
			fragmentShaders[masked] = CreateShader(gpuDevice, source.c_str(), SDL_GPU_SHADERSTAGE_FRAGMENT, SDL_GPU_SHADERFORMAT_MSL, 9, 2);
			shadowFragmentShaders[masked] = CreateShader(gpuDevice, shadowSource.c_str(), SDL_GPU_SHADERSTAGE_FRAGMENT, SDL_GPU_SHADERFORMAT_MSL, masked ? 1 : 0);
		}
		const auto releaseShaders = [&]() {
			if (vertexShader) SDL_ReleaseGPUShader(gpuDevice, vertexShader);
			if (instanceVertexShader) SDL_ReleaseGPUShader(gpuDevice, instanceVertexShader);
			if (shadowVertexShader) SDL_ReleaseGPUShader(gpuDevice, shadowVertexShader);
			if (instanceShadowShader) SDL_ReleaseGPUShader(gpuDevice, instanceShadowShader);
			if (skinnedVertexShader) SDL_ReleaseGPUShader(gpuDevice, skinnedVertexShader);
			if (skinnedShadowVertexShader) SDL_ReleaseGPUShader(gpuDevice, skinnedShadowVertexShader);
			for (auto* shader : fragmentShaders) if (shader) SDL_ReleaseGPUShader(gpuDevice, shader);
			for (auto* shader : shadowFragmentShaders) if (shader) SDL_ReleaseGPUShader(gpuDevice, shader);
		};
		if (!vertexShader || !instanceVertexShader || !shadowVertexShader || !instanceShadowShader || !skinnedVertexShader || !skinnedShadowVertexShader || !fragmentShaders[0] || !fragmentShaders[1] || !shadowFragmentShaders[0] || !shadowFragmentShaders[1]) {
			releaseShaders();
			return false;
		}
		SDL_GPUVertexBufferDescription vertexBufferDescription{};
		vertexBufferDescription.slot = 0;
		vertexBufferDescription.pitch = sizeof(MeshVertex);
		vertexBufferDescription.input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;
		SDL_GPUVertexAttribute vertexAttributes[4]{};
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
		vertexAttributes[3].location = 3;
		vertexAttributes[3].buffer_slot = 0;
		vertexAttributes[3].format = SDL_GPU_VERTEXELEMENTFORMAT_FLOAT4;
		vertexAttributes[3].offset = offsetof(MeshVertex, color);
		SDL_GPUVertexBufferDescription instanceDescriptions[2]{vertexBufferDescription, {}};
		instanceDescriptions[1].slot = 1;
		instanceDescriptions[1].pitch = sizeof(MeshInstance);
		instanceDescriptions[1].input_rate = SDL_GPU_VERTEXINPUTRATE_INSTANCE;
		SDL_GPUVertexAttribute instanceAttributes[8]{};
		std::copy_n(vertexAttributes, 4, instanceAttributes);
		instanceAttributes[4] = {4, 1, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(MeshInstance, position)};
		instanceAttributes[5] = {5, 1, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(MeshInstance, rotation)};
		instanceAttributes[6] = {6, 1, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT3, offsetof(MeshInstance, scale)};
		instanceAttributes[7] = {7, 1, SDL_GPU_VERTEXELEMENTFORMAT_FLOAT, offsetof(MeshInstance, customData)};
		SDL_GPUVertexBufferDescription skinnedDescriptions[2]{vertexBufferDescription, {}};
		skinnedDescriptions[1].slot = 1;
		skinnedDescriptions[1].pitch = sizeof(SkinInfluence);
		skinnedDescriptions[1].input_rate = SDL_GPU_VERTEXINPUTRATE_VERTEX;
		SDL_GPUVertexAttribute skinnedAttributes[6]{};
		std::copy_n(vertexAttributes, 4, skinnedAttributes);
		skinnedAttributes[4] = {4, 1, SDL_GPU_VERTEXELEMENTFORMAT_USHORT4, offsetof(SkinInfluence, boneIndices)};
		skinnedAttributes[5] = {5, 1, SDL_GPU_VERTEXELEMENTFORMAT_USHORT4_NORM, offsetof(SkinInfluence, weights)};
		try {
			try {
				const SDL_GPUTextureUsageFlags samplerUsage = SDL_GPU_TEXTUREUSAGE_SAMPLER;
				const SDL_GPUTextureUsageFlags fogTextureUsage = samplerUsage | SDL_GPU_TEXTUREUSAGE_COMPUTE_STORAGE_WRITE;
				if (SDL_GPUTextureSupportsFormat(gpuDevice, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, SDL_GPU_TEXTURETYPE_3D, samplerUsage)) {
					SDL_GPUSamplerCreateInfo fogSamplerInfo{};
					fogSamplerInfo.min_filter = SDL_GPU_FILTER_LINEAR;
					fogSamplerInfo.mag_filter = SDL_GPU_FILTER_LINEAR;
					fogSamplerInfo.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
					fogSamplerInfo.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
					fogSamplerInfo.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
					fogSamplerInfo.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
					fogSampler = CreateSampler(gpuDevice, fogSamplerInfo);
					SDL_GPUTextureCreateInfo emptyFogInfo{};
					emptyFogInfo.type = SDL_GPU_TEXTURETYPE_3D;
					emptyFogInfo.format = SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT;
					emptyFogInfo.usage = samplerUsage;
					emptyFogInfo.width = emptyFogInfo.height = emptyFogInfo.layer_count_or_depth = 1;
					emptyFogInfo.num_levels = 1;
					emptyFogInfo.sample_count = SDL_GPU_SAMPLECOUNT_1;
					emptyFogVolumeTexture = CreateTextureResource(gpuDevice, emptyFogInfo);
					if (SDL_GPUTextureSupportsFormat(gpuDevice, SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, SDL_GPU_TEXTURETYPE_3D, fogTextureUsage)) {
						SDL_GPUComputePipelineCreateInfo computeInfo{};
						computeInfo.code_size = fogComputeShaderSource.size() + 1;
						computeInfo.code = reinterpret_cast<const Uint8*>(fogComputeShaderSource.c_str());
						computeInfo.entrypoint = "main0";
						computeInfo.format = SDL_GPU_SHADERFORMAT_MSL;
						computeInfo.num_readwrite_storage_textures = 1;
						computeInfo.num_uniform_buffers = 1;
						computeInfo.threadcount_x = 8;
						computeInfo.threadcount_y = 8;
						computeInfo.threadcount_z = 1;
						fogComputePipeline = SDL_CreateGPUComputePipeline(gpuDevice, &computeInfo);
						volumetricFogSupported = fogComputePipeline != nullptr;
						if (!fogComputePipeline) SDL_Log("Volumetric fog compute pipeline unavailable: %s", SDL_GetError());
					}
					else SDL_Log("Volumetric fog storage format unavailable");
				}
				else SDL_Log("Volumetric fog sampling format unavailable");
			}
			catch (const std::exception& error) {
				SDL_Log("Volumetric fog initialization unavailable: %s", error.what());
				if (fogComputePipeline) SDL_ReleaseGPUComputePipeline(gpuDevice, fogComputePipeline);
				fogComputePipeline = nullptr;
				if (fogSampler) SDL_ReleaseGPUSampler(gpuDevice, fogSampler);
				fogSampler = nullptr;
				if (emptyFogVolumeTexture) ReleaseTexture(gpuDevice, emptyFogVolumeTexture);
				emptyFogVolumeTexture = nullptr;
				volumetricFogSupported = false;
			}
			const std::uint8_t whitePixel[4] = {255, 255, 255, 255};
			whiteTexture = CreateTexture(gpuDevice, 1, 1, whitePixel);
			SDL_GPUSamplerCreateInfo samplerInfo{};
			samplerInfo.min_filter = SDL_GPU_FILTER_LINEAR;
			samplerInfo.mag_filter = SDL_GPU_FILTER_LINEAR;
			samplerInfo.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR;
			samplerInfo.max_lod = 32.0f;
			samplerInfo.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
			samplerInfo.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
			samplerInfo.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
			sampler = CreateSampler(gpuDevice, samplerInfo);
			SDL_GPUSamplerCreateInfo shadowSamplerInfo{};
			shadowSamplerInfo.min_filter = SDL_GPU_FILTER_LINEAR;
			shadowSamplerInfo.mag_filter = SDL_GPU_FILTER_LINEAR;
			shadowSamplerInfo.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST;
			shadowSamplerInfo.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
			shadowSamplerInfo.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
			shadowSamplerInfo.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE;
			shadowSamplerInfo.compare_op = SDL_GPU_COMPAREOP_LESS_OR_EQUAL;
			shadowSamplerInfo.enable_compare = true;
			shadowSampler = CreateSampler(gpuDevice, shadowSamplerInfo);
			shadowMapResolution = 2048;
			emptyShadowTexture = CreateShadowTexture(1);
			SDL_GPUColorTargetDescription colorTarget{};
			colorTarget.format = window ? SDL_GetGPUSwapchainTextureFormat(gpuDevice, window) : SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
		pipelineColorFormat = colorTarget.format;
		try { PrepareFogBackgroundPipeline(); }
		catch (const std::exception& error) { SDL_Log("Volumetric fog background pass unavailable: %s", error.what()); }
			SDL_GPUGraphicsPipelineCreateInfo pipelineInfo{};
			pipelineInfo.vertex_shader = vertexShader;
			pipelineInfo.vertex_input_state.num_vertex_buffers = 1;
			pipelineInfo.vertex_input_state.vertex_buffer_descriptions = &vertexBufferDescription;
			pipelineInfo.vertex_input_state.num_vertex_attributes = 4;
			pipelineInfo.vertex_input_state.vertex_attributes = vertexAttributes;
			pipelineInfo.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
			pipelineInfo.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
			pipelineInfo.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_BACK;
			pipelineInfo.multisample_state.sample_count = SDL_GPU_SAMPLECOUNT_1;
			pipelineInfo.depth_stencil_state.enable_depth_test = true;
			pipelineInfo.depth_stencil_state.enable_depth_write = true;
			pipelineInfo.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_LESS;
			pipelineInfo.target_info.num_color_targets = 1;
			pipelineInfo.target_info.color_target_descriptions = &colorTarget;
			pipelineInfo.target_info.has_depth_stencil_target = true;
			pipelineInfo.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
			for (int instanced = 0; instanced < 2; ++instanced) {
				pipelineInfo.rasterizer_state.front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE;
				pipelineInfo.vertex_shader = instanced ? instanceVertexShader : vertexShader;
				pipelineInfo.vertex_input_state.num_vertex_buffers = instanced ? 2 : 1;
				pipelineInfo.vertex_input_state.vertex_buffer_descriptions = instanced ? instanceDescriptions : &vertexBufferDescription;
				pipelineInfo.vertex_input_state.num_vertex_attributes = instanced ? 8 : 4;
				pipelineInfo.vertex_input_state.vertex_attributes = instanced ? instanceAttributes : vertexAttributes;
				for (int masked = 0; masked < 2; ++masked) {
					pipelineInfo.fragment_shader = fragmentShaders[masked];
					for (int cull = 0; cull < 2; ++cull) {
						pipelineInfo.rasterizer_state.cull_mode = cull ? SDL_GPU_CULLMODE_BACK : SDL_GPU_CULLMODE_NONE;
						(instanced ? instancePipelines : pipelines)[masked][cull] = CreatePipeline(gpuDevice, pipelineInfo);
					}
				}
			}
			pipelineInfo.fragment_shader = fragmentShaders[0];
			pipelineInfo.depth_stencil_state.enable_depth_write = false;
			colorTarget.blend_state.enable_blend = true;
			colorTarget.blend_state.src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA;
			colorTarget.blend_state.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
			colorTarget.blend_state.color_blend_op = SDL_GPU_BLENDOP_ADD;
			colorTarget.blend_state.src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
			colorTarget.blend_state.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
			colorTarget.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_ADD;
			pipelineInfo.rasterizer_state.front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE;
			for (int instanced = 0; instanced < 2; ++instanced) {
				pipelineInfo.vertex_shader = instanced ? instanceVertexShader : vertexShader;
				pipelineInfo.vertex_input_state.num_vertex_buffers = instanced ? 2 : 1;
				pipelineInfo.vertex_input_state.vertex_buffer_descriptions = instanced ? instanceDescriptions : &vertexBufferDescription;
				pipelineInfo.vertex_input_state.num_vertex_attributes = instanced ? 8 : 4;
				pipelineInfo.vertex_input_state.vertex_attributes = instanced ? instanceAttributes : vertexAttributes;
				for (int cull = 0; cull < 2; ++cull) {
					pipelineInfo.rasterizer_state.cull_mode = cull ? SDL_GPU_CULLMODE_BACK : SDL_GPU_CULLMODE_NONE;
					(instanced ? instanceTransparentPipelines : transparentPipelines)[cull] = CreatePipeline(gpuDevice, pipelineInfo);
				}
			}
			pipelineInfo.depth_stencil_state.enable_depth_write = true;
			pipelineInfo.vertex_shader = shadowVertexShader;
			pipelineInfo.rasterizer_state.front_face = SDL_GPU_FRONTFACE_CLOCKWISE;
			pipelineInfo.vertex_input_state.num_vertex_attributes = 4;
			pipelineInfo.rasterizer_state.enable_depth_bias = true;
			pipelineInfo.rasterizer_state.depth_bias_constant_factor = 2.0f;
			pipelineInfo.rasterizer_state.depth_bias_slope_factor = shadowSlopeFactor;
			pipelineInfo.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_BACK;
			pipelineInfo.target_info.num_color_targets = 0;
			pipelineInfo.target_info.color_target_descriptions = nullptr;
			pipelineInfo.target_info.has_depth_stencil_target = true;
			pipelineInfo.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
			for (int instanced = 0; instanced < 2; ++instanced) {
				pipelineInfo.vertex_shader = instanced ? instanceShadowShader : shadowVertexShader;
				pipelineInfo.vertex_input_state.num_vertex_buffers = instanced ? 2 : 1;
				pipelineInfo.vertex_input_state.vertex_buffer_descriptions = instanced ? instanceDescriptions : &vertexBufferDescription;
				pipelineInfo.vertex_input_state.num_vertex_attributes = instanced ? 8 : 4;
				pipelineInfo.vertex_input_state.vertex_attributes = instanced ? instanceAttributes : vertexAttributes;
				for (int masked = 0; masked < 2; ++masked) {
					pipelineInfo.fragment_shader = shadowFragmentShaders[masked];
					for (int cull = 0; cull < 2; ++cull) {
						pipelineInfo.rasterizer_state.cull_mode = cull ? SDL_GPU_CULLMODE_BACK : SDL_GPU_CULLMODE_NONE;
						(instanced ? instanceShadowPipelines : shadowPipelines)[masked][cull] = CreatePipeline(gpuDevice, pipelineInfo);
					}
				}
			}
			pipelineInfo.vertex_input_state.num_vertex_buffers = 2;
			pipelineInfo.vertex_input_state.vertex_buffer_descriptions = skinnedDescriptions;
			pipelineInfo.vertex_input_state.num_vertex_attributes = 6;
			pipelineInfo.vertex_input_state.vertex_attributes = skinnedAttributes;
			pipelineInfo.vertex_shader = skinnedVertexShader;
			pipelineInfo.rasterizer_state.front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE;
			pipelineInfo.rasterizer_state.enable_depth_bias = false;
			colorTarget.blend_state.enable_blend = false;
			pipelineInfo.depth_stencil_state.enable_depth_write = true;
			pipelineInfo.target_info.num_color_targets = 1;
			pipelineInfo.target_info.color_target_descriptions = &colorTarget;
			for (int masked = 0; masked < 2; ++masked) {
				pipelineInfo.fragment_shader = fragmentShaders[masked];
				for (int cull = 0; cull < 2; ++cull) {
					pipelineInfo.rasterizer_state.cull_mode = cull ? SDL_GPU_CULLMODE_BACK : SDL_GPU_CULLMODE_NONE;
					skinnedPipelines[masked][cull] = CreatePipeline(gpuDevice, pipelineInfo);
				}
			}
			pipelineInfo.fragment_shader = fragmentShaders[0];
			pipelineInfo.depth_stencil_state.enable_depth_write = false;
			colorTarget.blend_state.enable_blend = true;
			colorTarget.blend_state.src_color_blendfactor = SDL_GPU_BLENDFACTOR_SRC_ALPHA;
			colorTarget.blend_state.dst_color_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
			colorTarget.blend_state.color_blend_op = SDL_GPU_BLENDOP_ADD;
			colorTarget.blend_state.src_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE;
			colorTarget.blend_state.dst_alpha_blendfactor = SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA;
			colorTarget.blend_state.alpha_blend_op = SDL_GPU_BLENDOP_ADD;
			for (int cull = 0; cull < 2; ++cull) {
				pipelineInfo.rasterizer_state.cull_mode = cull ? SDL_GPU_CULLMODE_BACK : SDL_GPU_CULLMODE_NONE;
				skinnedTransparentPipelines[cull] = CreatePipeline(gpuDevice, pipelineInfo);
			}
			pipelineInfo.depth_stencil_state.enable_depth_write = true;
			pipelineInfo.vertex_shader = skinnedShadowVertexShader;
			pipelineInfo.rasterizer_state.front_face = SDL_GPU_FRONTFACE_CLOCKWISE;
			pipelineInfo.rasterizer_state.enable_depth_bias = true;
			pipelineInfo.rasterizer_state.depth_bias_constant_factor = 2.0f;
			pipelineInfo.rasterizer_state.depth_bias_slope_factor = shadowSlopeFactor;
			pipelineInfo.target_info.num_color_targets = 0;
			pipelineInfo.target_info.color_target_descriptions = nullptr;
			colorTarget.blend_state.enable_blend = false;
			for (int masked = 0; masked < 2; ++masked) {
				pipelineInfo.fragment_shader = shadowFragmentShaders[masked];
				for (int cull = 0; cull < 2; ++cull) {
					pipelineInfo.rasterizer_state.cull_mode = cull ? SDL_GPU_CULLMODE_BACK : SDL_GPU_CULLMODE_NONE;
					skinnedShadowPipelines[masked][cull] = CreatePipeline(gpuDevice, pipelineInfo);
				}
			}
		}
		catch (...) {
			releaseShaders();
			throw;
		}
		releaseShaders();
		if (!whiteTexture || !sampler || !shadowSampler || !emptyShadowTexture) return false;
		for (int masked = 0; masked < 2; ++masked) for (int cull = 0; cull < 2; ++cull) if (!pipelines[masked][cull] || !shadowPipelines[masked][cull] || !instancePipelines[masked][cull] || !instanceShadowPipelines[masked][cull] || !skinnedPipelines[masked][cull] || !skinnedShadowPipelines[masked][cull]) return false;
		return transparentPipelines[0] && transparentPipelines[1] && instanceTransparentPipelines[0] && instanceTransparentPipelines[1] && skinnedTransparentPipelines[0] && skinnedTransparentPipelines[1];
	}

	static void ReleaseSceneResources() {
		while (!gpuMeshes.empty()) ReleaseMesh(gpuMeshes.begin()->first);
		for (auto& [skeleton, gpuSkeleton] : gpuSkeletons) ReleaseBuffer(gpuDevice, gpuSkeleton.poseBuffer);
		gpuSkeletons.clear();
		meshBounds.clear();
		skinBounds.clear();
		meshSources.clear();
		meshEntries.clear();
		lightEntries.clear();
		sceneTriangleCount = 0;
		dirtyCells.clear();
		shadowChanges.clear();
		staticBatches.clear();
		staticDraws.clear();
		staticBatchCells.clear();
		staticBatchOwners.clear();
		renderItems.clear();
		staticItems.clear();
		dynamicItems.clear();
		visibleItems.clear();
		shadowCasters.clear();
		faceCasters.clear();
		pointLights.clear();
		visibleCells.clear();
		batchCellSize = -1;
		for (auto& [key, shadowMap] : staticShadowMaps) if (shadowMap.texture) ReleaseTexture(gpuDevice, shadowMap.texture);
		staticShadowMaps.clear();
	}

	void PrepareScene(Scene& scene) {
		ReleaseSceneResources();
		UpdateSceneResources(scene);
		fallbackShadowReady = false;
		fallbackShadowMap = {};
		for (auto& state : pointShadowStates) {
			state.staticReady = false;
			state.staticValid = false;
			state.dynamicValid = false;
		}
		staticShadowDirection = {};
		shadowLight = nullptr;
		renderItems.reserve(staticBatches.size() + scene.ResidentRenderObjects().size());
		FlushUploads();
	}

	static void RecordFrameStatistics() {
		frameSamples[frameSampleIndex] = Time::deltaTime * 1000.0f;
		frameSampleIndex = (frameSampleIndex + 1) % frameSamples.size();
		frameSampleCount = std::min(frameSampleCount + 1, frameSamples.size());
		fpsElapsed += Time::deltaTime;
		++fpsFrames;
		if (fpsElapsed < 0.5f) return;
		fpsString = std::to_string(static_cast<int>(fpsFrames / fpsElapsed));
		fpsElapsed = 0.0f;
		fpsFrames = 0;
		auto sorted = frameSamples;
		std::sort(sorted.begin(), sorted.begin() + frameSampleCount);
		medianFrameTime = (sorted[(frameSampleCount - 1) / 2] + sorted[frameSampleCount / 2]) * 0.5f;
		slowFrameTime = sorted[frameSampleCount - 1];
	}

	static void SortDraws() {
		const auto transparent = std::partition(visibleItems.begin(), visibleItems.end(), [](const RenderItem* item) { return !item->mesh->GetMaterial(item->materialIndex).transparent; });
		std::sort(visibleItems.begin(), transparent, [](const RenderItem* left, const RenderItem* right) {
			if (left->depthBucket != right->depthBucket) return left->depthBucket < right->depthBucket;
			const auto* leftMaterial = left->gpu->materials[left->materialIndex].material;
			const auto* rightMaterial = right->gpu->materials[right->materialIndex].material;
			const int leftPipeline = (leftMaterial->masked ? 2 : 0) + (left->mesh->cullBackfaces ? 1 : 0);
			const int rightPipeline = (rightMaterial->masked ? 2 : 0) + (right->mesh->cullBackfaces ? 1 : 0);
			if (leftPipeline != rightPipeline) return leftPipeline < rightPipeline;
			if (leftMaterial != rightMaterial) return std::less<const GpuMaterial*>{}(leftMaterial, rightMaterial);
			return left->viewDepth < right->viewDepth;
		});
		std::sort(transparent, visibleItems.end(), [](const RenderItem* left, const RenderItem* right) { return left->viewDepth > right->viewDepth; });
	}

	void Render(Scene& scene, SDL_GPUCommandBuffer* commandBuffer, SDL_GPUTexture* colorTarget, Uint32 width, Uint32 height) {
		if (!colorTarget || !width || !height) return;
		const Uint64 renderStart = SDL_GetTicksNS();
		UpdateSceneResources(scene);
		EnsureShadowResolution(scene.shadowResolution);
		const float frameTime = static_cast<float>(renderStart / 1000000000ull) + static_cast<float>(renderStart % 1000000000ull) / 1000000000.0f;
		cullingTimeNanoseconds = 0;
		drawCallCount = 0;
		shadowDrawCount = 0;
		shadowTriangleCount = 0;
		RecordFrameStatistics();
		std::uint64_t visibleTriangles = 0;
		const LightData lightData = CollectLights();
		const LightData* light = lightData.source ? &lightData : nullptr;
		const bool directionalShadowsActive = scene.shadowsEnabled && light && light->castShadows;
		const bool pointShadowsActive = scene.shadowsEnabled && std::any_of(pointLights.begin(), pointLights.end(), [](const PointLightData& pointLight) { return pointLight.castShadows; });
		const bool shadowsActive = directionalShadowsActive || pointShadowsActive;
		EnsureDirectionalShadowTextures(directionalShadowsActive, directionalShadowsActive && light->renderStatic);
		if (depthWidth != width || depthHeight != height) {
			if (depthTexture) ReleaseTexture(gpuDevice, depthTexture);
			depthTexture = CreateDepthTexture(width, height);
			depthWidth = width;
			depthHeight = height;
		}
		renderItems.clear();
		staticItems.clear();
		dynamicItems.clear();
		visibleItems.clear();
		Transform cameraTransform{};
		const float aspectRatio = static_cast<float>(width) / height;
		ShadowMapData dynamicMap{};
		Culling::Frustum frustum{};
		if (scene.activeCamera && scene.activeCamera->owner) {
			cameraTransform = CameraWorldTransform(*scene.activeCamera);
			for (const auto& item : staticDraws) {
				renderItems.push_back(&item);
				if (shadowsActive) {
					if (item.mesh->vertexAnimated) dynamicItems.push_back(&item);
					else staticItems.push_back(&item);
				}
			}
			for (const auto& entry : meshEntries) {
				const Object* object = entry.object;
				auto& source = *entry.source;
				Mesh* mesh = source.mesh;
				if (!mesh || mesh->data.vertices.empty() || mesh->data.indices.empty() || (mesh->instanced && mesh->instances.empty()) || (object->renderStatic && !mesh->instanced && IsBatchable(*mesh))) continue;
				Transform transform = object->transform;
				if (auto* body = source.body) {
					const float alpha = Time::interpolationAlpha;
					transform.position = {body->previousPosition.x + (transform.position.x - body->previousPosition.x) * alpha, body->previousPosition.y + (transform.position.y - body->previousPosition.y) * alpha, body->previousPosition.z + (transform.position.z - body->previousPosition.z) * alpha};
				}
				if (!object->renderStatic || mesh->IsSkinned() || !source.drawReady) {
					const CellKey cell{scene.cellSize > 0 ? static_cast<int>(std::floor(transform.position.x / scene.cellSize)) : 0, scene.cellSize > 0 ? static_cast<int>(std::floor(transform.position.y / scene.cellSize)) : 0, scene.cellSize > 0 ? static_cast<int>(std::floor(transform.position.z / scene.cellSize)) : 0};
					source.draws.clear();
					for (std::size_t sectionIndex = 0; sectionIndex < mesh->SectionCount(); ++sectionIndex) source.draws.push_back(PrepareDraw(mesh, transform, cell, sectionIndex));
					source.drawReady = true;
				}
				for (const auto& item : source.draws) {
					renderItems.push_back(&item);
					if (shadowsActive) {
						if (object->renderStatic && !mesh->vertexAnimated && !mesh->IsSkinned()) staticItems.push_back(&item);
						else dynamicItems.push_back(&item);
					}
				}
			}
			FlushUploads();
			frustum = Culling::FromCamera(*scene.activeCamera, cameraTransform, aspectRatio);
			const Uint64 cullingStart = SDL_GetTicksNS();
			for (const auto* itemPointer : renderItems) {
				const RenderItem& item = *itemPointer;
				item.lightMask = 0;
				if (!Culling::IsVisible(item.bounds, frustum)) continue;
				for (std::size_t index = 0; index < pointLights.size(); ++index) if (IntersectsSphere(item.bounds, pointLights[index].position, pointLights[index].range)) item.lightMask |= 1u << index;
				const Vector3 center = Scale(Add(item.bounds.minimum, item.bounds.maximum), 0.5f);
				item.viewDepth = Dot(Subtract(center, frustum.position), frustum.forward);
				const float nearDepth = ProjectBounds(item.bounds, frustum.forward).first - Dot(frustum.position, frustum.forward);
				item.depthBucket = std::floor(nearDepth / std::max(scene.cellSize, 1));
				visibleItems.push_back(&item);
			}
			cullingTimeNanoseconds = SDL_GetTicksNS() - cullingStart;
			for (const auto* item : visibleItems) {
				item->gpu = &GetMesh(item->mesh);
				if (item->skeleton) GetSkeleton(item->skeleton);
			}
			FlushUploads();
			if (directionalShadowsActive) {
				if (light->renderStatic) {
					const Vector3 direction = Normalize(light->direction);
					if (shadowLight != light->source || direction.x != staticShadowDirection.x || direction.y != staticShadowDirection.y || direction.z != staticShadowDirection.z) {
						fallbackShadowReady = false;
						for (auto& [key, map] : staticShadowMaps) map.ready = false;
						shadowLight = light->source;
						staticShadowDirection = direction;
					}
					auto& casters = shadowCasters;
					if (!fallbackShadowReady) {
						fallbackShadowMap = BuildStaticShadowMap(direction, 0, 0, 0, 0, staticItems, casters);
						RenderShadowMap(commandBuffer, fallbackShadowTexture, fallbackShadowMap, casters, frameTime);
						fallbackShadowReady = true;
					}
					visibleCells.clear();
					for (const auto* item : visibleItems) visibleCells.emplace_back(item->cellX, item->cellY, item->cellZ);
					std::sort(visibleCells.begin(), visibleCells.end());
					visibleCells.erase(std::unique(visibleCells.begin(), visibleCells.end()), visibleCells.end());
					for (auto map = staticShadowMaps.begin(); map != staticShadowMaps.end();) {
						if (!std::binary_search(visibleCells.begin(), visibleCells.end(), map->first)) {
							if (map->second.texture) ReleaseTexture(gpuDevice, map->second.texture);
							map = staticShadowMaps.erase(map);
						}
						else ++map;
					}
					for (const auto& key : visibleCells) {
						auto& map = staticShadowMaps[key];
						if (map.ready) continue;
						map.data = BuildStaticShadowMap(direction, std::get<0>(key), std::get<1>(key), std::get<2>(key), scene.cellSize, staticItems, casters);
						if (map.data.valid) {
							if (!map.texture) map.texture = CreateShadowTexture(shadowMapResolution);
							RenderShadowMap(commandBuffer, map.texture, map.data, casters, frameTime);
						}
						map.ready = true;
					}
				}
				const auto& casters = light->renderStatic ? dynamicItems : renderItems;
				dynamicMap = BuildDynamicShadowMap(*scene.activeCamera, frustum, scene.shadowFar, *light, casters, shadowCasters);
				RenderShadowMap(commandBuffer, dynamicShadowTexture, dynamicMap, shadowCasters, frameTime);
			}
		}
		RenderPointShadows(scene, commandBuffer, frameTime);
		Uint32 colliderDebugVertices = 0;
		if (scene.debugMode && scene.activeCamera) {
			const std::vector<Vector3> lines = Physics::ColliderDebugLines(scene);
			if (!lines.empty()) {
				PrepareColliderDebugPipeline();
				const Uint32 bytes = static_cast<Uint32>(lines.size() * sizeof(Vector3));
				if (bytes > colliderDebugCapacity) {
					ReleaseBuffer(gpuDevice, colliderDebugBuffer);
					colliderDebugBuffer = nullptr;
					colliderDebugCapacity = bytes;
				}
				colliderDebugBuffer = UploadBuffer(SDL_GPU_BUFFERUSAGE_VERTEX, lines.data(), bytes, colliderDebugBuffer);
				colliderDebugVertices = static_cast<Uint32>(lines.size());
				FlushUploads();
			}
		}
		FrameUniforms frame{};
		frame.cameraPosition[0] = cameraTransform.position.x;
		frame.cameraPosition[1] = cameraTransform.position.y;
		frame.cameraPosition[2] = cameraTransform.position.z;
		frame.fogColor[0] = scene.fogColor.x;
		frame.fogColor[1] = scene.fogColor.y;
		frame.fogColor[2] = scene.fogColor.z;
		const float farClip = scene.activeCamera ? std::max(scene.activeCamera->far, 0.01f) : 1000.0f;
		const float nearClip = scene.activeCamera ? std::min(std::max(scene.activeCamera->near, 0.01f), farClip * 0.5f) : 0.1f;
		bool volumetricFogActive = scene.fogEnabled && scene.fogVolumetric && scene.activeCamera && volumetricFogSupported && fogComputePipeline && fogBackgroundPipeline && fogSampler && emptyFogVolumeTexture;
		Uint32 frameFogWidth = 1;
		Uint32 frameFogHeight = 1;
		if (volumetricFogActive) {
			frameFogWidth = std::min(fogMaxGridWidth, std::max(1u, (width + 7) / 8));
			frameFogHeight = std::min(fogMaxGridHeight, std::max(1u, (height + 7) / 8));
			volumetricFogActive = EnsureFogVolumeTexture(frameFogWidth, frameFogHeight);
		}
		const float fogEndDistance = volumetricFogActive ? farClip * volumetricFogEndDistanceScale : farClip;
		frame.fogParameters[1] = fogEndDistance;
		frame.fogParameters[2] = scene.fogEnabled ? 1.0f : 0.0f;
		frame.fogHeightParameters[0] = 0.35f / fogEndDistance;
		frame.fogHeightParameters[1] = std::isfinite(scene.fogFalloff) ? std::max(scene.fogFalloff, 0.0f) : 0.1f;
		frame.fogHeightParameters[2] = std::isfinite(scene.fogHeight) ? scene.fogHeight : 10.0f;
		frame.fogHeightParameters[3] = linearOutput ? 1.0f : 0.0f;
		frame.fogVolumeParameters[0] = volumetricFogActive ? 1.0f : 0.0f;
		frame.fogVolumeParameters[1] = std::isfinite(scene.fogIntensity) ? std::max(scene.fogIntensity, 0.0f) : 0.0f;
		frame.fogVolumeParameters[2] = static_cast<float>(frameFogWidth);
		frame.fogVolumeParameters[3] = static_cast<float>(frameFogHeight);
		frame.fogScreenParameters[0] = static_cast<float>(width);
		frame.fogScreenParameters[1] = static_cast<float>(height);
		frame.fogScreenParameters[2] = static_cast<float>(frameFogWidth);
		frame.fogScreenParameters[3] = static_cast<float>(frameFogHeight);
		frame.fogViewParameters[0] = frustum.tanHalfFov;
		frame.fogViewParameters[1] = aspectRatio;
		frame.fogViewParameters[2] = nearClip;
		frame.fogViewParameters[3] = farClip;
		frame.cameraRight[0] = frustum.right.x;
		frame.cameraRight[1] = frustum.right.y;
		frame.cameraRight[2] = frustum.right.z;
		frame.cameraUp[0] = frustum.up.x;
		frame.cameraUp[1] = frustum.up.y;
		frame.cameraUp[2] = frustum.up.z;
		frame.cameraForward[0] = frustum.forward.x;
		frame.cameraForward[1] = frustum.forward.y;
		frame.cameraForward[2] = frustum.forward.z;
		frame.ambientLight[0] = scene.ambientLightColor.x;
		frame.ambientLight[1] = scene.ambientLightColor.y;
		frame.ambientLight[2] = scene.ambientLightColor.z;
		frame.ambientLight[3] = scene.ambientLight ? 1.0f : 0.0f;
		if (light) {
			frame.directionalDirection[0] = light->direction.x;
			frame.directionalDirection[1] = light->direction.y;
			frame.directionalDirection[2] = light->direction.z;
			frame.directionalDirection[3] = light->intensity;
			frame.directionalColor[0] = light->color.x;
			frame.directionalColor[1] = light->color.y;
			frame.directionalColor[2] = light->color.z;
		}
		for (std::size_t index = 0; index < pointLights.size(); ++index) {
			frame.pointLightPositionRange[index][0] = pointLights[index].position.x;
			frame.pointLightPositionRange[index][1] = pointLights[index].position.y;
			frame.pointLightPositionRange[index][2] = pointLights[index].position.z;
			frame.pointLightPositionRange[index][3] = pointLights[index].range;
			frame.pointLightColorIntensity[index][0] = pointLights[index].color.x;
			frame.pointLightColorIntensity[index][1] = pointLights[index].color.y;
			frame.pointLightColorIntensity[index][2] = pointLights[index].color.z;
			frame.pointLightColorIntensity[index][3] = pointLights[index].intensity;
			frame.pointLightShadowInfo[index][0] = static_cast<float>(pointLights[index].staticShadowIndex);
			frame.pointLightShadowInfo[index][1] = static_cast<float>(pointLights[index].dynamicShadowIndex);
			frame.pointLightShadowInfo[index][2] = pointLights[index].staticShadowValid ? 1.0f : 0.0f;
			frame.pointLightShadowInfo[index][3] = pointLights[index].dynamicShadowValid ? 1.0f : 0.0f;
			frame.pointLightShadowBias[index][0] = std::max(pointLights[index].shadowBias, 0.0f);
		}
		frame.fogParameters[0] = static_cast<float>(pointLights.size());
		const Matrix4 view = scene.activeCamera ? ViewDirection(frustum.position, frustum.forward, frustum.up) : Identity();
		Matrix4 projection = scene.activeCamera ? Perspective(scene.activeCamera->fieldOfView, aspectRatio, scene.activeCamera->near, scene.activeCamera->far) : Identity();
		if (scene.activeCamera) projection.values[0] = -projection.values[0];
		const Matrix4 viewProjection = Multiply(projection, view);
		if (volumetricFogActive) {
			SDL_GPUStorageTextureReadWriteBinding fogWrite{};
			fogWrite.texture = fogVolumeTexture;
			fogWrite.mip_level = 0;
			fogWrite.layer = 0;
			SDL_PushGPUComputeUniformData(commandBuffer, 0, &frame, sizeof(frame));
			SDL_GPUComputePass* fogPass = SDL_BeginGPUComputePass(commandBuffer, &fogWrite, 1, nullptr, 0);
			if (fogPass) {
				SDL_BindGPUComputePipeline(fogPass, fogComputePipeline);
				SDL_DispatchGPUCompute(fogPass, (frameFogWidth + 7) / 8, (frameFogHeight + 7) / 8, 1);
				SDL_EndGPUComputePass(fogPass);
			}
			else {
				SDL_Log("Volumetric fog compute pass unavailable: %s", SDL_GetError());
				volumetricFogActive = false;
				frame.fogVolumeParameters[0] = 0.0f;
				frame.fogParameters[1] = farClip;
				frame.fogHeightParameters[0] = 0.35f / farClip;
				volumetricFogSupported = false;
			}
		}
		SDL_PushGPUFragmentUniformData(commandBuffer, 1, &frame, sizeof(frame));
		SDL_GPUColorTargetInfo colorInfo{};
		colorInfo.texture = colorTarget;
		Vector3 background = scene.backgroundColor;
		if (!linearOutput) {
			auto encode = [](float value) { return value <= 0.0031308f ? value * 12.92f : 1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f; };
			background = {encode(std::max(background.x, 0.0f)), encode(std::max(background.y, 0.0f)), encode(std::max(background.z, 0.0f))};
		}
		colorInfo.clear_color = {background.x, background.y, background.z, 1.0f};
		colorInfo.load_op = SDL_GPU_LOADOP_CLEAR;
		colorInfo.store_op = SDL_GPU_STOREOP_STORE;
		SDL_GPUDepthStencilTargetInfo depthInfo{};
		depthInfo.texture = depthTexture;
		depthInfo.clear_depth = 1.0f;
		depthInfo.load_op = SDL_GPU_LOADOP_CLEAR;
		depthInfo.store_op = SDL_GPU_STOREOP_DONT_CARE;
		depthInfo.stencil_load_op = SDL_GPU_LOADOP_DONT_CARE;
		depthInfo.stencil_store_op = SDL_GPU_STOREOP_DONT_CARE;
		SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(commandBuffer, &colorInfo, 1, &depthInfo);
		if (!pass) throw std::runtime_error(std::string("Begin main render pass: ") + SDL_GetError());
		if (volumetricFogActive) {
			struct BackgroundUniforms { float color[4]; } backgroundUniforms{{scene.backgroundColor.x, scene.backgroundColor.y, scene.backgroundColor.z, 1.0f}};
			SDL_PushGPUFragmentUniformData(commandBuffer, 0, &backgroundUniforms, sizeof(backgroundUniforms));
			SDL_BindGPUGraphicsPipeline(pass, fogBackgroundPipeline);
			SDL_GPUTextureSamplerBinding fogBinding{fogVolumeTexture, fogSampler};
			SDL_BindGPUFragmentSamplers(pass, 0, &fogBinding, 1);
			SDL_DrawGPUPrimitives(pass, 3, 1, 0, 0);
			++drawCallCount;
		}
		SortDraws();
		SDL_GPUGraphicsPipeline* activePipeline = nullptr;
		std::array<SDL_GPUTexture*, 9> activeTextures{};
		GpuMaterial* activeMaterial = nullptr;
		BufferBindings buffers;
		SkeletonBindings skeletonBindings;
		for (const auto* itemPointer : visibleItems) {
			const RenderItem& item = *itemPointer;
			Mesh* mesh = item.mesh;
			const auto selected = shadowsActive ? staticShadowMaps.find({item.cellX, item.cellY, item.cellZ}) : staticShadowMaps.end();
			const StaticShadowMap* detail = selected != staticShadowMaps.end() ? &selected->second : nullptr;
			const bool staticEnabled = directionalShadowsActive && light->renderStatic;
			const ShadowMapData maps[3] = {detail ? detail->data : ShadowMapData{}, fallbackShadowMap, dynamicMap};
			const bool enabled[3] = {staticEnabled && detail && detail->data.valid, staticEnabled && fallbackShadowMap.valid, directionalShadowsActive && dynamicMap.valid};
			GpuMesh& gpuMesh = *item.gpu;
			const Material& material = mesh->GetMaterial(item.materialIndex);
			GpuMeshMaterial& gpuMeshMaterial = gpuMesh.materials.at(item.materialIndex);
			GpuMaterial* activeGpuMaterial = gpuMeshMaterial.material;
			SDL_GPUGraphicsPipeline* meshPipeline = nullptr;
			if (item.skeleton) meshPipeline = material.vertexModifierSource.empty() ? (material.transparent ? skinnedTransparentPipelines[mesh->cullBackfaces ? 1 : 0] : skinnedPipelines[activeGpuMaterial->masked ? 1 : 0][mesh->cullBackfaces ? 1 : 0]) : (material.transparent ? GetModifierPipelines(material.vertexModifierSource, true).skinnedTransparent[mesh->cullBackfaces ? 1 : 0] : GetModifierPipelines(material.vertexModifierSource, true).skinnedMain[activeGpuMaterial->masked ? 1 : 0][mesh->cullBackfaces ? 1 : 0]);
			else meshPipeline = material.vertexModifierSource.empty() ? (material.transparent ? (mesh->instanced ? instanceTransparentPipelines : transparentPipelines)[mesh->cullBackfaces ? 1 : 0] : (mesh->instanced ? instancePipelines : pipelines)[activeGpuMaterial->masked ? 1 : 0][mesh->cullBackfaces ? 1 : 0]) : (material.transparent ? GetModifierPipelines(material.vertexModifierSource).transparent[mesh->instanced ? 1 : 0][mesh->cullBackfaces ? 1 : 0] : GetModifierPipelines(material.vertexModifierSource).main[mesh->instanced ? 1 : 0][activeGpuMaterial->masked ? 1 : 0][mesh->cullBackfaces ? 1 : 0]);
			if (meshPipeline != activePipeline) {
				SDL_BindGPUGraphicsPipeline(pass, meshPipeline);
				activePipeline = meshPipeline;
			}
			visibleTriangles += item.indexCount / 3 * (mesh->instanced ? mesh->instances.size() : 1);
			ObjectUniforms vertex{};
			const Matrix4 mvp = Multiply(viewProjection, item.model);
			std::memcpy(vertex.mvp, mvp.values, sizeof(vertex.mvp));
			std::memcpy(vertex.model, item.model.values, sizeof(vertex.model));
			std::memcpy(vertex.normal, item.normal.values, sizeof(vertex.normal));
			vertex.displacementParameters[0] = gpuMeshMaterial.displacementScale;
			std::memcpy(vertex.uvTransform, gpuMeshMaterial.uv, sizeof(vertex.uvTransform));
			vertex.modifierTime[0] = frameTime;
			FragmentUniforms fragment = gpuMeshMaterial.fragment;
			if (!mesh->cullBackfaces) fragment.lightingParameters[0] = static_cast<float>(static_cast<Uint32>(fragment.lightingParameters[0]) | 32u);
			Uint32 lightCount = 0;
			for (Uint32 index = 0; index < pointLights.size(); ++index) {
				if (!(item.lightMask & (1u << index))) continue;
				fragment.pointLightIndices[lightCount / 4][lightCount % 4] = index;
				++lightCount;
			}
			fragment.lightingParameters[3] = static_cast<float>(lightCount);
			for (int index = 0; index < 3; ++index) {
				if (directionalShadowsActive) std::memcpy(vertex.shadowMvp[index], maps[index].mvp.values, sizeof(vertex.shadowMvp[index]));
				fragment.shadowParameters[index][0] = enabled[index] ? 1.0f : 0.0f;
				fragment.shadowParameters[index][1] = directionalShadowsActive ? std::max(light->shadowBias, 0.0f) / std::max(maps[index].depthRange, 0.01f) : 0.0f;
				fragment.shadowParameters[index][2] = 1.0f / shadowMapResolution;
			}
			fragment.shadowParameters[1][3] = 1.0f;

			fragment.lightingParameters[2] = shadowsActive ? 1.0f : 0.0f;
			vertex.shadowState[0] = directionalShadowsActive ? 1.0f : 0.0f;
			SDL_PushGPUVertexUniformData(commandBuffer, 0, &vertex, sizeof(vertex));
			SDL_PushGPUFragmentUniformData(commandBuffer, 0, &fragment, sizeof(fragment));
			buffers.Bind(pass, gpuMesh);
			if (item.skeleton) {
				SDL_GPUBufferBinding skinBinding{gpuMesh.skinBuffer, 0};
				SDL_BindGPUVertexBuffers(pass, 1, &skinBinding, 1);
				skeletonBindings.Bind(pass, GetSkeleton(item.skeleton));
			}
			else if (mesh->instanced) {
				SDL_GPUBufferBinding instanceBinding{gpuMesh.instanceBuffer, 0};
				SDL_BindGPUVertexBuffers(pass, 1, &instanceBinding, 1);
			}
			SDL_GPUTextureSamplerBinding bindings[9]{};
			bindings[0] = {activeGpuMaterial->textures[0], sampler};
			bindings[1] = {enabled[0] ? detail->texture : emptyShadowTexture, shadowSampler};
			bindings[2] = {enabled[1] ? fallbackShadowTexture : emptyShadowTexture, shadowSampler};
			bindings[3] = {enabled[2] ? dynamicShadowTexture : emptyShadowTexture, shadowSampler};
			bindings[4] = {pointShadowTexture, shadowSampler};
			bindings[5] = {activeGpuMaterial->textures[1], sampler};
			bindings[6] = {activeGpuMaterial->textures[2], sampler};
			bindings[7] = {activeGpuMaterial->textures[3], sampler};
			bindings[8] = {volumetricFogActive ? fogVolumeTexture : emptyFogVolumeTexture, fogSampler ? fogSampler : sampler};
			bool texturesChanged = false;
			for (std::size_t index = 0; index < activeTextures.size(); ++index) {
				texturesChanged = texturesChanged || activeTextures[index] != bindings[index].texture;
				activeTextures[index] = bindings[index].texture;
			}
			if (texturesChanged) SDL_BindGPUFragmentSamplers(pass, 0, bindings, 9);
			SDL_GPUTextureSamplerBinding vertexBindings[2] = {{activeGpuMaterial->textures[4], sampler}, {activeGpuMaterial->textures[5], sampler}};
			if (activeMaterial != activeGpuMaterial) SDL_BindGPUVertexSamplers(pass, 0, vertexBindings, 2);
			activeMaterial = activeGpuMaterial;
			SDL_DrawGPUIndexedPrimitives(pass, item.indexCount, mesh->instanced ? gpuMesh.uploadedInstanceCount : 1, item.firstIndex, 0, 0);
			++drawCallCount;
		}
		if (colliderDebugVertices) {
			SDL_BindGPUGraphicsPipeline(pass, colliderDebugPipeline);
			SDL_GPUBufferBinding binding{colliderDebugBuffer, 0};
			SDL_BindGPUVertexBuffers(pass, 0, &binding, 1);
			SDL_PushGPUVertexUniformData(commandBuffer, 0, viewProjection.values, sizeof(viewProjection.values));
			SDL_DrawGPUPrimitives(pass, colliderDebugVertices, 1, 0, 0);
			++drawCallCount;
		}
		SDL_EndGPURenderPass(pass);
		triangleCountString = std::to_string(visibleTriangles) + "/" + std::to_string(sceneTriangleCount);
		renderTimeNanoseconds = SDL_GetTicksNS() - renderStart;
	}

	std::string FPS() { return fpsString; }
	std::string FrameTime() { return Milliseconds(Time::deltaTime * 1000.0); }
	std::string CullingTime() { return Milliseconds(cullingTimeNanoseconds / 1000000.0); }
	std::string RenderCpuTime() { return Milliseconds(renderTimeNanoseconds / 1000000.0); }
	std::string SubmitCpuTime() { return Milliseconds(submitTimeNanoseconds / 1000000.0); }
	std::string GPUCompletionTime() { return hasGpuCompletionLatency ? Milliseconds(gpuCompletionLatencyNanoseconds / 1000000.0) : "pending"; }
	std::string DrawCalls() { return std::to_string(drawCallCount); }
	std::string TriangleCount() { return triangleCountString; }
	std::string ShadowDrawCalls() { return std::to_string(shadowDrawCount); }
	std::string ShadowTriangles() { return std::to_string(shadowTriangleCount); }
	std::string UploadBytes() { return std::to_string(frameUploadedBytes); }
	std::string PeakUploadBytes() { return std::to_string(peakUploadBytes); }
	std::string TextureBytes() { return std::to_string(textureBytes); }
	std::string BufferBytes() { return std::to_string(bufferBytes); }
	std::string PeakResourceBytes() { return std::to_string(peakResourceBytes); }
	std::string BatchRebuilds() { return std::to_string(meshRebuildCount); }
	std::string MedianFrameTime() { return Milliseconds(medianFrameTime); }
	std::string SlowFrameTime() { return Milliseconds(slowFrameTime); }

	void Submit(SDL_GPUCommandBuffer* commandBuffer) {
		FlushUploads();
		frameUploadedBytes = uploadedBytes - previousUploadedBytes;
		previousUploadedBytes = uploadedBytes;
		UpdateGpuCompletionLatency();
		const Uint64 submitStart = SDL_GetTicksNS();
		SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(commandBuffer);
		if (!fence) throw std::runtime_error(std::string("Submit frame command buffer: ") + SDL_GetError());
		const Uint64 submittedAt = SDL_GetTicksNS();
		submitTimeNanoseconds = submittedAt - submitStart;
		if (fence) pendingGpuSubmissions.push_back({fence, submittedAt});
	}

	void Shutdown() {
		if (fogComputePipeline) SDL_ReleaseGPUComputePipeline(gpuDevice, fogComputePipeline);
		fogComputePipeline = nullptr;
		if (fogBackgroundPipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, fogBackgroundPipeline);
		fogBackgroundPipeline = nullptr;
		if (fogSampler) SDL_ReleaseGPUSampler(gpuDevice, fogSampler);
		fogSampler = nullptr;
		if (fogVolumeTexture) ReleaseTexture(gpuDevice, fogVolumeTexture);
		if (emptyFogVolumeTexture) ReleaseTexture(gpuDevice, emptyFogVolumeTexture);
		fogVolumeTexture = nullptr;
		emptyFogVolumeTexture = nullptr;
		fogGridWidth = 0;
		fogGridHeight = 0;
		volumetricFogSupported = false;
		if (colliderDebugPipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, colliderDebugPipeline);
		colliderDebugPipeline = nullptr;
		ReleaseBuffer(gpuDevice, colliderDebugBuffer);
		colliderDebugBuffer = nullptr;
		colliderDebugCapacity = 0;
		if (uploadCommandBuffer) {
			if (uploadPass) SDL_EndGPUCopyPass(uploadPass);
			SDL_CancelGPUCommandBuffer(uploadCommandBuffer);
			uploadCommandBuffer = nullptr;
			uploadPass = nullptr;
		}
		for (auto* transfer : uploadTransfers) ReleaseTransferBuffer(gpuDevice, transfer);
		uploadTransfers.clear();
		for (auto& [source, family] : modifierPipelines) {
			for (auto& instanced : family.main) for (auto& masked : instanced) for (auto* pipeline : masked) if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
			for (auto& instanced : family.transparent) for (auto* pipeline : instanced) if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
			for (auto& instanced : family.shadow) for (auto& masked : instanced) for (auto* pipeline : masked) if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
			for (auto& masked : family.skinnedMain) for (auto* pipeline : masked) if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
			for (auto* pipeline : family.skinnedTransparent) if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
			for (auto& masked : family.skinnedShadow) for (auto* pipeline : masked) if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
		}
		modifierPipelines.clear();
		for (const auto& submission : pendingGpuSubmissions) SDL_ReleaseGPUFence(gpuDevice, submission.fence);
		pendingGpuSubmissions.clear();
		ReleaseSceneResources();
		while (!textures.empty()) {
			auto* texture = textures.begin()->second;
			textureReferences[texture] = 1;
			ReleaseTexture(gpuDevice, texture);
		}
		staticShadowDirection = {};
		for (auto& variants : {&pipelines, &shadowPipelines, &instancePipelines, &instanceShadowPipelines}) {
			for (auto& row : *variants) {
				for (auto& pipeline : row) {
					if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
					pipeline = nullptr;
				}
			}
		}
		for (auto& masked : skinnedPipelines) for (auto*& pipeline : masked) {
			if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
			pipeline = nullptr;
		}
		for (auto& masked : skinnedShadowPipelines) for (auto*& pipeline : masked) {
			if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
			pipeline = nullptr;
		}
		for (auto*& pipeline : skinnedTransparentPipelines) {
			if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
			pipeline = nullptr;
		}
		for (auto& pipeline : transparentPipelines) {
			if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
			pipeline = nullptr;
		}
		for (auto& pipeline : instanceTransparentPipelines) {
			if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
			pipeline = nullptr;
		}
		if (whiteTexture) ReleaseTexture(gpuDevice, whiteTexture);
		if (sampler) SDL_ReleaseGPUSampler(gpuDevice, sampler);
		if (shadowSampler) SDL_ReleaseGPUSampler(gpuDevice, shadowSampler);
		if (depthTexture) ReleaseTexture(gpuDevice, depthTexture);
		if (emptyShadowTexture) ReleaseTexture(gpuDevice, emptyShadowTexture);
		if (dynamicShadowTexture) ReleaseTexture(gpuDevice, dynamicShadowTexture);
		if (fallbackShadowTexture) ReleaseTexture(gpuDevice, fallbackShadowTexture);
		if (pointShadowTexture) ReleaseTexture(gpuDevice, pointShadowTexture);
		depthTexture = nullptr;
		emptyShadowTexture = nullptr;
		dynamicShadowTexture = nullptr;
		fallbackShadowTexture = nullptr;
		pointShadowTexture = nullptr;
		pointShadowResolution = 0;
		pointShadowSlots.clear();
		pointShadowStates.clear();
		fallbackShadowReady = false;
		fallbackShadowMap = {};
		shadowLight = nullptr;
		shadowMapResolution = 2048;
		batchCellSize = -1;
		depthWidth = 0;
		depthHeight = 0;
		fpsElapsed = 0.0f;
		fpsFrames = 0;
		hasGpuCompletionLatency = false;
		shadowDrawCount = 0;
		shadowTriangleCount = 0;
		meshRebuildCount = 0;
		uploadedBytes = 0;
		previousUploadedBytes = 0;
		frameUploadedBytes = 0;
		peakUploadBytes = 0;
		allocations.clear();
		textureBytes = 0;
		bufferBytes = 0;
		peakResourceBytes = 0;
		frameSampleCount = 0;
		frameSampleIndex = 0;
		medianFrameTime = 0.0f;
		slowFrameTime = 0.0f;
		whiteTexture = nullptr;
		sampler = nullptr;
		shadowSampler = nullptr;
		gpuDevice = nullptr;
	}
}
