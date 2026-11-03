#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
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

namespace Sapphire::Graphics {
	SDL_GPUBuffer* CreateBuffer(SDL_GPUDevice* gpuDevice, SDL_GPUBufferUsageFlags usage, Uint32 size);
	SDL_GPUTexture* CreateTexture(SDL_GPUDevice* gpuDevice, int width, int height, const std::uint8_t* pixels);
	SDL_GPUTexture* CreateTexture(SDL_GPUDevice* gpuDevice, int width, int height, const std::uint8_t* pixels, bool srgb);
	SDL_GPUTexture* UploadTexture(SDL_GPUDevice* gpuDevice, int width, int height, const void* pixels, SDL_GPUTextureFormat format, Uint32 stride, Uint32 rowPitch = 0, bool mipmaps = false, bool masked = false);
	SDL_GPUTexture* LoadTexture(std::string_view fileName);
	SDL_GPUTexture* LoadTexture(std::string_view fileName, bool srgb);
	SDL_GPUTexture* LoadTexture(std::string_view fileName, bool srgb, bool mipmaps);

	SDL_GPUTransferBuffer* CreateTransferBuffer(SDL_GPUDevice* device, Uint32 size);
	void ReleaseTransferBuffer(SDL_GPUDevice* device, SDL_GPUTransferBuffer* buffer);
	void ReleaseBuffer(SDL_GPUDevice* device, SDL_GPUBuffer* buffer);
	void ReleaseTexture(SDL_GPUDevice* device, SDL_GPUTexture* texture);
	void FlushUploads();

	namespace {
		using namespace Math;
		constexpr int shadowMapCount = 3;
		constexpr int maxPointLights = 16;
		constexpr float shadowSlopeFactor = 3.0f;
		Uint32 shadowMapResolution = 2048;
		using ChunkKey = std::tuple<int, int, int>;

		struct GpuMesh;

		struct RenderItem {
			Mesh* mesh;
			Matrix4 model;
			Matrix4 normal;
			int cellX = 0;
			int cellY = 0;
			int cellZ = 0;
			Bounds bounds;
			GpuMesh* gpu = nullptr;
			Uint32 lightMask = 0;
		};

		struct ObjectUniforms {
			float mvp[16];
			float model[16];
			float normal[16];
			float displacementParameters[4];
			float uvTransform[4];
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
			std::vector<RenderItem> staticCasters;
			std::vector<RenderItem> dynamicCasters;
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

		using MaterialTextures = std::array<std::string, 6>;

		struct GpuMaterial {
			const MaterialTextures* source = nullptr;
			std::array<SDL_GPUTexture*, 6> textures{};
			std::size_t references = 0;
			Uint32 flags = 0;
			bool masked = false;
		};

		struct GpuMesh {
			SDL_GPUBuffer* vertexBuffer = nullptr;
			SDL_GPUBuffer* indexBuffer = nullptr;
			GpuMaterial* material = nullptr;
			Uint32 indexCount = 0;
		};

		struct PendingGpuSubmission {
			SDL_GPUFence* fence;
			Uint64 submittedAt;
		};

		SDL_GPUDevice* gpuDevice = nullptr;
		SDL_GPUGraphicsPipeline* pipelines[2][2]{};
		SDL_GPUGraphicsPipeline* transparentPipelines[2]{};
		SDL_GPUGraphicsPipeline* shadowPipelines[2][2]{};
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
		int batchChunkSize = -1;
		bool loadingFrame = true;
		const Light* shadowLight = nullptr;
		std::map<ChunkKey, StaticShadowMap> staticShadowMaps;
		std::unordered_map<Mesh*, ChunkKey> staticBatchCells;
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
		std::uint64_t submitTimeNanoseconds = 0;
		std::uint64_t gpuCompletionLatencyNanoseconds = 0;
		bool hasGpuCompletionLatency = false;
		std::string fpsString = "0";
		std::string triangleCountString = "0/0";
		std::unordered_map<Mesh*, GpuMesh> gpuMeshes;
		std::map<MaterialTextures, GpuMaterial> gpuMaterials;
		std::unordered_map<Mesh*, Bounds> meshBounds;
		std::unordered_map<std::string, SDL_GPUTexture*> textures;
		std::deque<PendingGpuSubmission> pendingGpuSubmissions;
		std::vector<std::unique_ptr<Mesh>> staticBatches;
		std::vector<RenderItem> staticDraws;
		std::vector<RenderItem> renderItems;
		std::vector<RenderItem> staticItems;
		std::vector<RenderItem> dynamicItems;
		std::vector<RenderItem> visibleItems;
		std::vector<RenderItem> shadowCasters;
		std::vector<RenderItem> faceCasters;
		std::vector<PointLightData> pointLights;
		std::vector<ChunkKey> visibleCells;
		struct MeshSource {
			Mesh* mesh = nullptr;
			std::uint64_t signature = 0;
			std::uint64_t geometrySignature = 0;
			bool batched = false;
			bool staticCaster = false;
			bool seen = false;
			std::set<ChunkKey> cells;
			RenderItem draw{};
			bool drawReady = false;
			Bounds shadowBounds{};
		};
		std::unordered_map<const Object*, MeshSource> meshSources;
		std::set<ChunkKey> dirtyCells;
		std::vector<Bounds> shadowChanges;

		std::unordered_map<SDL_GPUTexture*, bool> maskedTextures;
		std::unordered_map<SDL_GPUTexture*, std::size_t> textureReferences;
		SDL_GPUCommandBuffer* uploadCommandBuffer = nullptr;
		SDL_GPUCopyPass* uploadPass = nullptr;
		std::vector<SDL_GPUTransferBuffer*> uploadTransfers;


		SDL_GPUCopyPass* BeginUploads() {
			if (!uploadCommandBuffer) uploadCommandBuffer = SDL_AcquireGPUCommandBuffer(gpuDevice);
			if (!uploadPass) uploadPass = SDL_BeginGPUCopyPass(uploadCommandBuffer);
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

		std::uint64_t GeometrySignature(const Mesh& mesh) {
			std::uint64_t hash = 14695981039346656037ull;
			AddHash(hash, mesh.revision);
			AddHash(hash, mesh.data.vertices.size());
			AddHash(hash, mesh.data.indices.size());
			AddHash(hash, mesh.data.vertices.data());
			AddHash(hash, mesh.data.indices.data());
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
			AddHash(hash, mesh.windAnimated);
			AddHash(hash, mesh.castShadows);
			AddHash(hash, mesh.cullBackfaces);
			std::apply([&](const auto&... values) { (AddHash(hash, values), ...); }, mesh.material.Key());
			AddHash(hash, mesh.data.vertices.size());
			AddHash(hash, mesh.data.indices.size());
			AddHash(hash, mesh.data.vertices.data());
			AddHash(hash, mesh.data.indices.data());
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

		const std::string deformationShaderSource = R"(
			#include <metal_stdlib>
			using namespace metal;
			float3 Deform(float3 position, float3 normal, float2 uv, float4 color, float4x4 model, float4 parameters, texture2d<float> heightMap, sampler heightSampler, texture2d<float> maskMap, sampler maskSampler) {
				if (parameters.x != 0.0) position += normal * ((heightMap.sample(heightSampler, uv, level(0.0)).r - 0.5) * maskMap.sample(maskSampler, uv, level(0.0)).r * parameters.x);
				if (parameters.y > 0.5) position.x += color.r * length(model[1].xyz) / max(length(model[0].xyz), 0.0001) * sin(parameters.z * 1.5 + color.a * 6.2831853) * parameters.w;
				return position;
			}
		)";

		const std::string vertexShaderSource = deformationShaderSource + R"(
			struct VertexInput { float3 position [[attribute(0)]]; float3 normal [[attribute(1)]]; float2 uv [[attribute(2)]]; float4 color [[attribute(3)]]; };
			struct VertexOutput { float4 position [[position]]; float2 uv; float4 color; float3 worldPosition; float3 worldNormal; float4 shadowPosition0; float4 shadowPosition1; float4 shadowPosition2; };
			struct ObjectUniforms { float4x4 mvp; float4x4 model; float4x4 normal; float4 displacementParameters; float4 uvTransform; float4x4 shadowMvp[3]; float4 shadowState; };
			vertex VertexOutput main0(VertexInput input [[stage_in]], constant ObjectUniforms& uniforms [[buffer(0)]], texture2d<float> heightMap [[texture(0)]], sampler heightSampler [[sampler(0)]], texture2d<float> maskMap [[texture(1)]], sampler maskSampler [[sampler(1)]]) {
				VertexOutput output;
				float2 uv = input.uv * uniforms.uvTransform.xy + uniforms.uvTransform.zw;
				float3 position = Deform(input.position, input.normal, uv, input.color, uniforms.model, uniforms.displacementParameters, heightMap, heightSampler, maskMap, maskSampler);
				output.position = uniforms.mvp * float4(position, 1.0);
				output.uv = uv;
				output.color = uniforms.displacementParameters.y > 0.5 ? float4(1.0) : input.color;
				output.worldPosition = (uniforms.model * float4(position, 1.0)).xyz;
				output.worldNormal = normalize((uniforms.normal * float4(input.normal, 0.0)).xyz);
				output.shadowPosition0 = output.shadowPosition1 = output.shadowPosition2 = float4(0.0);
				if (uniforms.shadowState.x > 0.5) {
					output.shadowPosition0 = uniforms.shadowMvp[0] * float4(output.worldPosition, 1.0);
					output.shadowPosition1 = uniforms.shadowMvp[1] * float4(output.worldPosition, 1.0);
					output.shadowPosition2 = uniforms.shadowMvp[2] * float4(output.worldPosition, 1.0);
				}
				return output;
			}
		)";

		constexpr const char* fragmentShaderSource = R"(
			#include <metal_stdlib>
			using namespace metal;
			struct FragmentInput { float4 position [[position]]; float2 uv; float4 color; float3 worldPosition; float3 worldNormal; float4 shadowPosition0; float4 shadowPosition1; float4 shadowPosition2; };
			struct FragmentUniforms { float4 shadowParameters[3]; float4 lightingParameters; float4 baseColor; float4 surfaceParameters; uint4 pointLightIndices[4]; };
			struct FrameUniforms { float4 cameraPosition; float4 fogColor; float4 fogParameters; float4 ambientLight; float4 directionalDirection; float4 directionalColor; float4 pointLightPositionRange[16]; float4 pointLightColorIntensity[16]; float4 pointLightShadowInfo[16]; float4 pointLightShadowBias[16]; };
			float ShadowCoverage(float4 position, float4 parameters) {
				if (parameters.x < 0.5) return 0.0;
				float3 projected = position.xyz / position.w;
				float2 uv = float2(projected.x * 0.5 + 0.5, 0.5 - projected.y * 0.5);
				float border = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
				if (projected.z < 0.0 || (projected.z > 1.0 && parameters.w < 0.5)) return 0.0;
				return smoothstep(parameters.z * 2.0, parameters.z * 6.0, border);
			}
			float ShadowAmount(float4 position, depth2d<float> shadowMap, sampler shadowSampler, float4 parameters) {
				if (ShadowCoverage(position, parameters) == 0.0) return 0.0;
				float3 projected = position.xyz / position.w;
				float2 uv = float2(projected.x * 0.5 + 0.5, 0.5 - projected.y * 0.5);
				float visibility = 0.0;
				for (int y = 0; y < 2; ++y) {
					for (int x = 0; x < 2; ++x) visibility += shadowMap.sample_compare(shadowSampler, uv + (float2(x, y) - 0.5) * parameters.z, min(projected.z, 1.0) - parameters.y);
				}
				return 1.0 - visibility * 0.25;
			}
			float PointShadowVisibility(float3 direction, float receiverDistance, float range, float bias, uint cubeIndex, depth2d_array<float> shadowMap, sampler shadowSampler) {
				float faceDepth = max(max(abs(direction.x), abs(direction.y)), abs(direction.z)) * receiverDistance;
				float nearClip = min(0.1, range * 0.01);
				float compareDistance = max(faceDepth - bias, nearClip);
				float compareDepth = range / (range - nearClip) - nearClip * range / ((range - nearClip) * compareDistance);
				float3 magnitude = abs(direction);
				float2 uv;
				uint face;
				if (magnitude.x >= magnitude.y && magnitude.x >= magnitude.z) {
					face = direction.x > 0.0 ? 0u : 1u;
					uv = float2(direction.x > 0.0 ? -direction.z : direction.z, -direction.y) / magnitude.x;
				}
				else if (magnitude.y >= magnitude.z) {
					face = direction.y > 0.0 ? 2u : 3u;
					uv = float2(direction.x, direction.y > 0.0 ? direction.z : -direction.z) / magnitude.y;
				}
				else {
					face = direction.z > 0.0 ? 4u : 5u;
					uv = float2(direction.z > 0.0 ? direction.x : -direction.x, -direction.y) / magnitude.z;
				}
				return shadowMap.sample_compare(shadowSampler, uv * 0.5 + 0.5, cubeIndex * 6u + face, compareDepth);
			}
			float PointShadowAmount(float3 direction, float range, float bias, uint cubeIndex, depth2d_array<float> shadowMap, sampler shadowSampler, float texelSize) {
				float receiverDistance = length(direction);
				if (receiverDistance <= 0.0001 || receiverDistance >= range) return 0.0;
				float3 normalizedDirection = direction / receiverDistance;
				float3 axis = abs(normalizedDirection.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
				float3 tangent = normalize(cross(normalizedDirection, axis));
				float3 bitangent = cross(normalizedDirection, tangent);
				float filterSize = texelSize * 2.0;
				float visibility = 0.0;
				visibility += PointShadowVisibility(normalize(normalizedDirection + (tangent + bitangent) * filterSize), receiverDistance, range, bias, cubeIndex, shadowMap, shadowSampler);
				visibility += PointShadowVisibility(normalize(normalizedDirection + (tangent - bitangent) * filterSize), receiverDistance, range, bias, cubeIndex, shadowMap, shadowSampler);
				visibility += PointShadowVisibility(normalize(normalizedDirection + (-tangent + bitangent) * filterSize), receiverDistance, range, bias, cubeIndex, shadowMap, shadowSampler);
				visibility += PointShadowVisibility(normalize(normalizedDirection - (tangent + bitangent) * filterSize), receiverDistance, range, bias, cubeIndex, shadowMap, shadowSampler);
				return 1.0 - visibility * 0.25;
			}

			struct Surface {
				float3 normal;
				float3 view;
				float3 albedo;
				float3 f0;
				float metallic;
				float alphaSquared;
				float nDotV;
				float viewVisibility;
			};
			float3 EvaluateLight(Surface surface, float3 light) {
				float nDotL = max(dot(surface.normal, light), 0.0);
				if (nDotL <= 0.0) return float3(0.0);
				float3 halfVector = surface.view + light;
				halfVector *= rsqrt(max(dot(halfVector, halfVector), 0.000001));
				float nDotH = clamp(dot(surface.normal, halfVector), 0.0, 1.0);
				float vDotH = clamp(dot(surface.view, halfVector), 0.0, 1.0);
				float denominator = nDotH * nDotH * (surface.alphaSquared - 1.0) + 1.0;
				float distribution = surface.alphaSquared / max(3.14159265 * denominator * denominator, 0.000000000001);
				float visibility = 0.5 / max(nDotL * surface.viewVisibility + surface.nDotV * sqrt(nDotL * nDotL * (1.0 - surface.alphaSquared) + surface.alphaSquared), 0.00000001);
				float fresnelFactor = 1.0 - vDotH;
				float fresnelSquared = fresnelFactor * fresnelFactor;
				float fresnelWeight = fresnelSquared * fresnelSquared * fresnelFactor;
				float3 fresnel = surface.f0 + (1.0 - surface.f0) * fresnelWeight;
				float3 diffuse = (1.0 - fresnel) * (1.0 - surface.metallic) * surface.albedo / 3.14159265;
				return (diffuse + distribution * visibility * fresnel) * nDotL;
			}

			float3 EncodeColor(float3 color) {
				color = max(color, float3(0.0));
				return select(1.055 * pow(color, float3(1.0 / 2.4)) - 0.055, color * 12.92, color <= 0.0031308);
			}
			fragment float4 main0(FragmentInput input [[stage_in]], constant FragmentUniforms& uniforms [[buffer(0)]], constant FrameUniforms& frame [[buffer(1)]], texture2d<float> image [[texture(0)]], sampler imageSampler [[sampler(0)]], depth2d<float> detailMap [[texture(1)]], sampler detailSampler [[sampler(1)]], depth2d<float> fallbackMap [[texture(2)]], sampler fallbackSampler [[sampler(2)]], depth2d<float> dynamicMap [[texture(3)]], sampler dynamicSampler [[sampler(3)]], depth2d_array<float> pointShadowMap [[texture(4)]], sampler pointShadowSampler [[sampler(4)]], texture2d<float> normalMap [[texture(5)]], sampler normalSampler [[sampler(5)]], texture2d<float> roughnessMap [[texture(6)]], sampler roughnessSampler [[sampler(6)]], texture2d<float> aoMap [[texture(7)]], sampler aoSampler [[sampler(7)]]) {
				uint materialFlags = uint(uniforms.lightingParameters.x);
				float3 normal = normalize(input.worldNormal);
				if ((materialFlags & 9u) != 0u) {
					float3 dpdx = dfdx(input.worldPosition);
					float3 dpdy = dfdy(input.worldPosition);
					if ((materialFlags & 8u) != 0u) {
						float3 geometricNormal = cross(dpdx, dpdy);
						geometricNormal *= rsqrt(max(dot(geometricNormal, geometricNormal), 1e-20));
						if (dot(geometricNormal, normal) < 0.0) geometricNormal = -geometricNormal;
						normal = geometricNormal;
					}
					if ((materialFlags & 1u) != 0u) {
						float2 duvdx = dfdx(input.uv);
						float2 duvdy = dfdy(input.uv);
						float determinant = duvdx.x * duvdy.y - duvdx.y * duvdy.x;
						if (determinant * determinant > 1e-8 * dot(duvdx, duvdx) * dot(duvdy, duvdy)) {
							float3 tangent = (dpdx * duvdy.y - dpdy * duvdx.y) / determinant;
							float3 uvBitangent = (dpdy * duvdx.x - dpdx * duvdy.x) / determinant;
							tangent -= normal * dot(normal, tangent);
							tangent *= rsqrt(max(dot(tangent, tangent), 1e-20));
							float3 bitangent = cross(normal, tangent);
							if (dot(bitangent, uvBitangent) < 0.0) bitangent = -bitangent;
							float3 mappedNormal = normalMap.sample(normalSampler, input.uv).xyz * 2.0 - 1.0;
							mappedNormal.xy *= float2(uniforms.surfaceParameters.y, uniforms.surfaceParameters.y * uniforms.surfaceParameters.w);
							normal = normalize(tangent * mappedNormal.x + bitangent * mappedNormal.y + normal * mappedNormal.z);
						}
					}
				}
				float4 texel = image.sample(imageSampler, input.uv);
				#if ALPHA_MASKED
				if (texel.a < 0.5) discard_fragment();
				#endif
				float fog = frame.fogParameters.z > 0.5 ? clamp((distance(input.worldPosition, frame.cameraPosition.xyz) - frame.fogParameters.x) / max(frame.fogParameters.y - frame.fogParameters.x, 0.0001), 0.0, 1.0) : 0.0;
				float opacity = input.color.a * texel.a * uniforms.lightingParameters.y;
				if (fog >= 1.0) return float4(EncodeColor(frame.fogColor.rgb), opacity);
				if ((materialFlags & 16u) != 0u) {
					float3 tint = input.color.rgb * uniforms.baseColor.rgb * texel.rgb;
					return float4(EncodeColor(mix(tint, frame.fogColor.rgb, fog)), opacity);
				}
				float3 albedo = input.color.rgb * texel.rgb * uniforms.baseColor.rgb;
				float roughness = clamp(uniforms.surfaceParameters.x * ((materialFlags & 2u) != 0u ? roughnessMap.sample(roughnessSampler, input.uv).r : 1.0), 0.045, 1.0);
				float ao = (materialFlags & 4u) != 0u ? mix(1.0, aoMap.sample(aoSampler, input.uv).r, uniforms.surfaceParameters.z) : 1.0;
				float metallic = uniforms.baseColor.w;
				float3 view = frame.cameraPosition.xyz - input.worldPosition;
				view *= rsqrt(max(dot(view, view), 0.000001));
				float alphaSquared = roughness * roughness * roughness * roughness;
				float nDotV = max(dot(normal, view), 0.0001);
				Surface surface{normal, view, albedo, mix(float3(0.04), albedo, metallic), metallic, alphaSquared, nDotV, sqrt(nDotV * nDotV * (1.0 - alphaSquared) + alphaSquared)};
				float diffuse = max(dot(normal, -frame.directionalDirection.xyz), 0.0) * frame.directionalDirection.w;
				bool directionalShadow = uniforms.lightingParameters.z > 0.5 && diffuse > 0.0;
				float staticShadow = 0.0;
				if (directionalShadow) {
					float detailCoverage = ShadowCoverage(input.shadowPosition0, uniforms.shadowParameters[0]);
					if (detailCoverage < 1.0) staticShadow = ShadowAmount(input.shadowPosition1, fallbackMap, fallbackSampler, uniforms.shadowParameters[1]);
					if (detailCoverage > 0.0) staticShadow = mix(staticShadow, ShadowAmount(input.shadowPosition0, detailMap, detailSampler, uniforms.shadowParameters[0]), detailCoverage);
				}
				float dynamicShadow = 0.0;
				if (directionalShadow) dynamicShadow = ShadowAmount(input.shadowPosition2, dynamicMap, dynamicSampler, uniforms.shadowParameters[2]);
				float3 direct = EvaluateLight(surface, -frame.directionalDirection.xyz) * frame.directionalColor.rgb * frame.directionalDirection.w * (1.0 - staticShadow) * (1.0 - dynamicShadow);
				float3 pointLighting = float3(0.0);
				for (int light = 0; light < int(uniforms.lightingParameters.w); ++light) {
					uint index = uniforms.pointLightIndices[light / 4][light % 4];
					float4 positionRange = frame.pointLightPositionRange[index];
					float4 colorIntensity = frame.pointLightColorIntensity[index];
					float3 toLight = positionRange.xyz - input.worldPosition;
					float distanceSquared = dot(toLight, toLight);
					if (distanceSquared < positionRange.w * positionRange.w) {
						float distanceToLight = sqrt(distanceSquared);
						float pointDiffuse = max(dot(normal, toLight / max(distanceToLight, 0.0001)), 0.0);
						if (pointDiffuse <= 0.0) continue;
						float rangeFade = clamp(1.0 - distanceToLight / positionRange.w, 0.0, 1.0);
						float attenuation = rangeFade * rangeFade / max(distanceSquared, 1.0);
						float4 shadowInfo = frame.pointLightShadowInfo[index];
						float shadowBias = frame.pointLightShadowBias[index].x;
						float staticPointShadow = uniforms.lightingParameters.z > 0.5 && shadowInfo.z > 0.5 ? PointShadowAmount(-toLight, positionRange.w, shadowBias, uint(shadowInfo.x), pointShadowMap, pointShadowSampler, uniforms.shadowParameters[0].z) : 0.0;
						float dynamicPointShadow = uniforms.lightingParameters.z > 0.5 && shadowInfo.w > 0.5 ? PointShadowAmount(-toLight, positionRange.w, shadowBias, uint(shadowInfo.y), pointShadowMap, pointShadowSampler, uniforms.shadowParameters[0].z) : 0.0;
						pointLighting += EvaluateLight(surface, toLight / max(distanceToLight, 0.0001)) * colorIntensity.rgb * colorIntensity.a * attenuation * (1.0 - staticPointShadow) * (1.0 - dynamicPointShadow);
					}
				}
				float3 ambient = frame.ambientLight.a > 0.5 ? frame.ambientLight.rgb : float3(0.0);
				float3 color = albedo * ambient * ao * (1.0 - metallic) + direct + pointLighting;
				return float4(EncodeColor(mix(color, frame.fogColor.rgb, fog)), opacity);
			}
		)";

		const std::string shadowVertexShaderSource = deformationShaderSource + R"(
			struct VertexInput { float3 position [[attribute(0)]]; float3 normal [[attribute(1)]]; float2 uv [[attribute(2)]]; float4 color [[attribute(3)]]; };
			struct VertexOutput { float4 position [[position]]; float2 uv; };
			struct ShadowUniforms { float4x4 mvp; float4x4 model; float4 displacementParameters; float4 uvTransform; };
			vertex VertexOutput main0(VertexInput input [[stage_in]], constant ShadowUniforms& uniforms [[buffer(0)]], texture2d<float> heightMap [[texture(0)]], sampler heightSampler [[sampler(0)]], texture2d<float> maskMap [[texture(1)]], sampler maskSampler [[sampler(1)]]) {
				VertexOutput output;
				float2 uv = input.uv * uniforms.uvTransform.xy + uniforms.uvTransform.zw;
				float3 position = Deform(input.position, input.normal, uv, input.color, uniforms.model, uniforms.displacementParameters, heightMap, heightSampler, maskMap, maskSampler);
				output.position = uniforms.mvp * float4(position, 1.0);
				output.uv = uv;
				return output;
			}
		)";

		constexpr const char* shadowFragmentShaderSource = R"(
			#include <metal_stdlib>
			using namespace metal;
			struct FragmentInput { float4 position [[position]]; float2 uv; };
			#if ALPHA_MASKED
			fragment void main0(FragmentInput input [[stage_in]], texture2d<float> image [[texture(0)]], sampler imageSampler [[sampler(0)]]) {
				if (image.sample(imageSampler, input.uv).a < 0.5) discard_fragment();
			}
			#else
			fragment void main0() {}
			#endif
		)";

		Transform CameraWorldTransform(const Camera& camera) {
			Transform transform = camera.owner->transform;
			Vector3 axes[3];
			GetAxes(transform, axes);
			transform.position += Add(Scale(axes[0], camera.localPosition.x), Add(Scale(axes[1], camera.localPosition.y), Scale(axes[2], camera.localPosition.z)));
			transform.rotation += camera.localRotation;
			transform.scale = {1.0f, 1.0f, 1.0f};
			return transform;
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

		LightData CollectLights(Scene& scene) {
			LightData directional{};
			pointLights.clear();
			for (auto& object : scene.Objects()) {
				auto* light = object->GetComponent<Light>();
				if (!light || !light->enabled || light->intensity <= 0.0f) continue;
				if (light->type == Light::Type::Directional && !directional.source) directional = {Normalize(object->transform.Forward()), light->color, light->intensity, light->castShadows, light->shadowBias, object->renderStatic, light};
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
			};

			struct Basic {
				float behindCameraDistance = 1.0f;
				float fovPadding = 1.05f;
			};
			constexpr Basic basic{};

			Frustum FromCamera(const Camera& camera, float aspectRatio) {
				const Transform transform = CameraWorldTransform(camera);
				const float pitch = transform.rotation.x * Math::degreesToRadians;
				const float yaw = transform.rotation.y * Math::degreesToRadians;
				const float cosPitch = std::cos(pitch);
				const float sinPitch = std::sin(pitch);
				const float cosYaw = std::cos(yaw);
				const float sinYaw = std::sin(yaw);
				return {
					transform.position,
					{cosYaw, 0.0f, -sinYaw},
					{sinYaw * sinPitch, cosPitch, cosYaw * sinPitch},
					{sinYaw * cosPitch, -sinPitch, cosYaw * cosPitch},
					std::tan(camera.fieldOfView * Math::degreesToRadians * 0.5f),
					aspectRatio,
					camera.farClip
				};
			}

			const Bounds& GetBounds(Mesh* mesh) {
				auto [entry, inserted] = meshBounds.try_emplace(mesh);
				if (inserted) {
					float windPadding = 0.0f;
					entry->second.minimum = mesh->data.vertices.front().position;
					entry->second.maximum = mesh->data.vertices.front().position;
					for (const auto& vertex : mesh->data.vertices) {
						Include(entry->second, vertex.position);
						if (mesh->windAnimated) windPadding = std::max(windPadding, std::abs(vertex.color[0]) * 0.08f);
					}
					if (mesh->windAnimated) {
						entry->second.minimum.x -= windPadding;
						entry->second.maximum.x += windPadding;
					}
					if (!mesh->material.displacementTexturePath.empty()) {
						const float displacementPadding = std::abs(mesh->material.displacementScale);
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
				bool outsideLeft = true;
				bool outsideRight = true;
				bool outsideBottom = true;
				bool outsideTop = true;
				bool outsideFar = true;
				bool outsideBehind = true;
				const float halfHeightScale = frustum.tanHalfFov * basic.fovPadding;
				const float halfWidthScale = halfHeightScale * frustum.aspectRatio;
				for (int corner = 0; corner < 8; ++corner) {
					const Vector3 localPoint{
						corner & 1 ? bounds.maximum.x : bounds.minimum.x,
						corner & 2 ? bounds.maximum.y : bounds.minimum.y,
						corner & 4 ? bounds.maximum.z : bounds.minimum.z
					};
					const Vector3& point = localPoint;
					const float x = point.x - frustum.position.x;
					const float y = point.y - frustum.position.y;
					const float z = point.z - frustum.position.z;
					const float depth = x * frustum.forward.x + y * frustum.forward.y + z * frustum.forward.z;
					const float horizontal = x * frustum.right.x + y * frustum.right.y + z * frustum.right.z;
					const float vertical = x * frustum.up.x + y * frustum.up.y + z * frustum.up.z;
					outsideLeft = outsideLeft && horizontal < -depth * halfWidthScale;
					outsideRight = outsideRight && horizontal > depth * halfWidthScale;
					outsideBottom = outsideBottom && vertical < -depth * halfHeightScale;
					outsideTop = outsideTop && vertical > depth * halfHeightScale;
					outsideFar = outsideFar && depth > frustum.farClip;
					outsideBehind = outsideBehind && depth < -basic.behindCameraDistance;
				}
				return !(outsideLeft || outsideRight || outsideBottom || outsideTop || outsideFar || outsideBehind);
			}
		}

		void PointShadowCasters(const std::vector<RenderItem>& items, const Vector3& position, float range, std::vector<RenderItem>& casters) {
			casters.clear();
			for (const auto& item : items) if (item.mesh->castShadows && IntersectsSphere(item.bounds, position, range)) casters.push_back(item);
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

		ShadowMapData BuildStaticShadowMap(const Vector3& lightDirection, int cellX, int cellY, int cellZ, int chunkSize, const std::vector<RenderItem>& items, std::vector<RenderItem>& casters) {
			ShadowMapData shadowMap{};
			casters.clear();
			if (items.empty()) return shadowMap;
			const Vector3 direction = Normalize(lightDirection);
			const Vector3 referenceUp = std::abs(Dot(direction, {0.0f, 1.0f, 0.0f})) > 0.99f ? Vector3{1.0f, 0.0f, 0.0f} : Vector3{0.0f, 1.0f, 0.0f};
			const Vector3 right = Normalize(Cross(referenceUp, direction));
			const Vector3 up = Normalize(Cross(direction, right));
			float minimumX = std::numeric_limits<float>::max();
			float maximumX = std::numeric_limits<float>::lowest();
			float minimumY = std::numeric_limits<float>::max();
			float maximumY = std::numeric_limits<float>::lowest();
			float minimumDepth = std::numeric_limits<float>::max();
			float maximumDepth = std::numeric_limits<float>::lowest();
			if (chunkSize > 0) {
				for (int corner = 0; corner < 8; ++corner) {
					const Vector3 point{
						(corner & 1 ? cellX + 1 : cellX) * static_cast<float>(chunkSize),
						(corner & 2 ? cellY + 1 : cellY) * static_cast<float>(chunkSize),
						(corner & 4 ? cellZ + 1 : cellZ) * static_cast<float>(chunkSize)
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
				for (const RenderItem& item : items) {
					const Bounds& bounds = item.bounds;
					for (int corner = 0; corner < 8; ++corner) {
						const Vector3 localPoint{corner & 1 ? bounds.maximum.x : bounds.minimum.x, corner & 2 ? bounds.maximum.y : bounds.minimum.y, corner & 4 ? bounds.maximum.z : bounds.minimum.z};
						const Vector3 point = localPoint;
						minimumX = std::min(minimumX, Dot(right, point));
						maximumX = std::max(maximumX, Dot(right, point));
						minimumY = std::min(minimumY, Dot(up, point));
						maximumY = std::max(maximumY, Dot(up, point));
						minimumDepth = std::min(minimumDepth, Dot(direction, point));
						maximumDepth = std::max(maximumDepth, Dot(direction, point));
					}
				}
			}
			const float extent = std::max(maximumX - minimumX, maximumY - minimumY);
			const float radius = extent * 0.5f + std::max(1.0f, extent * 4.0f / shadowMapResolution);
			const float texel = radius * 2.0f / shadowMapResolution;
			const float centerX = std::round((minimumX + maximumX) * 0.5f / texel) * texel;
			const float centerY = std::round((minimumY + maximumY) * 0.5f / texel) * texel;
			const Vector3 center = Add(Scale(right, centerX), Scale(up, centerY));
			bool hasCaster = false;
			for (const RenderItem& item : items) {
				if (!item.mesh->castShadows) continue;
				const Bounds& bounds = item.bounds;
				float itemMinimumX = std::numeric_limits<float>::max();
				float itemMaximumX = std::numeric_limits<float>::lowest();
				float itemMinimumY = std::numeric_limits<float>::max();
				float itemMaximumY = std::numeric_limits<float>::lowest();
				float itemMinimumDepth = std::numeric_limits<float>::max();
				for (int corner = 0; corner < 8; ++corner) {
					const Vector3 localPoint{corner & 1 ? bounds.maximum.x : bounds.minimum.x, corner & 2 ? bounds.maximum.y : bounds.minimum.y, corner & 4 ? bounds.maximum.z : bounds.minimum.z};
					const Vector3 point = localPoint;
					itemMinimumX = std::min(itemMinimumX, Dot(right, point));
					itemMaximumX = std::max(itemMaximumX, Dot(right, point));
					itemMinimumY = std::min(itemMinimumY, Dot(up, point));
					itemMaximumY = std::max(itemMaximumY, Dot(up, point));
					itemMinimumDepth = std::min(itemMinimumDepth, Dot(direction, point));
				}
				const bool overlaps = !(itemMaximumX < centerX - radius || itemMinimumX > centerX + radius || itemMaximumY < centerY - radius || itemMinimumY > centerY + radius);
				if (!overlaps || itemMinimumDepth > maximumDepth) continue;
				minimumDepth = std::min(minimumDepth, itemMinimumDepth);
				casters.push_back(item);
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

		ShadowMapData BuildDynamicShadowMap(const Camera& camera, float aspectRatio, float shadowFar, const LightData& light, const std::vector<RenderItem>& items, std::vector<RenderItem>& casters) {
			ShadowMapData map{};
			casters.clear();
			const auto frustum = Culling::FromCamera(camera, aspectRatio);
			const float near = std::max(camera.nearClip, 0.01f);
			const auto corners = FrustumCorners(frustum, near, std::max(near + 0.1f, std::min(camera.farClip, shadowFar)));
			const Vector3 direction = Normalize(light.direction);
			const Vector3 referenceUp = std::abs(direction.y) > 0.99f ? Vector3{1.0f, 0.0f, 0.0f} : Vector3{0.0f, 1.0f, 0.0f};
			const Vector3 right = Normalize(Cross(referenceUp, direction));
			const Vector3 up = Normalize(Cross(direction, right));
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
			for (const auto& item : items) {
				if (!item.mesh->castShadows) continue;
				const auto& bounds = item.bounds;
				float minX = std::numeric_limits<float>::max();
				float maxX = std::numeric_limits<float>::lowest();
				float minY = minX;
				float maxY = maxX;
				float depth = minX;
				for (int corner = 0; corner < 8; ++corner) {
					const Vector3 local{corner & 1 ? bounds.maximum.x : bounds.minimum.x, corner & 2 ? bounds.maximum.y : bounds.minimum.y, corner & 4 ? bounds.maximum.z : bounds.minimum.z};
					const Vector3 point = local;
					minX = std::min(minX, Dot(right, point));
					maxX = std::max(maxX, Dot(right, point));
					minY = std::min(minY, Dot(up, point));
					maxY = std::max(maxY, Dot(up, point));
					depth = std::min(depth, Dot(direction, point));
				}
				if (maxX < receiverMinX || minX > receiverMaxX || maxY < receiverMinY || minY > receiverMaxY || depth > maximumDepth) continue;
				casters.push_back(item);
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
			auto [entry, inserted] = gpuMaterials.try_emplace(std::move(key));
			auto& gpu = entry->second;
			if (inserted) {
				gpu.source = &entry->first;
				for (std::size_t index = 0; index < gpu.textures.size(); ++index) {
					SDL_GPUTexture* texture = entry->first[index].empty() ? nullptr : LoadTexture(entry->first[index], index == 0, index < 4);
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

		RenderItem PrepareDraw(Mesh* mesh, const Transform& transform, const ChunkKey& cell) {
			RenderItem item{mesh, Model(transform), NormalMatrix(transform), std::get<0>(cell), std::get<1>(cell), std::get<2>(cell)};
			item.bounds = TransformBounds(Culling::GetBounds(mesh), item.model);
			item.gpu = &GetMesh(mesh);
			return item;
		}

		void ReleaseMesh(Mesh* mesh, bool geometryOnly = false) {
			meshBounds.erase(mesh);
			const auto entry = gpuMeshes.find(mesh);
			if (entry == gpuMeshes.end()) return;
			auto& gpuMesh = entry->second;
			ReleaseBuffer(gpuDevice, gpuMesh.vertexBuffer);
			ReleaseBuffer(gpuDevice, gpuMesh.indexBuffer);
			gpuMesh.vertexBuffer = nullptr;
			gpuMesh.indexBuffer = nullptr;
			gpuMesh.indexCount = 0;
			if (geometryOnly) return;
			ReleaseMaterial(gpuMesh.material);
			gpuMeshes.erase(entry);
		}

		void BuildStaticBatches(Scene& scene, bool rebuildAll) {
			Loading::Timer loading("Static mesh batching and upload recording");
			std::vector<GpuMaterial*> retainedMaterials;
			for (auto batch = staticBatches.begin(); batch != staticBatches.end();) {
				Mesh* mesh = batch->get();
				if (rebuildAll || dirtyCells.count(staticBatchCells.at(mesh))) {
					const auto gpu = gpuMeshes.find(mesh);
					if (gpu != gpuMeshes.end() && gpu->second.material) {
						++gpu->second.material->references;
						retainedMaterials.push_back(gpu->second.material);
					}
					ReleaseMesh(mesh);
					staticBatchCells.erase(mesh);
					batch = staticBatches.erase(batch);
				}
				else ++batch;
			}
			using BatchKey = std::tuple<int, int, int, Material, bool, bool, bool>;
			using CellKey = std::tuple<int, int, int>;
			std::map<BatchKey, MeshData> batchData;
			for (auto& object : scene.Objects()) {
				if (!object->renderStatic) continue;
				Mesh* mesh = object->GetComponent<Mesh>();
				if (mesh && (mesh->material.transparent || !mesh->material.displacementTexturePath.empty())) continue;
				if (!mesh || mesh->data.vertices.empty() || mesh->data.indices.empty()) continue;
				if (!rebuildAll) {
					const auto& cells = meshSources.at(object.get()).cells;
					if (std::none_of(cells.begin(), cells.end(), [&](const ChunkKey& cell) { return dirtyCells.count(cell); })) continue;
				}
				const Matrix4 model = Model(object->transform);
				const Matrix4 normal = NormalMatrix(object->transform);
				std::map<CellKey, std::unordered_map<std::uint32_t, std::uint32_t>> vertexMaps;
				for (std::size_t index = 0; index + 2 < mesh->data.indices.size(); index += 3) {
					const std::uint32_t sourceIndices[3] = {mesh->data.indices[index], mesh->data.indices[index + 1], mesh->data.indices[index + 2]};
					const Vector3 worldPositions[3] = {
						TransformPoint(model, mesh->data.vertices[sourceIndices[0]].position),
						TransformPoint(model, mesh->data.vertices[sourceIndices[1]].position),
						TransformPoint(model, mesh->data.vertices[sourceIndices[2]].position)
					};
					const float centerX = (worldPositions[0].x + worldPositions[1].x + worldPositions[2].x) / 3.0f;
					const float centerY = (worldPositions[0].y + worldPositions[1].y + worldPositions[2].y) / 3.0f;
					const float centerZ = (worldPositions[0].z + worldPositions[1].z + worldPositions[2].z) / 3.0f;
					const int cellX = scene.chunkSize > 0 ? static_cast<int>(std::floor(centerX / scene.chunkSize)) : 0;
					const int cellY = scene.chunkSize > 0 ? static_cast<int>(std::floor(centerY / scene.chunkSize)) : 0;
					const int cellZ = scene.chunkSize > 0 ? static_cast<int>(std::floor(centerZ / scene.chunkSize)) : 0;
					const CellKey cellKey{cellX, cellY, cellZ};
					if (rebuildAll) meshSources.at(object.get()).cells.insert(cellKey);
					if (!rebuildAll && !dirtyCells.count(cellKey)) continue;
					auto& output = batchData[{cellX, cellY, cellZ, mesh->material, mesh->cullBackfaces, mesh->castShadows, mesh->windAnimated}];
					auto& vertexMap = vertexMaps[cellKey];
					for (int corner = 0; corner < 3; ++corner) {
						auto [entry, inserted] = vertexMap.try_emplace(sourceIndices[corner], static_cast<std::uint32_t>(output.vertices.size()));
						if (inserted) {
							MeshVertex vertex = mesh->data.vertices[sourceIndices[corner]];
							vertex.position = worldPositions[corner];
							vertex.normal = TransformNormal(normal, vertex.normal);
							if (mesh->windAnimated) vertex.color[0] *= std::abs(object->transform.scale.y);
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
				batch->material = std::get<3>(key);
				batch->cullBackfaces = std::get<4>(key);
				batch->castShadows = std::get<5>(key);
				batch->windAnimated = std::get<6>(key);
				const ChunkKey cellKey{std::get<0>(key), std::get<1>(key), std::get<2>(key)};
				staticBatchCells.emplace(batch.get(), cellKey);
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

		MeshData CreatePlane(int subdivisions) {
			MeshData mesh;
			subdivisions = std::max(subdivisions, 1);
			const int rowSize = subdivisions + 1;
			mesh.vertices.reserve(static_cast<std::size_t>(rowSize) * rowSize);
			mesh.indices.reserve(static_cast<std::size_t>(subdivisions) * subdivisions * 6);
			for (int z = 0; z <= subdivisions; ++z) {
				for (int x = 0; x <= subdivisions; ++x) {
					const float u = static_cast<float>(x) / subdivisions;
					const float v = static_cast<float>(z) / subdivisions;
					mesh.vertices.push_back(MakeVertex({u - 0.5f, 0.0f, v - 0.5f}, {0.0f, 1.0f, 0.0f}, u, v));
				}
			}
			for (int z = 0; z < subdivisions; ++z) {
				for (int x = 0; x < subdivisions; ++x) {
					const std::uint32_t first = z * rowSize + x;
					const std::uint32_t nextRow = first + rowSize;
					mesh.indices.insert(mesh.indices.end(), {first, nextRow, nextRow + 1, first, nextRow + 1, first + 1});
				}
			}
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
			SDL_GPUTransferBuffer* transferBuffer = CreateTransferBuffer(gpuDevice, size);
			void* mappedData = SDL_MapGPUTransferBuffer(gpuDevice, transferBuffer, false);
			std::memcpy(mappedData, data, size);
			SDL_UnmapGPUTransferBuffer(gpuDevice, transferBuffer);
			SDL_GPUTransferBufferLocation source{transferBuffer, 0};
			SDL_GPUBufferRegion destination{buffer, 0, size};
			SDL_UploadToGPUBuffer(BeginUploads(), &source, &destination, false);
			uploadTransfers.push_back(transferBuffer);
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
			textureInfo.format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
			textureInfo.usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET;
			textureInfo.width = width;
			textureInfo.height = height;
			textureInfo.layer_count_or_depth = 1;
			textureInfo.num_levels = 1;
			textureInfo.sample_count = SDL_GPU_SAMPLECOUNT_1;
			return SDL_CreateGPUTexture(gpuDevice, &textureInfo);
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
			return SDL_CreateGPUTexture(gpuDevice, &textureInfo);
		}

		void EnsurePointShadowTexture(bool layoutChanged) {
			const Uint32 resolution = pointShadowSlots.empty() ? 1 : shadowMapResolution;
			if (!layoutChanged && resolution == pointShadowResolution && pointShadowTexture) return;
			if (pointShadowTexture) ReleaseTexture(gpuDevice, pointShadowTexture);
			pointShadowResolution = resolution;
			for (auto& state : pointShadowStates) state.staticReady = false;
			SDL_GPUTextureCreateInfo textureInfo{};
			textureInfo.type = SDL_GPU_TEXTURETYPE_2D_ARRAY;
			textureInfo.format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
			textureInfo.usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;
			textureInfo.width = resolution;
			textureInfo.height = resolution;
			textureInfo.layer_count_or_depth = static_cast<Uint32>(std::max<std::size_t>(pointShadowSlots.size(), 1) * 6);
			textureInfo.num_levels = 1;
			textureInfo.sample_count = SDL_GPU_SAMPLECOUNT_1;
			pointShadowTexture = SDL_CreateGPUTexture(gpuDevice, &textureInfo);
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

		GpuMesh& GetMesh(Mesh* mesh) {
			GpuMesh& gpuMesh = gpuMeshes.try_emplace(mesh).first->second;
			if (!gpuMesh.vertexBuffer && !mesh->data.vertices.empty() && !mesh->data.indices.empty()) {
				auto uploaded = UploadMesh(mesh->data);
				gpuMesh.vertexBuffer = uploaded.vertexBuffer;
				gpuMesh.indexBuffer = uploaded.indexBuffer;
				gpuMesh.indexCount = uploaded.indexCount;
			}
			const auto paths = MaterialPaths(mesh->material);
			bool changed = !gpuMesh.material;
			for (std::size_t index = 0; gpuMesh.material && index < paths.size(); ++index) changed = changed || *paths[index] != (*gpuMesh.material->source)[index];
			if (changed) {
				GpuMaterial* next = AcquireMaterial(mesh->material);
				ReleaseMaterial(gpuMesh.material);
				gpuMesh.material = next;
			}
			return gpuMesh;
		}

		void UpdateSceneResources(Scene& scene) {
			const bool rebuildAll = batchChunkSize != scene.chunkSize;
			shadowChanges.clear();
			dirtyCells.clear();
			sceneTriangleCount = 0;
			for (auto& [object, source] : meshSources) source.seen = false;
			for (auto& object : scene.Objects()) {
				Mesh* mesh = object->GetComponent<Mesh>();
				if (!mesh) continue;
				sceneTriangleCount += mesh->data.indices.size() / 3;
				auto [entry, inserted] = meshSources.try_emplace(object.get());
				auto& source = entry->second;
				source.seen = true;
				const auto signature = MeshSignature(*object, *mesh);
				if (!inserted && source.signature == signature && (!rebuildAll || !source.batched)) continue;
				dirtyCells.insert(source.cells.begin(), source.cells.end());
				if (source.staticCaster) shadowChanges.push_back(source.shadowBounds);
				const bool batched = object->renderStatic && !mesh->material.transparent && mesh->material.displacementTexturePath.empty();
				const auto geometrySignature = GeometrySignature(*mesh);
				if (source.mesh && (source.mesh != mesh || source.batched != batched)) ReleaseMesh(source.mesh);
				else if (!batched && source.geometrySignature != geometrySignature) ReleaseMesh(mesh, true);
				meshBounds.erase(mesh);
				source.geometrySignature = geometrySignature;
				source.mesh = mesh;
				source.signature = signature;
				source.drawReady = false;
				source.batched = batched;
				source.staticCaster = object->renderStatic && mesh->castShadows && !mesh->windAnimated && !mesh->data.vertices.empty() && !mesh->data.indices.empty();
				if (source.staticCaster) {
					source.shadowBounds = TransformBounds(Culling::GetBounds(mesh), Model(object->transform));
					shadowChanges.push_back(source.shadowBounds);
				}
				source.cells.clear();
				if (batched && !rebuildAll) {
					const Matrix4 model = Model(object->transform);
					for (std::size_t index = 0; index + 2 < mesh->data.indices.size(); index += 3) {
						Vector3 center{};
						for (int corner = 0; corner < 3; ++corner) center += TransformPoint(model, mesh->data.vertices[mesh->data.indices[index + corner]].position);
						center = center * (1.0f / 3.0f);
						source.cells.emplace(scene.chunkSize > 0 ? static_cast<int>(std::floor(center.x / scene.chunkSize)) : 0, scene.chunkSize > 0 ? static_cast<int>(std::floor(center.y / scene.chunkSize)) : 0, scene.chunkSize > 0 ? static_cast<int>(std::floor(center.z / scene.chunkSize)) : 0);
					}
					dirtyCells.insert(source.cells.begin(), source.cells.end());
				}
			}
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
			batchChunkSize = scene.chunkSize;
			if (rebuildAll || !shadowChanges.empty()) {
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
		}

		void MaterialUv(const Material& material, float (&uv)[4]) {
			uv[0] = material.uvScale[0];
			uv[1] = material.uvScale[1];
			uv[2] = material.uvOffset[0];
			uv[3] = material.uvOffset[1];
		}

		void RenderShadowMap(SDL_GPUCommandBuffer* commandBuffer, SDL_GPUTexture* texture, const ShadowMapData& shadowMap, const std::vector<RenderItem>& items, float time, Uint8 layer = 0, bool cullBackfaces = true) {
			SDL_GPUDepthStencilTargetInfo depthInfo{};
			depthInfo.texture = texture;
			depthInfo.layer = layer;
			depthInfo.clear_depth = 1.0f;
			depthInfo.load_op = SDL_GPU_LOADOP_CLEAR;
			depthInfo.store_op = SDL_GPU_STOREOP_STORE;
			depthInfo.stencil_load_op = SDL_GPU_LOADOP_DONT_CARE;
			depthInfo.stencil_store_op = SDL_GPU_STOREOP_DONT_CARE;
			SDL_GPURenderPass* shadowPass = SDL_BeginGPURenderPass(commandBuffer, nullptr, 0, &depthInfo);
			if (!shadowMap.valid) {
				SDL_EndGPURenderPass(shadowPass);
				return;
			}
			SDL_GPUGraphicsPipeline* activePipeline = nullptr;
			for (const RenderItem& item : items) {
				Mesh* mesh = item.mesh;
				if (!mesh->castShadows || !item.gpu || !item.gpu->indexCount) continue;
				GpuMesh& gpuMesh = *item.gpu;
				SDL_GPUGraphicsPipeline* meshPipeline = shadowPipelines[gpuMesh.material->masked ? 1 : 0][cullBackfaces && mesh->cullBackfaces ? 1 : 0];
				if (meshPipeline != activePipeline) {
					SDL_BindGPUGraphicsPipeline(shadowPass, meshPipeline);
					activePipeline = meshPipeline;
				}
				struct ShadowUniforms {
					float mvp[16];
					float model[16];
					float displacementParameters[4];
					float uvTransform[4];
				} uniforms{};
				const Matrix4 mvp = Multiply(shadowMap.mvp, item.model);
				std::memcpy(uniforms.mvp, mvp.values, sizeof(uniforms.mvp));
				std::memcpy(uniforms.model, item.model.values, sizeof(uniforms.model));
				uniforms.displacementParameters[0] = (gpuMesh.material->flags & 8u) ? mesh->material.displacementScale : 0.0f;
				MaterialUv(mesh->material, uniforms.uvTransform);
				uniforms.displacementParameters[1] = mesh->windAnimated ? 1.0f : 0.0f;
				uniforms.displacementParameters[2] = time;
				uniforms.displacementParameters[3] = 0.08f;
				SDL_PushGPUVertexUniformData(commandBuffer, 0, &uniforms, sizeof(uniforms));
				SDL_GPUBufferBinding vertexBinding{};
				vertexBinding.buffer = gpuMesh.vertexBuffer;
				SDL_BindGPUVertexBuffers(shadowPass, 0, &vertexBinding, 1);
				SDL_GPUBufferBinding indexBinding{};
				indexBinding.buffer = gpuMesh.indexBuffer;
				SDL_BindGPUIndexBuffer(shadowPass, &indexBinding, SDL_GPU_INDEXELEMENTSIZE_32BIT);
				SDL_GPUTextureSamplerBinding textureBinding{};
				textureBinding.texture = gpuMesh.material->textures[0];
				textureBinding.sampler = sampler;
				if (gpuMesh.material->masked) SDL_BindGPUFragmentSamplers(shadowPass, 0, &textureBinding, 1);
				SDL_GPUTextureSamplerBinding vertexTextureBindings[2]{};
				vertexTextureBindings[0].texture = gpuMesh.material->textures[4];
				vertexTextureBindings[0].sampler = sampler;
				vertexTextureBindings[1].texture = gpuMesh.material->textures[5];
				vertexTextureBindings[1].sampler = sampler;
				SDL_BindGPUVertexSamplers(shadowPass, 0, vertexTextureBindings, 2);
				SDL_DrawGPUIndexedPrimitives(shadowPass, gpuMesh.indexCount, 1, 0, 0, 0);
				++drawCallCount;
			}
			SDL_EndGPURenderPass(shadowPass);
		}

		void RenderPointShadowMap(SDL_GPUCommandBuffer* commandBuffer, int slot, const PointLightData& light, const std::vector<RenderItem>& casters, float time) {
			for (int face = 0; face < 6; ++face) {
				faceCasters.clear();
				const Culling::Frustum frustum{light.position, Cross(pointUps[face], pointDirections[face]), pointUps[face], pointDirections[face], 1.0f, 1.0f, light.range};
				for (const auto& item : casters) if (Culling::IsVisible(item.bounds, frustum)) faceCasters.push_back(item);
				RenderShadowMap(commandBuffer, pointShadowTexture, BuildPointShadowMap(light.position, light.range, face), faceCasters, time, static_cast<Uint8>(slot * 6 + face), false);
			}
		}

		void RenderPointShadows(Scene& scene, SDL_GPUCommandBuffer* commandBuffer, float time) {
			auto activeLight = [&](const Light* source) {
				return std::find_if(pointLights.begin(), pointLights.end(), [&](const PointLightData& light) { return scene.shadowsEnabled && light.castShadows && light.source == source; });
			};
			const auto previousSlotCount = pointShadowSlots.size();
			pointShadowSlots.erase(std::remove_if(pointShadowSlots.begin(), pointShadowSlots.end(), [&](const auto& slot) {
				const auto light = activeLight(slot.first);
				return light == pointLights.end() || (!slot.second && !light->renderStatic);
			}), pointShadowSlots.end());
			bool layoutChanged = previousSlotCount != pointShadowSlots.size();
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
				state->relevant = std::any_of(visibleItems.begin(), visibleItems.end(), [&](const RenderItem& item) { return (item.lightMask & (1u << static_cast<Uint32>(&light - pointLights.data()))) != 0; });
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
						layoutChanged = true;
					}
				}
			}
			EnsurePointShadowTexture(layoutChanged);
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

	SDL_GPUTexture* UploadTexture(SDL_GPUDevice* gpuDevice, int width, int height, const void* pixels, SDL_GPUTextureFormat format, Uint32 stride, Uint32 rowPitch, bool mipmaps, bool masked) {
		const bool generateOnGpu = mipmaps && !masked;
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
		SDL_GPUTexture* texture = SDL_CreateGPUTexture(gpuDevice, &textureInfo);
		const Uint32 rowBytes = static_cast<Uint32>(width) * stride;
		if (!rowPitch) rowPitch = rowBytes;
		Uint32 totalBytes = 0;
		for (Uint32 level = 0; level < uploadLevels; ++level) {
			totalBytes = (totalBytes + 255u) & ~255u;
			totalBytes += std::max(1, width >> level) * std::max(1, height >> level) * stride;
		}
		SDL_GPUTransferBuffer* transferBuffer = CreateTransferBuffer(gpuDevice, totalBytes);
		auto* mappedPixels = static_cast<std::uint8_t*>(SDL_MapGPUTransferBuffer(gpuDevice, transferBuffer, false));
		for (int row = 0; row < height; ++row) std::memcpy(mappedPixels + row * rowBytes, static_cast<const std::uint8_t*>(pixels) + row * rowPitch, rowBytes);
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
			for (int pixel = 0; pixel < width * height; ++pixel) if (mappedPixels[pixel * 4 + 3] >= 128) targetCoverage += 1.0f;
			targetCoverage /= static_cast<float>(width * height);
		}
		Uint32 offset = 0;
		Uint32 previousOffset = 0;
		for (Uint32 level = 0; level < uploadLevels; ++level) {
			offset = (offset + 255u) & ~255u;
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
						for (int sampleY = y * previousHeight / levelHeight; sampleY < (y + 1) * previousHeight / levelHeight; ++sampleY) {
							for (int sampleX = x * previousWidth / levelWidth; sampleX < (x + 1) * previousWidth / levelWidth; ++sampleX) {
								const auto* pixel = sourcePixels + (sampleY * previousWidth + sampleX) * 4;
								for (int channel = 0; channel < 3; ++channel) colors[channel] += linearColors[pixel[channel]] * pixel[3];
								alpha += pixel[3];
								++samples;
							}
						}
						auto* pixel = destinationPixels + (y * levelWidth + x) * 4;
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
						const float coverage = static_cast<float>(covered) / (levelWidth * levelHeight);
						const float error = std::abs(coverage - targetCoverage);
						if (error < bestError) {
							bestError = error;
							bestScale = scale;
						}
						if (coverage < targetCoverage) low = scale;
						else high = scale;
					}
					for (int pixel = 0; pixel < levelWidth * levelHeight; ++pixel) destinationPixels[pixel * 4 + 3] = static_cast<std::uint8_t>(std::min(255.0f, std::round(destinationPixels[pixel * 4 + 3] * bestScale)));
				}
			}
			previousOffset = offset;
			offset += levelWidth * levelHeight * stride;
		}
		SDL_UnmapGPUTransferBuffer(gpuDevice, transferBuffer);
		offset = 0;
		for (Uint32 level = 0; level < uploadLevels; ++level) {
			offset = (offset + 255u) & ~255u;
			SDL_GPUTextureTransferInfo source{};
			source.transfer_buffer = transferBuffer;
			source.offset = offset;
			source.pixels_per_row = std::max(1, width >> level);
			source.rows_per_layer = std::max(1, height >> level);
			SDL_GPUTextureRegion destination{};
			destination.texture = texture;
			destination.mip_level = level;
			destination.w = source.pixels_per_row;
			destination.h = source.rows_per_layer;
			destination.d = 1;
			SDL_UploadToGPUTexture(BeginUploads(), &source, &destination, false);
			offset += destination.w * destination.h * stride;
		}
		uploadTransfers.push_back(transferBuffer);
		if (generateOnGpu && textureInfo.num_levels > 1) {
			SDL_EndGPUCopyPass(uploadPass);
			uploadPass = nullptr;
			SDL_GenerateMipmapsForGPUTexture(uploadCommandBuffer, texture);
		}
		return texture;
	}

	SDL_GPUTexture* CreateTexture(SDL_GPUDevice* gpuDevice, int width, int height, const std::uint8_t* pixels) {
		return CreateTexture(gpuDevice, width, height, pixels, false);
	}

	SDL_GPUTexture* CreateTexture(SDL_GPUDevice* gpuDevice, int width, int height, const std::uint8_t* pixels, bool srgb) {
		return UploadTexture(gpuDevice, width, height, pixels, srgb ? SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB : SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, 4);
	}

	SDL_GPUTexture* LoadTexture(std::string_view fileName) { return LoadTexture(fileName, false); }

	SDL_GPUTexture* LoadTexture(std::string_view fileName, bool srgb) { return LoadTexture(fileName, srgb, srgb); }

	SDL_GPUTexture* LoadTexture(std::string_view fileName, bool srgb, bool mipmaps) {
		const std::string path = (std::filesystem::path(SDL_GetBasePath()) / ".." / "resources" / std::string(fileName)).lexically_normal().string();
		const std::string key = std::string(srgb ? "color:" : "data:") + (mipmaps ? "mips:" : "single:") + path;
		if (auto found = textures.find(key); found != textures.end()) {
			++textureReferences[found->second];
			return found->second;
		}
		Loading::Timer loading("Texture decode, mipmaps and upload recording: " + std::string(fileName));
		SDL_Surface* loadedSurface = SDL_LoadSurface(path.c_str());
		if (!loadedSurface) return nullptr;
		SDL_Surface* rgbaSurface = SDL_ConvertSurface(loadedSurface, SDL_PIXELFORMAT_RGBA32);
		SDL_DestroySurface(loadedSurface);
		if (!rgbaSurface) return nullptr;
		bool masked = false;
		for (int row = 0; row < rgbaSurface->h && !masked; ++row) {
			const auto* pixels = static_cast<const std::uint8_t*>(rgbaSurface->pixels) + row * rgbaSurface->pitch;
			for (int column = 0; column < rgbaSurface->w; ++column) if (pixels[column * 4 + 3] < 128) {
				masked = true;
				break;
			}
		}
		SDL_GPUTexture* texture = UploadTexture(gpuDevice, rgbaSurface->w, rgbaSurface->h, rgbaSurface->pixels, srgb ? SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB : SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, 4, static_cast<Uint32>(rgbaSurface->pitch), mipmaps, srgb && masked);
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
		SDL_ReleaseGPUTexture(device, texture);
	}

	void ReleaseBuffer(SDL_GPUDevice* device, SDL_GPUBuffer* buffer) {
		if (!buffer) return;
		SDL_ReleaseGPUBuffer(device, buffer);
	}

	SDL_GPUTransferBuffer* CreateTransferBuffer(SDL_GPUDevice* device, Uint32 size) {
		SDL_GPUTransferBufferCreateInfo info{};
		info.usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD;
		info.size = size;
		SDL_GPUTransferBuffer* buffer = SDL_CreateGPUTransferBuffer(device, &info);
		return buffer;
	}

	void ReleaseTransferBuffer(SDL_GPUDevice* device, SDL_GPUTransferBuffer* buffer) {
		if (!buffer) return;
		SDL_ReleaseGPUTransferBuffer(device, buffer);
	}

	void FlushUploads() {
		if (!uploadCommandBuffer) return;
		if (uploadPass) SDL_EndGPUCopyPass(uploadPass);
		SDL_SubmitGPUCommandBuffer(uploadCommandBuffer);
		uploadCommandBuffer = nullptr;
		uploadPass = nullptr;
		for (auto* buffer : uploadTransfers) ReleaseTransferBuffer(gpuDevice, buffer);
		uploadTransfers.clear();
	}

	SDL_GPUBuffer* CreateBuffer(SDL_GPUDevice* gpuDevice, SDL_GPUBufferUsageFlags usage, Uint32 size) {
		SDL_GPUBufferCreateInfo bufferInfo{};
		bufferInfo.usage = usage;
		bufferInfo.size = size;
		SDL_GPUBuffer* buffer = SDL_CreateGPUBuffer(gpuDevice, &bufferInfo);
		return buffer;
	}

	SDL_GPUSampler* CreateSampler(SDL_GPUDevice* gpuDevice, const SDL_GPUSamplerCreateInfo& samplerInfo) {
		return SDL_CreateGPUSampler(gpuDevice, &samplerInfo);
	}

	SDL_GPUGraphicsPipeline* CreatePipeline(SDL_GPUDevice* gpuDevice, const SDL_GPUGraphicsPipelineCreateInfo& pipelineInfo) {
		return SDL_CreateGPUGraphicsPipeline(gpuDevice, &pipelineInfo);
	}

	void Initialize(SDL_GPUDevice* device, SDL_Window* window) {
		Loading::Timer loading("Graphics shaders and pipelines");
		gpuDevice = device;
		SDL_GPUShader* vertexShader = CreateShader(gpuDevice, vertexShaderSource.c_str(), SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 2, 1);
		SDL_GPUShader* fragmentShaders[2]{};
		SDL_GPUShader* shadowVertexShader = CreateShader(gpuDevice, shadowVertexShaderSource.c_str(), SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 2, 1);
		SDL_GPUShader* shadowFragmentShaders[2]{};
		for (int masked = 0; masked < 2; ++masked) {
			const std::string define = masked ? "#define ALPHA_MASKED 1\n" : "#define ALPHA_MASKED 0\n";
			const std::string source = define + fragmentShaderSource;
			const std::string shadowSource = define + shadowFragmentShaderSource;
			fragmentShaders[masked] = CreateShader(gpuDevice, source.c_str(), SDL_GPU_SHADERSTAGE_FRAGMENT, SDL_GPU_SHADERFORMAT_MSL, 8, 2);
			shadowFragmentShaders[masked] = CreateShader(gpuDevice, shadowSource.c_str(), SDL_GPU_SHADERSTAGE_FRAGMENT, SDL_GPU_SHADERFORMAT_MSL, masked ? 1 : 0);
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
		SDL_GPUGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.vertex_shader = vertexShader;
		pipelineInfo.vertex_input_state.num_vertex_buffers = 1;
		pipelineInfo.vertex_input_state.vertex_buffer_descriptions = &vertexBufferDescription;
		pipelineInfo.vertex_input_state.num_vertex_attributes = 4;
		pipelineInfo.vertex_input_state.vertex_attributes = vertexAttributes;
		pipelineInfo.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
		pipelineInfo.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
		pipelineInfo.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_BACK;
		pipelineInfo.rasterizer_state.front_face = SDL_GPU_FRONTFACE_CLOCKWISE;
		pipelineInfo.multisample_state.sample_count = SDL_GPU_SAMPLECOUNT_1;
		pipelineInfo.depth_stencil_state.enable_depth_test = true;
		pipelineInfo.depth_stencil_state.enable_depth_write = true;
		pipelineInfo.depth_stencil_state.compare_op = SDL_GPU_COMPAREOP_LESS;
		pipelineInfo.target_info.num_color_targets = 1;
		pipelineInfo.target_info.color_target_descriptions = &colorTarget;
		pipelineInfo.target_info.has_depth_stencil_target = true;
		pipelineInfo.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
		for (int masked = 0; masked < 2; ++masked) {
			pipelineInfo.fragment_shader = fragmentShaders[masked];
			for (int cull = 0; cull < 2; ++cull) {
				pipelineInfo.rasterizer_state.cull_mode = cull ? SDL_GPU_CULLMODE_BACK : SDL_GPU_CULLMODE_NONE;
				pipelines[masked][cull] = CreatePipeline(gpuDevice, pipelineInfo);
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
			transparentPipelines[cull] = CreatePipeline(gpuDevice, pipelineInfo);
		}
		pipelineInfo.depth_stencil_state.enable_depth_write = true;
		pipelineInfo.vertex_shader = shadowVertexShader;
		pipelineInfo.vertex_input_state.num_vertex_attributes = 4;
		pipelineInfo.rasterizer_state.enable_depth_bias = true;
		pipelineInfo.rasterizer_state.depth_bias_constant_factor = 2.0f;
		pipelineInfo.rasterizer_state.depth_bias_slope_factor = shadowSlopeFactor;
		pipelineInfo.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_BACK;
		pipelineInfo.target_info.num_color_targets = 0;
		pipelineInfo.target_info.color_target_descriptions = nullptr;
		pipelineInfo.target_info.has_depth_stencil_target = true;
		pipelineInfo.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
		for (int masked = 0; masked < 2; ++masked) {
			pipelineInfo.fragment_shader = shadowFragmentShaders[masked];
			for (int cull = 0; cull < 2; ++cull) {
				pipelineInfo.rasterizer_state.cull_mode = cull ? SDL_GPU_CULLMODE_BACK : SDL_GPU_CULLMODE_NONE;
				shadowPipelines[masked][cull] = CreatePipeline(gpuDevice, pipelineInfo);
			}
		}
		SDL_ReleaseGPUShader(gpuDevice, vertexShader);
		for (auto* shader : fragmentShaders) SDL_ReleaseGPUShader(gpuDevice, shader);
		SDL_ReleaseGPUShader(gpuDevice, shadowVertexShader);
		for (auto* shader : shadowFragmentShaders) SDL_ReleaseGPUShader(gpuDevice, shader);
	}

	static void ReleaseSceneResources() {
		while (!gpuMeshes.empty()) ReleaseMesh(gpuMeshes.begin()->first);
		meshBounds.clear();
		meshSources.clear();
		sceneTriangleCount = 0;
		dirtyCells.clear();
		shadowChanges.clear();
		staticBatches.clear();
		staticDraws.clear();
		staticBatchCells.clear();
		renderItems.clear();
		staticItems.clear();
		dynamicItems.clear();
		visibleItems.clear();
		shadowCasters.clear();
		faceCasters.clear();
		pointLights.clear();
		visibleCells.clear();
		batchChunkSize = -1;
		for (auto& [key, shadowMap] : staticShadowMaps) if (shadowMap.texture) ReleaseTexture(gpuDevice, shadowMap.texture);
		staticShadowMaps.clear();
	}

	void PrepareScene(Scene& scene) {
		Loading::Timer loading("Graphics scene preparation");
		loadingFrame = true;
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
		renderItems.reserve(staticBatches.size() + scene.Objects().size());
		FlushUploads();
	}

	void Render(Scene& scene, SDL_GPUCommandBuffer* commandBuffer, SDL_GPUTexture* colorTarget, Uint32 width, Uint32 height) {
		if (!colorTarget || !width || !height) return;
		const bool logLoading = loadingFrame;
		loadingFrame = false;
		Loading::Timer loading("First frame graphics preparation and command recording", logLoading);
		const Uint64 renderStart = SDL_GetTicksNS();
		UpdateSceneResources(scene);
		EnsureShadowResolution(scene.shadowResolution);
		const float frameTime = static_cast<float>(renderStart / 1000000000ull) + static_cast<float>(renderStart % 1000000000ull) / 1000000000.0f;
		cullingTimeNanoseconds = 0;
		drawCallCount = 0;
		fpsElapsed += Time::deltaTime;
		++fpsFrames;
		if (fpsElapsed >= 0.5f) {
			fpsString = std::to_string(static_cast<int>(fpsFrames / fpsElapsed));
			fpsElapsed = 0.0f;
			fpsFrames = 0;
		}
		std::uint64_t totalTriangles = 0;
		std::uint64_t visibleTriangles = 0;
		const LightData lightData = CollectLights(scene);
		const LightData* light = lightData.source ? &lightData : nullptr;
		const bool directionalShadowsActive = scene.shadowsEnabled && light && light->castShadows;
		const bool pointShadowsActive = scene.shadowsEnabled && std::any_of(pointLights.begin(), pointLights.end(), [](const PointLightData& pointLight) { return pointLight.castShadows; });
		const bool shadowsActive = directionalShadowsActive || pointShadowsActive;
		EnsureDirectionalShadowTextures(directionalShadowsActive, directionalShadowsActive && light->renderStatic);
		totalTriangles = sceneTriangleCount;
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
		if (scene.activeCamera && scene.activeCamera->owner) {
			cameraTransform = CameraWorldTransform(*scene.activeCamera);
			for (const auto& item : staticDraws) {
				renderItems.push_back(item);
				if (shadowsActive) {
					if (item.mesh->windAnimated) dynamicItems.push_back(item);
					else staticItems.push_back(item);
				}
			}
			for (auto& object : scene.Objects()) {
				Mesh* mesh = object->GetComponent<Mesh>();
				if (!mesh || mesh->data.vertices.empty() || mesh->data.indices.empty() || (object->renderStatic && !mesh->material.transparent && mesh->material.displacementTexturePath.empty())) continue;
				Transform transform = object->transform;
				if (auto* body = object->GetComponent<PhysicsBody>()) {
					const float alpha = Time::interpolationAlpha;
					transform.position = {body->previousPosition.x + (transform.position.x - body->previousPosition.x) * alpha, body->previousPosition.y + (transform.position.y - body->previousPosition.y) * alpha, body->previousPosition.z + (transform.position.z - body->previousPosition.z) * alpha};
				}
				auto& source = meshSources.at(object.get());
				if (!object->renderStatic || !source.drawReady) {
					const ChunkKey cell{scene.chunkSize > 0 ? static_cast<int>(std::floor(transform.position.x / scene.chunkSize)) : 0, scene.chunkSize > 0 ? static_cast<int>(std::floor(transform.position.y / scene.chunkSize)) : 0, scene.chunkSize > 0 ? static_cast<int>(std::floor(transform.position.z / scene.chunkSize)) : 0};
					source.draw = PrepareDraw(mesh, transform, cell);
					source.drawReady = true;
				}
				const RenderItem& item = source.draw;
				renderItems.push_back(item);
				if (shadowsActive) {
					if (object->renderStatic && !mesh->windAnimated) staticItems.push_back(item);
					else dynamicItems.push_back(item);
				}
			}
			FlushUploads();
			const auto frustum = Culling::FromCamera(*scene.activeCamera, aspectRatio);
			const Uint64 cullingStart = SDL_GetTicksNS();
			for (auto item : renderItems) {
				if (!Culling::IsVisible(item.bounds, frustum)) continue;
				for (std::size_t index = 0; index < pointLights.size(); ++index) if (IntersectsSphere(item.bounds, pointLights[index].position, pointLights[index].range)) item.lightMask |= 1u << index;
				visibleItems.push_back(item);
			}
			cullingTimeNanoseconds = SDL_GetTicksNS() - cullingStart;
			if (directionalShadowsActive) {
				Loading::Timer shadowLoading("Initial directional shadow preparation and command recording", logLoading);
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
					for (const auto& item : visibleItems) visibleCells.emplace_back(item.cellX, item.cellY, item.cellZ);
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
						map.data = BuildStaticShadowMap(direction, std::get<0>(key), std::get<1>(key), std::get<2>(key), scene.chunkSize, staticItems, casters);
						if (map.data.valid) {
							if (!map.texture) map.texture = CreateShadowTexture(shadowMapResolution);
							RenderShadowMap(commandBuffer, map.texture, map.data, casters, frameTime);
						}
						map.ready = true;
					}
				}
				const auto& casters = light->renderStatic ? dynamicItems : renderItems;
				dynamicMap = BuildDynamicShadowMap(*scene.activeCamera, aspectRatio, scene.shadowFar, *light, casters, shadowCasters);
				RenderShadowMap(commandBuffer, dynamicShadowTexture, dynamicMap, shadowCasters, frameTime);
			}
		}
		Loading::Timer pointShadowLoading("Initial point shadow preparation and command recording", logLoading);
		RenderPointShadows(scene, commandBuffer, frameTime);
		pointShadowLoading.Stop();
		FrameUniforms frame{};
		frame.cameraPosition[0] = cameraTransform.position.x;
		frame.cameraPosition[1] = cameraTransform.position.y;
		frame.cameraPosition[2] = cameraTransform.position.z;
		frame.fogColor[0] = scene.fogColor.x;
		frame.fogColor[1] = scene.fogColor.y;
		frame.fogColor[2] = scene.fogColor.z;
		frame.fogParameters[0] = scene.fogNear;
		frame.fogParameters[1] = scene.fogFar;
		frame.fogParameters[2] = scene.fogEnabled ? 1.0f : 0.0f;
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
		SDL_PushGPUFragmentUniformData(commandBuffer, 1, &frame, sizeof(frame));
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
		SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(commandBuffer, &colorInfo, 1, &depthInfo);
		const Matrix4 view = View(cameraTransform);
		const Matrix4 projection = scene.activeCamera ? Perspective(scene.activeCamera->fieldOfView, aspectRatio, scene.activeCamera->nearClip, scene.activeCamera->farClip) : Identity();
		const Matrix4 viewProjection = Multiply(projection, view);
		const auto transparentBegin = std::stable_partition(visibleItems.begin(), visibleItems.end(), [](const RenderItem& item) { return !item.mesh->material.transparent; });
		const Vector3 cameraForward = cameraTransform.Forward();
		std::sort(transparentBegin, visibleItems.end(), [&](const RenderItem& left, const RenderItem& right) {
			const Vector3 leftCenter = Add(left.bounds.minimum, left.bounds.maximum) * 0.5f;
			const Vector3 rightCenter = Add(right.bounds.minimum, right.bounds.maximum) * 0.5f;
			return Dot(Subtract(leftCenter, cameraTransform.position), cameraForward) > Dot(Subtract(rightCenter, cameraTransform.position), cameraForward);
		});
		SDL_GPUGraphicsPipeline* activePipeline = nullptr;
		for (const auto& item : visibleItems) {
			Mesh* mesh = item.mesh;
			const auto selected = shadowsActive ? staticShadowMaps.find({item.cellX, item.cellY, item.cellZ}) : staticShadowMaps.end();
			const StaticShadowMap* detail = selected != staticShadowMaps.end() ? &selected->second : nullptr;
			const bool staticEnabled = directionalShadowsActive && light->renderStatic;
			const ShadowMapData maps[3] = {detail ? detail->data : ShadowMapData{}, fallbackShadowMap, dynamicMap};
			const bool enabled[3] = {staticEnabled && detail && detail->data.valid, staticEnabled && fallbackShadowMap.valid, directionalShadowsActive && dynamicMap.valid};
			GpuMesh& gpuMesh = *item.gpu;
			SDL_GPUGraphicsPipeline* meshPipeline = mesh->material.transparent ? transparentPipelines[mesh->cullBackfaces ? 1 : 0] : pipelines[gpuMesh.material->masked ? 1 : 0][mesh->cullBackfaces ? 1 : 0];
			if (meshPipeline != activePipeline) {
				SDL_BindGPUGraphicsPipeline(pass, meshPipeline);
				activePipeline = meshPipeline;
			}
			visibleTriangles += gpuMesh.indexCount / 3;
			ObjectUniforms vertex{};
			const Matrix4 mvp = Multiply(viewProjection, item.model);
			std::memcpy(vertex.mvp, mvp.values, sizeof(vertex.mvp));
			std::memcpy(vertex.model, item.model.values, sizeof(vertex.model));
			std::memcpy(vertex.normal, item.normal.values, sizeof(vertex.normal));
			vertex.displacementParameters[0] = (gpuMesh.material->flags & 8u) ? mesh->material.displacementScale : 0.0f;
			MaterialUv(mesh->material, vertex.uvTransform);
			vertex.displacementParameters[1] = mesh->windAnimated ? 1.0f : 0.0f;
			vertex.displacementParameters[2] = frameTime;
			vertex.displacementParameters[3] = 0.08f;
			FragmentUniforms fragment{};
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
			const Material& material = mesh->material;
			Uint32 materialFlags = gpuMesh.material->flags;
			if (material.normalScale <= 0.0f) materialFlags &= ~1u;
			if (material.aoStrength <= 0.0f) materialFlags &= ~4u;
			if (material.displacementScale == 0.0f) materialFlags &= ~8u;
			if (material.unlit) materialFlags |= 16u;
			fragment.lightingParameters[0] = static_cast<float>(materialFlags);
			fragment.lightingParameters[1] = std::clamp(material.opacity, 0.0f, 1.0f);
			fragment.baseColor[0] = material.baseColor.x;
			fragment.baseColor[1] = material.baseColor.y;
			fragment.baseColor[2] = material.baseColor.z;
			fragment.baseColor[3] = std::clamp(material.metallic, 0.0f, 1.0f);
			fragment.surfaceParameters[0] = std::clamp(material.roughness, 0.0f, 1.0f);
			fragment.surfaceParameters[1] = std::max(material.normalScale, 0.0f);
			fragment.surfaceParameters[2] = std::clamp(material.aoStrength, 0.0f, 1.0f);
			fragment.surfaceParameters[3] = material.flipNormalY ? -1.0f : 1.0f;
			fragment.lightingParameters[2] = shadowsActive ? 1.0f : 0.0f;
			vertex.shadowState[0] = directionalShadowsActive ? 1.0f : 0.0f;
			SDL_PushGPUVertexUniformData(commandBuffer, 0, &vertex, sizeof(vertex));
			SDL_PushGPUFragmentUniformData(commandBuffer, 0, &fragment, sizeof(fragment));
			SDL_GPUBufferBinding vertexBinding{gpuMesh.vertexBuffer, 0};
			SDL_GPUBufferBinding indexBinding{gpuMesh.indexBuffer, 0};
			SDL_BindGPUVertexBuffers(pass, 0, &vertexBinding, 1);
			SDL_BindGPUIndexBuffer(pass, &indexBinding, SDL_GPU_INDEXELEMENTSIZE_32BIT);
			SDL_GPUTextureSamplerBinding bindings[8]{};
			bindings[0] = {gpuMesh.material->textures[0], sampler};
			bindings[1] = {enabled[0] ? detail->texture : emptyShadowTexture, shadowSampler};
			bindings[2] = {enabled[1] ? fallbackShadowTexture : emptyShadowTexture, shadowSampler};
			bindings[3] = {enabled[2] ? dynamicShadowTexture : emptyShadowTexture, shadowSampler};
			bindings[4] = {pointShadowTexture, shadowSampler};
			bindings[5] = {gpuMesh.material->textures[1], sampler};
			bindings[6] = {gpuMesh.material->textures[2], sampler};
			bindings[7] = {gpuMesh.material->textures[3], sampler};
			SDL_BindGPUFragmentSamplers(pass, 0, bindings, 8);
			SDL_GPUTextureSamplerBinding vertexBindings[2] = {{gpuMesh.material->textures[4], sampler}, {gpuMesh.material->textures[5], sampler}};
			SDL_BindGPUVertexSamplers(pass, 0, vertexBindings, 2);
			SDL_DrawGPUIndexedPrimitives(pass, gpuMesh.indexCount, 1, 0, 0, 0);
			++drawCallCount;
		}
		SDL_EndGPURenderPass(pass);
		triangleCountString = std::to_string(visibleTriangles) + "/" + std::to_string(totalTriangles);
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

	void Submit(SDL_GPUCommandBuffer* commandBuffer) {
		FlushUploads();
		UpdateGpuCompletionLatency();
		const Uint64 submitStart = SDL_GetTicksNS();
		SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(commandBuffer);
		const Uint64 submittedAt = SDL_GetTicksNS();
		submitTimeNanoseconds = submittedAt - submitStart;
		if (fence) pendingGpuSubmissions.push_back({fence, submittedAt});
	}

	void Shutdown() {
		FlushUploads();
		for (const auto& submission : pendingGpuSubmissions) SDL_ReleaseGPUFence(gpuDevice, submission.fence);
		pendingGpuSubmissions.clear();
		ReleaseSceneResources();
		while (!textures.empty()) {
			auto* texture = textures.begin()->second;
			textureReferences[texture] = 1;
			ReleaseTexture(gpuDevice, texture);
		}
		staticShadowDirection = {};
		for (auto& variants : {&pipelines, &shadowPipelines}) {
			for (auto& row : *variants) {
				for (auto& pipeline : row) {
					if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
					pipeline = nullptr;
				}
			}
		}
		for (auto& pipeline : transparentPipelines) {
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
		batchChunkSize = -1;
		depthWidth = 0;
		depthHeight = 0;
		fpsElapsed = 0.0f;
		fpsFrames = 0;
		hasGpuCompletionLatency = false;
		whiteTexture = nullptr;
		sampler = nullptr;
		shadowSampler = nullptr;
		gpuDevice = nullptr;
	}
}

namespace Sapphire {
	MeshData Mesh::Cube() { return Graphics::CreateCube(); }
	MeshData Mesh::Plane(int subdivisions) { return Graphics::CreatePlane(subdivisions); }
}
