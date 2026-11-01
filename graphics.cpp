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
	SDL_GPUTexture* UploadTexture(SDL_GPUDevice* gpuDevice, int width, int height, const void* pixels, SDL_GPUTextureFormat format, Uint32 stride);
	SDL_GPUTexture* LoadTexture(std::string_view fileName);
	SDL_GPUTexture* LoadTexture(std::string_view fileName, bool srgb);

	namespace {
		constexpr int shadowMapCount = 3;
		constexpr int maxPointLights = 16;
		constexpr float shadowSlopeFactor = 3.0f;
		Uint32 shadowMapResolution = 2048;
		int shadowDebugMode = 0;
		using ChunkKey = std::tuple<int, int, int>;

		struct Matrix4 {
			float values[16]{};
		};

		struct RenderItem {
			Mesh* mesh;
			Matrix4 model;
			Matrix4 normal;
			int cellX = 0;
			int cellY = 0;
			int cellZ = 0;
		};

		struct ObjectUniforms {
			float mvp[16];
			float model[16];
			float normal[16];
			float displacementParameters[4];
			float shadowMvp[shadowMapCount][16];
			float shadowState[4];
		};

		struct FragmentUniforms {
			float cameraPosition[4];
			float fogColor[4];
			float fogParameters[4];
			float ambientLight[4];
			float directionalDirection[4];
			float directionalColor[4];
			float shadowParameters[shadowMapCount][4];
			float lightingParameters[4];
			float pointLightCount[4];
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
			int shadowIndex = -1;
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

		struct Bounds {
			Vector3 minimum;
			Vector3 maximum;
		};

		struct GpuMesh {
			SDL_GPUBuffer* vertexBuffer = nullptr;
			SDL_GPUBuffer* indexBuffer = nullptr;
			SDL_GPUTexture* texture = nullptr;
			SDL_GPUTexture* displacementTexture = nullptr;
			SDL_GPUTexture* maskTexture = nullptr;
			Uint32 indexCount = 0;
		};

		struct PendingGpuSubmission {
			SDL_GPUFence* fence;
			Uint64 submittedAt;
		};

		SDL_GPUDevice* gpuDevice = nullptr;
		SDL_GPUGraphicsPipeline* pipeline = nullptr;
		SDL_GPUGraphicsPipeline* noCullingPipeline = nullptr;
		SDL_GPUGraphicsPipeline* shadowPipeline = nullptr;
		SDL_GPUGraphicsPipeline* shadowNoCullingPipeline = nullptr;
		SDL_GPUTexture* depthTexture = nullptr;
		SDL_GPUTexture* dynamicShadowTexture = nullptr;
		SDL_GPUTexture* fallbackShadowTexture = nullptr;
		SDL_GPUTexture* pointShadowTexture = nullptr;
		Uint32 pointShadowResolution = 0;
		std::vector<const Light*> pointShadowSources;
		std::vector<PointShadowState> pointShadowStates;
		ShadowMapData fallbackShadowMap;
		bool fallbackShadowReady = false;
		std::uint64_t sourceSignature = 0;
		const Light* shadowLight = nullptr;
		SDL_GPUTexture* staticShadowDebugTexture = nullptr;
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
		std::uint64_t submitTimeNanoseconds = 0;
		std::uint64_t gpuCompletionTimeNanoseconds = 0;
		bool hasGpuCompletionTime = false;
		std::string fpsString = "0";
		std::string triangleCountString = "0/0";
		std::unordered_map<Mesh*, GpuMesh> gpuMeshes;
		std::unordered_map<Mesh*, Bounds> meshBounds;
		std::unordered_map<std::string, SDL_GPUTexture*> textures;
		std::deque<PendingGpuSubmission> pendingGpuSubmissions;
		std::vector<std::unique_ptr<Mesh>> staticBatches;
		std::vector<RenderItem> renderItems;
		bool staticBatchesBuilt = false;

		std::uint64_t Hash(std::uint64_t hash, const void* data, std::size_t size) {
			const auto* bytes = static_cast<const unsigned char*>(data);
			for (std::size_t index = 0; index < size; ++index) hash = (hash ^ bytes[index]) * 1099511628211ull;
			return hash;
		}

		template <typename T>
		void AddHash(std::uint64_t& hash, const T& value) { hash = Hash(hash, &value, sizeof(value)); }

		void AddString(std::uint64_t& hash, const std::string& value) { hash = Hash(hash, value.c_str(), value.size() + 1); }

		std::uint64_t SceneSignature(Scene& scene) {
			std::uint64_t hash = 14695981039346656037ull;
			AddHash(hash, scene.chunkSize);
			std::map<std::string, std::filesystem::file_time_type> assets;
			for (auto& object : scene.Objects()) {
				Mesh* mesh = object->GetComponent<Mesh>();
				if (!mesh) continue;
				AddHash(hash, object->renderStatic);
				if (object->renderStatic) {
					AddHash(hash, object->transform.position);
					AddHash(hash, object->transform.rotation);
					AddHash(hash, object->transform.scale);
				}
				AddHash(hash, mesh);
				AddHash(hash, mesh->revision);
				AddHash(hash, mesh->windAnimated);
				AddHash(hash, mesh->castShadows);
				AddHash(hash, mesh->cullBackfaces);
				AddHash(hash, mesh->displacementScale);
				AddHash(hash, mesh->data.vertices.size());
				AddHash(hash, mesh->data.indices.size());
				AddHash(hash, mesh->data.vertices.data());
				AddHash(hash, mesh->data.indices.data());
				for (const auto& name : {mesh->texturePath, mesh->displacementTexturePath, mesh->maskTexturePath}) {
					AddString(hash, name);
					if (name.empty()) continue;
					auto [entry, inserted] = assets.try_emplace(name);
					if (inserted) {
						std::error_code error;
						entry->second = std::filesystem::last_write_time(std::filesystem::path(SDL_GetBasePath()) / ".." / "resources" / name, error);
					}
					AddHash(hash, entry->second.time_since_epoch().count());
				}
			}
			return hash;
		}

		std::string Milliseconds(double value) {
			std::ostringstream stream;
			stream << std::fixed << std::setprecision(2) << value << " ms";
			return stream.str();
		}

		void UpdateGpuCompletionTime() {
			const Uint64 now = SDL_GetTicksNS();
			while (!pendingGpuSubmissions.empty()) {
				auto& submission = pendingGpuSubmissions.front();
				if (!SDL_QueryGPUFence(gpuDevice, submission.fence)) return;
				gpuCompletionTimeNanoseconds = now - submission.submittedAt;
				hasGpuCompletionTime = true;
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
			struct ObjectUniforms { float4x4 mvp; float4x4 model; float4x4 normal; float4 displacementParameters; float4x4 shadowMvp[3]; float4 shadowState; };
			vertex VertexOutput main0(VertexInput input [[stage_in]], constant ObjectUniforms& uniforms [[buffer(0)]], texture2d<float> heightMap [[texture(0)]], sampler heightSampler [[sampler(0)]], texture2d<float> maskMap [[texture(1)]], sampler maskSampler [[sampler(1)]]) {
				VertexOutput output;
				float3 position = Deform(input.position, input.normal, input.uv, input.color, uniforms.model, uniforms.displacementParameters, heightMap, heightSampler, maskMap, maskSampler);
				output.position = uniforms.mvp * float4(position, 1.0);
				output.uv = input.uv;
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
			struct FragmentUniforms { float4 cameraPosition; float4 fogColor; float4 fogParameters; float4 ambientLight; float4 directionalDirection; float4 directionalColor; float4 shadowParameters[3]; float4 lightingParameters; float4 pointLightCount; float4 pointLightPositionRange[16]; float4 pointLightColorIntensity[16]; float4 pointLightShadowInfo[16]; float4 pointLightShadowBias[16]; };
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
			float PointShadowVisibility(float3 direction, float receiverDistance, float range, float bias, uint cubeIndex, depthcube_array<float> shadowMap, sampler shadowSampler) {
				float faceDepth = max(max(abs(direction.x), abs(direction.y)), abs(direction.z)) * receiverDistance;
				float nearClip = min(0.1, range * 0.01);
				float compareDistance = max(faceDepth - bias, nearClip);
				float compareDepth = range / (range - nearClip) - nearClip * range / ((range - nearClip) * compareDistance);
				return shadowMap.sample_compare(shadowSampler, direction, cubeIndex, compareDepth);
			}
			float PointShadowAmount(float3 direction, float range, float bias, uint cubeIndex, depthcube_array<float> shadowMap, sampler shadowSampler, float texelSize) {
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

			float3 EncodeColor(float3 color) {
				color = max(color, float3(0.0));
				return select(1.055 * pow(color, float3(1.0 / 2.4)) - 0.055, color * 12.92, color <= 0.0031308);
			}
			fragment float4 main0(FragmentInput input [[stage_in]], constant FragmentUniforms& uniforms [[buffer(0)]], texture2d<float> image [[texture(0)]], sampler imageSampler [[sampler(0)]], depth2d<float> detailMap [[texture(1)]], sampler detailSampler [[sampler(1)]], depth2d<float> fallbackMap [[texture(2)]], sampler fallbackSampler [[sampler(2)]], depth2d<float> dynamicMap [[texture(3)]], sampler dynamicSampler [[sampler(3)]], depthcube_array<float> pointShadowMap [[texture(4)]], sampler pointShadowSampler [[sampler(4)]]) {
				float3 normal = normalize(input.worldNormal);
				float3 geometricNormal = normalize(cross(dfdx(input.worldPosition), dfdy(input.worldPosition)));
				if (dot(geometricNormal, normal) < 0.0) geometricNormal = -geometricNormal;
				if (uniforms.lightingParameters.x > 0.5) normal = geometricNormal;
				float4 texel = image.sample(imageSampler, input.uv);
				if (texel.a < 0.5) discard_fragment();
				float3 albedo = input.color.rgb * texel.rgb;
				float staticShadow = 0.0;
				if (uniforms.lightingParameters.z > 0.5) {
					float detailCoverage = ShadowCoverage(input.shadowPosition0, uniforms.shadowParameters[0]);
					if (detailCoverage < 1.0) staticShadow = ShadowAmount(input.shadowPosition1, fallbackMap, fallbackSampler, uniforms.shadowParameters[1]);
					if (detailCoverage > 0.0) staticShadow = mix(staticShadow, ShadowAmount(input.shadowPosition0, detailMap, detailSampler, uniforms.shadowParameters[0]), detailCoverage);
				}
				float dynamicShadow = 0.0;
				if (uniforms.lightingParameters.z > 0.5) dynamicShadow = ShadowAmount(input.shadowPosition2, dynamicMap, dynamicSampler, uniforms.shadowParameters[2]);
				float diffuse = max(dot(normal, -uniforms.directionalDirection.xyz), 0.0) * uniforms.directionalDirection.w;
				float3 direct = uniforms.directionalColor.rgb * diffuse * (1.0 - staticShadow);
				float3 pointLighting = float3(0.0);
				float pointStaticShadow = 0.0;
				float pointDynamicShadow = 0.0;
				for (int index = 0; index < int(uniforms.pointLightCount.x); ++index) {
					float4 positionRange = uniforms.pointLightPositionRange[index];
					float4 colorIntensity = uniforms.pointLightColorIntensity[index];
					float3 toLight = positionRange.xyz - input.worldPosition;
					float distanceSquared = dot(toLight, toLight);
					if (distanceSquared < positionRange.w * positionRange.w) {
						float distanceToLight = sqrt(distanceSquared);
						float rangeFade = clamp(1.0 - distanceToLight / positionRange.w, 0.0, 1.0);
						float attenuation = rangeFade * rangeFade / max(distanceSquared, 1.0);
						float4 shadowInfo = uniforms.pointLightShadowInfo[index];
						float shadowBias = uniforms.pointLightShadowBias[index].x;
						float staticPointShadow = uniforms.lightingParameters.z > 0.5 && shadowInfo.z > 0.5 ? PointShadowAmount(-toLight, positionRange.w, shadowBias, uint(shadowInfo.x), pointShadowMap, pointShadowSampler, uniforms.shadowParameters[0].z) : 0.0;
						float dynamicPointShadow = uniforms.lightingParameters.z > 0.5 && shadowInfo.w > 0.5 ? PointShadowAmount(-toLight, positionRange.w, shadowBias, uint(shadowInfo.y), pointShadowMap, pointShadowSampler, uniforms.shadowParameters[0].z) : 0.0;
						pointStaticShadow = max(pointStaticShadow, staticPointShadow);
						pointDynamicShadow = max(pointDynamicShadow, dynamicPointShadow);
						pointLighting += colorIntensity.rgb * colorIntensity.a * max(dot(normal, toLight / max(distanceToLight, 0.0001)), 0.0) * attenuation * (1.0 - staticPointShadow) * (1.0 - dynamicPointShadow);
					}
				}
				float3 ambient = uniforms.ambientLight.a > 0.5 ? uniforms.ambientLight.rgb : float3(0.0);
				float3 color = albedo * (ambient + direct * (1.0 - dynamicShadow) + pointLighting);
				float fog = uniforms.fogParameters.z > 0.5 ? clamp((distance(input.worldPosition, uniforms.cameraPosition.xyz) - uniforms.fogParameters.x) / max(uniforms.fogParameters.y - uniforms.fogParameters.x, 0.0001), 0.0, 1.0) : 0.0;
				if (uniforms.lightingParameters.y > 0.5) return float4(mix(float3(max(staticShadow, pointStaticShadow), 1.0 - max(max(staticShadow, pointStaticShadow), max(dynamicShadow, pointDynamicShadow)), max(dynamicShadow, pointDynamicShadow)), uniforms.fogColor.rgb, fog), 1.0);
				return float4(EncodeColor(mix(color, uniforms.fogColor.rgb, fog)), input.color.a * texel.a);
			}
		)";

		const std::string shadowVertexShaderSource = deformationShaderSource + R"(
			struct VertexInput { float3 position [[attribute(0)]]; float3 normal [[attribute(1)]]; float2 uv [[attribute(2)]]; float4 color [[attribute(3)]]; };
			struct VertexOutput { float4 position [[position]]; float2 uv; };
			struct ShadowUniforms { float4x4 mvp; float4x4 model; float4 displacementParameters; };
			vertex VertexOutput main0(VertexInput input [[stage_in]], constant ShadowUniforms& uniforms [[buffer(0)]], texture2d<float> heightMap [[texture(0)]], sampler heightSampler [[sampler(0)]], texture2d<float> maskMap [[texture(1)]], sampler maskSampler [[sampler(1)]]) {
				VertexOutput output;
				float3 position = Deform(input.position, input.normal, input.uv, input.color, uniforms.model, uniforms.displacementParameters, heightMap, heightSampler, maskMap, maskSampler);
				output.position = uniforms.mvp * float4(position, 1.0);
				output.uv = input.uv;
				return output;
			}
		)";

		constexpr const char* shadowFragmentShaderSource = R"(
			#include <metal_stdlib>
			using namespace metal;
			struct FragmentInput { float4 position [[position]]; float2 uv; };
			fragment void main0(FragmentInput input [[stage_in]], texture2d<float> image [[texture(0)]], sampler imageSampler [[sampler(0)]]) {
				if (image.sample(imageSampler, input.uv).a < 0.5) discard_fragment();
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

		Matrix4 NormalMatrix(const Transform& transform) {
			Transform normalTransform = transform;
			normalTransform.position = {};
			normalTransform.scale = {
				std::abs(transform.scale.x) > 0.0001f ? 1.0f / transform.scale.x : 0.0f,
				std::abs(transform.scale.y) > 0.0001f ? 1.0f / transform.scale.y : 0.0f,
				std::abs(transform.scale.z) > 0.0001f ? 1.0f / transform.scale.z : 0.0f
			};
			return Model(normalTransform);
		}

		Vector3 TransformNormal(const Matrix4& normal, const Vector3& value) {
			Vector3 result{
				normal.values[0] * value.x + normal.values[4] * value.y + normal.values[8] * value.z,
				normal.values[1] * value.x + normal.values[5] * value.y + normal.values[9] * value.z,
				normal.values[2] * value.x + normal.values[6] * value.y + normal.values[10] * value.z
			};
			const float length = std::sqrt(result.x * result.x + result.y * result.y + result.z * result.z);
			if (length > 0.0001f) {
				result.x /= length;
				result.y /= length;
				result.z /= length;
			}
			return result;
		}

		Transform CameraWorldTransform(const Camera& camera) {
			Transform transform = camera.owner->transform;
			const float x = transform.rotation.x * Math::degreesToRadians;
			const float y = transform.rotation.y * Math::degreesToRadians;
			const float z = transform.rotation.z * Math::degreesToRadians;
			const float cx = std::cos(x);
			const float sx = std::sin(x);
			const float cy = std::cos(y);
			const float sy = std::sin(y);
			const float cz = std::cos(z);
			const float sz = std::sin(z);
			const Vector3& offset = camera.localPosition;
			transform.position += {
				cz * cy * offset.x + (cz * sy * sx - sz * cx) * offset.y + (cz * sy * cx + sz * sx) * offset.z,
				sz * cy * offset.x + (sz * sy * sx + cz * cx) * offset.y + (sz * sy * cx - cz * sx) * offset.z,
				-sy * offset.x + cy * sx * offset.y + cy * cx * offset.z
			};
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

		float Dot(const Vector3& left, const Vector3& right) {
			return left.x * right.x + left.y * right.y + left.z * right.z;
		}

		Vector3 Add(const Vector3& left, const Vector3& right) {
			return {left.x + right.x, left.y + right.y, left.z + right.z};
		}

		Vector3 Scale(const Vector3& value, float scalar) {
			return {value.x * scalar, value.y * scalar, value.z * scalar};
		}

		Vector3 Cross(const Vector3& left, const Vector3& right) {
			return {left.y * right.z - left.z * right.y, left.z * right.x - left.x * right.z, left.x * right.y - left.y * right.x};
		}

		Vector3 Normalize(const Vector3& value) {
			const float length = std::sqrt(Dot(value, value));
			return length > 0.0001f ? Scale(value, 1.0f / length) : Vector3{};
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

		LightData CollectDirectionalLight(Scene& scene) {
			for (auto& object : scene.Objects()) {
				auto* light = object->GetComponent<Light>();
				if (light && light->enabled && light->type == Light::Type::Directional) return {Normalize(object->transform.Forward()), light->color, light->intensity, light->castShadows, light->shadowBias, object->renderStatic, light};
			}
			return {};
		}

		std::vector<PointLightData> CollectPointLights(Scene& scene) {
			std::vector<PointLightData> pointLights;
			int shadowIndex = 0;
			for (auto& object : scene.Objects()) {
				auto* light = object->GetComponent<Light>();
				if (light && light->enabled && light->type == Light::Type::Point && light->range > 0.0f && light->intensity > 0.0f && pointLights.size() < maxPointLights) {
					PointLightData pointLight{object->transform.position, light->color, light->intensity, light->range, light->shadowBias, light->castShadows, object->renderStatic};
					pointLight.source = light;
					if (pointLight.castShadows) pointLight.shadowIndex = shadowIndex++;
					pointLights.push_back(pointLight);
				}
			}
			return pointLights;
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

			Vector3 TransformPoint(const Matrix4& model, const Vector3& point) {
				return {
					model.values[0] * point.x + model.values[4] * point.y + model.values[8] * point.z + model.values[12],
					model.values[1] * point.x + model.values[5] * point.y + model.values[9] * point.z + model.values[13],
					model.values[2] * point.x + model.values[6] * point.y + model.values[10] * point.z + model.values[14]
				};
			}

			const Bounds& GetBounds(Mesh* mesh) {
				auto [entry, inserted] = meshBounds.try_emplace(mesh);
				if (inserted) {
					float windPadding = 0.0f;
					entry->second.minimum = mesh->data.vertices.front().position;
					entry->second.maximum = mesh->data.vertices.front().position;
					for (const auto& vertex : mesh->data.vertices) {
						entry->second.minimum.x = std::min(entry->second.minimum.x, vertex.position.x);
						entry->second.minimum.y = std::min(entry->second.minimum.y, vertex.position.y);
						entry->second.minimum.z = std::min(entry->second.minimum.z, vertex.position.z);
						entry->second.maximum.x = std::max(entry->second.maximum.x, vertex.position.x);
						entry->second.maximum.y = std::max(entry->second.maximum.y, vertex.position.y);
						entry->second.maximum.z = std::max(entry->second.maximum.z, vertex.position.z);
						if (mesh->windAnimated) windPadding = std::max(windPadding, std::abs(vertex.color[0]) * 0.08f);
					}
					if (mesh->windAnimated) {
						entry->second.minimum.x -= windPadding;
						entry->second.maximum.x += windPadding;
					}
					if (!mesh->displacementTexturePath.empty()) {
						const float displacementPadding = std::abs(mesh->displacementScale);
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

			bool IsVisible(const Bounds& bounds, const Matrix4& model, const Frustum& frustum) {
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
					const Vector3 point = TransformPoint(model, localPoint);
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

		bool IntersectsSphere(const Bounds& bounds, const Matrix4& model, const Vector3& center, float radius) {
			Vector3 minimum{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
			Vector3 maximum{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
			for (int corner = 0; corner < 8; ++corner) {
				const Vector3 localPoint{corner & 1 ? bounds.maximum.x : bounds.minimum.x, corner & 2 ? bounds.maximum.y : bounds.minimum.y, corner & 4 ? bounds.maximum.z : bounds.minimum.z};
				const Vector3 point = Culling::TransformPoint(model, localPoint);
				minimum.x = std::min(minimum.x, point.x);
				minimum.y = std::min(minimum.y, point.y);
				minimum.z = std::min(minimum.z, point.z);
				maximum.x = std::max(maximum.x, point.x);
				maximum.y = std::max(maximum.y, point.y);
				maximum.z = std::max(maximum.z, point.z);
			}
			const Vector3 closest{std::clamp(center.x, minimum.x, maximum.x), std::clamp(center.y, minimum.y, maximum.y), std::clamp(center.z, minimum.z, maximum.z)};
			const Vector3 difference = {closest.x - center.x, closest.y - center.y, closest.z - center.z};
			return Dot(difference, difference) <= radius * radius;
		}

		std::vector<RenderItem> PointShadowCasters(const std::vector<RenderItem>& items, const Vector3& position, float range) {
			std::vector<RenderItem> casters;
			for (const auto& item : items) if (item.mesh->castShadows && IntersectsSphere(Culling::GetBounds(item.mesh), item.model, position, range)) casters.push_back(item);
			return casters;
		}

		ShadowMapData BuildPointShadowMap(const Vector3& position, float range, int face) {
			static const Vector3 directions[6] = {{1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}};
			static const Vector3 ups[6] = {{0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
			const float nearClip = std::min(0.1f, range * 0.01f);
			const Matrix4 projection = Perspective(90.0f, 1.0f, nearClip, range);
			ShadowMapData map{};
			map.mvp = Multiply(projection, ViewDirection(position, directions[face], ups[face]));
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
					const Bounds& bounds = Culling::GetBounds(item.mesh);
					for (int corner = 0; corner < 8; ++corner) {
						const Vector3 localPoint{corner & 1 ? bounds.maximum.x : bounds.minimum.x, corner & 2 ? bounds.maximum.y : bounds.minimum.y, corner & 4 ? bounds.maximum.z : bounds.minimum.z};
						const Vector3 point = Culling::TransformPoint(item.model, localPoint);
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
				const Bounds& bounds = Culling::GetBounds(item.mesh);
				float itemMinimumX = std::numeric_limits<float>::max();
				float itemMaximumX = std::numeric_limits<float>::lowest();
				float itemMinimumY = std::numeric_limits<float>::max();
				float itemMaximumY = std::numeric_limits<float>::lowest();
				float itemMinimumDepth = std::numeric_limits<float>::max();
				for (int corner = 0; corner < 8; ++corner) {
					const Vector3 localPoint{corner & 1 ? bounds.maximum.x : bounds.minimum.x, corner & 2 ? bounds.maximum.y : bounds.minimum.y, corner & 4 ? bounds.maximum.z : bounds.minimum.z};
					const Vector3 point = Culling::TransformPoint(item.model, localPoint);
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

		ShadowMapData BuildDynamicShadowMap(const Camera& camera, float aspectRatio, float shadowFar, const LightData& light, const std::vector<RenderItem>& items) {
			ShadowMapData map{};
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
				const auto& bounds = Culling::GetBounds(item.mesh);
				float minX = std::numeric_limits<float>::max();
				float maxX = std::numeric_limits<float>::lowest();
				float minY = minX;
				float maxY = maxX;
				float depth = minX;
				for (int corner = 0; corner < 8; ++corner) {
					const Vector3 local{corner & 1 ? bounds.maximum.x : bounds.minimum.x, corner & 2 ? bounds.maximum.y : bounds.minimum.y, corner & 4 ? bounds.maximum.z : bounds.minimum.z};
					const Vector3 point = Culling::TransformPoint(item.model, local);
					minX = std::min(minX, Dot(right, point));
					maxX = std::max(maxX, Dot(right, point));
					minY = std::min(minY, Dot(up, point));
					maxY = std::max(maxY, Dot(up, point));
					depth = std::min(depth, Dot(direction, point));
				}
				if (maxX < receiverMinX || minX > receiverMaxX || maxY < receiverMinY || minY > receiverMaxY || depth > maximumDepth) continue;
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

		void BuildStaticBatches(Scene& scene) {
			if (staticBatchesBuilt) return;
			using BatchKey = std::tuple<int, int, int, std::string, std::string, std::string, float, bool, bool, bool>;
			using CellKey = std::tuple<int, int, int>;
			std::map<BatchKey, MeshData> batchData;
			for (auto& object : scene.Objects()) {
				if (!object->renderStatic) continue;
				Mesh* mesh = object->GetComponent<Mesh>();
				if (mesh && !mesh->displacementTexturePath.empty()) continue;
				if (!mesh || mesh->data.vertices.empty() || mesh->data.indices.empty()) continue;
				const Matrix4 model = Model(object->transform);
				const Matrix4 normal = NormalMatrix(object->transform);
				std::map<CellKey, std::unordered_map<std::uint32_t, std::uint32_t>> vertexMaps;
				for (std::size_t index = 0; index + 2 < mesh->data.indices.size(); index += 3) {
					const std::uint32_t sourceIndices[3] = {mesh->data.indices[index], mesh->data.indices[index + 1], mesh->data.indices[index + 2]};
					const Vector3 worldPositions[3] = {
						Culling::TransformPoint(model, mesh->data.vertices[sourceIndices[0]].position),
						Culling::TransformPoint(model, mesh->data.vertices[sourceIndices[1]].position),
						Culling::TransformPoint(model, mesh->data.vertices[sourceIndices[2]].position)
					};
					const float centerX = (worldPositions[0].x + worldPositions[1].x + worldPositions[2].x) / 3.0f;
					const float centerY = (worldPositions[0].y + worldPositions[1].y + worldPositions[2].y) / 3.0f;
					const float centerZ = (worldPositions[0].z + worldPositions[1].z + worldPositions[2].z) / 3.0f;
					const int cellX = scene.chunkSize > 0 ? static_cast<int>(std::floor(centerX / scene.chunkSize)) : 0;
					const int cellY = scene.chunkSize > 0 ? static_cast<int>(std::floor(centerY / scene.chunkSize)) : 0;
					const int cellZ = scene.chunkSize > 0 ? static_cast<int>(std::floor(centerZ / scene.chunkSize)) : 0;
					const CellKey cellKey{cellX, cellY, cellZ};
					auto& output = batchData[{cellX, cellY, cellZ, mesh->texturePath, mesh->displacementTexturePath, mesh->maskTexturePath, mesh->displacementScale, mesh->cullBackfaces, mesh->castShadows, mesh->windAnimated}];
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
			staticBatches.reserve(batchData.size());
			for (auto& [key, data] : batchData) {
				auto batch = std::make_unique<Mesh>();
				batch->data = std::move(data);
				batch->texturePath = std::get<3>(key);
				batch->displacementTexturePath = std::get<4>(key);
				batch->maskTexturePath = std::get<5>(key);
				batch->displacementScale = std::get<6>(key);
				batch->cullBackfaces = std::get<7>(key);
				batch->castShadows = std::get<8>(key);
				batch->windAnimated = std::get<9>(key);
				const ChunkKey cellKey{std::get<0>(key), std::get<1>(key), std::get<2>(key)};
				staticBatchCells.emplace(batch.get(), cellKey);

				staticBatches.push_back(std::move(batch));
			}
			staticBatchesBuilt = true;
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

		void EnsurePointShadowTexture(const std::vector<PointLightData>& pointLights, bool enabled) {
			std::vector<const Light*> sources;
			if (enabled) for (const auto& light : pointLights) if (light.castShadows) sources.push_back(light.source);
			const Uint32 resolution = sources.empty() ? 1 : shadowMapResolution;
			if (sources == pointShadowSources && resolution == pointShadowResolution && pointShadowTexture) return;
			if (pointShadowTexture) SDL_ReleaseGPUTexture(gpuDevice, pointShadowTexture);
			pointShadowSources = sources;
			pointShadowResolution = resolution;
			pointShadowStates.clear();
			for (const auto& light : pointLights) {
				if (!enabled || !light.castShadows) continue;
				pointShadowStates.push_back({light.source, light.position, light.range, light.renderStatic});
			}
			SDL_GPUTextureCreateInfo textureInfo{};
			textureInfo.type = SDL_GPU_TEXTURETYPE_CUBE_ARRAY;
			textureInfo.format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
			textureInfo.usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET | SDL_GPU_TEXTUREUSAGE_SAMPLER;
			textureInfo.width = resolution;
			textureInfo.height = resolution;
			textureInfo.layer_count_or_depth = static_cast<Uint32>(std::max<std::size_t>(sources.size() * 2, 1) * 6);
			textureInfo.num_levels = 1;
			textureInfo.sample_count = SDL_GPU_SAMPLECOUNT_1;
			pointShadowTexture = SDL_CreateGPUTexture(gpuDevice, &textureInfo);
		}

		void EnsureShadowResolution(int requestedResolution) {
			const Uint32 resolution = static_cast<Uint32>(std::max(requestedResolution, 1));
			if (resolution == shadowMapResolution && dynamicShadowTexture && fallbackShadowTexture) return;
			if (dynamicShadowTexture) SDL_ReleaseGPUTexture(gpuDevice, dynamicShadowTexture);
			if (fallbackShadowTexture) SDL_ReleaseGPUTexture(gpuDevice, fallbackShadowTexture);
			if (pointShadowTexture) SDL_ReleaseGPUTexture(gpuDevice, pointShadowTexture);
			pointShadowTexture = nullptr;
			pointShadowResolution = 0;
			pointShadowSources.clear();
			pointShadowStates.clear();
			for (auto& [key, map] : staticShadowMaps) if (map.texture) SDL_ReleaseGPUTexture(gpuDevice, map.texture);
			staticShadowMaps.clear();
			shadowMapResolution = resolution;
			dynamicShadowTexture = CreateShadowTexture(shadowMapResolution);
			fallbackShadowTexture = CreateShadowTexture(shadowMapResolution);
			fallbackShadowMap = {};
			fallbackShadowReady = false;
			staticShadowDebugTexture = nullptr;
		}

		SDL_GPUTexture* GetTexture(const std::string& fileName, bool srgb = false) {
			const std::string key = (srgb ? "color:" : "data:") + fileName;
			if (auto found = textures.find(key); found != textures.end()) return found->second;
			SDL_GPUTexture* texture = LoadTexture(fileName, srgb);
			if (texture) textures[key] = texture;
			return texture;
		}

		GpuMesh& GetMesh(Mesh* mesh) {
			auto [entry, inserted] = gpuMeshes.try_emplace(mesh);
			if (inserted) {
				entry->second = UploadMesh(mesh->data);
				entry->second.texture = mesh->texturePath.empty() ? whiteTexture : GetTexture(mesh->texturePath, true);
				if (!entry->second.texture) entry->second.texture = whiteTexture;
				entry->second.displacementTexture = mesh->displacementTexturePath.empty() ? whiteTexture : GetTexture(mesh->displacementTexturePath);
				if (!entry->second.displacementTexture) entry->second.displacementTexture = whiteTexture;
				entry->second.maskTexture = mesh->maskTexturePath.empty() ? whiteTexture : GetTexture(mesh->maskTexturePath);
				if (!entry->second.maskTexture) entry->second.maskTexture = whiteTexture;
			}
			return entry->second;
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
				if (!mesh->castShadows || mesh->data.vertices.empty() || mesh->data.indices.empty()) continue;
				SDL_GPUGraphicsPipeline* meshPipeline = cullBackfaces && mesh->cullBackfaces ? shadowPipeline : shadowNoCullingPipeline;
				if (meshPipeline != activePipeline) {
					SDL_BindGPUGraphicsPipeline(shadowPass, meshPipeline);
					activePipeline = meshPipeline;
				}
				GpuMesh& gpuMesh = GetMesh(mesh);
				struct ShadowUniforms {
					float mvp[16];
					float model[16];
					float displacementParameters[4];
				} uniforms{};
				const Matrix4 mvp = Multiply(shadowMap.mvp, item.model);
				std::memcpy(uniforms.mvp, mvp.values, sizeof(uniforms.mvp));
				std::memcpy(uniforms.model, item.model.values, sizeof(uniforms.model));
				uniforms.displacementParameters[0] = mesh->displacementTexturePath.empty() ? 0.0f : mesh->displacementScale;
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
				textureBinding.texture = gpuMesh.texture;
				textureBinding.sampler = sampler;
				SDL_BindGPUFragmentSamplers(shadowPass, 0, &textureBinding, 1);
				SDL_GPUTextureSamplerBinding vertexTextureBindings[2]{};
				vertexTextureBindings[0].texture = gpuMesh.displacementTexture;
				vertexTextureBindings[0].sampler = sampler;
				vertexTextureBindings[1].texture = gpuMesh.maskTexture;
				vertexTextureBindings[1].sampler = sampler;
				SDL_BindGPUVertexSamplers(shadowPass, 0, vertexTextureBindings, 2);
				SDL_DrawGPUIndexedPrimitives(shadowPass, gpuMesh.indexCount, 1, 0, 0, 0);
				++drawCallCount;
			}
			SDL_EndGPURenderPass(shadowPass);
		}

		void RenderPointShadowMap(SDL_GPUCommandBuffer* commandBuffer, SDL_GPUTexture* texture, int cubeIndex, const PointLightData& light, const std::vector<RenderItem>& casters, float time) {
			for (int face = 0; face < 6; ++face) RenderShadowMap(commandBuffer, texture, BuildPointShadowMap(light.position, light.range, face), casters, time, static_cast<Uint8>(cubeIndex * 6 + face), false);
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

	SDL_GPUTexture* UploadTexture(SDL_GPUDevice* gpuDevice, int width, int height, const void* pixels, SDL_GPUTextureFormat format, Uint32 stride) {
		SDL_GPUTextureCreateInfo textureInfo{};
		textureInfo.type = SDL_GPU_TEXTURETYPE_2D;
		textureInfo.format = format;
		textureInfo.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER;
		textureInfo.width = width;
		textureInfo.height = height;
		textureInfo.layer_count_or_depth = 1;
		textureInfo.num_levels = 1;
		textureInfo.sample_count = SDL_GPU_SAMPLECOUNT_1;
		SDL_GPUTexture* texture = SDL_CreateGPUTexture(gpuDevice, &textureInfo);
		const Uint32 pixelBytes = static_cast<Uint32>(width * height * stride);
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

	SDL_GPUTexture* CreateTexture(SDL_GPUDevice* gpuDevice, int width, int height, const std::uint8_t* pixels) {
		return CreateTexture(gpuDevice, width, height, pixels, false);
	}

	SDL_GPUTexture* CreateTexture(SDL_GPUDevice* gpuDevice, int width, int height, const std::uint8_t* pixels, bool srgb) {
		return UploadTexture(gpuDevice, width, height, pixels, srgb ? SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM_SRGB : SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, 4);
	}

	SDL_GPUTexture* LoadTexture(std::string_view fileName) { return LoadTexture(fileName, false); }

	SDL_GPUTexture* LoadTexture(std::string_view fileName, bool srgb) {
		const std::string path = (std::filesystem::path(SDL_GetBasePath()) / ".." / "resources" / std::string(fileName)).string();
		SDL_Surface* loadedSurface = SDL_LoadSurface(path.c_str());
		if (!loadedSurface) return nullptr;
		SDL_Surface* rgbaSurface = SDL_ConvertSurface(loadedSurface, SDL_PIXELFORMAT_RGBA32);
		SDL_DestroySurface(loadedSurface);
		if (!rgbaSurface) return nullptr;
		const int width = rgbaSurface->w;
		const int height = rgbaSurface->h;
		std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * height * 4);
		for (int row = 0; row < height; ++row) {
			const auto* source = static_cast<const std::uint8_t*>(rgbaSurface->pixels) + row * rgbaSurface->pitch;
			std::memcpy(pixels.data() + static_cast<std::size_t>(row) * width * 4, source, static_cast<std::size_t>(width) * 4);
		}
		SDL_DestroySurface(rgbaSurface);
		return CreateTexture(gpuDevice, width, height, pixels.data(), srgb);
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
		SDL_GPUShader* vertexShader = CreateShader(gpuDevice, vertexShaderSource.c_str(), SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 2, 1);
		SDL_GPUShader* fragmentShader = CreateShader(gpuDevice, fragmentShaderSource, SDL_GPU_SHADERSTAGE_FRAGMENT, SDL_GPU_SHADERFORMAT_MSL, 5, 1);
		SDL_GPUShader* shadowVertexShader = CreateShader(gpuDevice, shadowVertexShaderSource.c_str(), SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 2, 1);
		SDL_GPUShader* shadowFragmentShader = CreateShader(gpuDevice, shadowFragmentShaderSource, SDL_GPU_SHADERSTAGE_FRAGMENT, SDL_GPU_SHADERFORMAT_MSL, 1);
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
		dynamicShadowTexture = CreateShadowTexture(shadowMapResolution);
		fallbackShadowTexture = CreateShadowTexture(shadowMapResolution);
		SDL_GPUColorTargetDescription colorTarget{};
		colorTarget.format = window ? SDL_GetGPUSwapchainTextureFormat(gpuDevice, window) : SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
		SDL_GPUGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.vertex_shader = vertexShader;
		pipelineInfo.fragment_shader = fragmentShader;
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
		pipeline = CreatePipeline(gpuDevice, pipelineInfo);
		pipelineInfo.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
		noCullingPipeline = CreatePipeline(gpuDevice, pipelineInfo);
		pipelineInfo.vertex_shader = shadowVertexShader;
		pipelineInfo.fragment_shader = shadowFragmentShader;
		pipelineInfo.vertex_input_state.num_vertex_attributes = 4;
		pipelineInfo.rasterizer_state.enable_depth_bias = true;
		pipelineInfo.rasterizer_state.depth_bias_constant_factor = 2.0f;
		pipelineInfo.rasterizer_state.depth_bias_slope_factor = shadowSlopeFactor;
		pipelineInfo.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_BACK;
		pipelineInfo.target_info.num_color_targets = 0;
		pipelineInfo.target_info.color_target_descriptions = nullptr;
		pipelineInfo.target_info.has_depth_stencil_target = true;
		pipelineInfo.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D32_FLOAT;
		shadowPipeline = CreatePipeline(gpuDevice, pipelineInfo);
		pipelineInfo.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
		shadowNoCullingPipeline = CreatePipeline(gpuDevice, pipelineInfo);
		SDL_ReleaseGPUShader(gpuDevice, vertexShader);
		SDL_ReleaseGPUShader(gpuDevice, fragmentShader);
		SDL_ReleaseGPUShader(gpuDevice, shadowVertexShader);
		SDL_ReleaseGPUShader(gpuDevice, shadowFragmentShader);
	}

	static void ReleaseSceneResources() {
		for (auto& [mesh, gpuMesh] : gpuMeshes) {
			if (gpuMesh.vertexBuffer) SDL_ReleaseGPUBuffer(gpuDevice, gpuMesh.vertexBuffer);
			if (gpuMesh.indexBuffer) SDL_ReleaseGPUBuffer(gpuDevice, gpuMesh.indexBuffer);
		}
		gpuMeshes.clear();
		meshBounds.clear();
		staticBatches.clear();
		staticBatchCells.clear();
		renderItems.clear();
		staticBatchesBuilt = false;
		for (auto& [key, shadowMap] : staticShadowMaps) if (shadowMap.texture) SDL_ReleaseGPUTexture(gpuDevice, shadowMap.texture);
		staticShadowMaps.clear();
		for (auto& [key, texture] : textures) SDL_ReleaseGPUTexture(gpuDevice, texture);
		textures.clear();
	}

	void PrepareScene(Scene& scene) {
		ReleaseSceneResources();
		BuildStaticBatches(scene);
		fallbackShadowReady = false;
		fallbackShadowMap = {};
		for (auto& state : pointShadowStates) {
			state.staticReady = false;
			state.staticValid = false;
			state.dynamicValid = false;
		}
		staticShadowDirection = {};
		staticShadowDebugTexture = nullptr;
		shadowLight = nullptr;
		renderItems.reserve(staticBatches.size() + scene.Objects().size());
		for (auto& batch : staticBatches) {
			Culling::GetBounds(batch.get());
			GetMesh(batch.get());
		}
		sourceSignature = SceneSignature(scene);
	}

	void Render(Scene& scene, SDL_GPUCommandBuffer* commandBuffer, SDL_GPUTexture* colorTarget, Uint32 width, Uint32 height) {
		if (!colorTarget || !width || !height) return;
		const Uint64 renderStart = SDL_GetTicksNS();
		if (sourceSignature != SceneSignature(scene)) PrepareScene(scene);
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
		const LightData lightData = CollectDirectionalLight(scene);
		const LightData* light = lightData.source ? &lightData : nullptr;
		std::vector<PointLightData> pointLights = CollectPointLights(scene);
		const bool directionalShadowsActive = scene.shadowsEnabled && light && light->castShadows;
		const bool pointShadowsActive = scene.shadowsEnabled && std::any_of(pointLights.begin(), pointLights.end(), [](const PointLightData& pointLight) { return pointLight.castShadows; });
		const bool shadowsActive = directionalShadowsActive || pointShadowsActive;
		EnsurePointShadowTexture(pointLights, scene.shadowsEnabled);
		for (auto& object : scene.Objects()) if (auto* mesh = object->GetComponent<Mesh>()) totalTriangles += mesh->data.indices.size() / 3;
		if (depthWidth != width || depthHeight != height) {
			if (depthTexture) SDL_ReleaseGPUTexture(gpuDevice, depthTexture);
			depthTexture = CreateDepthTexture(width, height);
			depthWidth = width;
			depthHeight = height;
		}
		renderItems.clear();
		std::vector<RenderItem> staticItems;
		std::vector<RenderItem> dynamicItems;
		std::vector<RenderItem> visibleItems;
		Transform cameraTransform{};
		const float aspectRatio = static_cast<float>(width) / height;
		ShadowMapData dynamicMap{};
		ChunkKey cameraCell{};
		if (scene.activeCamera && scene.activeCamera->owner) {
			cameraTransform = CameraWorldTransform(*scene.activeCamera);
			cameraCell = {scene.chunkSize > 0 ? static_cast<int>(std::floor(cameraTransform.position.x / scene.chunkSize)) : 0, scene.chunkSize > 0 ? static_cast<int>(std::floor(cameraTransform.position.y / scene.chunkSize)) : 0, scene.chunkSize > 0 ? static_cast<int>(std::floor(cameraTransform.position.z / scene.chunkSize)) : 0};
			for (auto& batch : staticBatches) {
				const auto& cell = staticBatchCells.at(batch.get());
				const RenderItem item{batch.get(), Identity(), Identity(), std::get<0>(cell), std::get<1>(cell), std::get<2>(cell)};
				renderItems.push_back(item);
				if (shadowsActive) {
					if (batch->windAnimated) dynamicItems.push_back(item);
					else staticItems.push_back(item);
				}
			}
			for (auto& object : scene.Objects()) {
				Mesh* mesh = object->GetComponent<Mesh>();
				if (!mesh || mesh->data.vertices.empty() || mesh->data.indices.empty() || (object->renderStatic && mesh->displacementTexturePath.empty())) continue;
				Transform transform = object->transform;
				if (auto* body = object->GetComponent<PhysicsBody>()) {
					const float alpha = Time::interpolationAlpha;
					transform.position = {body->previousPosition.x + (transform.position.x - body->previousPosition.x) * alpha, body->previousPosition.y + (transform.position.y - body->previousPosition.y) * alpha, body->previousPosition.z + (transform.position.z - body->previousPosition.z) * alpha};
				}
				const RenderItem item{mesh, Model(transform), NormalMatrix(transform), scene.chunkSize > 0 ? static_cast<int>(std::floor(transform.position.x / scene.chunkSize)) : 0, scene.chunkSize > 0 ? static_cast<int>(std::floor(transform.position.y / scene.chunkSize)) : 0, scene.chunkSize > 0 ? static_cast<int>(std::floor(transform.position.z / scene.chunkSize)) : 0};
				renderItems.push_back(item);
				if (shadowsActive) {
					if (object->renderStatic && !mesh->windAnimated) staticItems.push_back(item);
					else dynamicItems.push_back(item);
				}
			}
			const auto frustum = Culling::FromCamera(*scene.activeCamera, aspectRatio);
			const Uint64 cullingStart = SDL_GetTicksNS();
			for (const auto& item : renderItems) if (Culling::IsVisible(Culling::GetBounds(item.mesh), item.model, frustum)) visibleItems.push_back(item);
			cullingTimeNanoseconds = SDL_GetTicksNS() - cullingStart;
			if (directionalShadowsActive) {
				if (light->renderStatic) {
					const Vector3 direction = Normalize(light->direction);
					if (shadowLight != light->source || direction.x != staticShadowDirection.x || direction.y != staticShadowDirection.y || direction.z != staticShadowDirection.z) {
						fallbackShadowReady = false;
						for (auto& [key, map] : staticShadowMaps) map.ready = false;
						shadowLight = light->source;
						staticShadowDirection = direction;
					}
					std::vector<RenderItem> casters;
					if (!fallbackShadowReady) {
						fallbackShadowMap = BuildStaticShadowMap(direction, 0, 0, 0, 0, staticItems, casters);
						RenderShadowMap(commandBuffer, fallbackShadowTexture, fallbackShadowMap, casters, frameTime);
						fallbackShadowReady = true;
					}
					std::set<ChunkKey> visibleCells;
					for (const auto& item : visibleItems) visibleCells.emplace(item.cellX, item.cellY, item.cellZ);
					for (auto map = staticShadowMaps.begin(); map != staticShadowMaps.end();) {
						if (!visibleCells.count(map->first)) {
							if (map->second.texture) SDL_ReleaseGPUTexture(gpuDevice, map->second.texture);
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
					const auto preview = staticShadowMaps.find(cameraCell);
					staticShadowDebugTexture = preview != staticShadowMaps.end() && preview->second.data.valid ? preview->second.texture : fallbackShadowTexture;
				}
				const auto& casters = light->renderStatic ? dynamicItems : renderItems;
				dynamicMap = BuildDynamicShadowMap(*scene.activeCamera, aspectRatio, scene.shadowFar, *light, casters);
				RenderShadowMap(commandBuffer, dynamicShadowTexture, dynamicMap, casters, frameTime);
			}
			if (pointShadowsActive) {
				std::size_t stateIndex = 0;
				for (auto& pointLight : pointLights) {
					if (!pointLight.castShadows) continue;
					auto& state = pointShadowStates[stateIndex++];
					if (state.source != pointLight.source || state.position.x != pointLight.position.x || state.position.y != pointLight.position.y || state.position.z != pointLight.position.z || state.range != pointLight.range || state.renderStatic != pointLight.renderStatic) {
						state.source = pointLight.source;
						state.position = pointLight.position;
						state.range = pointLight.range;
						state.renderStatic = pointLight.renderStatic;
						state.staticReady = false;
						state.staticValid = false;
						state.dynamicValid = false;
					}
					if (pointLight.renderStatic) {
						if (!state.staticReady) {
							const auto casters = PointShadowCasters(staticItems, pointLight.position, pointLight.range);
							state.staticValid = !casters.empty();
							if (state.staticValid) RenderPointShadowMap(commandBuffer, pointShadowTexture, pointLight.shadowIndex * 2, pointLight, casters, frameTime);
							state.staticReady = true;
						}
						const auto casters = PointShadowCasters(dynamicItems, pointLight.position, pointLight.range);
						state.dynamicValid = !casters.empty();
						if (state.dynamicValid) RenderPointShadowMap(commandBuffer, pointShadowTexture, pointLight.shadowIndex * 2 + 1, pointLight, casters, frameTime);
					}
					else {
						const auto casters = PointShadowCasters(renderItems, pointLight.position, pointLight.range);
						state.staticValid = false;
						state.staticReady = true;
						state.dynamicValid = !casters.empty();
						if (state.dynamicValid) RenderPointShadowMap(commandBuffer, pointShadowTexture, pointLight.shadowIndex * 2 + 1, pointLight, casters, frameTime);
					}
					pointLight.staticShadowValid = state.staticValid;
					pointLight.dynamicShadowValid = state.dynamicValid;
				}
			}
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
		SDL_GPURenderPass* pass = SDL_BeginGPURenderPass(commandBuffer, &colorInfo, 1, &depthInfo);
		const Matrix4 view = View(cameraTransform);
		const Matrix4 projection = scene.activeCamera ? Perspective(scene.activeCamera->fieldOfView, aspectRatio, scene.activeCamera->nearClip, scene.activeCamera->farClip) : Identity();
		SDL_GPUGraphicsPipeline* activePipeline = nullptr;
		for (const auto& item : visibleItems) {
			Mesh* mesh = item.mesh;
			const auto selected = shadowsActive ? staticShadowMaps.find({item.cellX, item.cellY, item.cellZ}) : staticShadowMaps.end();
			const StaticShadowMap* detail = selected != staticShadowMaps.end() ? &selected->second : nullptr;
			const bool staticEnabled = directionalShadowsActive && light->renderStatic;
			const ShadowMapData maps[3] = {detail ? detail->data : ShadowMapData{}, fallbackShadowMap, dynamicMap};
			const bool enabled[3] = {staticEnabled && detail && detail->data.valid, staticEnabled && fallbackShadowMap.valid, directionalShadowsActive && dynamicMap.valid};
			SDL_GPUGraphicsPipeline* meshPipeline = mesh->cullBackfaces ? pipeline : noCullingPipeline;
			if (meshPipeline != activePipeline) {
				SDL_BindGPUGraphicsPipeline(pass, meshPipeline);
				activePipeline = meshPipeline;
			}
			GpuMesh& gpuMesh = GetMesh(mesh);
			visibleTriangles += gpuMesh.indexCount / 3;
			ObjectUniforms vertex{};
			const Matrix4 mvp = Multiply(projection, Multiply(view, item.model));
			std::memcpy(vertex.mvp, mvp.values, sizeof(vertex.mvp));
			std::memcpy(vertex.model, item.model.values, sizeof(vertex.model));
			std::memcpy(vertex.normal, item.normal.values, sizeof(vertex.normal));
			vertex.displacementParameters[0] = mesh->displacementTexturePath.empty() ? 0.0f : mesh->displacementScale;
			vertex.displacementParameters[1] = mesh->windAnimated ? 1.0f : 0.0f;
			vertex.displacementParameters[2] = frameTime;
			vertex.displacementParameters[3] = 0.08f;
			FragmentUniforms fragment{};
			fragment.cameraPosition[0] = cameraTransform.position.x;
			fragment.cameraPosition[1] = cameraTransform.position.y;
			fragment.cameraPosition[2] = cameraTransform.position.z;
			fragment.fogColor[0] = scene.fogColor.x;
			fragment.fogColor[1] = scene.fogColor.y;
			fragment.fogColor[2] = scene.fogColor.z;
			fragment.fogParameters[0] = scene.fogNear;
			fragment.fogParameters[1] = scene.fogFar;
			fragment.fogParameters[2] = scene.fogEnabled ? 1.0f : 0.0f;
			fragment.ambientLight[0] = scene.ambientLightColor.x;
			fragment.ambientLight[1] = scene.ambientLightColor.y;
			fragment.ambientLight[2] = scene.ambientLightColor.z;
			fragment.ambientLight[3] = scene.ambientLight ? 1.0f : 0.0f;
			if (light) {
				fragment.directionalDirection[0] = light->direction.x;
				fragment.directionalDirection[1] = light->direction.y;
				fragment.directionalDirection[2] = light->direction.z;
				fragment.directionalDirection[3] = light->intensity;
				fragment.directionalColor[0] = light->color.x;
				fragment.directionalColor[1] = light->color.y;
				fragment.directionalColor[2] = light->color.z;
			}
			for (int index = 0; index < 3; ++index) {
				if (directionalShadowsActive) std::memcpy(vertex.shadowMvp[index], maps[index].mvp.values, sizeof(vertex.shadowMvp[index]));
				fragment.shadowParameters[index][0] = enabled[index] ? 1.0f : 0.0f;
				fragment.shadowParameters[index][1] = directionalShadowsActive ? std::max(light->shadowBias, 0.0f) / std::max(maps[index].depthRange, 0.01f) : 0.0f;
				fragment.shadowParameters[index][2] = 1.0f / shadowMapResolution;
			}
			fragment.shadowParameters[1][3] = 1.0f;
			fragment.lightingParameters[0] = mesh->displacementTexturePath.empty() ? 0.0f : 1.0f;
			fragment.lightingParameters[1] = scene.shadowsEnabled && shadowDebugMode == 1 ? 1.0f : 0.0f;
			fragment.lightingParameters[2] = shadowsActive ? 1.0f : 0.0f;
			fragment.pointLightCount[0] = static_cast<float>(pointLights.size());
			for (std::size_t index = 0; index < pointLights.size(); ++index) {
				fragment.pointLightPositionRange[index][0] = pointLights[index].position.x;
				fragment.pointLightPositionRange[index][1] = pointLights[index].position.y;
				fragment.pointLightPositionRange[index][2] = pointLights[index].position.z;
				fragment.pointLightPositionRange[index][3] = pointLights[index].range;
				fragment.pointLightColorIntensity[index][0] = pointLights[index].color.x;
				fragment.pointLightColorIntensity[index][1] = pointLights[index].color.y;
				fragment.pointLightColorIntensity[index][2] = pointLights[index].color.z;
				fragment.pointLightColorIntensity[index][3] = pointLights[index].intensity;
				fragment.pointLightShadowInfo[index][0] = static_cast<float>(pointLights[index].shadowIndex * 2);
				fragment.pointLightShadowInfo[index][1] = static_cast<float>(pointLights[index].shadowIndex * 2 + 1);
				fragment.pointLightShadowInfo[index][2] = pointLights[index].staticShadowValid ? 1.0f : 0.0f;
				fragment.pointLightShadowInfo[index][3] = pointLights[index].dynamicShadowValid ? 1.0f : 0.0f;
				fragment.pointLightShadowBias[index][0] = std::max(pointLights[index].shadowBias, 0.0f);
			}
			vertex.shadowState[0] = directionalShadowsActive ? 1.0f : 0.0f;
			SDL_PushGPUVertexUniformData(commandBuffer, 0, &vertex, sizeof(vertex));
			SDL_PushGPUFragmentUniformData(commandBuffer, 0, &fragment, sizeof(fragment));
			SDL_GPUBufferBinding vertexBinding{gpuMesh.vertexBuffer, 0};
			SDL_GPUBufferBinding indexBinding{gpuMesh.indexBuffer, 0};
			SDL_BindGPUVertexBuffers(pass, 0, &vertexBinding, 1);
			SDL_BindGPUIndexBuffer(pass, &indexBinding, SDL_GPU_INDEXELEMENTSIZE_32BIT);
			SDL_GPUTextureSamplerBinding bindings[5]{};
			bindings[0] = {gpuMesh.texture, sampler};
			bindings[1] = {enabled[0] ? detail->texture : fallbackShadowTexture, shadowSampler};
			bindings[2] = {fallbackShadowTexture, shadowSampler};
			bindings[3] = {dynamicShadowTexture, shadowSampler};
			bindings[4] = {pointShadowTexture, shadowSampler};
			SDL_BindGPUFragmentSamplers(pass, 0, bindings, 5);
			SDL_GPUTextureSamplerBinding vertexBindings[2] = {{gpuMesh.displacementTexture, sampler}, {gpuMesh.maskTexture, sampler}};
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
	std::string GPUCompletionTime() { return hasGpuCompletionTime ? Milliseconds(gpuCompletionTimeNanoseconds / 1000000.0) : "pending"; }
	std::string DrawCalls() { return std::to_string(drawCallCount); }
	std::string TriangleCount() { return triangleCountString; }
	void CycleShadowDebugMode() { shadowDebugMode = (shadowDebugMode + 1) % 3; }
	int ShadowDebugMode() { return shadowDebugMode; }
	SDL_GPUTexture* ShadowMap(int mapIndex) { return mapIndex == 0 ? staticShadowDebugTexture : mapIndex == 1 ? dynamicShadowTexture : nullptr; }

	void Submit(SDL_GPUCommandBuffer* commandBuffer) {
		UpdateGpuCompletionTime();
		const Uint64 submitStart = SDL_GetTicksNS();
		SDL_GPUFence* fence = SDL_SubmitGPUCommandBufferAndAcquireFence(commandBuffer);
		const Uint64 submittedAt = SDL_GetTicksNS();
		submitTimeNanoseconds = submittedAt - submitStart;
		if (fence) pendingGpuSubmissions.push_back({fence, submittedAt});
	}

	void Shutdown() {
		for (const auto& submission : pendingGpuSubmissions) SDL_ReleaseGPUFence(gpuDevice, submission.fence);
		pendingGpuSubmissions.clear();
		ReleaseSceneResources();
		staticShadowDebugTexture = nullptr;
		staticShadowDirection = {};
		if (whiteTexture) SDL_ReleaseGPUTexture(gpuDevice, whiteTexture);
		if (sampler) SDL_ReleaseGPUSampler(gpuDevice, sampler);
		if (shadowSampler) SDL_ReleaseGPUSampler(gpuDevice, shadowSampler);
		if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
		if (noCullingPipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, noCullingPipeline);
		if (shadowPipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, shadowPipeline);
		if (shadowNoCullingPipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, shadowNoCullingPipeline);
		if (depthTexture) SDL_ReleaseGPUTexture(gpuDevice, depthTexture);
		if (dynamicShadowTexture) SDL_ReleaseGPUTexture(gpuDevice, dynamicShadowTexture);
		if (fallbackShadowTexture) SDL_ReleaseGPUTexture(gpuDevice, fallbackShadowTexture);
		if (pointShadowTexture) SDL_ReleaseGPUTexture(gpuDevice, pointShadowTexture);
		pipeline = nullptr;
		noCullingPipeline = nullptr;
		shadowPipeline = nullptr;
		shadowNoCullingPipeline = nullptr;
		depthTexture = nullptr;
		dynamicShadowTexture = nullptr;
		fallbackShadowTexture = nullptr;
		pointShadowTexture = nullptr;
		pointShadowResolution = 0;
		pointShadowSources.clear();
		pointShadowStates.clear();
		fallbackShadowReady = false;
		fallbackShadowMap = {};
		shadowLight = nullptr;
		shadowMapResolution = 2048;
		sourceSignature = 0;
		depthWidth = 0;
		depthHeight = 0;
		fpsElapsed = 0.0f;
		fpsFrames = 0;
		hasGpuCompletionTime = false;
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
