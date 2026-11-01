#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <map>
#include <memory>
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
	SDL_GPUTexture* LoadTexture(std::string_view fileName);

	namespace {
		struct Matrix4 {
			float values[16]{};
		};

		struct RenderItem {
			Mesh* mesh;
			Matrix4 model;
			Matrix4 normal;
		};

		struct ObjectUniforms {
			float mvp[16];
			float model[16];
			float cameraPosition[4];
			float fogColor[4];
			float fogParameters[4];
			float ambientLight[4];
			float normal[16];
			float directionalDirection[4];
			float directionalColor[4];
			float displacementParameters[4];
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
		SDL_GPUTexture* depthTexture = nullptr;
		SDL_GPUTexture* whiteTexture = nullptr;
		SDL_GPUSampler* sampler = nullptr;
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

		constexpr const char* vertexShaderSource = R"(
			#include <metal_stdlib>
			using namespace metal;
			struct VertexInput { float3 position [[attribute(0)]]; float3 normal [[attribute(1)]]; float2 uv [[attribute(2)]]; float4 color [[attribute(3)]]; };
			struct VertexOutput { float4 position [[position]]; float2 uv; float4 color; float3 worldPosition; float3 worldNormal; float3 cameraPosition; float4 fogColor; float4 fogParameters; float4 ambientLight; float4 directionalDirection; float4 directionalColor; float displacementEnabled; };
			struct ObjectUniforms { float4x4 mvp; float4x4 model; float4 cameraPosition; float4 fogColor; float4 fogParameters; float4 ambientLight; float4x4 normal; float4 directionalDirection; float4 directionalColor; float4 displacementParameters; };
			vertex VertexOutput main0(VertexInput input [[stage_in]], constant ObjectUniforms& uniforms [[buffer(0)]], texture2d<float> displacementTexture [[texture(0)]], sampler displacementSampler [[sampler(0)]], texture2d<float> maskTexture [[texture(1)]], sampler maskSampler [[sampler(1)]]) {
				VertexOutput output;
				float3 position = input.position;
				if (uniforms.displacementParameters.x != 0.0) {
					float height = displacementTexture.sample(displacementSampler, input.uv, level(0.0)).r;
					float mask = maskTexture.sample(maskSampler, input.uv, level(0.0)).r;
					position += input.normal * ((height - 0.5) * mask * uniforms.displacementParameters.x);
				}
				output.position = uniforms.mvp * float4(position, 1.0);
				output.uv = input.uv;
				output.color = input.color;
				output.worldPosition = (uniforms.model * float4(position, 1.0)).xyz;
				output.worldNormal = normalize((uniforms.normal * float4(input.normal, 0.0)).xyz);
				output.cameraPosition = uniforms.cameraPosition.xyz;
				output.fogColor = uniforms.fogColor;
				output.fogParameters = uniforms.fogParameters;
				output.ambientLight = uniforms.ambientLight;
				output.directionalDirection = uniforms.directionalDirection;
				output.directionalColor = uniforms.directionalColor;
				output.displacementEnabled = uniforms.displacementParameters.x != 0.0 ? 1.0 : 0.0;
				return output;
			}
		)";

		constexpr const char* fragmentShaderSource = R"(
			#include <metal_stdlib>
			using namespace metal;
			struct FragmentInput { float4 position [[position]]; float2 uv; float4 color; float3 worldPosition; float3 worldNormal; float3 cameraPosition; float4 fogColor; float4 fogParameters; float4 ambientLight; float4 directionalDirection; float4 directionalColor; float displacementEnabled; };
			fragment float4 main0(FragmentInput input [[stage_in]], texture2d<float> image [[texture(0)]], sampler imageSampler [[sampler(0)]]) {
				float4 texel = image.sample(imageSampler, input.uv);
				if (texel.a < 0.5) discard_fragment();
				float3 albedo = input.color.rgb * texel.rgb;
				float3 ambientColor = input.ambientLight.a > 0.5 ? input.ambientLight.rgb : float3(0.0);
				float3 normal = normalize(input.worldNormal);
				if (input.displacementEnabled > 0.5) {
					float3 displacedNormal = normalize(cross(dfdx(input.worldPosition), dfdy(input.worldPosition)));
					if (dot(displacedNormal, normal) < 0.0) displacedNormal = -displacedNormal;
					normal = displacedNormal;
				}
				float diffuse = max(dot(normal, normalize(-input.directionalDirection.xyz)), 0.0) * input.directionalDirection.w;
				float3 color = albedo * (ambientColor + input.directionalColor.rgb * diffuse);
				float fogFactor = input.fogParameters.z > 0.5 ? clamp((distance(input.worldPosition, input.cameraPosition) - input.fogParameters.x) / max(input.fogParameters.y - input.fogParameters.x, 0.0001), 0.0, 1.0) : 0.0;
				return float4(mix(color, input.fogColor.rgb, fogFactor), input.color.a * texel.a);
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

		namespace Culling {
			struct Frustum {
				Vector3 position;
				Vector3 right;
				Vector3 up;
				Vector3 forward;
				float tanHalfFov;
				float aspectRatio;
				float nearClip;
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
					camera.nearClip,
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
					entry->second.minimum = mesh->data.vertices.front().position;
					entry->second.maximum = mesh->data.vertices.front().position;
					for (const auto& vertex : mesh->data.vertices) {
						entry->second.minimum.x = std::min(entry->second.minimum.x, vertex.position.x);
						entry->second.minimum.y = std::min(entry->second.minimum.y, vertex.position.y);
						entry->second.minimum.z = std::min(entry->second.minimum.z, vertex.position.z);
						entry->second.maximum.x = std::max(entry->second.maximum.x, vertex.position.x);
						entry->second.maximum.y = std::max(entry->second.maximum.y, vertex.position.y);
						entry->second.maximum.z = std::max(entry->second.maximum.z, vertex.position.z);
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

		void BuildStaticBatches(Scene& scene) {
			if (staticBatchesBuilt) return;
			using BatchKey = std::tuple<int, int, int, std::string, std::string, std::string, float, bool>;
			using CellKey = std::tuple<int, int, int>;
			std::map<BatchKey, MeshData> batchData;
			for (auto& object : scene.Objects()) {
				if (!object->renderStatic) continue;
				Mesh* mesh = object->GetComponent<Mesh>();
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
					auto& output = batchData[{cellX, cellY, cellZ, mesh->texturePath, mesh->displacementTexturePath, mesh->maskTexturePath, mesh->displacementScale, mesh->cullBackfaces}];
					auto& vertexMap = vertexMaps[cellKey];
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
			staticBatches.reserve(batchData.size());
			for (auto& [key, data] : batchData) {
				auto batch = std::make_unique<Mesh>();
				batch->data = std::move(data);
				batch->texturePath = std::get<3>(key);
				batch->displacementTexturePath = std::get<4>(key);
				batch->maskTexturePath = std::get<5>(key);
				batch->displacementScale = std::get<6>(key);
				batch->cullBackfaces = std::get<7>(key);
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
			textureInfo.format = SDL_GPU_TEXTUREFORMAT_D16_UNORM;
			textureInfo.usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET;
			textureInfo.width = width;
			textureInfo.height = height;
			textureInfo.layer_count_or_depth = 1;
			textureInfo.num_levels = 1;
			textureInfo.sample_count = SDL_GPU_SAMPLECOUNT_1;
			return SDL_CreateGPUTexture(gpuDevice, &textureInfo);
		}

		SDL_GPUTexture* GetTexture(const std::string& fileName) {
			if (auto found = textures.find(fileName); found != textures.end()) return found->second;
			SDL_GPUTexture* texture = LoadTexture(fileName);
			if (texture) textures[fileName] = texture;
			return texture;
		}

		GpuMesh& GetMesh(Mesh* mesh) {
			auto [entry, inserted] = gpuMeshes.try_emplace(mesh);
			if (inserted) {
				entry->second = UploadMesh(mesh->data);
				entry->second.texture = mesh->texturePath.empty() ? whiteTexture : GetTexture(mesh->texturePath);
				if (!entry->second.texture) entry->second.texture = whiteTexture;
				entry->second.displacementTexture = mesh->displacementTexturePath.empty() ? whiteTexture : GetTexture(mesh->displacementTexturePath);
				if (!entry->second.displacementTexture) entry->second.displacementTexture = whiteTexture;
				entry->second.maskTexture = mesh->maskTexturePath.empty() ? whiteTexture : GetTexture(mesh->maskTexturePath);
				if (!entry->second.maskTexture) entry->second.maskTexture = whiteTexture;
			}
			return entry->second;
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

	SDL_GPUTexture* LoadTexture(std::string_view fileName) {
		const std::string path = (std::filesystem::path(SDL_GetBasePath()) / "resources" / std::string(fileName)).string();
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
		return CreateTexture(gpuDevice, width, height, pixels.data());
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
		SDL_GPUShader* vertexShader = CreateShader(gpuDevice, vertexShaderSource, SDL_GPU_SHADERSTAGE_VERTEX, SDL_GPU_SHADERFORMAT_MSL, 2, 1);
		SDL_GPUShader* fragmentShader = CreateShader(gpuDevice, fragmentShaderSource, SDL_GPU_SHADERSTAGE_FRAGMENT, SDL_GPU_SHADERFORMAT_MSL, 1);
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
		SDL_GPUColorTargetDescription colorTarget{};
		colorTarget.format = SDL_GetGPUSwapchainTextureFormat(gpuDevice, window);
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
		pipelineInfo.target_info.depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D16_UNORM;
		pipeline = CreatePipeline(gpuDevice, pipelineInfo);
		pipelineInfo.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
		noCullingPipeline = CreatePipeline(gpuDevice, pipelineInfo);
		SDL_ReleaseGPUShader(gpuDevice, vertexShader);
		SDL_ReleaseGPUShader(gpuDevice, fragmentShader);
	}

	void PrepareScene(Scene& scene) {
		BuildStaticBatches(scene);
		renderItems.reserve(staticBatches.size() + scene.Objects().size());
		for (auto& batch : staticBatches) {
			Culling::GetBounds(batch.get());
			GetMesh(batch.get());
		}
	}

	void Render(Scene& scene, SDL_GPUCommandBuffer* commandBuffer, SDL_GPUTexture* colorTarget, Uint32 width, Uint32 height) {
		if (!colorTarget) return;
		const Uint64 renderStart = SDL_GetTicksNS();
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
		Vector3 directionalDirection{0.0f, -1.0f, 0.0f};
		Vector3 directionalColor{1.0f, 1.0f, 1.0f};
		float directionalIntensity = 0.0f;
		bool hasDirectionalLight = false;
		for (auto& object : scene.Objects()) {
			if (auto* mesh = object->GetComponent<Mesh>()) totalTriangles += mesh->data.indices.size() / 3;
			if (auto* light = object->GetComponent<Light>(); light && light->type == Light::Type::Directional && !hasDirectionalLight) {
				directionalDirection = object->transform.Forward();
				directionalColor = light->color;
				directionalIntensity = light->intensity;
				hasDirectionalLight = true;
			}
		}
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
			const Transform cameraTransform = CameraWorldTransform(*scene.activeCamera);
			const Matrix4 view = View(cameraTransform);
			const float aspectRatio = static_cast<float>(width) / height;
			const Matrix4 projection = Perspective(scene.activeCamera->fieldOfView, aspectRatio, scene.activeCamera->nearClip, scene.activeCamera->farClip);
			const Culling::Frustum frustum = Culling::FromCamera(*scene.activeCamera, aspectRatio);
			renderItems.clear();
			for (auto& batch : staticBatches) renderItems.push_back({batch.get(), Identity(), Identity()});
			for (auto& object : scene.Objects()) {
				if (object->renderStatic) continue;
				Mesh* mesh = object->GetComponent<Mesh>();
				if (!mesh || mesh->data.vertices.empty() || mesh->data.indices.empty()) continue;
				Transform renderTransform = object->transform;
				if (auto* body = object->GetComponent<PhysicsBody>()) {
					const float alpha = Time::interpolationAlpha;
					renderTransform.position = {
						body->previousPosition.x + (object->transform.position.x - body->previousPosition.x) * alpha,
						body->previousPosition.y + (object->transform.position.y - body->previousPosition.y) * alpha,
						body->previousPosition.z + (object->transform.position.z - body->previousPosition.z) * alpha
					};
				}
				renderItems.push_back({mesh, Model(renderTransform), NormalMatrix(renderTransform)});
			}
			SDL_GPUGraphicsPipeline* activePipeline = nullptr;
			for (const auto& item : renderItems) {
				Mesh* mesh = item.mesh;
				SDL_GPUGraphicsPipeline* meshPipeline = mesh->cullBackfaces ? pipeline : noCullingPipeline;
				if (meshPipeline != activePipeline) {
					SDL_BindGPUGraphicsPipeline(renderPass, meshPipeline);
					activePipeline = meshPipeline;
				}
				const Matrix4& model = item.model;
				const Uint64 cullingStart = SDL_GetTicksNS();
				const bool visible = Culling::IsVisible(Culling::GetBounds(mesh), model, frustum);
				cullingTimeNanoseconds += SDL_GetTicksNS() - cullingStart;
				if (!visible) continue;
				GpuMesh& gpuMesh = GetMesh(mesh);
				visibleTriangles += gpuMesh.indexCount / 3;
				const Matrix4 mvp = Multiply(projection, Multiply(view, model));
				ObjectUniforms uniforms{};
				std::memcpy(uniforms.mvp, mvp.values, sizeof(uniforms.mvp));
				std::memcpy(uniforms.model, model.values, sizeof(uniforms.model));
				std::memcpy(uniforms.normal, item.normal.values, sizeof(uniforms.normal));
				uniforms.cameraPosition[0] = cameraTransform.position.x;
				uniforms.cameraPosition[1] = cameraTransform.position.y;
				uniforms.cameraPosition[2] = cameraTransform.position.z;
				uniforms.cameraPosition[3] = 1.0f;
				uniforms.fogColor[0] = scene.fogColor.x;
				uniforms.fogColor[1] = scene.fogColor.y;
				uniforms.fogColor[2] = scene.fogColor.z;
				uniforms.fogColor[3] = 1.0f;
				uniforms.fogParameters[0] = scene.fogNear;
				uniforms.fogParameters[1] = scene.fogFar;
				uniforms.fogParameters[2] = scene.fogEnabled ? 1.0f : 0.0f;
				uniforms.ambientLight[0] = scene.ambientLightColor.x;
				uniforms.ambientLight[1] = scene.ambientLightColor.y;
				uniforms.ambientLight[2] = scene.ambientLightColor.z;
				uniforms.ambientLight[3] = scene.ambientLight ? 1.0f : 0.0f;
				uniforms.directionalDirection[0] = directionalDirection.x;
				uniforms.directionalDirection[1] = directionalDirection.y;
				uniforms.directionalDirection[2] = directionalDirection.z;
				uniforms.directionalDirection[3] = directionalIntensity;
				uniforms.directionalColor[0] = directionalColor.x;
				uniforms.directionalColor[1] = directionalColor.y;
				uniforms.directionalColor[2] = directionalColor.z;
				uniforms.directionalColor[3] = 1.0f;
				uniforms.displacementParameters[0] = mesh->displacementTexturePath.empty() ? 0.0f : mesh->displacementScale;
				SDL_PushGPUVertexUniformData(commandBuffer, 0, &uniforms, sizeof(uniforms));
				SDL_GPUBufferBinding vertexBinding{};
				vertexBinding.buffer = gpuMesh.vertexBuffer;
				SDL_BindGPUVertexBuffers(renderPass, 0, &vertexBinding, 1);
				SDL_GPUBufferBinding indexBinding{};
				indexBinding.buffer = gpuMesh.indexBuffer;
				SDL_BindGPUIndexBuffer(renderPass, &indexBinding, SDL_GPU_INDEXELEMENTSIZE_32BIT);
				SDL_GPUTextureSamplerBinding textureBinding{};
				textureBinding.texture = gpuMesh.texture;
				textureBinding.sampler = sampler;
				SDL_BindGPUFragmentSamplers(renderPass, 0, &textureBinding, 1);
				SDL_GPUTextureSamplerBinding vertexTextureBindings[2]{};
				vertexTextureBindings[0].texture = gpuMesh.displacementTexture;
				vertexTextureBindings[0].sampler = sampler;
				vertexTextureBindings[1].texture = gpuMesh.maskTexture;
				vertexTextureBindings[1].sampler = sampler;
				SDL_BindGPUVertexSamplers(renderPass, 0, vertexTextureBindings, 2);
				SDL_DrawGPUIndexedPrimitives(renderPass, gpuMesh.indexCount, 1, 0, 0, 0);
				++drawCallCount;
			}
		}
		SDL_EndGPURenderPass(renderPass);
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
		for (auto& [mesh, gpuMesh] : gpuMeshes) {
			if (gpuMesh.vertexBuffer) SDL_ReleaseGPUBuffer(gpuDevice, gpuMesh.vertexBuffer);
			if (gpuMesh.indexBuffer) SDL_ReleaseGPUBuffer(gpuDevice, gpuMesh.indexBuffer);
		}
		gpuMeshes.clear();
		meshBounds.clear();
		staticBatches.clear();
		renderItems.clear();
		staticBatchesBuilt = false;
		for (auto& [path, texture] : textures) SDL_ReleaseGPUTexture(gpuDevice, texture);
		textures.clear();
		if (whiteTexture) SDL_ReleaseGPUTexture(gpuDevice, whiteTexture);
		if (sampler) SDL_ReleaseGPUSampler(gpuDevice, sampler);
		if (pipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, pipeline);
		if (noCullingPipeline) SDL_ReleaseGPUGraphicsPipeline(gpuDevice, noCullingPipeline);
		if (depthTexture) SDL_ReleaseGPUTexture(gpuDevice, depthTexture);
		pipeline = nullptr;
		noCullingPipeline = nullptr;
		depthTexture = nullptr;
		whiteTexture = nullptr;
		sampler = nullptr;
		gpuDevice = nullptr;
	}
}

namespace Sapphire {
	MeshData Mesh::Cube() { return Graphics::CreateCube(); }
	MeshData Mesh::Plane(int subdivisions) { return Graphics::CreatePlane(subdivisions); }
}
