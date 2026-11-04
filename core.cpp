#include "loading.cpp"
#include <algorithm>
#include <any>
#include <limits>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <memory>
#include <map>
#include <string>
#include <sstream>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <utility>
#include <stdexcept>

namespace Sapphire {
	class Object;
	class Scene;

	namespace Time {
		inline float deltaTime = 0.0f;
		inline float fixedDeltaTime = 1.0f / 60.0f;
		inline float interpolationAlpha = 0.0f;
	}

	namespace Math {
		inline constexpr float degreesToRadians = 0.01745329252f;
	}

	namespace Screen {
		inline int height = 720;
		inline int width = 1280;
	};

	struct Vector3 {
		float x = 0.0f;
		float y = 0.0f;
		float z = 0.0f;

		std::string String() const {
			std::ostringstream oss;
			oss << std::fixed << std::setprecision(2) << x << ", " << y << ", " << z;
			return oss.str();
		}

		Vector3 operator*(float scalar) const {
			return {x * scalar, y * scalar, z * scalar};
		}

		Vector3& operator+=(const Vector3& other) {
			x += other.x;
			y += other.y;
			z += other.z;
			return *this;
		}
	};

	struct Transform {
		Vector3 position;
		Vector3 rotation;
		Vector3 scale{1.0f, 1.0f, 1.0f};

		Vector3 Forward() const {
			const float yaw = rotation.y * Math::degreesToRadians;
			const float pitch = rotation.x * Math::degreesToRadians;
			return {std::sin(yaw) * std::cos(pitch), -std::sin(pitch), std::cos(yaw) * std::cos(pitch)};
		}
	};

	namespace Math {
		struct Matrix4 {
			float values[16]{};
		};

		struct Bounds {
			Vector3 minimum;
			Vector3 maximum;
		};

		Vector3 Subtract(const Vector3& a, const Vector3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

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

		void GetAxes(const Transform& transform, Vector3 (&axes)[3]) {
			const float x = transform.rotation.x * Math::degreesToRadians;
			const float y = transform.rotation.y * Math::degreesToRadians;
			const float z = transform.rotation.z * Math::degreesToRadians;
			const float cx = std::cos(x), sx = std::sin(x), cy = std::cos(y), sy = std::sin(y), cz = std::cos(z), sz = std::sin(z);
			axes[0] = {cz * cy, sz * cy, -sy};
			axes[1] = {cz * sy * sx - sz * cx, sz * sy * sx + cz * cx, cy * sx};
			axes[2] = {cz * sy * cx + sz * sx, sz * sy * cx - cz * sx, cy * cx};
		}

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

		Matrix4 Model(const Transform& transform, const Vector3* axes) {
			Matrix4 matrix = Identity();
			const float scale[3] = {transform.scale.x, transform.scale.y, transform.scale.z};
			for (int axis = 0; axis < 3; ++axis) {
				matrix.values[axis * 4] = axes[axis].x * scale[axis];
				matrix.values[axis * 4 + 1] = axes[axis].y * scale[axis];
				matrix.values[axis * 4 + 2] = axes[axis].z * scale[axis];
			}
			matrix.values[12] = transform.position.x;
			matrix.values[13] = transform.position.y;
			matrix.values[14] = transform.position.z;
			return matrix;
		}

		Matrix4 Model(const Transform& transform) {
			Vector3 axes[3];
			GetAxes(transform, axes);
			return Model(transform, axes);
		}

		Matrix4 NormalMatrix(const Transform& transform, const Vector3* axes) {
			Transform normalTransform = transform;
			normalTransform.position = {};
			normalTransform.scale = {
				std::abs(transform.scale.x) > 0.0001f ? 1.0f / transform.scale.x : 0.0f,
				std::abs(transform.scale.y) > 0.0001f ? 1.0f / transform.scale.y : 0.0f,
				std::abs(transform.scale.z) > 0.0001f ? 1.0f / transform.scale.z : 0.0f
			};
			return Model(normalTransform, axes);
		}

		Matrix4 NormalMatrix(const Transform& transform) {
			Vector3 axes[3];
			GetAxes(transform, axes);
			return NormalMatrix(transform, axes);
		}

		Vector3 TransformNormal(const Matrix4& normal, const Vector3& value) {
			Vector3 result{
				normal.values[0] * value.x + normal.values[4] * value.y + normal.values[8] * value.z,
				normal.values[1] * value.x + normal.values[5] * value.y + normal.values[9] * value.z,
				normal.values[2] * value.x + normal.values[6] * value.y + normal.values[10] * value.z
			};
			const float length = std::sqrt(result.x * result.x + result.y * result.y + result.z * result.z);
			if (length > 0.0f) {
				result.x /= length;
				result.y /= length;
				result.z /= length;
			}
			return result;
		}

		Bounds EmptyBounds() {
			const float maximum = std::numeric_limits<float>::max();
			return {{maximum, maximum, maximum}, {-maximum, -maximum, -maximum}};
		}

		void Include(Bounds& bounds, const Vector3& point) {
			bounds.minimum.x = std::min(bounds.minimum.x, point.x);
			bounds.minimum.y = std::min(bounds.minimum.y, point.y);
			bounds.minimum.z = std::min(bounds.minimum.z, point.z);
			bounds.maximum.x = std::max(bounds.maximum.x, point.x);
			bounds.maximum.y = std::max(bounds.maximum.y, point.y);
			bounds.maximum.z = std::max(bounds.maximum.z, point.z);
		}

		bool Overlaps(const Bounds& a, const Bounds& b) {
			return a.minimum.x <= b.maximum.x && a.maximum.x >= b.minimum.x && a.minimum.y <= b.maximum.y && a.maximum.y >= b.minimum.y && a.minimum.z <= b.maximum.z && a.maximum.z >= b.minimum.z;
		}

		Vector3 TransformPoint(const Matrix4& model, const Vector3& point) {
			return {
				model.values[0] * point.x + model.values[4] * point.y + model.values[8] * point.z + model.values[12],
				model.values[1] * point.x + model.values[5] * point.y + model.values[9] * point.z + model.values[13],
				model.values[2] * point.x + model.values[6] * point.y + model.values[10] * point.z + model.values[14]
			};
		}

		float Length(const Vector3& value) { return std::sqrt(Dot(value, value)); }

		Vector3 TransformPoint(const Transform& transform, const Vector3 (&axes)[3], const Vector3& point) {
			return Add(transform.position, Add(Scale(axes[0], point.x * transform.scale.x), Add(Scale(axes[1], point.y * transform.scale.y), Scale(axes[2], point.z * transform.scale.z))));
		}

		Bounds TransformBounds(const Bounds& bounds, const Matrix4& matrix) {
			const Vector3 center = TransformPoint(matrix, Scale(Add(bounds.minimum, bounds.maximum), 0.5f));
			const Vector3 halfSize = Scale(Subtract(bounds.maximum, bounds.minimum), 0.5f);
			const Vector3 extent{std::abs(matrix.values[0]) * halfSize.x + std::abs(matrix.values[4]) * halfSize.y + std::abs(matrix.values[8]) * halfSize.z, std::abs(matrix.values[1]) * halfSize.x + std::abs(matrix.values[5]) * halfSize.y + std::abs(matrix.values[9]) * halfSize.z, std::abs(matrix.values[2]) * halfSize.x + std::abs(matrix.values[6]) * halfSize.y + std::abs(matrix.values[10]) * halfSize.z};
			return {Subtract(center, extent), Add(center, extent)};
		}

		std::pair<float, float> ProjectBounds(const Bounds& bounds, const Vector3& axis) {
			const Vector3 center = Scale(Add(bounds.minimum, bounds.maximum), 0.5f);
			const Vector3 extent = Scale(Subtract(bounds.maximum, bounds.minimum), 0.5f);
			const float projection = Dot(center, axis);
			const float radius = std::abs(axis.x) * extent.x + std::abs(axis.y) * extent.y + std::abs(axis.z) * extent.z;
			return {projection - radius, projection + radius};
		}

		bool IntersectsSphere(const Bounds& bounds, const Vector3& center, float radius) {
			const Vector3 closest{std::clamp(center.x, bounds.minimum.x, bounds.maximum.x), std::clamp(center.y, bounds.minimum.y, bounds.maximum.y), std::clamp(center.z, bounds.minimum.z, bounds.maximum.z)};
			const Vector3 difference = Subtract(closest, center);
			return Dot(difference, difference) <= radius * radius;
		}
	}

	class Component {
		public:
			Object* owner = nullptr;
			virtual ~Component() = default;
	};

	class Camera : public Component {
		public:
			Vector3 localPosition;
			Vector3 localRotation;
			float fieldOfView = 60.0f;
			float nearClip = 0.1f;
			float farClip = 1000.0f;
	};

	class Light : public Component {
		public:
			enum class Type { Directional, Point, Spot };
			bool enabled = true;
			Type type = Type::Directional;
			Vector3 color{1.0f, 1.0f, 1.0f};
			float intensity = 1.0f;
			float range = 10.0f;
			float innerSpotAngle = 25.0f;
			float outerSpotAngle = 30.0f;
			bool castShadows = true;
			float shadowBias = 0.02f;
	};

	struct MeshVertex {
		Vector3 position;
		Vector3 normal;
		float uv[2];
		float color[4]{1.0f, 1.0f, 1.0f, 1.0f};
	};

	struct MeshData {
		std::vector<MeshVertex> vertices;
		std::vector<std::uint32_t> indices;

		void Validate() const {
			if (vertices.size() > std::numeric_limits<std::uint32_t>::max() / sizeof(MeshVertex) || indices.size() > std::numeric_limits<std::uint32_t>::max() / sizeof(std::uint32_t)) throw std::length_error("Mesh exceeds GPU buffer size");
			for (const auto& vertex : vertices) if (!std::isfinite(vertex.position.x) || !std::isfinite(vertex.position.y) || !std::isfinite(vertex.position.z)) throw std::invalid_argument("Mesh position must be finite");
			for (auto index : indices) if (index >= vertices.size()) throw std::out_of_range("Mesh index exceeds vertex count");
		}
	};

	struct Material {
		Vector3 baseColor{1.0f, 1.0f, 1.0f};
		std::string baseTexturePath;
		std::string normalTexturePath;
		std::string roughnessTexturePath;
		std::string aoTexturePath;
		std::string displacementTexturePath;
		std::string displacementMaskTexturePath;
		std::string vertexModifierSource;
		float roughness = 1.0f;
		float metallic = 0.0f;
		float normalScale = 1.0f;
		float aoStrength = 1.0f;
		float displacementScale = 0.05f;
		float uvScale[2]{1.0f, 1.0f};
		float uvOffset[2]{};
		bool flipNormalY = false;
		bool transparent = false;
		bool unlit = false;
		float opacity = 1.0f;

		auto Key() const {
			return std::tie(baseTexturePath, normalTexturePath, roughnessTexturePath, aoTexturePath, displacementTexturePath, displacementMaskTexturePath, vertexModifierSource, baseColor.x, baseColor.y, baseColor.z, roughness, metallic, normalScale, aoStrength, displacementScale, uvScale[0], uvScale[1], uvOffset[0], uvOffset[1], flipNormalY, transparent, unlit, opacity);
		}

		bool operator<(const Material& other) const { return Key() < other.Key(); }

	};

	struct MeshInstance {
		Vector3 position;
		Vector3 rotation;
		Vector3 scale{1.0f, 1.0f, 1.0f};
		float customData = 0.0f;
	};

	class Mesh : public Component {
		public:
			MeshData data;
			std::vector<MeshInstance> instances;
			bool instanced = false;
			Material material;
			bool cullBackfaces = true;
			bool castShadows = true;
			Vector3 vertexBoundsPadding{};
			bool vertexAnimated = false;
			std::uint64_t revision = 0;
			std::uint64_t instanceRevision = 0;
			void MarkDirty() { ++revision; }
			void MarkInstancesDirty() { ++instanceRevision; }
			static MeshData Cube();
			static MeshData Plane(int subdivisions = 1);
	};

	struct MeshRevision {
		std::uint64_t revision = 0;
		const MeshVertex* vertices = nullptr;
		const std::uint32_t* indices = nullptr;
		std::size_t vertexCount = 0;
		std::size_t indexCount = 0;
		bool Matches(const Mesh& mesh) const { return revision == mesh.revision && vertices == mesh.data.vertices.data() && indices == mesh.data.indices.data() && vertexCount == mesh.data.vertices.size() && indexCount == mesh.data.indices.size(); }
		void Set(const Mesh& mesh) { revision = mesh.revision; vertices = mesh.data.vertices.data(); indices = mesh.data.indices.data(); vertexCount = mesh.data.vertices.size(); indexCount = mesh.data.indices.size(); }
	};

	class Object {
		public:
			std::string name;
			Object* parent = nullptr;
			Scene* scene = nullptr;
			Transform transform;
			bool renderStatic = true;
			bool spatiallyLoaded = true;
			bool worldPartitionedMesh = false;
			bool worldPartitionCell = false;
			bool initializedForApp = true;
			Object* worldPartitionSourceObject = nullptr;
			explicit Object(std::string name) : name(std::move(name)) {}
			virtual void Initialize() {}
			virtual void Update() {}
			virtual void FixedUpdate() {}
			virtual void GUI() {}
			virtual ~Object() = default;

			template <typename T>
			T& AddComponent() {
				components.push_back(std::make_unique<T>());
				componentCache.clear();
				components.back()->owner = this;
				return static_cast<T&>(*components.back());
			}

			template <typename T>
			T* GetComponent() {
				static const char type = 0;
				for (const auto& entry : componentCache) if (entry.first == &type) return static_cast<T*>(entry.second);
				for (auto& component : components) {
					if (auto* match = dynamic_cast<T*>(component.get())) {
						componentCache.emplace_back(&type, match);
						return match;
					}
				}
				componentCache.emplace_back(&type, nullptr);
				return nullptr;
			}

			template <typename T>
			T& GetProperty(const std::string& key) {
				return std::any_cast<T&>(properties.at(key));
			}

			template <typename T>
			void SetProperty(const std::string& key, const T& value) {
				properties[key] = value;
			}
		
		private:
			std::vector<std::unique_ptr<Component>> components;
			std::vector<std::pair<const void*, Component*>> componentCache;
			std::unordered_map<std::string, std::any> properties;
	};

	class WorldPartitioner {
		using CellKey = std::tuple<int, int, int>;
		std::map<CellKey, std::vector<Object*>> cells;
		std::map<CellKey, Math::Bounds> cellBounds;
		std::vector<Object*> alwaysResidentObjects;
		std::vector<Object*> dynamicObjects;
		std::vector<Object*> residentObjects;
		std::vector<Object*> residentRenderObjects;
		std::vector<CellKey> residentCells;
		std::set<CellKey> residentCellKeys;
		std::deque<CellKey> pendingCellLoads;
		std::deque<CellKey> pendingCellUnloads;
		std::set<Object*> residentDynamicObjects;
		std::vector<std::unique_ptr<Object>> cellsData;
		struct SourceSnapshot {
			Object* object;
			Transform transform;
			Mesh* mesh;
			MeshRevision revision;
			Material material;
			bool renderStatic;
			bool spatiallyLoaded;
			bool instanced;
			bool cullBackfaces;
			bool castShadows;
			Vector3 padding;
			bool animated;
		};
		std::vector<SourceSnapshot> sources;
		int partitionSize = 0;
		float loadingRange = 0.0f;
		CellKey sourceCell{};
		bool initialized = false;
		bool dirty = true;
		bool hasCellSelection = false;
		bool initialCellLoadComplete = false;
		bool streamLoadNext = true;
		std::chrono::steady_clock::time_point nextCellStreamTime{};

		static CellKey Key(const Vector3& position, int size) {
			return {static_cast<int>(std::floor(position.x / size)), static_cast<int>(std::floor(position.y / size)), static_cast<int>(std::floor(position.z / size))};
		}

		static void IncludeBounds(std::map<CellKey, Math::Bounds>& boundsByCell, const CellKey& key, const Vector3& point) {
			auto [entry, inserted] = boundsByCell.try_emplace(key, Math::Bounds{point, point});
			if (inserted) return;
			Math::Include(entry->second, point);
		}

		void Rebuild(const std::vector<std::unique_ptr<Object>>& objects, int size) {
			const auto previousResidentCells = residentCells;
			const bool hadSelection = hasCellSelection;
			cells.clear();
			cellBounds.clear();
			alwaysResidentObjects.clear();
			dynamicObjects.clear();
			residentObjects.clear();
			residentRenderObjects.clear();
			residentCells.clear();
			residentCellKeys.clear();
			pendingCellLoads.clear();
			pendingCellUnloads.clear();
			residentDynamicObjects.clear();
			cellsData.clear();
			sources.clear();
			partitionSize = size;
			for (const auto& object : objects) {
				if (!object->initializedForApp) continue;
				object->worldPartitionedMesh = false;
				Mesh* mesh = object->GetComponent<Mesh>();
				SourceSnapshot snapshot{object.get(), object->transform, mesh, {}, mesh ? mesh->material : Material{}, object->renderStatic, object->spatiallyLoaded, mesh && mesh->instanced, mesh && mesh->cullBackfaces, mesh && mesh->castShadows, mesh ? mesh->vertexBoundsPadding : Vector3{}, mesh && mesh->vertexAnimated};
				if (mesh) snapshot.revision.Set(*mesh);
				sources.push_back(std::move(snapshot));
				if (!object->spatiallyLoaded) {
					alwaysResidentObjects.push_back(object.get());
					continue;
				}
				if (!object->renderStatic) {
					dynamicObjects.push_back(object.get());
					continue;
				}
				if (!mesh || mesh->instanced || mesh->data.indices.size() < 3 || mesh->data.vertices.empty()) {
					const CellKey key = Key(object->transform.position, size);
					cells[key].push_back(object.get());
					IncludeBounds(cellBounds, key, object->transform.position);
					continue;
				}
				using VertexMap = std::unordered_map<std::uint32_t, std::uint32_t>;
				std::map<CellKey, MeshData> cellMeshes;
				std::map<CellKey, VertexMap> cellVertices;
				std::map<CellKey, Math::Bounds> cellMeshBounds;
				const Math::Matrix4 model = Math::Model(object->transform);
				for (std::size_t index = 0; index + 2 < mesh->data.indices.size(); index += 3) {
					const std::uint32_t sourceIndices[3] = {mesh->data.indices[index], mesh->data.indices[index + 1], mesh->data.indices[index + 2]};
					if (sourceIndices[0] >= mesh->data.vertices.size() || sourceIndices[1] >= mesh->data.vertices.size() || sourceIndices[2] >= mesh->data.vertices.size()) continue;
					const Vector3 a = Math::TransformPoint(model, mesh->data.vertices[sourceIndices[0]].position);
					const Vector3 b = Math::TransformPoint(model, mesh->data.vertices[sourceIndices[1]].position);
					const Vector3 c = Math::TransformPoint(model, mesh->data.vertices[sourceIndices[2]].position);
					const CellKey key = Key(Math::Scale(Math::Add(Math::Add(a, b), c), 1.0f / 3.0f), size);
					IncludeBounds(cellMeshBounds, key, a);
					IncludeBounds(cellMeshBounds, key, b);
					IncludeBounds(cellMeshBounds, key, c);
					MeshData& cellMesh = cellMeshes[key];
					VertexMap& vertices = cellVertices[key];
					for (const auto sourceIndex : sourceIndices) {
						auto [entry, inserted] = vertices.try_emplace(sourceIndex, static_cast<std::uint32_t>(cellMesh.vertices.size()));
						if (inserted) cellMesh.vertices.push_back(mesh->data.vertices[sourceIndex]);
						cellMesh.indices.push_back(entry->second);
					}
				}
				if (cellMeshes.empty()) {
					const CellKey key = Key(object->transform.position, size);
					cells[key].push_back(object.get());
					IncludeBounds(cellBounds, key, object->transform.position);
					continue;
				}
				object->worldPartitionedMesh = true;
				for (auto& [key, data] : cellMeshes) {
					auto cellObject = std::make_unique<Object>(object->name);
					cellObject->scene = object->scene;
					cellObject->transform = object->transform;
					cellObject->renderStatic = true;
					cellObject->worldPartitionCell = true;
					cellObject->worldPartitionSourceObject = object.get();
					Mesh& cellMesh = cellObject->AddComponent<Mesh>();
					cellMesh.data = std::move(data);
					cellMesh.material = mesh->material;
					cellMesh.cullBackfaces = mesh->cullBackfaces;
					cellMesh.castShadows = mesh->castShadows;
					cellMesh.vertexBoundsPadding = mesh->vertexBoundsPadding;
					cellMesh.vertexAnimated = mesh->vertexAnimated;
					cells[key].push_back(object.get());
					cells[key].push_back(cellObject.get());
					const Math::Bounds cellMeshBoundsValue = cellMeshBounds.at(key);
					IncludeBounds(cellBounds, key, cellMeshBoundsValue.minimum);
					IncludeBounds(cellBounds, key, cellMeshBoundsValue.maximum);
					cellsData.push_back(std::move(cellObject));
				}
			}
			for (const auto& key : previousResidentCells) {
				if (cells.count(key) && residentCellKeys.insert(key).second) residentCells.push_back(key);
			}
			hasCellSelection = false;
			if (!hadSelection) initialCellLoadComplete = false;
			initialized = true;
			dirty = false;
		}

	public:
		void MarkDirty() { dirty = true; }

		void Update(const std::vector<std::unique_ptr<Object>>& objects, const Camera* camera, int size, float range) {
			if (!camera || !camera->owner || size <= 0) {
				cells.clear();
				cellBounds.clear();
				alwaysResidentObjects.clear();
				dynamicObjects.clear();
				residentObjects.clear();
				residentRenderObjects.clear();
				residentCells.clear();
				residentCellKeys.clear();
				pendingCellLoads.clear();
				pendingCellUnloads.clear();
				residentDynamicObjects.clear();
				cellsData.clear();
				hasCellSelection = false;
				initialCellLoadComplete = false;
				nextCellStreamTime = {};
				for (const auto& object : objects) object->worldPartitionedMesh = false;
				for (const auto& object : objects) {
					if (!object->initializedForApp) continue;
					residentObjects.push_back(object.get());
					residentRenderObjects.push_back(object.get());
				}
				partitionSize = size;
				initialized = false;
				dirty = true;
				return;
			}
			Vector3 sourcePosition = camera->owner->transform.position;
			Vector3 axes[3];
			Math::GetAxes(camera->owner->transform, axes);
			sourcePosition += Math::Scale(axes[0], camera->localPosition.x);
			sourcePosition += Math::Scale(axes[1], camera->localPosition.y);
			sourcePosition += Math::Scale(axes[2], camera->localPosition.z);
			const CellKey nextSource = Key(sourcePosition, size);
			std::size_t sourceIndex = 0;
			bool materialChanged = false;
			for (const auto& object : objects) {
				if (!object->initializedForApp) continue;
				if (sourceIndex >= sources.size() || sources[sourceIndex].object != object.get()) { dirty = true; break; }
				auto& source = sources[sourceIndex++];
				Mesh* mesh = object->GetComponent<Mesh>();
				const auto& a = source.transform;
				const auto& b = object->transform;
				const bool transformChanged = source.renderStatic && source.spatiallyLoaded && !object->GetComponent<Camera>() && std::tie(a.position.x, a.position.y, a.position.z, a.rotation.x, a.rotation.y, a.rotation.z, a.scale.x, a.scale.y, a.scale.z) != std::tie(b.position.x, b.position.y, b.position.z, b.rotation.x, b.rotation.y, b.rotation.z, b.scale.x, b.scale.y, b.scale.z);
				if (source.renderStatic != object->renderStatic || source.spatiallyLoaded != object->spatiallyLoaded || source.mesh != mesh || transformChanged || (mesh && (!source.revision.Matches(*mesh) || source.instanced != mesh->instanced))) { dirty = true; break; }
				if (mesh && (source.material.Key() != mesh->material.Key() || source.cullBackfaces != mesh->cullBackfaces || source.castShadows != mesh->castShadows || source.animated != mesh->vertexAnimated || std::tie(source.padding.x, source.padding.y, source.padding.z) != std::tie(mesh->vertexBoundsPadding.x, mesh->vertexBoundsPadding.y, mesh->vertexBoundsPadding.z))) materialChanged = true;
			}
			if (sourceIndex != sources.size()) dirty = true;
			if (materialChanged && !dirty) {
				for (auto& cell : cellsData) {
					Mesh* sourceMesh = cell->worldPartitionSourceObject->GetComponent<Mesh>();
					Mesh* cellMesh = cell->GetComponent<Mesh>();
					cellMesh->material = sourceMesh->material;
					cellMesh->cullBackfaces = sourceMesh->cullBackfaces;
					cellMesh->castShadows = sourceMesh->castShadows;
					cellMesh->vertexBoundsPadding = sourceMesh->vertexBoundsPadding;
					cellMesh->vertexAnimated = sourceMesh->vertexAnimated;
				}
				for (auto& source : sources) if (source.mesh) { source.material = source.mesh->material; source.cullBackfaces = source.mesh->cullBackfaces; source.castShadows = source.mesh->castShadows; source.padding = source.mesh->vertexBoundsPadding; source.animated = source.mesh->vertexAnimated; }
			}
			if (!initialized || dirty || partitionSize != size) Rebuild(objects, size);
			const float loadRange = std::max(range, 0.0f);
			const float unloadRange = loadRange + size;
			const float loadRangeSquared = loadRange * loadRange;
			const float unloadRangeSquared = unloadRange * unloadRange;
			const bool refreshCells = !hasCellSelection || sourceCell != nextSource || loadingRange != range;
			if (refreshCells) {
				std::vector<std::pair<float, CellKey>> ordered;
				for (const auto& [key, members] : cells) {
					const auto& bounds = cellBounds.at(key);
					const Vector3 closest{std::clamp(sourcePosition.x, bounds.minimum.x, bounds.maximum.x), std::clamp(sourcePosition.y, bounds.minimum.y, bounds.maximum.y), std::clamp(sourcePosition.z, bounds.minimum.z, bounds.maximum.z)};
					const Vector3 difference{closest.x - sourcePosition.x, closest.y - sourcePosition.y, closest.z - sourcePosition.z};
					const float distanceSquared = Math::Dot(difference, difference);
					if (key == nextSource || distanceSquared <= loadRangeSquared || (residentCellKeys.count(key) && distanceSquared <= unloadRangeSquared)) ordered.emplace_back(distanceSquared, key);
				}
				std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) {
					if (left.first != right.first) return left.first < right.first;
					return left.second < right.second;
				});
				pendingCellLoads.clear();
				pendingCellUnloads.clear();
				std::set<CellKey> desiredCells;
				for (const auto& [distance, key] : ordered) {
					desiredCells.insert(key);
					if ((key == nextSource || distance <= loadRangeSquared) && !residentCellKeys.count(key)) pendingCellLoads.push_back(key);
				}
				for (const auto& key : residentCellKeys) if (!desiredCells.count(key)) pendingCellUnloads.push_back(key);
				std::sort(pendingCellUnloads.begin(), pendingCellUnloads.end(), [&](const CellKey& left, const CellKey& right) {
					const auto& leftBounds = cellBounds.at(left);
					const auto& rightBounds = cellBounds.at(right);
					const Vector3 leftPoint{std::clamp(sourcePosition.x, leftBounds.minimum.x, leftBounds.maximum.x), std::clamp(sourcePosition.y, leftBounds.minimum.y, leftBounds.maximum.y), std::clamp(sourcePosition.z, leftBounds.minimum.z, leftBounds.maximum.z)};
					const Vector3 rightPoint{std::clamp(sourcePosition.x, rightBounds.minimum.x, rightBounds.maximum.x), std::clamp(sourcePosition.y, rightBounds.minimum.y, rightBounds.maximum.y), std::clamp(sourcePosition.z, rightBounds.minimum.z, rightBounds.maximum.z)};
					const Vector3 leftDelta{leftPoint.x - sourcePosition.x, leftPoint.y - sourcePosition.y, leftPoint.z - sourcePosition.z};
					const Vector3 rightDelta{rightPoint.x - sourcePosition.x, rightPoint.y - sourcePosition.y, rightPoint.z - sourcePosition.z};
					return Math::Dot(leftDelta, leftDelta) > Math::Dot(rightDelta, rightDelta);
				});
				if (!initialCellLoadComplete) {
					if (cells.count(nextSource) && residentCellKeys.insert(nextSource).second) residentCells.push_back(nextSource);
					pendingCellLoads.erase(std::remove(pendingCellLoads.begin(), pendingCellLoads.end(), nextSource), pendingCellLoads.end());
					initialCellLoadComplete = true;
					nextCellStreamTime = std::chrono::steady_clock::now() + std::chrono::milliseconds(8);
				}
				sourceCell = nextSource;
				loadingRange = range;
				hasCellSelection = true;
			}
			const auto now = std::chrono::steady_clock::now();
			if (now >= nextCellStreamTime) {
				const bool canLoad = !pendingCellLoads.empty();
				const bool canUnload = !pendingCellUnloads.empty();
				if (canLoad && (!canUnload || streamLoadNext)) {
					const CellKey key = pendingCellLoads.front();
					pendingCellLoads.pop_front();
					if (residentCellKeys.insert(key).second) {
						residentCells.push_back(key);
					}
					streamLoadNext = false;
					nextCellStreamTime = now + std::chrono::milliseconds(8);
				}
				else if (canUnload) {
					const CellKey key = pendingCellUnloads.front();
					pendingCellUnloads.pop_front();
					residentCellKeys.erase(key);
					residentCells.erase(std::remove(residentCells.begin(), residentCells.end(), key), residentCells.end());
					streamLoadNext = true;
					nextCellStreamTime = now + std::chrono::milliseconds(8);
				}
			}
			residentObjects.clear();
			residentRenderObjects.clear();
			std::unordered_set<Object*> residentSet;
			auto addResident = [&](Object* object) {
				if (!residentSet.insert(object).second) return;
				residentObjects.push_back(object);
				if (!object->worldPartitionedMesh || object->worldPartitionCell) residentRenderObjects.push_back(object);
			};
			for (const auto& key : residentCells) {
				for (auto* object : cells.at(key)) addResident(object);
			}
			std::set<Object*> nextDynamicObjects;
			for (auto* object : dynamicObjects) {
				const Vector3 difference{object->transform.position.x - sourcePosition.x, object->transform.position.y - sourcePosition.y, object->transform.position.z - sourcePosition.z};
				const float distanceSquared = Math::Dot(difference, difference);
				if (distanceSquared <= loadRangeSquared || (residentDynamicObjects.count(object) && distanceSquared <= unloadRangeSquared)) {
					nextDynamicObjects.insert(object);
					addResident(object);
				}
			}
			residentDynamicObjects.swap(nextDynamicObjects);
			if (camera->owner->parent) addResident(camera->owner->parent);
			addResident(camera->owner);
			for (auto* object : alwaysResidentObjects) addResident(object);
		}

		const std::vector<Object*>& Objects() const { return residentObjects; }
		const std::vector<Object*>& RenderObjects() const { return residentRenderObjects; }
	};

	class Scene {
		public:
			Camera* activeCamera = nullptr;
			Vector3 gravity{0.0f, -9.81f, 0.0f};
			int cellSize = 20;
			int partitionSize = 200;
			float partitionLoadingRange = -1.0f;
			bool shadowsEnabled = true;
			float shadowFar = 10.0f;
			int shadowResolution = 2048;
			bool ambientLight = true;
			Vector3 ambientLightColor{1.0f, 1.0f, 1.0f};
			bool fogEnabled = false;
			Vector3 fogColor;
			Vector3 backgroundColor;

			template <typename T = Object>
			T& CreateObject(std::string name) {
				objects.push_back(std::make_unique<T>(std::move(name)));
				objects.back()->scene = this;
				if (appActive) objects.back()->initializedForApp = false;
				worldPartitioner.MarkDirty();
				return static_cast<T&>(*objects.back());
			}

			template <typename T>
			void AddObject(std::unique_ptr<T> object) {
				objects.push_back(std::move(object));
				objects.back()->scene = this;
				if (appActive) objects.back()->initializedForApp = false;
				worldPartitioner.MarkDirty();
			}

			Object& Find(const std::string& name) {
				for (auto& object : objects) {
					if (object->name == name) return *object;
				}
				throw std::out_of_range("Object not found");
			}

			void SelectFirstCamera() {
				activeCamera = nullptr;
				for (auto& object : objects) {
					if (auto* camera = object->GetComponent<Camera>()) {
						activeCamera = camera;
						break;
					}
				}
				UpdateWorldPartition();
			}

			void UpdateWorldPartition() {
				const float range = partitionLoadingRange >= 0.0f ? partitionLoadingRange : activeCamera ? activeCamera->farClip : 0.0f;
				worldPartitioner.Update(objects, activeCamera, partitionSize, range);
			}

			void SetAppActive(bool active) {
				appActive = active;
				if (!active) for (auto& object : objects) object->initializedForApp = true;
				worldPartitioner.MarkDirty();
			}

			const std::vector<Object*>& ResidentObjects() const { return worldPartitioner.Objects(); }
			const std::vector<Object*>& ResidentRenderObjects() const { return worldPartitioner.RenderObjects(); }

			std::vector<std::unique_ptr<Object>>& Objects() {
				return objects;
			}

		private:
			std::vector<std::unique_ptr<Object>> objects;
			bool appActive = false;
			WorldPartitioner worldPartitioner;
	};

}
