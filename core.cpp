#include "loading.cpp"
#include <algorithm>
#include <any>
#include <limits>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <string>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <vector>

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
		Vector3 localPosition;
		Vector3 localRotation;

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

		Matrix4 Model(const Transform& transform) {
			Vector3 axes[3];
			GetAxes(transform, axes);
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
	};

	struct Material {
		Vector3 baseColor{1.0f, 1.0f, 1.0f};
		std::string baseTexturePath;
		std::string normalTexturePath;
		std::string roughnessTexturePath;
		std::string aoTexturePath;
		std::string displacementTexturePath;
		std::string displacementMaskTexturePath;
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
			return std::tie(baseTexturePath, normalTexturePath, roughnessTexturePath, aoTexturePath, displacementTexturePath, displacementMaskTexturePath, baseColor.x, baseColor.y, baseColor.z, roughness, metallic, normalScale, aoStrength, displacementScale, uvScale[0], uvScale[1], uvOffset[0], uvOffset[1], flipNormalY, transparent, unlit, opacity);
		}

		bool operator<(const Material& other) const { return Key() < other.Key(); }

	};

	class Mesh : public Component {
		public:
			MeshData data;
			Material material;
			bool cullBackfaces = true;
			bool castShadows = true;
			bool windAnimated = false;
			std::uint64_t revision = 0;
			void MarkDirty() { ++revision; }
			static MeshData Cube();
			static MeshData Plane(int subdivisions = 1);
	};

	class Object {
		public:
			std::string name;
			Object* parent = nullptr;
			Scene* scene = nullptr;
			Transform transform;
			bool renderStatic = true;
			explicit Object(std::string name) : name(name) {}
			virtual void Initialize() {}
			virtual void Update() {}
			virtual void FixedUpdate() {}
			virtual void GUI() {}
			virtual ~Object() = default;

			template <typename T>
			T& AddComponent() {
				components.push_back(std::make_unique<T>());
				components.back()->owner = this;
				return static_cast<T&>(*components.back());
			}

			template <typename T>
			T* GetComponent() {
				for (auto& component : components) {
					if (auto* match = dynamic_cast<T*>(component.get())) {
						return match;
					}
				}
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
			std::unordered_map<std::string, std::any> properties;
	};

	class Scene {
		public:
			Camera* activeCamera = nullptr;
			Vector3 gravity{0.0f, -9.81f, 0.0f};
			int chunkSize = 20;
			bool shadowsEnabled = true;
			float shadowFar = 10.0f;
			int shadowResolution = 2048;
			bool ambientLight = true;
			Vector3 ambientLightColor{1.0f, 1.0f, 1.0f};
			bool fogEnabled = false;
			Vector3 fogColor;
			float fogNear = 0.0f;
			float fogFar = 1000.0f;

			template <typename T = Object>
			T& CreateObject(std::string name) {
				objects.push_back(std::make_unique<T>(name));
				objects.back()->scene = this;
				return static_cast<T&>(*objects.back());
			}

			template <typename T>
			void AddObject(const T& object) {
				objects.push_back(std::make_unique<T>(object));
				objects.back()->scene = this;
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
						return;
					}
				}
			}

			std::vector<std::unique_ptr<Object>>& Objects() {
				return objects;
			}

		private:
			std::vector<std::unique_ptr<Object>> objects;
	};

}
