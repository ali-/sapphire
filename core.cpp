#include <any>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <string>
#include <sstream>
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
			Type type = Type::Directional;
			Vector3 color{1.0f, 1.0f, 1.0f};
			float intensity = 1.0f;
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

	class Mesh : public Component {
		public:
			MeshData data;
			std::string texturePath;
			std::string displacementTexturePath;
			std::string maskTexturePath;
			float displacementScale = 0.05f;
			bool cullBackfaces = true;
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
