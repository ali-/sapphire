#include <algorithm>
#include <any>
#include <limits>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <map>
#include <string>
#include <sstream>
#include <set>
#include <tuple>
#include <type_traits>
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
		Vector3 localPosition;
		Vector3 localRotation;
		Vector3 localScale{1.0f, 1.0f, 1.0f};

		Vector3 Forward() const {
			const float pitch = rotation.x * Math::degreesToRadians;
			const float roll = rotation.y * Math::degreesToRadians;
			const float yaw = rotation.z * Math::degreesToRadians;
			const float cosinePitch = std::cos(pitch), sinePitch = std::sin(pitch), cosineRoll = std::cos(roll), sineRoll = std::sin(roll), cosineYaw = std::cos(yaw), sineYaw = std::sin(yaw);
			return {sineYaw * cosinePitch - cosineYaw * sineRoll * sinePitch, -sineYaw * sineRoll * sinePitch - cosineYaw * cosinePitch, -cosineRoll * sinePitch};
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

		Vector3 ComposeRotation(const Vector3& parent, const Vector3& local) {
			Vector3 parentAxes[3];
			Vector3 localAxes[3];
			Transform parentTransform;
			Transform localTransform;
			parentTransform.rotation = parent;
			localTransform.rotation = local;
			GetAxes(parentTransform, parentAxes);
			GetAxes(localTransform, localAxes);
			Vector3 axes[3];
			for (int index = 0; index < 3; ++index) axes[index] = Add(Scale(parentAxes[0], localAxes[index].x), Add(Scale(parentAxes[1], localAxes[index].y), Scale(parentAxes[2], localAxes[index].z)));
			const float roll = std::asin(std::clamp(-axes[0].z, -1.0f, 1.0f));
			const float cosineRoll = std::cos(roll);
			const float pitch = std::abs(cosineRoll) > 0.0001f ? std::atan2(axes[1].z, axes[2].z) : std::atan2(-axes[2].y, axes[1].y);
			const float yaw = std::abs(cosineRoll) > 0.0001f ? std::atan2(axes[0].y, axes[0].x) : 0.0f;
			return {pitch / degreesToRadians, roll / degreesToRadians, yaw / degreesToRadians};
		}

		Vector3 RelativeRotation(const Vector3& parent, const Vector3& world) {
			Vector3 parentAxes[3];
			Vector3 worldAxes[3];
			Transform parentTransform;
			Transform worldTransform;
			parentTransform.rotation = parent;
			worldTransform.rotation = world;
			GetAxes(parentTransform, parentAxes);
			GetAxes(worldTransform, worldAxes);
			Vector3 localAxes[3];
			for (int index = 0; index < 3; ++index) localAxes[index] = {Dot(parentAxes[0], worldAxes[index]), Dot(parentAxes[1], worldAxes[index]), Dot(parentAxes[2], worldAxes[index])};
			const float roll = std::asin(std::clamp(-localAxes[0].z, -1.0f, 1.0f));
			const float cosineRoll = std::cos(roll);
			const float pitch = std::abs(cosineRoll) > 0.0001f ? std::atan2(localAxes[1].z, localAxes[2].z) : std::atan2(-localAxes[2].y, localAxes[1].y);
			const float yaw = std::abs(cosineRoll) > 0.0001f ? std::atan2(localAxes[0].y, localAxes[0].x) : 0.0f;
			return {pitch / degreesToRadians, roll / degreesToRadians, yaw / degreesToRadians};
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
			float fieldOfView = 60.0f;
			float near = 0.1f;
			float far = 1000.0f;
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

	struct Bone {
		std::string name;
		std::int32_t parentIndex = -1;
		Transform bindTransform;
		Transform poseTransform;
	};

	class Skeleton : public Component {
		public:
			std::vector<Bone> bones;
			std::uint64_t poseRevision = 1;

			void Validate() const {
				if (bones.size() > static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max()) + 1) throw std::length_error("Skeleton bone count exceeds supported range");
				for (std::size_t index = 0; index < bones.size(); ++index) {
					const Bone& bone = bones[index];
					if (bone.parentIndex < -1 || bone.parentIndex >= static_cast<std::int32_t>(bones.size()) || bone.parentIndex == static_cast<std::int32_t>(index)) throw std::out_of_range("Skeleton bone parent index is invalid");
					auto validateTransform = [](const Transform& transform) {
						if (!std::isfinite(transform.position.x) || !std::isfinite(transform.position.y) || !std::isfinite(transform.position.z) || !std::isfinite(transform.rotation.x) || !std::isfinite(transform.rotation.y) || !std::isfinite(transform.rotation.z) || !std::isfinite(transform.scale.x) || !std::isfinite(transform.scale.y) || !std::isfinite(transform.scale.z)) throw std::invalid_argument("Skeleton transform must be finite");
					};
					validateTransform(bone.bindTransform);
					validateTransform(bone.poseTransform);
				}
				std::vector<std::uint8_t> states(bones.size());
				for (std::size_t index = 0; index < bones.size(); ++index) {
					std::size_t current = index;
					while (states[current] == 0) {
						states[current] = 1;
						const std::int32_t parentIndex = bones[current].parentIndex;
						if (parentIndex < 0 || states[parentIndex] == 2) break;
						if (states[parentIndex] == 1) throw std::invalid_argument("Skeleton hierarchy contains a cycle");
						current = static_cast<std::size_t>(parentIndex);
					}
					current = index;
					while (states[current] == 1) {
						states[current] = 2;
						const std::int32_t parentIndex = bones[current].parentIndex;
						if (parentIndex < 0) break;
						current = static_cast<std::size_t>(parentIndex);
					}
				}
			}

			void InitializePose() {
				for (auto& bone : bones) bone.poseTransform = bone.bindTransform;
				bindMatrices.clear();
				inverseBindMatrices.clear();
				poseGlobalMatrices.clear();
				poseMatrices.clear();
				++poseRevision;
			}

			void MarkPoseDirty() { ++poseRevision; }
			void SetPoseTransform(std::size_t boneIndex, const Transform& transform) { bones.at(boneIndex).poseTransform = transform; MarkPoseDirty(); }

			const std::vector<Math::Matrix4>& PoseMatrices() {
				if (cachedPoseRevision == poseRevision) return poseMatrices;
				Validate();
				auto buildMatrices = [&](bool bindPose) {
					std::vector<Math::Matrix4> matrices(bones.size());
					std::vector<std::uint8_t> complete(bones.size());
					std::vector<std::size_t> path;
					for (std::size_t index = 0; index < bones.size(); ++index) {
						if (complete[index]) continue;
						path.clear();
						std::size_t current = index;
						while (!complete[current]) {
							path.push_back(current);
							const std::int32_t parentIndex = bones[current].parentIndex;
							if (parentIndex < 0) break;
							current = static_cast<std::size_t>(parentIndex);
						}
						for (auto boneIndex = path.rbegin(); boneIndex != path.rend(); ++boneIndex) {
							const Bone& bone = bones[*boneIndex];
							const Transform& transform = bindPose ? bone.bindTransform : bone.poseTransform;
							const Math::Matrix4 local = Math::Model(transform);
							matrices[*boneIndex] = bone.parentIndex < 0 ? local : Math::Multiply(matrices[bone.parentIndex], local);
							complete[*boneIndex] = 1;
						}
					}
					return matrices;
				};
				if (bindMatrices.size() != bones.size()) bindMatrices = buildMatrices(true);
				if (inverseBindMatrices.size() != bones.size()) {
					inverseBindMatrices.resize(bones.size());
					for (std::size_t index = 0; index < bones.size(); ++index) inverseBindMatrices[index] = InverseAffine(bindMatrices[index]);
				}
				const auto poseGlobals = buildMatrices(false);
				poseGlobalMatrices = poseGlobals;
				poseMatrices.resize(bones.size());
				for (std::size_t index = 0; index < bones.size(); ++index) poseMatrices[index] = Math::Multiply(poseGlobals[index], inverseBindMatrices[index]);
				cachedPoseRevision = poseRevision;
				return poseMatrices;
			}

			const std::vector<Math::Matrix4>& InverseBindMatrices() { PoseMatrices(); return inverseBindMatrices; }
			const std::vector<Math::Matrix4>& PoseGlobalMatrices() { PoseMatrices(); return poseGlobalMatrices; }

		private:
			std::vector<Math::Matrix4> bindMatrices;
			std::vector<Math::Matrix4> inverseBindMatrices;
			std::vector<Math::Matrix4> poseGlobalMatrices;
			std::vector<Math::Matrix4> poseMatrices;
			std::uint64_t cachedPoseRevision = 0;

			static Math::Matrix4 InverseAffine(const Math::Matrix4& matrix) {
				const float a00 = matrix.values[0], a01 = matrix.values[4], a02 = matrix.values[8];
				const float a10 = matrix.values[1], a11 = matrix.values[5], a12 = matrix.values[9];
				const float a20 = matrix.values[2], a21 = matrix.values[6], a22 = matrix.values[10];
				const float determinant = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);
				if (!std::isfinite(determinant) || std::abs(determinant) < 0.000001f) throw std::invalid_argument("Skeleton bind transform is not invertible");
				const float inverseDeterminant = 1.0f / determinant;
				Math::Matrix4 inverse = Math::Identity();
				inverse.values[0] = (a11 * a22 - a12 * a21) * inverseDeterminant;
				inverse.values[4] = (a02 * a21 - a01 * a22) * inverseDeterminant;
				inverse.values[8] = (a01 * a12 - a02 * a11) * inverseDeterminant;
				inverse.values[1] = (a12 * a20 - a10 * a22) * inverseDeterminant;
				inverse.values[5] = (a00 * a22 - a02 * a20) * inverseDeterminant;
				inverse.values[9] = (a02 * a10 - a00 * a12) * inverseDeterminant;
				inverse.values[2] = (a10 * a21 - a11 * a20) * inverseDeterminant;
				inverse.values[6] = (a01 * a20 - a00 * a21) * inverseDeterminant;
				inverse.values[10] = (a00 * a11 - a01 * a10) * inverseDeterminant;
				const Vector3 translation{matrix.values[12], matrix.values[13], matrix.values[14]};
				inverse.values[12] = -(inverse.values[0] * translation.x + inverse.values[4] * translation.y + inverse.values[8] * translation.z);
				inverse.values[13] = -(inverse.values[1] * translation.x + inverse.values[5] * translation.y + inverse.values[9] * translation.z);
				inverse.values[14] = -(inverse.values[2] * translation.x + inverse.values[6] * translation.y + inverse.values[10] * translation.z);
				return inverse;
			}
	};

	struct SkinInfluence {
		std::uint16_t boneIndices[4]{};
		std::uint16_t weights[4]{};
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
		std::vector<SkinInfluence> skinWeights;

		void Validate() const {
			if (vertices.size() > std::numeric_limits<std::uint32_t>::max() / sizeof(MeshVertex) || indices.size() > std::numeric_limits<std::uint32_t>::max() / sizeof(std::uint32_t) || skinWeights.size() > std::numeric_limits<std::uint32_t>::max() / sizeof(SkinInfluence)) throw std::length_error("Mesh exceeds GPU buffer size");
			if (!skinWeights.empty() && skinWeights.size() != vertices.size()) throw std::invalid_argument("Mesh skin weights must match vertex count");
			for (const auto& vertex : vertices) if (!std::isfinite(vertex.position.x) || !std::isfinite(vertex.position.y) || !std::isfinite(vertex.position.z)) throw std::invalid_argument("Mesh position must be finite");
			for (auto index : indices) if (index >= vertices.size()) throw std::out_of_range("Mesh index exceeds vertex count");
			for (const auto& skin : skinWeights) {
				const std::uint32_t weightSum = skin.weights[0] + skin.weights[1] + skin.weights[2] + skin.weights[3];
				if (weightSum != std::numeric_limits<std::uint16_t>::max()) throw std::invalid_argument("Mesh skin weights must be normalized");
			}
		}

		void ValidateSkeleton(const Skeleton& skeleton) const {
			if (skinWeights.empty()) return;
			skeleton.Validate();
			if (skeleton.bones.empty()) throw std::invalid_argument("Skinned meshes require a skeleton with bones");
			for (const auto& skin : skinWeights) for (const auto boneIndex : skin.boneIndices) if (boneIndex >= skeleton.bones.size()) throw std::out_of_range("Mesh skin weight references an invalid bone");
		}
	};

	struct Primitive {
		enum class Shape { Cube, Capsule, Plane };
		Shape shape;
		static const Primitive Cube;
		static const Primitive Capsule;
		static const Primitive Plane;
		MeshData Mesh(Vector3 size = {1.0f, 1.0f, 1.0f}) const;
		MeshData Mesh(int subdivisions) const;
		static float CapsuleRadius(Vector3 size) {
			return std::min({size.x, size.y, size.z}) * 0.5f;
		}
	};

	inline const Primitive Primitive::Cube{Primitive::Shape::Cube};
	inline const Primitive Primitive::Capsule{Primitive::Shape::Capsule};
	inline const Primitive Primitive::Plane{Primitive::Shape::Plane};

	inline MeshData Primitive::Mesh(int subdivisions) const {
		if (shape != Shape::Plane) throw std::invalid_argument("Only planes support subdivisions");
		MeshData mesh;
		subdivisions = std::max(subdivisions, 1);
		const std::uint64_t rowCount = static_cast<std::uint64_t>(subdivisions) + 1;
		const std::uint64_t vertexCount = rowCount * rowCount;
		const std::uint64_t quadCount = static_cast<std::uint64_t>(subdivisions) * subdivisions;
		if (vertexCount > std::numeric_limits<std::uint32_t>::max() / sizeof(MeshVertex) || quadCount > std::numeric_limits<std::uint32_t>::max() / sizeof(std::uint32_t) / 6) throw std::length_error("Plane exceeds GPU buffer size");
		const int rowSize = static_cast<int>(rowCount);
		mesh.vertices.reserve(static_cast<std::size_t>(vertexCount));
		mesh.indices.reserve(static_cast<std::size_t>(quadCount * 6));
		for (int y = 0; y <= subdivisions; ++y) {
			for (int x = 0; x <= subdivisions; ++x) {
				const float u = static_cast<float>(x) / subdivisions;
				const float v = static_cast<float>(y) / subdivisions;
				mesh.vertices.push_back({{u - 0.5f, v - 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {u, v}});
			}
		}
		for (int y = 0; y < subdivisions; ++y) {
			for (int x = 0; x < subdivisions; ++x) {
				const std::uint32_t first = y * rowSize + x;
				const std::uint32_t nextRow = first + rowSize;
				mesh.indices.insert(mesh.indices.end(), {first, nextRow + 1, nextRow, first, first + 1, nextRow + 1});
			}
		}
		return mesh;
	}

	inline MeshData Primitive::Mesh(Vector3 size) const {
		if (shape == Shape::Plane) {
			MeshData mesh = Mesh(1);
			for (auto& vertex : mesh.vertices) {
				vertex.position.x *= size.x;
				vertex.position.y *= size.y;
			}
			return mesh;
		}
		if (shape == Shape::Cube) {
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
			for (int face = 0; face < 6; ++face) {
				const std::uint32_t first = static_cast<std::uint32_t>(mesh.vertices.size());
				for (int corner = 0; corner < 4; ++corner) mesh.vertices.push_back({{faces[face][corner].x * size.x, faces[face][corner].y * size.y, faces[face][corner].z * size.z}, normals[face], {corner == 1 || corner == 2 ? 1.0f : 0.0f, corner >= 2 ? 1.0f : 0.0f}});
				mesh.indices.insert(mesh.indices.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
			}
			return mesh;
		}
		MeshData mesh;
		const float radius = CapsuleRadius(size);
		const float halfStraight = std::max(0.0f, size.z * 0.5f - radius);
		constexpr int sides = 16;
		constexpr int hemisphereRings = 8;
		constexpr float pi = 3.14159265358979323846f;
		auto addRing = [&](float angle, float centerZ) {
			const float ringRadius = radius * std::cos(angle);
			const float z = centerZ + radius * std::sin(angle);
			for (int side = 0; side <= sides; ++side) {
				const float u = static_cast<float>(side) / sides;
				const float azimuth = u * 2.0f * pi;
				const float cosine = std::cos(azimuth);
				const float sine = std::sin(azimuth);
				mesh.vertices.push_back({{ringRadius * cosine, ringRadius * sine, z}, {std::cos(angle) * cosine, std::cos(angle) * sine, std::sin(angle)}, {u, size.z > 0.0f ? z / size.z + 0.5f : 0.5f}});
			}
		};
		for (int ring = 0; ring <= hemisphereRings; ++ring) addRing(-pi * 0.5f + static_cast<float>(ring) / hemisphereRings * pi * 0.5f, -halfStraight);
		for (int ring = halfStraight > 0.0f ? 0 : 1; ring <= hemisphereRings; ++ring) addRing(static_cast<float>(ring) / hemisphereRings * pi * 0.5f, halfStraight);
		const std::uint32_t ringCount = static_cast<std::uint32_t>(mesh.vertices.size() / (sides + 1));
		for (std::uint32_t ring = 0; ring + 1 < ringCount; ++ring) for (int side = 0; side < sides; ++side) {
			const std::uint32_t first = ring * (sides + 1) + side;
			const std::uint32_t next = first + sides + 1;
			mesh.indices.insert(mesh.indices.end(), {first, first + 1, next, first + 1, next + 1, next});
		}
		return mesh;
	}

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

	struct MeshSection {
		std::uint32_t firstIndex = 0;
		std::uint32_t indexCount = 0;
		std::uint32_t materialIndex = 0;
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
			std::vector<Material> materials;
			std::vector<MeshSection> sections;
			bool cullBackfaces = true;
			bool castShadows = true;
			Vector3 vertexBoundsPadding{};
			bool vertexAnimated = false;
			std::uint64_t revision = 0;
			std::uint64_t instanceRevision = 0;
			bool IsSkinned() const { return !data.skinWeights.empty(); }
			std::size_t MaterialCount() const { return materials.empty() ? 1 : materials.size(); }
			const Material& GetMaterial(std::size_t index) const { return materials.empty() ? material : materials.at(index); }
			std::size_t SectionCount() const { return sections.empty() ? (data.indices.empty() ? 0 : 1) : sections.size(); }
			MeshSection GetSection(std::size_t index) const { return sections.empty() ? MeshSection{0, static_cast<std::uint32_t>(data.indices.size()), 0} : sections.at(index); }
			std::uint32_t MaterialIndexAt(std::size_t index) const {
				if (sections.empty()) return 0;
				auto section = std::upper_bound(sections.begin(), sections.end(), index, [](std::size_t position, const MeshSection& value) { return position < value.firstIndex; });
				return section == sections.begin() ? 0 : (section - 1)->materialIndex;
			}
			void ValidateSections() const {
				if (materials.size() > std::numeric_limits<std::uint32_t>::max()) throw std::length_error("Mesh material count exceeds supported range");
				if (materials.size() > 1 && sections.empty()) throw std::invalid_argument("Meshes with multiple materials require sections");
				std::size_t nextIndex = 0;
				for (const auto& section : sections) {
					if (section.firstIndex != nextIndex || section.indexCount == 0 || section.indexCount % 3 != 0 || static_cast<std::size_t>(section.firstIndex) + section.indexCount > data.indices.size()) throw std::invalid_argument("Mesh sections must cover contiguous triangle index ranges");
					if (section.materialIndex >= MaterialCount()) throw std::out_of_range("Mesh section material index is invalid");
					nextIndex += section.indexCount;
				}
				if (!sections.empty() && nextIndex != data.indices.size()) throw std::invalid_argument("Mesh sections must cover all mesh indices");
			}
			void Validate() const {
				data.Validate();
				ValidateSections();
				if (IsSkinned() && instanced) throw std::invalid_argument("Skinned meshes cannot be instanced");
			}
			void ValidateSkeleton(const Skeleton& skeleton) const { data.ValidateSkeleton(skeleton); }
			void MarkDirty() { ++revision; }
			void MarkInstancesDirty() { ++instanceRevision; }
	};

	struct MeshRevision {
		std::uint64_t revision = 0;
		const MeshVertex* vertices = nullptr;
		const std::uint32_t* indices = nullptr;
			const SkinInfluence* skinWeights = nullptr;
		std::size_t vertexCount = 0;
		std::size_t indexCount = 0;
			std::size_t skinWeightCount = 0;
			bool Matches(const Mesh& mesh) const { return revision == mesh.revision && vertices == mesh.data.vertices.data() && indices == mesh.data.indices.data() && skinWeights == mesh.data.skinWeights.data() && vertexCount == mesh.data.vertices.size() && indexCount == mesh.data.indices.size() && skinWeightCount == mesh.data.skinWeights.size(); }
			void Set(const Mesh& mesh) { revision = mesh.revision; vertices = mesh.data.vertices.data(); indices = mesh.data.indices.data(); skinWeights = mesh.data.skinWeights.data(); vertexCount = mesh.data.vertices.size(); indexCount = mesh.data.indices.size(); skinWeightCount = mesh.data.skinWeights.size(); }
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
			std::uint64_t transformGeneration = 0;
			explicit Object(std::string name) : name(std::move(name)) {}
			virtual void Initialize() {}
			virtual void Update() {}
			virtual void FixedUpdate() {}
			virtual void GUI() {}
			virtual ~Object() = default;
			bool LoadS3D(const std::string& fileName);

			void UpdateWorldTransform() {
				if (parent) parent->UpdateWorldTransform();
				ApplyWorldTransform();
			}

			void UpdateWorldTransformSubtree() {
				UpdateWorldTransform();
				UpdateDescendantWorldTransforms();
			}

			bool HasChildren() const { return !children.empty(); }

			void ApplyWorldTransform() {
				if (!parent) {
					transform.localPosition = transform.position;
					transform.localRotation = transform.rotation;
					transform.localScale = transform.scale;
					return;
				}
				Vector3 axes[3];
				Math::GetAxes(parent->transform, axes);
				transform.position = Math::Add(parent->transform.position, Math::Add(Math::Scale(axes[0], transform.localPosition.x * parent->transform.scale.x), Math::Add(Math::Scale(axes[1], transform.localPosition.y * parent->transform.scale.y), Math::Scale(axes[2], transform.localPosition.z * parent->transform.scale.z))));
				transform.rotation = Math::ComposeRotation(parent->transform.rotation, transform.localRotation);
				transform.scale = {parent->transform.scale.x * transform.localScale.x, parent->transform.scale.y * transform.localScale.y, parent->transform.scale.z * transform.localScale.z};
			}

			void UpdateDescendantWorldTransforms() {
				for (Object* child : children) {
					child->ApplyWorldTransform();
					child->UpdateDescendantWorldTransforms();
				}
			}

			void UpdateWorldTransform(std::uint64_t generation) {
				if (transformGeneration == generation) return;
				if (parent) parent->UpdateWorldTransform(generation);
				ApplyWorldTransform();
				transformGeneration = generation;
			}

			void SetWorldPosition(const Vector3& position) {
				transform.position = position;
				if (!parent) {
					UpdateDescendantWorldTransforms();
					return;
				}
				parent->UpdateWorldTransform();
				Vector3 axes[3];
				Math::GetAxes(parent->transform, axes);
				const Vector3 offset = Math::Subtract(position, parent->transform.position);
				transform.localPosition = {std::abs(parent->transform.scale.x) > 0.0001f ? Math::Dot(offset, axes[0]) / parent->transform.scale.x : 0.0f, std::abs(parent->transform.scale.y) > 0.0001f ? Math::Dot(offset, axes[1]) / parent->transform.scale.y : 0.0f, std::abs(parent->transform.scale.z) > 0.0001f ? Math::Dot(offset, axes[2]) / parent->transform.scale.z : 0.0f};
				UpdateDescendantWorldTransforms();
			}

			void SetWorldRotation(const Vector3& rotation) {
				transform.rotation = rotation;
				if (parent) parent->UpdateWorldTransform();
				transform.localRotation = parent ? Math::RelativeRotation(parent->transform.rotation, rotation) : rotation;
				UpdateDescendantWorldTransforms();
			}

			void SetWorldScale(const Vector3& scale) {
				transform.scale = scale;
				if (parent) parent->UpdateWorldTransform();
				transform.localScale = parent ? Vector3{std::abs(parent->transform.scale.x) > 0.0001f ? scale.x / parent->transform.scale.x : 0.0f, std::abs(parent->transform.scale.y) > 0.0001f ? scale.y / parent->transform.scale.y : 0.0f, std::abs(parent->transform.scale.z) > 0.0001f ? scale.z / parent->transform.scale.z : 0.0f} : scale;
				UpdateDescendantWorldTransforms();
			}

			void SetParent(Object* nextParent, bool worldPositionStays = true) {
				if (parent == nextParent) return;
				for (Object* ancestor = nextParent; ancestor; ancestor = ancestor->parent) if (ancestor == this) throw std::invalid_argument("Object cannot be parented to itself or a descendant");
				const Vector3 worldPosition = transform.position;
				const Vector3 worldRotation = transform.rotation;
				const Vector3 worldScale = transform.scale;
				if (parent) {
					auto& siblings = parent->children;
					siblings.erase(std::remove(siblings.begin(), siblings.end(), this), siblings.end());
				}
				parent = nextParent;
				if (parent) parent->children.push_back(this);
				if (worldPositionStays) {
					SetWorldPosition(worldPosition);
					SetWorldRotation(worldRotation);
					SetWorldScale(worldScale);
				}
				else {
					if (!parent) {
						transform.position = transform.localPosition;
						transform.rotation = transform.localRotation;
						transform.scale = transform.localScale;
					}
					UpdateWorldTransform();
					UpdateDescendantWorldTransforms();
				}
			}

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
			std::vector<Object*> children;
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
			std::vector<Material> materials;
			std::vector<MeshSection> sections;
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
		Vector3 sourceForward{};
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
				SourceSnapshot snapshot{object.get(), object->transform, mesh, {}, mesh ? mesh->material : Material{}, mesh ? mesh->materials : std::vector<Material>{}, mesh ? mesh->sections : std::vector<MeshSection>{}, object->renderStatic, object->spatiallyLoaded, mesh && mesh->instanced, mesh && mesh->cullBackfaces, mesh && mesh->castShadows, mesh ? mesh->vertexBoundsPadding : Vector3{}, mesh && mesh->vertexAnimated};
				if (mesh) {
					mesh->Validate();
					if (mesh->IsSkinned()) {
						Skeleton* skeleton = object->GetComponent<Skeleton>();
						if (!skeleton) throw std::invalid_argument("Skinned meshes require a skeleton on the same object");
						mesh->ValidateSkeleton(*skeleton);
					}
					snapshot.revision.Set(*mesh);
				}
				sources.push_back(std::move(snapshot));
				if (!object->spatiallyLoaded) {
					alwaysResidentObjects.push_back(object.get());
					continue;
				}
				if (!object->renderStatic) {
					dynamicObjects.push_back(object.get());
					continue;
				}
				if (!mesh || mesh->instanced || mesh->IsSkinned() || mesh->data.indices.size() < 3 || mesh->data.vertices.empty()) {
					const CellKey key = Key(object->transform.position, size);
					cells[key].push_back(object.get());
					IncludeBounds(cellBounds, key, object->transform.position);
					continue;
				}
				using VertexMap = std::unordered_map<std::uint32_t, std::uint32_t>;
				std::map<CellKey, MeshData> cellMeshes;
				std::map<CellKey, VertexMap> cellVertices;
				std::map<CellKey, std::vector<MeshSection>> cellSections;
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
					const std::uint32_t materialIndex = mesh->MaterialIndexAt(index);
					auto& sections = cellSections[key];
					if (sections.empty() || sections.back().materialIndex != materialIndex) sections.push_back({static_cast<std::uint32_t>(cellMesh.indices.size()), 0, materialIndex});
					VertexMap& vertices = cellVertices[key];
					for (const auto sourceIndex : sourceIndices) {
						auto [entry, inserted] = vertices.try_emplace(sourceIndex, static_cast<std::uint32_t>(cellMesh.vertices.size()));
						if (inserted) cellMesh.vertices.push_back(mesh->data.vertices[sourceIndex]);
						cellMesh.indices.push_back(entry->second);
					}
					sections.back().indexCount += 3;
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
					cellMesh.materials = mesh->materials;
					cellMesh.sections = std::move(cellSections.at(key));
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
			const Vector3 forward = camera->owner->transform.Forward();
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
				if (mesh) {
					bool sectionsChanged = source.sections.size() != mesh->sections.size();
					for (std::size_t index = 0; !sectionsChanged && index < source.sections.size(); ++index) sectionsChanged = std::tie(source.sections[index].firstIndex, source.sections[index].indexCount, source.sections[index].materialIndex) != std::tie(mesh->sections[index].firstIndex, mesh->sections[index].indexCount, mesh->sections[index].materialIndex);
					if (sectionsChanged) { dirty = true; break; }
					bool materialsChanged = source.material.Key() != mesh->material.Key() || source.materials.size() != mesh->materials.size();
					for (std::size_t index = 0; !materialsChanged && index < source.materials.size(); ++index) materialsChanged = source.materials[index].Key() != mesh->materials[index].Key();
					if (materialsChanged || source.cullBackfaces != mesh->cullBackfaces || source.castShadows != mesh->castShadows || source.animated != mesh->vertexAnimated || std::tie(source.padding.x, source.padding.y, source.padding.z) != std::tie(mesh->vertexBoundsPadding.x, mesh->vertexBoundsPadding.y, mesh->vertexBoundsPadding.z)) materialChanged = true;
				}
			}
			if (sourceIndex != sources.size()) dirty = true;
			if (materialChanged && !dirty) {
				for (auto& cell : cellsData) {
					Mesh* sourceMesh = cell->worldPartitionSourceObject->GetComponent<Mesh>();
					Mesh* cellMesh = cell->GetComponent<Mesh>();
					cellMesh->material = sourceMesh->material;
					cellMesh->materials = sourceMesh->materials;
					cellMesh->cullBackfaces = sourceMesh->cullBackfaces;
					cellMesh->castShadows = sourceMesh->castShadows;
					cellMesh->vertexBoundsPadding = sourceMesh->vertexBoundsPadding;
					cellMesh->vertexAnimated = sourceMesh->vertexAnimated;
				}
				for (auto& source : sources) if (source.mesh) { source.material = source.mesh->material; source.materials = source.mesh->materials; source.cullBackfaces = source.mesh->cullBackfaces; source.castShadows = source.mesh->castShadows; source.padding = source.mesh->vertexBoundsPadding; source.animated = source.mesh->vertexAnimated; }
			}
			if (!initialized || dirty || partitionSize != size) Rebuild(objects, size);
			const float loadRange = std::max(range, 0.0f);
			const float unloadRange = loadRange + size;
			const float loadRangeSquared = loadRange * loadRange;
			const float unloadRangeSquared = unloadRange * unloadRange;
			const float priorityAngle = std::min(camera->fieldOfView * 0.5f + 15.0f, 89.0f) * Math::degreesToRadians;
			const float forwardThreshold = std::cos(priorityAngle);
			const bool directionChanged = !pendingCellLoads.empty() && Math::Dot(sourceForward, forward) < 0.9659258f;
			const bool refreshCells = !hasCellSelection || sourceCell != nextSource || loadingRange != range || directionChanged;
			if (refreshCells) {
				std::vector<std::tuple<int, float, CellKey>> ordered;
				for (const auto& [key, members] : cells) {
					const auto& bounds = cellBounds.at(key);
					const Vector3 closest{std::clamp(sourcePosition.x, bounds.minimum.x, bounds.maximum.x), std::clamp(sourcePosition.y, bounds.minimum.y, bounds.maximum.y), std::clamp(sourcePosition.z, bounds.minimum.z, bounds.maximum.z)};
					const Vector3 difference{closest.x - sourcePosition.x, closest.y - sourcePosition.y, closest.z - sourcePosition.z};
					const float distanceSquared = Math::Dot(difference, difference);
					if (key == nextSource || distanceSquared <= loadRangeSquared || (residentCellKeys.count(key) && distanceSquared <= unloadRangeSquared)) {
						const float distance = std::sqrt(distanceSquared);
						const float alignment = distance > 0.0001f ? Math::Dot(Math::Scale(difference, 1.0f / distance), forward) : 1.0f;
						const int priority = key == nextSource ? -1 : alignment >= forwardThreshold ? 0 : 1;
						ordered.emplace_back(priority, distanceSquared, key);
					}
				}
				std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) {
					if (std::get<0>(left) != std::get<0>(right)) return std::get<0>(left) < std::get<0>(right);
					if (std::get<1>(left) != std::get<1>(right)) return std::get<1>(left) < std::get<1>(right);
					return std::get<2>(left) < std::get<2>(right);
				});
				pendingCellLoads.clear();
				pendingCellUnloads.clear();
				std::set<CellKey> desiredCells;
				for (const auto& entry : ordered) {
					const float distance = std::get<1>(entry);
					const CellKey& key = std::get<2>(entry);
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
				sourceForward = forward;
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
			Vector3 gravity{0.0f, 0.0f, -9.81f};
			int cellSize = 20;
			int partitionSize = 200;
			float partitionBuffer = 1.50f;
			bool shadowsEnabled = true;
			float shadowFar = 10.0f;
			int shadowResolution = 2048;
			bool ambientLight = true;
			Vector3 ambientLightColor{1.0f, 1.0f, 1.0f};
			bool fogEnabled = false;
			bool fogVolumetric = false;
			float fogIntensity = 1.0f;
			float fogHeight = 10.0f; // World-space height where density is defined.
			float fogFalloff = 0.1f; // Exponential density falloff above fogHeight.
			bool debugMode = false;
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
				UpdateTransforms();
				const float range = activeCamera ? activeCamera->far * std::max(partitionBuffer, 0.0f) : 0.0f;
				worldPartitioner.Update(objects, activeCamera, partitionSize, range);
			}

			void UpdateTransforms() {
				++transformGeneration;
				for (auto& object : objects) object->UpdateWorldTransform(transformGeneration);
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
			std::uint64_t transformGeneration = 0;
			WorldPartitioner worldPartitioner;
	};

	class ParticleEmitter : public Object {
		struct Particle {
			Vector3 velocity;
			float age = 0.0f;
		};
	
		std::vector<Particle> particles;
		std::mt19937 random;
		float emissionRemainder = 0.0f;
		std::size_t pendingBurst = 0;
	
		public:
			float rate = 24.0f;
			float lifetime = 1.0f;
			float speed = 1.0f;
			float spread = 0.0f;
			float size = 0.15f;
			std::size_t maxParticles = 256;
			Vector3 direction{0.0f, 0.0f, 1.0f};
			Vector3 startColor{1.0f, 1.0f, 1.0f};
			Vector3 endColor{-1.0f, -1.0f, -1.0f};
			Material material;
			using Object::Object;
	
			void Emit(std::size_t count) { pendingBurst = std::min(maxParticles, pendingBurst + count); }
	
			void Initialize() override {
				renderStatic = false;
				auto& mesh = AddComponent<Mesh>();
				mesh.data = Primitive::Plane.Mesh();
				mesh.material = material;
				if (endColor.x >= 0.0f) mesh.material.vertexModifierSource = "void ModifyVertex(thread float3& position, thread float3& normal, thread float4& color, float instanceData, float time, float4x4 model) { color.rgb *= mix(float3(" + std::to_string(startColor.x) + ", " + std::to_string(startColor.y) + ", " + std::to_string(startColor.z) + "), float3(" + std::to_string(endColor.x) + ", " + std::to_string(endColor.y) + ", " + std::to_string(endColor.z) + "), clamp(instanceData, 0.0, 1.0)); }";
				else mesh.material.baseColor = {material.baseColor.x * startColor.x, material.baseColor.y * startColor.y, material.baseColor.z * startColor.z};
				mesh.instanced = true;
				mesh.cullBackfaces = false;
				mesh.castShadows = false;
				particles.reserve(maxParticles);
				mesh.instances.reserve(maxParticles);
			}
	
			void Update() override {
				auto& mesh = *GetComponent<Mesh>();
				const float deltaTime = Time::deltaTime;
				bool changed = false;
				for (std::size_t index = 0; index < particles.size();) {
					Particle& particle = particles[index];
					particle.age += deltaTime;
					if (particle.age >= lifetime) {
						particles[index] = particles.back();
						particles.pop_back();
						mesh.instances[index] = mesh.instances.back();
						mesh.instances.pop_back();
						changed = true;
						continue;
					}
					mesh.instances[index].position += particle.velocity * deltaTime;
					mesh.instances[index].customData = particle.age / lifetime;
					++index;
					changed = true;
				}
				emissionRemainder += std::max(rate, 0.0f) * deltaTime;
				const std::size_t scheduled = static_cast<std::size_t>(emissionRemainder);
				emissionRemainder -= static_cast<float>(scheduled);
				const std::size_t available = maxParticles > particles.size() ? maxParticles - particles.size() : 0;
				const std::size_t count = std::min(available, scheduled + pendingBurst);
				pendingBurst = 0;
				const Vector3 launchDirection = Math::Normalize(direction);
				std::uniform_real_distribution<float> deviation(-spread, spread);
				for (std::size_t index = 0; index < count; ++index) {
					Vector3 travel = launchDirection;
					if (spread > 0.0f) travel = Math::Normalize(Math::Add(travel, {deviation(random), deviation(random), deviation(random)}));
					particles.push_back({Math::Scale(travel, speed), 0.0f});
					MeshInstance instance{};
					instance.scale = {size, size, size};
					mesh.instances.push_back(instance);
					changed = true;
				}
				if (!changed) return;
				if (scene->activeCamera && scene->activeCamera->owner) {
					Vector3 facing = scene->activeCamera->owner->transform.rotation;
					facing.x -= 90.0f;
					const Vector3 rotation = Math::RelativeRotation(transform.rotation, facing);
					for (auto& instance : mesh.instances) instance.rotation = rotation;
				}
				mesh.MarkInstancesDirty();
			}
	};

	namespace Physics { std::vector<Vector3> ColliderDebugLines(Scene& scene); }

	namespace {
		struct S3DRecord {
			std::string name;
			std::string source;
			std::uint8_t type;
			std::vector<std::pair<std::string, std::string>> parameters;
			Transform transform;
			MeshData data;
			std::vector<Material> materials;
			std::vector<MeshSection> sections;
			std::vector<Bone> bones;
			Light light;
		};

		class S3DReader {
			std::vector<std::uint8_t> data;
			std::size_t offset = 0;
			std::filesystem::path path;

		public:
			explicit S3DReader(const std::string& fileName) {
				path = fileName;
				std::ifstream file(path, std::ios::binary | std::ios::ate);
				if (!file && path.is_relative()) {
					const char* base = SDL_GetBasePath();
					if (base) {
						path = std::filesystem::path(base) / ".." / "resources" / path;
						file.clear();
						file.open(path, std::ios::binary | std::ios::ate);
					}
				}
				if (!file) throw std::runtime_error("Could not open S3D file");
				const auto length = file.tellg();
				if (length < 8 || static_cast<std::uint64_t>(length) > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error("Invalid S3D file size");
				data.resize(static_cast<std::size_t>(length));
				file.seekg(0);
				file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
				if (!file) throw std::runtime_error("Could not read S3D file");
				path = std::filesystem::absolute(path);
			}

			template <typename T> T Read() {
				if (sizeof(T) > data.size() - offset) throw std::runtime_error("Truncated S3D file");
				if constexpr (std::is_floating_point_v<T>) {
					static_assert(sizeof(T) == sizeof(std::uint32_t));
					const std::uint32_t bits = Read<std::uint32_t>();
					T value;
					std::memcpy(&value, &bits, sizeof(value));
					if (!std::isfinite(value)) throw std::runtime_error("Non-finite S3D value");
					return value;
				}
				else {
					using Unsigned = std::make_unsigned_t<T>;
					Unsigned value = 0;
					for (std::size_t index = 0; index < sizeof(T); ++index) value |= static_cast<Unsigned>(data[offset++]) << (index * 8);
					return static_cast<T>(value);
				}
			}

			std::uint32_t Count(std::size_t itemSize) {
				const auto count = Read<std::uint32_t>();
				if (itemSize && count > (data.size() - offset) / itemSize) throw std::runtime_error("Invalid S3D count");
				return count;
			}

			std::string String() {
				const auto size = Count(1);
				std::string value(reinterpret_cast<const char*>(data.data() + offset), size);
				offset += size;
				return value;
			}

			Vector3 Vector() { return {Read<float>(), Read<float>(), Read<float>()}; }
			Transform TransformValue() { return {Vector(), Vector(), Vector()}; }
			const std::filesystem::path& FilePath() const { return path; }
			bool Finished() const { return offset == data.size(); }
		};

		void ReadS3DModel(S3DReader& reader, S3DRecord& record) {
			record.data.vertices.resize(reader.Count(48));
			record.data.indices.resize(reader.Count(4));
			record.sections.resize(reader.Count(12));
			record.materials.resize(reader.Count(1));
			record.bones.resize(reader.Count(1));
			for (auto& vertex : record.data.vertices) {
				vertex.position = reader.Vector();
				vertex.normal = reader.Vector();
				vertex.uv[0] = reader.Read<float>();
				vertex.uv[1] = reader.Read<float>();
				for (float& color : vertex.color) color = reader.Read<float>();
			}
			for (auto& index : record.data.indices) index = reader.Read<std::uint32_t>();
			for (auto& section : record.sections) section = {reader.Read<std::uint32_t>(), reader.Read<std::uint32_t>(), reader.Read<std::uint32_t>()};
			for (auto& material : record.materials) {
				material.baseColor = reader.Vector();
				material.roughness = reader.Read<float>(); material.metallic = reader.Read<float>(); material.normalScale = reader.Read<float>(); material.aoStrength = reader.Read<float>(); material.displacementScale = reader.Read<float>();
				for (float& value : material.uvScale) value = reader.Read<float>();
				for (float& value : material.uvOffset) value = reader.Read<float>();
				material.flipNormalY = reader.Read<std::uint8_t>() != 0; material.transparent = reader.Read<std::uint8_t>() != 0; material.unlit = reader.Read<std::uint8_t>() != 0;
				material.opacity = reader.Read<float>();
				std::string* paths[] = {&material.baseTexturePath, &material.normalTexturePath, &material.roughnessTexturePath, &material.aoTexturePath, &material.displacementTexturePath, &material.displacementMaskTexturePath};
				for (auto* path : paths) *path = reader.String();
				material.vertexModifierSource = reader.String();
			}
			for (auto& bone : record.bones) {
				bone.name = reader.String();
				bone.parentIndex = reader.Read<std::int32_t>(); bone.bindTransform = reader.TransformValue();
			}
			record.data.skinWeights.resize(reader.Count(16));
			for (auto& skin : record.data.skinWeights) {
				for (auto& index : skin.boneIndices) index = reader.Read<std::uint16_t>();
				for (auto& weight : skin.weights) weight = reader.Read<std::uint16_t>();
			}
			record.data.Validate();
			Mesh validation;
			validation.data = std::move(record.data);
			validation.materials = std::move(record.materials);
			validation.sections = std::move(record.sections);
			validation.ValidateSections();
			record.data = std::move(validation.data);
			record.materials = std::move(validation.materials);
			record.sections = std::move(validation.sections);
			if (!record.bones.empty()) {
				Skeleton skeleton;
				skeleton.bones = std::move(record.bones);
				skeleton.Validate();
				if (!record.data.skinWeights.empty()) record.data.ValidateSkeleton(skeleton);
				record.bones = std::move(skeleton.bones);
			}
			else if (!record.data.skinWeights.empty()) throw std::runtime_error("Skinned mesh has no skeleton");
		}

		std::vector<S3DRecord> ReadS3D(const std::string& fileName, std::filesystem::path& filePath, std::unordered_map<std::string, S3DRecord>& models) {
			S3DReader reader(fileName);
			filePath = reader.FilePath();
			if (reader.Read<std::uint8_t>() != 'S' || reader.Read<std::uint8_t>() != '3' || reader.Read<std::uint8_t>() != 'D' || reader.Read<std::uint8_t>() != 0) throw std::runtime_error("Invalid S3D signature");
			const auto modelCount = reader.Count(1);
			models.reserve(modelCount);
			for (std::uint32_t modelIndex = 0; modelIndex < modelCount; ++modelIndex) {
				S3DRecord model{};
				model.name = reader.String();
				ReadS3DModel(reader, model);
				std::string name = model.name;
				if (!models.emplace(std::move(name), std::move(model)).second) throw std::runtime_error("Duplicate S3D model name");
			}
			const auto objectCount = reader.Count(1);
			if (!objectCount) throw std::runtime_error("S3D contains no objects");
			std::vector<S3DRecord> records;
			records.reserve(objectCount);
			std::unordered_set<std::string> names;
			std::unordered_set<std::string> parents;
			for (std::uint32_t objectIndex = 0; objectIndex < objectCount; ++objectIndex) {
				S3DRecord record{};
				record.name = reader.String();
				if (!names.insert(record.name).second) throw std::runtime_error("Duplicate S3D object name");
				record.type = reader.Read<std::uint8_t>();
				const auto parameterCount = reader.Count(8);
				if (!parameterCount) throw std::runtime_error("S3D object has no hierarchy parameter");
				record.parameters.reserve(parameterCount);
				std::unordered_set<std::string> keys;
				for (std::uint32_t parameterIndex = 0; parameterIndex < parameterCount; ++parameterIndex) {
					std::string key = reader.String();
					std::string value = reader.String();
					if (!keys.insert(key).second || (parameterIndex && (key == "isParent" || key == "parent"))) throw std::runtime_error("Invalid S3D parameter key");
					record.parameters.emplace_back(std::move(key), std::move(value));
				}
				const auto& hierarchy = record.parameters.front();
				if (hierarchy.first == "isParent") {
					if (hierarchy.second != "True" && hierarchy.second != "False") throw std::runtime_error("Invalid S3D isParent value");
					if (hierarchy.second == "True") parents.insert(record.name);
				}
				else if (hierarchy.first == "parent") {
					if (!parents.count(hierarchy.second)) throw std::runtime_error("S3D child references a missing or later parent");
				}
				else throw std::runtime_error("Invalid S3D hierarchy parameter");
				record.transform = reader.TransformValue();
				if (record.type == 0) {
					record.source = reader.String();
					if (!models.count(record.source)) throw std::runtime_error("S3D object references a missing model");
				}
				else if (record.type == 1) {
					const auto type = reader.Read<std::uint8_t>();
					if (type > 2) throw std::runtime_error("Invalid S3D light type");
					record.light.type = static_cast<Light::Type>(type);
					record.light.color = reader.Vector();
					record.light.intensity = reader.Read<float>(); record.light.range = reader.Read<float>(); record.light.innerSpotAngle = reader.Read<float>(); record.light.outerSpotAngle = reader.Read<float>();
					record.light.castShadows = reader.Read<std::uint8_t>() != 0; record.light.shadowBias = reader.Read<float>();
				}
				else throw std::runtime_error("Invalid S3D object type");
				records.push_back(std::move(record));
			}
			if (!reader.Finished()) throw std::runtime_error("Invalid S3D structure");
			return records;
		}
	}

	namespace Physics { void EnsureMeshCollider(Object& object); }

	bool Object::LoadS3D(const std::string& fileName) {
		std::filesystem::path sourcePath;
		std::vector<S3DRecord> records;
		std::unordered_map<std::string, S3DRecord> models;
		try {
			records = ReadS3D(fileName, sourcePath, models);
			for (auto& [name, model] : models) for (auto& material : model.materials) {
				std::string* paths[] = {&material.baseTexturePath, &material.normalTexturePath, &material.roughnessTexturePath, &material.aoTexturePath, &material.displacementTexturePath, &material.displacementMaskTexturePath};
				for (auto* path : paths) if (!path->empty() && std::filesystem::path(*path).is_relative()) *path = (sourcePath.parent_path() / *path).lexically_normal().string();
			}
		}
		catch (const std::exception&) { return false; }
		if (records.size() > 1 && !scene) return false;
		std::unordered_map<std::string, Object*> created;
		created.reserve(records.size());
		for (std::size_t index = 0; index < records.size(); ++index) {
			auto& record = records[index];
			Object* object = index ? &scene->CreateObject<>(record.name) : this;
			object->name = record.name;
			const auto& hierarchy = record.parameters.front();
			Object* parent = hierarchy.first == "parent" ? created.at(hierarchy.second) : nullptr;
			object->SetParent(parent, false);
			if (parent) {
				object->transform.localPosition = record.transform.position;
				object->transform.localRotation = record.transform.rotation;
				object->transform.localScale = record.transform.scale;
			}
			else {
				object->transform.position = record.transform.position;
				object->transform.rotation = record.transform.rotation;
				object->transform.scale = record.transform.scale;
			}
			object->UpdateWorldTransform();
			for (const auto& [key, value] : record.parameters) object->SetProperty<std::string>(key, value);
			if (record.type == 0) {
				const auto& model = models.at(record.source);
				Mesh* mesh = object->GetComponent<Mesh>();
				if (!mesh) mesh = &object->AddComponent<Mesh>();
				mesh->data = model.data;
				mesh->material = Material{};
				mesh->materials = model.materials;
				mesh->sections = model.sections;
				mesh->MarkDirty();
				Physics::EnsureMeshCollider(*object);
				if (!model.bones.empty()) {
					Skeleton* skeleton = object->GetComponent<Skeleton>();
					if (!skeleton) skeleton = &object->AddComponent<Skeleton>();
					skeleton->bones = model.bones;
					skeleton->InitializePose();
				}
			}
			else {
				Light* light = object->GetComponent<Light>();
				if (!light) light = &object->AddComponent<Light>();
				*light = record.light;
				light->owner = object;
				if (light->type == Light::Type::Directional) object->spatiallyLoaded = false;
			}
			created.emplace(record.name, object);
		}
		if (scene) scene->UpdateWorldPartition();
		return true;
	}
}
