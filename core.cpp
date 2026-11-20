#include <algorithm>
#include <array>
#include <any>
#include <limits>
#include <cmath>
#include <complex>
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

	struct Vector4 {
		float x = 0.0f;
		float y = 0.0f;
		float z = 0.0f;
		float w = 0.0f;
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
			return value * scalar;
		}

		Vector3 Cross(const Vector3& left, const Vector3& right) {
			return {left.y * right.z - left.z * right.y, left.z * right.x - left.x * right.z, left.x * right.y - left.y * right.x};
		}

		Vector3 Normalize(const Vector3& value) {
			const float length = std::sqrt(Dot(value, value));
			return length > 0.0001f ? Scale(value, 1.0f / length) : Vector3{};
		}

		void GetRotationAxes(const Vector3& rotation, Vector3 (&axes)[3]) {
			const float x = rotation.x * Math::degreesToRadians;
			const float y = rotation.y * Math::degreesToRadians;
			const float z = rotation.z * Math::degreesToRadians;
			const float cx = std::cos(x), sx = std::sin(x), cy = std::cos(y), sy = std::sin(y), cz = std::cos(z), sz = std::sin(z);
			axes[0] = {cz * cy, sz * cy, -sy};
			axes[1] = {cz * sy * sx - sz * cx, sz * sy * sx + cz * cx, cy * sx};
			axes[2] = {cz * sy * cx + sz * sx, sz * sy * cx - cz * sx, cy * cx};
		}

		void GetAxes(const Transform& transform, Vector3 (&axes)[3]) { GetRotationAxes(transform.rotation, axes); }

		Vector3 RotationFromAxes(const Vector3 (&axes)[3]) {
			const float roll = std::asin(std::clamp(-axes[0].z, -1.0f, 1.0f));
			const float cosineRoll = std::cos(roll);
			const float pitch = std::abs(cosineRoll) > 0.0001f ? std::atan2(axes[1].z, axes[2].z) : std::atan2(-axes[2].y, axes[1].y);
			const float yaw = std::abs(cosineRoll) > 0.0001f ? std::atan2(axes[0].y, axes[0].x) : 0.0f;
			return {pitch / degreesToRadians, roll / degreesToRadians, yaw / degreesToRadians};
		}

		Vector3 ComposeRotation(const Vector3& parent, const Vector3& local, const Vector3* preparedParentAxes = nullptr) {
			Vector3 parentAxes[3];
			Vector3 localAxes[3];
			if (preparedParentAxes) std::copy_n(preparedParentAxes, 3, parentAxes);
			else GetRotationAxes(parent, parentAxes);
			GetRotationAxes(local, localAxes);
			Vector3 axes[3];
			for (int index = 0; index < 3; ++index) axes[index] = Add(Scale(parentAxes[0], localAxes[index].x), Add(Scale(parentAxes[1], localAxes[index].y), Scale(parentAxes[2], localAxes[index].z)));
			return RotationFromAxes(axes);
		}

		Vector3 RelativeRotation(const Vector3& parent, const Vector3& world, const Vector3* preparedParentAxes) {
			Vector3 parentAxes[3];
			Vector3 worldAxes[3];
			if (preparedParentAxes) std::copy_n(preparedParentAxes, 3, parentAxes);
			else GetRotationAxes(parent, parentAxes);
			GetRotationAxes(world, worldAxes);
			Vector3 localAxes[3];
			for (int index = 0; index < 3; ++index) localAxes[index] = {Dot(parentAxes[0], worldAxes[index]), Dot(parentAxes[1], worldAxes[index]), Dot(parentAxes[2], worldAxes[index])};
			return RotationFromAxes(localAxes);
		}

		Vector3 RelativeRotation(const Vector3& parent, const Vector3& world) {
			Vector3 parentAxes[3];
			Vector3 worldAxes[3];
			GetRotationAxes(parent, parentAxes);
			GetRotationAxes(world, worldAxes);
			Vector3 localAxes[3];
			for (int index = 0; index < 3; ++index) localAxes[index] = {Dot(parentAxes[0], worldAxes[index]), Dot(parentAxes[1], worldAxes[index]), Dot(parentAxes[2], worldAxes[index])};
			return RotationFromAxes(localAxes);
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
			float shadowBias = 0.04f;
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
				auto buildMatrices = [&](bool bindPose, std::vector<Math::Matrix4>& matrices, std::vector<std::uint8_t>& complete, std::vector<std::size_t>& path) {
					matrices.resize(bones.size());
					std::fill(complete.begin(), complete.end(), 0);
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
				};
				std::vector<std::uint8_t> complete(bones.size());
				std::vector<std::size_t> path;
				if (inverseBindMatrices.size() != bones.size()) {
					std::vector<Math::Matrix4> bindMatrices(bones.size());
					buildMatrices(true, bindMatrices, complete, path);
					std::vector<Math::Matrix4> inverseBindMatricesNext(bones.size());
					for (std::size_t index = 0; index < bones.size(); ++index) inverseBindMatricesNext[index] = InverseAffine(bindMatrices[index]);
					inverseBindMatrices = std::move(inverseBindMatricesNext);
				}
				buildMatrices(false, poseGlobalMatrices, complete, path);
				poseMatrices.resize(bones.size());
				for (std::size_t index = 0; index < bones.size(); ++index) poseMatrices[index] = Math::Multiply(poseGlobalMatrices[index], inverseBindMatrices[index]);
				cachedPoseRevision = poseRevision;
				return poseMatrices;
			}

			const std::vector<Math::Matrix4>& InverseBindMatrices() { PoseMatrices(); return inverseBindMatrices; }
			const std::vector<Math::Matrix4>& PoseGlobalMatrices() { PoseMatrices(); return poseGlobalMatrices; }

		private:
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
			mesh.vertices.reserve(24);
			mesh.indices.reserve(36);
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
		static const std::array<std::array<float, 2>, sides + 1> azimuths = [] {
			std::array<std::array<float, 2>, sides + 1> values{};
			for (int side = 0; side <= sides; ++side) {
				const float azimuth = static_cast<float>(side) / sides * 2.0f * pi;
				values[side] = {std::cos(azimuth), std::sin(azimuth)};
			}
			return values;
		}();
		const int upperStart = halfStraight > 0.0f ? 0 : 1;
		const std::size_t ringCount = static_cast<std::size_t>(hemisphereRings + 1 + hemisphereRings - upperStart + 1);
		mesh.vertices.reserve(ringCount * (sides + 1));
		mesh.indices.reserve((ringCount - 1) * sides * 6);
		auto addRing = [&](float angle, float centerZ) {
			const float cosineAngle = std::cos(angle);
			const float sineAngle = std::sin(angle);
			const float ringRadius = radius * cosineAngle;
			const float z = centerZ + radius * sineAngle;
			for (int side = 0; side <= sides; ++side) {
				const float u = static_cast<float>(side) / sides;
				const float cosine = azimuths[side][0];
				const float sine = azimuths[side][1];
				mesh.vertices.push_back({{ringRadius * cosine, ringRadius * sine, z}, {cosineAngle * cosine, cosineAngle * sine, sineAngle}, {u, size.z > 0.0f ? z / size.z + 0.5f : 0.5f}});
			}
		};
		for (int ring = 0; ring <= hemisphereRings; ++ring) addRing(-pi * 0.5f + static_cast<float>(ring) / hemisphereRings * pi * 0.5f, -halfStraight);
		for (int ring = upperStart; ring <= hemisphereRings; ++ring) addRing(static_cast<float>(ring) / hemisphereRings * pi * 0.5f, halfStraight);
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
		std::string fragmentNormalSource;
		std::string depthSurfaceSource;
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
		bool excludeFromSceneDepth = false;
		bool singleLayer = false;
		float opacity = 1.0f;
		std::array<float, 16> customData{};

		auto Key() const {
			return std::tie(baseTexturePath, normalTexturePath, roughnessTexturePath, aoTexturePath, displacementTexturePath, displacementMaskTexturePath, vertexModifierSource, fragmentNormalSource, depthSurfaceSource, baseColor.x, baseColor.y, baseColor.z, roughness, metallic, normalScale, aoStrength, displacementScale, uvScale[0], uvScale[1], uvOffset[0], uvOffset[1], flipNormalY, transparent, unlit, excludeFromSceneDepth, singleLayer, opacity, customData);
		}

		bool UsesSingleLayer() const { return transparent && singleLayer; }
		bool HasShaderModifiers() const { return !vertexModifierSource.empty() || !fragmentNormalSource.empty() || !depthSurfaceSource.empty() || UsesSingleLayer(); }
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
		float customData[4]{};
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
			static void ValidateSections(std::size_t indexCount, std::size_t suppliedMaterialCount, const std::vector<MeshSection>& sectionList) {
				if (suppliedMaterialCount > std::numeric_limits<std::uint32_t>::max()) throw std::length_error("Mesh material count exceeds supported range");
				if (suppliedMaterialCount > 1 && sectionList.empty()) throw std::invalid_argument("Meshes with multiple materials require sections");
				std::size_t nextIndex = 0;
				for (const auto& section : sectionList) {
					if (section.firstIndex != nextIndex || section.indexCount == 0 || section.indexCount % 3 != 0 || static_cast<std::size_t>(section.firstIndex) + section.indexCount > indexCount) throw std::invalid_argument("Mesh sections must cover contiguous triangle index ranges");
					if (section.materialIndex >= (suppliedMaterialCount ? suppliedMaterialCount : 1)) throw std::out_of_range("Mesh section material index is invalid");
					nextIndex += section.indexCount;
				}
				if (!sectionList.empty() && nextIndex != indexCount) throw std::invalid_argument("Mesh sections must cover all mesh indices");
			}
			void ValidateSections() const { ValidateSections(data.indices.size(), materials.size(), sections); }
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
				transform.rotation = Math::ComposeRotation(parent->transform.rotation, transform.localRotation, axes);
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
				transform.localPosition = LocalPosition(position, parent->transform, axes);
				UpdateDescendantWorldTransforms();
			}

			void SetWorldRotation(const Vector3& rotation) {
				transform.rotation = rotation;
				Vector3 axes[3];
				if (parent) {
					parent->UpdateWorldTransform();
					Math::GetAxes(parent->transform, axes);
				}
				transform.localRotation = parent ? Math::RelativeRotation(parent->transform.rotation, rotation, axes) : rotation;
				UpdateDescendantWorldTransforms();
			}

			void SetWorldScale(const Vector3& scale) {
				transform.scale = scale;
				if (parent) parent->UpdateWorldTransform();
				transform.localScale = parent ? RelativeScale(scale, parent->transform.scale) : scale;
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
					transform.position = worldPosition;
					transform.rotation = worldRotation;
					transform.scale = worldScale;
					if (parent) {
						parent->UpdateWorldTransform();
						Vector3 axes[3];
						Math::GetAxes(parent->transform, axes);
						transform.localPosition = LocalPosition(worldPosition, parent->transform, axes);
						transform.localRotation = Math::RelativeRotation(parent->transform.rotation, worldRotation, axes);
						transform.localScale = RelativeScale(worldScale, parent->transform.scale);
					}
					else {
						transform.localRotation = worldRotation;
						transform.localScale = worldScale;
					}
					UpdateDescendantWorldTransforms();
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
			static Vector3 LocalPosition(const Vector3& position, const Transform& parentTransform, const Vector3 (&axes)[3]) {
				const Vector3 offset = Math::Subtract(position, parentTransform.position);
				return {std::abs(parentTransform.scale.x) > 0.0001f ? Math::Dot(offset, axes[0]) / parentTransform.scale.x : 0.0f, std::abs(parentTransform.scale.y) > 0.0001f ? Math::Dot(offset, axes[1]) / parentTransform.scale.y : 0.0f, std::abs(parentTransform.scale.z) > 0.0001f ? Math::Dot(offset, axes[2]) / parentTransform.scale.z : 0.0f};
			}

			static Vector3 RelativeScale(const Vector3& scale, const Vector3& parentScale) {
				return {std::abs(parentScale.x) > 0.0001f ? scale.x / parentScale.x : 0.0f, std::abs(parentScale.y) > 0.0001f ? scale.y / parentScale.y : 0.0f, std::abs(parentScale.z) > 0.0001f ? scale.z / parentScale.z : 0.0f};
			}

			std::vector<std::unique_ptr<Component>> components;
			std::vector<std::pair<const void*, Component*>> componentCache;
			std::unordered_map<std::string, std::any> properties;
			std::vector<Object*> children;
	};

	struct FogDensityInput {
		enum class Encoding : std::uint32_t { ExponentialHeight, SampledGrid };
		Encoding encoding = Encoding::SampledGrid;
		Math::Bounds bounds;
		float density = 0.0f;
		float height = 0.0f;
		float falloff = 0.0f;
		float endDistance = 0.0f;
		std::uint32_t width = 0;
		std::uint32_t heightSamples = 0;
		std::uint32_t depth = 0;
		std::vector<float> samples;
		std::uint64_t revision = 0;
	};

	class FogDensitySource {
		public:
			virtual const FogDensityInput& GetFogDensityInput() const = 0;
			virtual ~FogDensitySource() = default;
	};

	class WorldPartitioner {
		using CellKey = std::tuple<int, int, int>;
		std::map<CellKey, std::vector<Object*>> cells;
		std::map<CellKey, Math::Bounds> cellBounds;
		std::vector<Object*> alwaysResidentObjects;
		struct DynamicObject { Object* object; bool resident = false; };
		std::vector<DynamicObject> dynamicObjects;
		std::vector<Object*> residentObjects;
		std::vector<Object*> residentRenderObjects;
		std::vector<CellKey> residentCells;
		std::set<CellKey> residentCellKeys;
		std::deque<CellKey> pendingCellLoads;
		std::deque<CellKey> pendingCellUnloads;
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
					dynamicObjects.push_back({object.get(), false});
					continue;
				}
				if (!mesh || mesh->instanced || mesh->IsSkinned() || mesh->data.indices.size() < 3 || mesh->data.vertices.empty()) {
					const CellKey key = Key(object->transform.position, size);
					cells[key].push_back(object.get());
					IncludeBounds(cellBounds, key, object->transform.position);
					continue;
				}
				using VertexMap = std::unordered_map<std::uint32_t, std::uint32_t>;
				struct CellMesh {
					MeshData data;
					VertexMap vertices;
					std::vector<MeshSection> sections;
					Math::Bounds bounds = Math::EmptyBounds();
				};
				std::map<CellKey, CellMesh> cellMeshes;
				const Math::Matrix4 model = Math::Model(object->transform);
				const std::size_t sectionCount = mesh->sections.empty() ? 1 : mesh->sections.size();
				for (std::size_t sectionIndex = 0; sectionIndex < sectionCount; ++sectionIndex) {
					const MeshSection section = mesh->sections.empty() ? MeshSection{0, static_cast<std::uint32_t>(mesh->data.indices.size()), 0} : mesh->GetSection(sectionIndex);
					for (std::size_t index = section.firstIndex; index + 2 < static_cast<std::size_t>(section.firstIndex) + section.indexCount; index += 3) {
					const std::uint32_t sourceIndices[3] = {mesh->data.indices[index], mesh->data.indices[index + 1], mesh->data.indices[index + 2]};
					const Vector3 a = Math::TransformPoint(model, mesh->data.vertices[sourceIndices[0]].position);
					const Vector3 b = Math::TransformPoint(model, mesh->data.vertices[sourceIndices[1]].position);
					const Vector3 c = Math::TransformPoint(model, mesh->data.vertices[sourceIndices[2]].position);
					const CellKey key = Key(Math::Scale(Math::Add(Math::Add(a, b), c), 1.0f / 3.0f), size);
					auto [cellEntry, insertedCell] = cellMeshes.try_emplace(key);
					CellMesh& cell = cellEntry->second;
					if (insertedCell) cell.bounds = {a, a};
					else Math::Include(cell.bounds, a);
					Math::Include(cell.bounds, b);
					Math::Include(cell.bounds, c);
					MeshData& cellMesh = cell.data;
					const std::uint32_t materialIndex = section.materialIndex;
					auto& sections = cell.sections;
					if (sections.empty() || sections.back().materialIndex != materialIndex) sections.push_back({static_cast<std::uint32_t>(cellMesh.indices.size()), 0, materialIndex});
					VertexMap& vertices = cell.vertices;
					for (const auto sourceIndex : sourceIndices) {
						auto [entry, inserted] = vertices.try_emplace(sourceIndex, static_cast<std::uint32_t>(cellMesh.vertices.size()));
						if (inserted) cellMesh.vertices.push_back(mesh->data.vertices[sourceIndex]);
						cellMesh.indices.push_back(entry->second);
					}
					sections.back().indexCount += 3;
					}
				}
				if (cellMeshes.empty()) {
					const CellKey key = Key(object->transform.position, size);
					cells[key].push_back(object.get());
					IncludeBounds(cellBounds, key, object->transform.position);
					continue;
				}
				object->worldPartitionedMesh = true;
				for (auto& [key, cellData] : cellMeshes) {
					auto cellObject = std::make_unique<Object>(object->name);
					cellObject->scene = object->scene;
					cellObject->transform = object->transform;
					cellObject->renderStatic = true;
					cellObject->worldPartitionCell = true;
					cellObject->worldPartitionSourceObject = object.get();
					Mesh& cellMesh = cellObject->AddComponent<Mesh>();
					cellMesh.data = std::move(cellData.data);
					cellMesh.material = mesh->material;
					cellMesh.materials = mesh->materials;
					cellMesh.sections = std::move(cellData.sections);
					cellMesh.cullBackfaces = mesh->cullBackfaces;
					cellMesh.castShadows = mesh->castShadows;
					cellMesh.vertexBoundsPadding = mesh->vertexBoundsPadding;
					cellMesh.vertexAnimated = mesh->vertexAnimated;
					cells[key].push_back(object.get());
					cells[key].push_back(cellObject.get());
					const Math::Bounds cellMeshBoundsValue = cellData.bounds;
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
				cellsData.clear();
				hasCellSelection = false;
				initialCellLoadComplete = false;
				nextCellStreamTime = {};
				for (const auto& object : objects) {
					object->worldPartitionedMesh = false;
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
			const bool directionChanged = !pendingCellLoads.empty() && Math::Dot(sourceForward, forward) < 0.9659258f;
			const bool refreshCells = !hasCellSelection || sourceCell != nextSource || loadingRange != range || directionChanged;
			if (refreshCells) {
				const float priorityAngle = std::min(camera->fieldOfView * 0.5f + 15.0f, 89.0f) * Math::degreesToRadians;
				const float forwardThreshold = std::cos(priorityAngle);
				pendingCellLoads.clear();
				pendingCellUnloads.clear();
				std::vector<std::tuple<int, float, CellKey>> ordered;
				const auto distanceSquaredToBounds = [&](const Math::Bounds& bounds, Vector3* delta) {
					const Vector3 closest{std::clamp(sourcePosition.x, bounds.minimum.x, bounds.maximum.x), std::clamp(sourcePosition.y, bounds.minimum.y, bounds.maximum.y), std::clamp(sourcePosition.z, bounds.minimum.z, bounds.maximum.z)};
					const Vector3 difference{closest.x - sourcePosition.x, closest.y - sourcePosition.y, closest.z - sourcePosition.z};
					if (delta) *delta = difference;
					return Math::Dot(difference, difference);
				};
				for (const auto& [key, members] : cells) {
					if (!residentCellKeys.count(key)) continue;
					const float distanceSquared = distanceSquaredToBounds(cellBounds.at(key), nullptr);
					if (key != nextSource && distanceSquared > loadRangeSquared && distanceSquared > unloadRangeSquared) pendingCellUnloads.push_back(key);
				}
				for (const auto& [key, members] : cells) {
					const auto& bounds = cellBounds.at(key);
					Vector3 difference;
					const float distanceSquared = distanceSquaredToBounds(bounds, key == nextSource ? nullptr : &difference);
					if (key == nextSource || distanceSquared <= loadRangeSquared) {
						int priority = -1;
						if (key != nextSource) {
							const float distance = std::sqrt(distanceSquared);
							const float alignment = distance > 0.0001f ? Math::Dot(Math::Scale(difference, 1.0f / distance), forward) : 1.0f;
							priority = alignment >= forwardThreshold ? 0 : 1;
						}
						ordered.emplace_back(priority, distanceSquared, key);
					}
				}
				std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) {
					if (std::get<0>(left) != std::get<0>(right)) return std::get<0>(left) < std::get<0>(right);
					if (std::get<1>(left) != std::get<1>(right)) return std::get<1>(left) < std::get<1>(right);
					return std::get<2>(left) < std::get<2>(right);
				});
				for (const auto& entry : ordered) {
					const CellKey& key = std::get<2>(entry);
					if (!residentCellKeys.count(key)) pendingCellLoads.push_back(key);
				}
				std::sort(pendingCellUnloads.begin(), pendingCellUnloads.end(), [&](const CellKey& left, const CellKey& right) {
					const float leftDistance = distanceSquaredToBounds(cellBounds.at(left), nullptr);
					const float rightDistance = distanceSquaredToBounds(cellBounds.at(right), nullptr);
					return leftDistance > rightDistance;
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
			for (auto& entry : dynamicObjects) {
				auto* object = entry.object;
				const Vector3 difference{object->transform.position.x - sourcePosition.x, object->transform.position.y - sourcePosition.y, object->transform.position.z - sourcePosition.z};
				const float distanceSquared = Math::Dot(difference, difference);
				entry.resident = distanceSquared <= loadRangeSquared || (entry.resident && distanceSquared <= unloadRangeSquared);
				if (entry.resident) addResident(object);
			}
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
			bool worldPartition = true;
			int partitionSize = 200;
			float partitionBuffer = 1.50f;
			bool shadowsEnabled = true;
			float shadowFar = 10.0f;
			int shadowResolution = 2048;
			int anisotropy = 0;
			bool ambientLight = true;
			Vector3 ambientLightColor{1.0f, 1.0f, 1.0f};
			bool fogEnabled = false;
			bool fogVolumetric = false;
			float fogQuality = 1.0f;
			float fogIntensity = 1.0f;
			float fogHeight = 10.0f;
			float fogFalloff = 0.1f;
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
				worldPartitioner.Update(objects, activeCamera, worldPartition ? partitionSize : 0, range);
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
			float sizeStart = 0.1f;
			float sizeEnd = 0.1f;
			std::size_t maxParticles = 256;
			Vector3 direction{0.0f, 0.0f, 1.0f};
			Vector4 colorStart{1.0f, 1.0f, 1.0f, 1.0f};
			Vector4 colorEnd{-1.0f, -1.0f, -1.0f, 1.0f};
			Material material;
			using Object::Object;
	
			void Emit(std::size_t count) { pendingBurst = std::min(maxParticles, pendingBurst + count); }
	
			void Initialize() override {
				renderStatic = false;
				auto& mesh = AddComponent<Mesh>();
				mesh.data = Primitive::Plane.Mesh();
				mesh.material = material;
				mesh.material.vertexModifierSource = "void ModifyVertex(thread float3& position, thread float3& normal, thread float4& color, float4 instanceData, float time, float4x4 model) { color *= instanceData; }";
				mesh.instanced = true;
				mesh.cullBackfaces = false;
				mesh.castShadows = false;
				particles.reserve(maxParticles);
				mesh.instances.reserve(maxParticles);
			}
	
			void Update() override {
				auto& mesh = *GetComponent<Mesh>();
				const float deltaTime = Time::deltaTime;
				bool changed = !particles.empty();
				for (std::size_t index = 0; index < particles.size();) {
					Particle& particle = particles[index];
					particle.age += deltaTime;
					if (particle.age >= lifetime) {
						particles[index] = particles.back();
						particles.pop_back();
						mesh.instances[index] = mesh.instances.back();
						mesh.instances.pop_back();
						continue;
					}
					mesh.instances[index].position += particle.velocity * deltaTime;
					const float age = particle.age / lifetime;
					const float particleSize = sizeStart + (sizeEnd - sizeStart) * age;
					mesh.instances[index].scale.x = particleSize;
					mesh.instances[index].scale.y = particleSize;
					const Vector4 target = colorEnd.x >= 0.0f ? colorEnd : colorStart;
					auto& data = mesh.instances[index].customData;
					data[0] = colorStart.x + (target.x - colorStart.x) * age;
					data[1] = colorStart.y + (target.y - colorStart.y) * age;
					data[2] = colorStart.z + (target.z - colorStart.z) * age;
					data[3] = colorStart.w + (target.w - colorStart.w) * age;
					++index;
				}
				emissionRemainder += std::max(rate, 0.0f) * deltaTime;
				const std::size_t scheduled = static_cast<std::size_t>(emissionRemainder);
				emissionRemainder -= static_cast<float>(scheduled);
				const std::size_t available = maxParticles > particles.size() ? maxParticles - particles.size() : 0;
				const std::size_t count = std::min(available, scheduled + pendingBurst);
				pendingBurst = 0;
				if (count) {
					const Vector3 launchDirection = Math::Normalize(direction);
					std::uniform_real_distribution<float> deviation(-spread, spread);
					for (std::size_t index = 0; index < count; ++index) {
						Vector3 travel = launchDirection;
						if (spread > 0.0f) travel = Math::Normalize(Math::Add(travel, {deviation(random), deviation(random), deviation(random)}));
						particles.push_back({Math::Scale(travel, speed), 0.0f});
						MeshInstance instance{};
						instance.scale = {sizeStart, sizeStart, sizeStart};
						instance.customData[0] = colorStart.x;
						instance.customData[1] = colorStart.y;
						instance.customData[2] = colorStart.z;
						instance.customData[3] = colorStart.w;
						mesh.instances.push_back(instance);
					}
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


	namespace Graphics::Shaders {
		extern const std::string waterVertexModifierSource;
		extern const std::string waterSurfaceSource;
	}

	namespace Graphics {
		void EnsureGeneratedTexture(const std::string& name, int width, int height, std::vector<std::uint8_t> (*generate)());
	}

	class WaterBody : public Object {
		static constexpr int resolution = 128;
		static constexpr int tile = resolution + 2;
		using Complex = std::complex<float>;

		static std::vector<std::uint8_t> BakeSpectrum() {
			constexpr int n = resolution, count = n * n, frames = 64;
			constexpr float tau = 6.28318530718f, dk = tau / 160.0f;
			constexpr float wind = 10.0f, age = 0.84f, peakK = 9.81f * age * age / (wind * wind);
			auto omega = [](float k) { return std::sqrt(9.81f * k * (1.0f + k * k / (370.0f * 370.0f))); };
			const float cp = omega(peakK) / peakK;
			const float friction = 0.41f * wind / std::log(10.0f / (0.000037f * wind * wind / 9.81f * std::pow(wind / cp, 0.9f)));
			const float sigma = 0.08f * (1.0f + 4.0f / (age * age * age));
			struct Mode { Complex seed; int rate; float x, y; };
			std::vector<Mode> modes(count);
			std::mt19937 random(48271);
			std::uniform_real_distribution<float> phase(0.0f, tau);
			for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x) {
				if ((!x && !y) || x == n / 2 || y == n / 2) continue;
				const float kx = (x < n / 2 ? x : x - n) * dk, ky = (y < n / 2 ? y : y - n) * dk;
				const float k = std::hypot(kx, ky), c = omega(k) / k, ratio = std::sqrt(k / peakK) - 1.0f;
				const float cutoff = std::exp(-1.25f * peakK * peakK / (k * k));
				const float peakShape = std::pow(1.7f, std::exp(-0.5f * ratio * ratio * sigma * sigma));
				const float gravity = 0.003f * std::sqrt(age) * cp / c * cutoff * peakShape * std::exp(-age * ratio / std::sqrt(10.0f));
				const float capillary = 0.005f * (1 + (friction >= 0.23f ? 3 : 1) * std::log(friction / 0.23f)) *
					0.23f / c * std::exp(-0.25f * std::pow(k / 370.0f - 1, 2.0f)) * cutoff;
				const float spread = std::tanh(std::log(2.0f) / 4 + 4 * std::pow(c / cp, 2.5f) + 0.13f * friction / 0.23f * std::pow(0.23f / c, 2.5f));
				const float density = (gravity + capillary) * (1 + spread * (2 * kx * kx / (k * k) - 1)) / (tau * k * k * k * k);
				modes[y * n + x] = {std::polar(std::sqrt(density * 0.5f) * dk, phase(random)), static_cast<int>(std::round(omega(k) * 16 / tau)), kx / k, ky / k};
			}

			std::array<Complex, n> roots;
			std::array<int, n> reverse{};
			for (int i = 0; i < n; ++i) {
				roots[i] = std::polar(1.0f, tau * i / n);
				reverse[i] = (reverse[i / 2] >> 1) | ((i & 1) * (n / 2));
			}
			auto inverse = [&](Complex* line, int stride) {
				for (int i = 0; i < n; ++i) if (i < reverse[i]) std::swap(line[i * stride], line[reverse[i] * stride]);
				for (int half = 1; half < n; half *= 2) {
					for (int start = 0; start < n; start += 2 * half) for (int j = 0; j < half; ++j) {
						auto& a = line[(start + j) * stride];
						auto& b = line[(start + j + half) * stride];
						const Complex odd = b * roots[j * n / (2 * half)];
						b = a - odd;
						a += odd;
					}
				}
			};
			std::vector<Complex> evolved(count), height(count), field(count);
			std::vector<float> samples(frames * 3 * count);
			float peak = 0.0f;
			for (int frame = 0; frame < frames; ++frame) {
				for (int i = 0; i < count; ++i) evolved[i] = modes[i].seed * std::conj(roots[(modes[i].rate * frame % frames) * (n / frames)]);
				for (int y = 0; y < n; ++y) for (int x = 0; x < n; ++x)
					height[y * n + x] = evolved[y * n + x] + std::conj(evolved[(n - y) % n * n + (n - x) % n]);
				for (int plane = 0; plane < 3; ++plane) {
					for (int i = 0; i < count; ++i) field[i] = plane == 0 ? height[i] : height[i] * Complex(0, plane == 1 ? modes[i].x : modes[i].y);
					for (int row = 0; row < n; ++row) inverse(field.data() + row * n, 1);
					for (int col = 0; col < n; ++col) inverse(field.data() + col, n);
					float* values = samples.data() + (frame * 3 + plane) * count;
					for (int i = 0; i < count; ++i) {
						values[i] = field[i].real();
						peak = std::max(peak, std::abs(values[i]));
					}
				}
			}
			const float scale = 127.5f / peak;
			auto encode = [scale](float value) { return static_cast<std::uint8_t>(std::clamp(value * scale + 128.0f, 0.0f, 255.0f)); };
			std::vector<std::uint8_t> pixels(tile * 24 * tile * 8 * 4);
			for (int frame = 0; frame < frames; ++frame) for (int plane = 0; plane < 3; ++plane) {
				const float* values = samples.data() + (frame * 3 + plane) * count;
				for (int y = 0; y < tile; ++y) for (int x = 0; x < tile; ++x) {
					const int col = (x + n - 1) % n, row = (y + n - 1) % n;
					const int pixel = ((frame / 8 * tile + y) * tile * 24 + (plane * 8 + frame % 8) * tile + x) * 4;
					pixels[pixel] = encode(values[row * n + col]);
					pixels[pixel + 1] = encode(values[row * n + (col + 1) % n] - values[row * n + (col + n - 1) % n]);
					pixels[pixel + 2] = encode(values[(row + 1) % n * n + col] - values[(row + n - 1) % n * n + col]);
					pixels[pixel + 3] = 255;
				}
			}
			return pixels;
		}

		public:
			using Object::Object;
			float choppiness = 1.0f;
			float size = 160.0f;
			float direction = 0.0f;
			Vector4 color{0.01f, 0.06f, 0.18f, 0.5f};
			bool foam = true;

			void Initialize() override {
				if (!std::isfinite(size) || size <= 0.0f || !std::isfinite(choppiness) || !std::isfinite(direction) || !std::isfinite(color.w)) return;
				if (!std::isfinite(transform.scale.x) || transform.scale.x == 0.0f || !std::isfinite(transform.scale.y) || transform.scale.y == 0.0f || !std::isfinite(transform.scale.z) || transform.scale.z < 0.0f) return;
				const float patch = std::max(size, 0.01f);
				const float chop = std::clamp(choppiness, 0.0f, 2.5f);
				const float angle = direction * Math::degreesToRadians;
				const float windX = std::cos(angle), windY = std::sin(angle);
				auto divisions = [&](float extent) {
					return static_cast<int>(std::clamp(std::ceil(std::abs(extent) / patch * 48.0f), 1.0f, 448.0f));
				};
				const int columns = divisions(transform.scale.x);
				const int rows = divisions(transform.scale.y);
				const float surfaceOpacity = std::clamp(color.w, 0.0f, 1.0f);
				const int rowSize = columns + 1;
				MeshData data;
				data.vertices.reserve(static_cast<std::size_t>(rowSize) * (rows + 1));
				data.indices.reserve(static_cast<std::size_t>(columns) * rows * 6);
				for (int y = 0; y <= rows; ++y) {
					for (int x = 0; x <= columns; ++x) {
						const float u = static_cast<float>(x) / columns;
						const float v = static_cast<float>(y) / rows;
						data.vertices.push_back({{u - 0.5f, v - 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {u, v}, {1.0f, 1.0f, 1.0f, surfaceOpacity}});
					}
				}
				for (int y = 0; y < rows; ++y) {
					for (int x = 0; x < columns; ++x) {
						const std::uint32_t first = y * rowSize + x;
						const std::uint32_t next = first + rowSize;
						data.indices.insert(data.indices.end(), {first, next + 1, next, first, first + 1, next + 1});
					}
				}
				auto& mesh = AddComponent<Mesh>();
				mesh.data = std::move(data);
				mesh.material.transparent = surfaceOpacity < 1.0f;
				mesh.material.singleLayer = true;
				mesh.material.baseColor = {color.x, color.y, color.z};
				mesh.material.roughness = 0.20f;
				mesh.material.depthSurfaceSource = Graphics::Shaders::waterSurfaceSource;
				Graphics::EnsureGeneratedTexture("generated:water-spectrum", tile * 24, tile * 8, BakeSpectrum);
				mesh.material.displacementTexturePath = "generated:water-spectrum";
				mesh.material.normalTexturePath = mesh.material.displacementTexturePath;
				mesh.material.displacementScale = 0.0f;
				mesh.material.normalScale = 0.0f;
				mesh.material.customData = {0.35f, 8.0f, foam ? 1.0f : 0.0f, 0.0f, windX, windY, chop};
				mesh.material.vertexModifierSource = Graphics::Shaders::waterVertexModifierSource;
				auto& vertexSource = mesh.material.vertexModifierSource;
				auto replaceMarker = [&](const char* marker, float value) {
					const std::string replacement = std::to_string(value);
					const auto length = std::char_traits<char>::length(marker);
					for (auto position = vertexSource.find(marker); position != std::string::npos; position = vertexSource.find(marker, position + replacement.size())) {
						vertexSource.replace(position, length, replacement);
					}
				};
				replaceMarker("PATCH_METRES", patch);
				replaceMarker("CHOP", chop);
				replaceMarker("WIND_X", windX);
				replaceMarker("WIND_Y", windY);
				mesh.vertexAnimated = true;
				const float padding = 0.707107f * chop * transform.scale.z;
				mesh.vertexBoundsPadding = {padding / std::abs(transform.scale.x), padding / std::abs(transform.scale.y), 0.5f};
				mesh.castShadows = false;
			}
	};


	namespace Physics { std::vector<Vector3> ColliderDebugLines(Scene& scene); }

	namespace {
		inline constexpr std::array<std::string Material::*, 6> materialTexturePaths{
			&Material::baseTexturePath, &Material::normalTexturePath, &Material::roughnessTexturePath,
			&Material::aoTexturePath, &Material::displacementTexturePath, &Material::displacementMaskTexturePath
		};

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
				for (const auto path : materialTexturePaths) material.*path = reader.String();
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
			Mesh::ValidateSections(record.data.indices.size(), record.materials.size(), record.sections);
			if (!record.bones.empty()) {
				Skeleton skeleton;
				skeleton.bones = std::move(record.bones);
				if (!record.data.skinWeights.empty()) record.data.ValidateSkeleton(skeleton);
				else skeleton.Validate();
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
					record.light.castShadows = reader.Read<std::uint8_t>() != 0;
					reader.Read<float>(); // Consume the stored bias; loaded lights use the current Light default.
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
				for (const auto path : materialTexturePaths) {
					auto& value = material.*path;
					if (!value.empty() && std::filesystem::path(value).is_relative()) value = (sourcePath.parent_path() / value).lexically_normal().string();
				}
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
