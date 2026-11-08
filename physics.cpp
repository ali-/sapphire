#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace Sapphire {

	class PhysicsBody : public Component {
		public:
			Vector3 velocity;
			Vector3 previousPosition;
			bool useGravity = true;
			bool grounded = false;
			float slopeLimit = 45.0f;
			float stepHeight = 0.3f;
			float SlopeCosine() const {
				if (cachedSlope != slopeLimit) {
					cachedSlope = slopeLimit;
					cachedCosine = std::cos(slopeLimit * Math::degreesToRadians);
				}
				return cachedCosine;
			}

		private:
			mutable float cachedSlope = std::numeric_limits<float>::quiet_NaN();
			mutable float cachedCosine = 0.0f;
	};

	class Collider : public Component {
		public:
			std::variant<std::monostate, Primitive, MeshData> mesh;
			Vector3 localPosition;
			Vector3 localRotation;
			Vector3 size{};
	};
	struct RaycastHit {
		Object* object = nullptr;
		Collider* collider = nullptr;
		Vector3 point;
		Vector3 normal;
		float distance = 0.0f;
	};

	namespace Physics {
		void EnsureMeshCollider(Object& object) {
			if (!object.GetComponent<Collider>()) object.AddComponent<Collider>();
		}

		namespace {
			using namespace Math;

			constexpr float skinWidth = 0.005f;
			constexpr float floorProbeDistance = 0.05f;
			constexpr float axisEpsilon = 0.000001f;
			constexpr int maxMoveIterations = 5;
			constexpr int maxOverlapIterations = 5;
			constexpr std::size_t trianglesPerLeaf = 8;

			struct Box {
				Vector3 center;
				Vector3 axes[3];
				float halfSize[3];
			};

			struct Capsule {
				Vector3 start;
				Vector3 end;
				float radius;
			};

			struct Triangle {
				std::uint32_t indices[3];
			};

			struct BvhNode {
				Bounds bounds;
				std::uint32_t first = 0;
				std::uint32_t count = 0;
				int left = -1;
				int right = -1;
			};


			struct MeshBounds {
				Bounds bounds;
				MeshRevision revision;
				std::uint64_t generation = 0;
			};

			struct MeshTree {
				MeshRevision revision;
				std::uint64_t signature = 0;
				bool ownerSource = false;
				std::uint64_t generation = 0;
				std::uint64_t checkedGeneration = 0;
				std::vector<Triangle> triangles;
				std::vector<std::uint32_t> order;
				std::vector<BvhNode> nodes;
				float winding = 1.0f;
				bool closed = false;
			};

			struct Contact {
				float time = 1.0f;
				float penetration = 0.0f;
				Vector3 normal;
				Vector3 surfaceNormal;
				Vector3 point;
				bool hit = false;
			};

			bool IsWalkable(const PhysicsBody& body, const Contact& contact) {
				return contact.hit && contact.surfaceNormal.z >= body.SlopeCosine();
			}

			bool IsCapsuleFootContact(const Capsule& capsule, const Contact& contact) {
				const float x = contact.point.x - capsule.start.x;
				const float y = contact.point.y - capsule.start.y;
				const float radius = std::max(0.0f, capsule.radius - skinWidth);
				return contact.hit && contact.point.z <= capsule.start.z && x * x + y * y <= radius * radius;
			}

			bool IsCapsuleSupport(const Capsule& capsule, const PhysicsBody& body, const Contact& contact) {
				return IsWalkable(body, contact) && IsCapsuleFootContact(capsule, contact);
			}

			bool PreferCapsuleHit(const Contact& candidate, const Contact& current, const Capsule& capsule, float displacementSquared, const PhysicsBody* floorBody) {
				if (!current.hit) return true;
				if (floorBody && (candidate.time - current.time) * (candidate.time - current.time) * displacementSquared <= axisEpsilon * axisEpsilon) {
					const bool candidateSupport = IsCapsuleSupport(capsule, *floorBody, candidate);
					const bool currentSupport = IsCapsuleSupport(capsule, *floorBody, current);
					if (candidateSupport != currentSupport) return candidateSupport;
					if (candidateSupport) return candidate.surfaceNormal.z > current.surfaceNormal.z;
				}
				return candidate.time < current.time;
			}

			std::unordered_map<const Mesh*, MeshBounds> meshBounds;
			std::vector<std::uint32_t> triangleCandidates;
			std::uint64_t cacheGeneration = 0;
			std::unordered_map<const MeshData*, MeshTree> meshTrees;
			enum class CollisionType { None, Cube, Capsule, Mesh };

			struct ColliderRecord {
				Object* object;
				Collider* collider;
				PhysicsBody* body;
				Box box;
				Capsule capsule;
				Bounds bounds;
				Vector3 axes[3]{};
				CollisionType type = CollisionType::None;
				const MeshData* data = nullptr;
				Transform transform;
				Mesh* ownerMesh = nullptr;
			};
			std::vector<ColliderRecord> colliderRecords;
			struct BodyRecord {
				Object* object;
				PhysicsBody* body;
				Collider* collider;
			};
			std::vector<BodyRecord> bodyRecords;

			const Bounds& GetMeshBounds(const Mesh& mesh) {
				auto [entry, inserted] = meshBounds.try_emplace(&mesh);
				if (inserted || !entry->second.revision.Matches(mesh)) {
					entry->second.bounds = EmptyBounds();
					for (const auto& vertex : mesh.data.vertices) Include(entry->second.bounds, vertex.position);
					entry->second.revision.Set(mesh);
				}
				entry->second.generation = cacheGeneration;
				return entry->second.bounds;
			}

			const MeshData* CollisionMesh(Object& object, const Collider& collider) {
				if (const auto* primitive = std::get_if<Primitive>(&collider.mesh)) {
					if (primitive->shape != Primitive::Shape::Plane) return nullptr;
					static const MeshData plane = Primitive::Plane.Mesh();
					return &plane;
				}
				if (const auto* data = std::get_if<MeshData>(&collider.mesh)) return data;
				if (auto* mesh = object.GetComponent<Mesh>()) return &mesh->data;
				return nullptr;
			}

			CollisionType ColliderType(Object& object, const Collider& collider) {
				if (const auto* primitive = std::get_if<Primitive>(&collider.mesh)) {
					if (primitive->shape == Primitive::Shape::Cube) return CollisionType::Cube;
					if (primitive->shape == Primitive::Shape::Capsule) return CollisionType::Capsule;
					return CollisionType::Mesh;
				}
				const MeshData* data = CollisionMesh(object, collider);
				return data && !data->vertices.empty() ? CollisionType::Mesh : CollisionType::None;
			}

			Bounds DataBounds(const MeshData& data) {
				Bounds bounds = EmptyBounds();
				for (const auto& vertex : data.vertices) Include(bounds, vertex.position);
				return bounds;
			}

			Vector3 ColliderSize(Object& object, const Collider& collider, const MeshData* source = nullptr) {
				for (float value : {collider.size.x, collider.size.y, collider.size.z}) if (!std::isfinite(value) || value < 0.0f) throw std::invalid_argument("Collider size must be finite and nonnegative");
				const Mesh* ownerMesh = object.GetComponent<Mesh>();
				const MeshData* reference = ownerMesh && !ownerMesh->data.vertices.empty() ? &ownerMesh->data : source;
				Vector3 dimensions{1.0f, 1.0f, 1.0f};
				if (reference && !reference->vertices.empty()) {
					const Bounds bounds = ownerMesh && reference == &ownerMesh->data ? GetMeshBounds(*ownerMesh) : DataBounds(*reference);
					dimensions = Subtract(bounds.maximum, bounds.minimum);
				}
				const Vector3& scale = object.transform.scale;
				return {collider.size.x > 0.0f ? collider.size.x : dimensions.x * std::abs(scale.x), collider.size.y > 0.0f ? collider.size.y : dimensions.y * std::abs(scale.y), collider.size.z > 0.0f ? collider.size.z : dimensions.z * std::abs(scale.z)};
			}

			Transform ColliderPose(Object& object, const Collider& collider) {
				Transform pose = object.transform;
				Vector3 axes[3];
				GetAxes(object.transform, axes);
				const Vector3 offset{collider.localPosition.x * object.transform.scale.x, collider.localPosition.y * object.transform.scale.y, collider.localPosition.z * object.transform.scale.z};
				pose.position = Add(object.transform.position, Add(Add(Scale(axes[0], offset.x), Scale(axes[1], offset.y)), Scale(axes[2], offset.z)));
				pose.rotation = ComposeRotation(object.transform.rotation, collider.localRotation);
				return pose;
			}

			Transform CollisionMeshTransform(Object& object, const Collider& collider, const MeshData& data) {
				Transform pose = ColliderPose(object, collider);
				const Mesh* ownerMesh = object.GetComponent<Mesh>();
				const Bounds bounds = ownerMesh && &ownerMesh->data == &data ? GetMeshBounds(*ownerMesh) : DataBounds(data);
				const Vector3 sourceSize = Subtract(bounds.maximum, bounds.minimum);
				const Vector3 targetSize = ColliderSize(object, collider, &data);
				const Vector3& ownerScale = object.transform.scale;
				pose.scale = {sourceSize.x > 0.0f ? std::copysign(targetSize.x / sourceSize.x, ownerScale.x) : ownerScale.x, sourceSize.y > 0.0f ? std::copysign(targetSize.y / sourceSize.y, ownerScale.y) : ownerScale.y, sourceSize.z > 0.0f ? std::copysign(targetSize.z / sourceSize.z, ownerScale.z) : ownerScale.z};
				return pose;
			}

			Box WorldBox(Object& object, const Collider& collider) {
				const Transform transform = ColliderPose(object, collider);
				Box box;
				GetAxes(transform, box.axes);
				const Vector3 size = ColliderSize(object, collider, CollisionMesh(object, collider));
				box.center = transform.position;
				box.halfSize[0] = size.x * 0.5f;
				box.halfSize[1] = size.y * 0.5f;
				box.halfSize[2] = size.z * 0.5f;
				return box;
			}

			Capsule WorldCapsule(Object& object, const Collider& collider) {
				const Transform transform = ColliderPose(object, collider);
				Vector3 axes[3];
				GetAxes(transform, axes);
				const Vector3 size = ColliderSize(object, collider);
				const float height = size.z;
				const float radius = Primitive::CapsuleRadius(size);
				const Vector3 segment = Scale(axes[2], std::max(0.0f, height * 0.5f - radius));
				Vector3 start = Subtract(transform.position, segment), end = Add(transform.position, segment);
				if (start.z > end.z) std::swap(start, end);
				return {start, end, radius};
			}

			Bounds BoxBounds(const Box& box) {
				const Vector3 extent{box.halfSize[0] * std::abs(box.axes[0].x) + box.halfSize[1] * std::abs(box.axes[1].x) + box.halfSize[2] * std::abs(box.axes[2].x), box.halfSize[0] * std::abs(box.axes[0].y) + box.halfSize[1] * std::abs(box.axes[1].y) + box.halfSize[2] * std::abs(box.axes[2].y), box.halfSize[0] * std::abs(box.axes[0].z) + box.halfSize[1] * std::abs(box.axes[1].z) + box.halfSize[2] * std::abs(box.axes[2].z)};
				return {{box.center.x - extent.x, box.center.y - extent.y, box.center.z - extent.z}, {box.center.x + extent.x, box.center.y + extent.y, box.center.z + extent.z}};
			}

			Bounds CapsuleBounds(const Capsule& capsule) {
				const Vector3 extent{capsule.radius, capsule.radius, capsule.radius};
				return {{std::min(capsule.start.x, capsule.end.x) - extent.x, std::min(capsule.start.y, capsule.end.y) - extent.y, std::min(capsule.start.z, capsule.end.z) - extent.z}, {std::max(capsule.start.x, capsule.end.x) + extent.x, std::max(capsule.start.y, capsule.end.y) + extent.y, std::max(capsule.start.z, capsule.end.z) + extent.z}};
			}

			void MoveObject(Object& object, const Vector3& offset) { object.SetWorldPosition(Add(object.transform.position, offset)); }

			bool IsDefaultOwnerCollider(const Collider& collider) {
				return std::holds_alternative<std::monostate>(collider.mesh) && collider.size.x == 0.0f && collider.size.y == 0.0f && collider.size.z == 0.0f && collider.localPosition.x == 0.0f && collider.localPosition.y == 0.0f && collider.localPosition.z == 0.0f && collider.localRotation.x == 0.0f && collider.localRotation.y == 0.0f && collider.localRotation.z == 0.0f;
			}

			template <typename T>
			void PruneCache(T& cache) {
				for (auto entry = cache.begin(); entry != cache.end();) {
					if (entry->second.generation != cacheGeneration) entry = cache.erase(entry);
					else ++entry;
				}
			}

			void SyncColliders(Scene& scene) {
				scene.UpdateWorldPartition();
				++cacheGeneration;
				colliderRecords.clear();
				bodyRecords.clear();
				for (auto* object : scene.ResidentObjects()) {
					Collider* collider = object->GetComponent<Collider>();
					if (object->worldPartitionCell && object->worldPartitionSourceObject) {
						if (auto* sourceCollider = object->worldPartitionSourceObject->GetComponent<Collider>(); sourceCollider && IsDefaultOwnerCollider(*sourceCollider)) {
							if (!collider) collider = &object->AddComponent<Collider>();
						}
						else continue;
					}
					if (object->worldPartitionedMesh && collider && IsDefaultOwnerCollider(*collider)) collider = nullptr;
					auto* body = object->GetComponent<PhysicsBody>();
					if (body) bodyRecords.push_back({object, body, collider && ColliderType(*object, *collider) != CollisionType::None ? collider : nullptr});
					if (!collider) continue;
					ColliderRecord record{object, collider, body, {}, {}, {}};
					record.type = ColliderType(*object, *collider);
					if (record.type == CollisionType::None) continue;
					if (record.type == CollisionType::Capsule) {
						record.capsule = WorldCapsule(*object, *collider);
						record.bounds = CapsuleBounds(record.capsule);
					}
					else if (record.type == CollisionType::Mesh) {
						record.data = CollisionMesh(*object, *collider);
						if (!record.data || record.data->vertices.empty()) continue;
						record.ownerMesh = object->GetComponent<Mesh>();
						if (!std::holds_alternative<std::monostate>(collider->mesh)) record.ownerMesh = nullptr;
						if (auto tree = meshTrees.find(record.data); tree != meshTrees.end()) tree->second.generation = cacheGeneration;
						record.transform = CollisionMeshTransform(*object, *collider, *record.data);
						GetAxes(record.transform, record.axes);
						record.bounds = TransformBounds(record.ownerMesh ? GetMeshBounds(*record.ownerMesh) : DataBounds(*record.data), Model(record.transform, record.axes));
					}
					else {
						record.box = WorldBox(*object, *collider);
						record.bounds = BoxBounds(record.box);
					}
					colliderRecords.push_back(record);
				}
				PruneCache(meshBounds);
				PruneCache(meshTrees);
			}

			Bounds SweptBounds(const Capsule& capsule, const Vector3& displacement) {
				Bounds bounds = CapsuleBounds(capsule);
				const Vector3 margin{skinWidth, skinWidth, skinWidth};
				bounds.minimum = Subtract(bounds.minimum, margin);
				bounds.maximum = Add(bounds.maximum, margin);
				bounds.minimum.x += std::min(0.0f, displacement.x);
				bounds.minimum.y += std::min(0.0f, displacement.y);
				bounds.minimum.z += std::min(0.0f, displacement.z);
				bounds.maximum.x += std::max(0.0f, displacement.x);
				bounds.maximum.y += std::max(0.0f, displacement.y);
				bounds.maximum.z += std::max(0.0f, displacement.z);
				return bounds;
			}

			Bounds SweptBounds(const Box& box, const Vector3& displacement) {
				Bounds bounds = BoxBounds(box);
				bounds.minimum.x += std::min(0.0f, displacement.x);
				bounds.minimum.y += std::min(0.0f, displacement.y);
				bounds.minimum.z += std::min(0.0f, displacement.z);
				bounds.maximum.x += std::max(0.0f, displacement.x);
				bounds.maximum.y += std::max(0.0f, displacement.y);
				bounds.maximum.z += std::max(0.0f, displacement.z);
				return bounds;
			}

			float Radius(const Box& box, const Vector3& axis) {
				return box.halfSize[0] * std::abs(Dot(box.axes[0], axis)) + box.halfSize[1] * std::abs(Dot(box.axes[1], axis)) + box.halfSize[2] * std::abs(Dot(box.axes[2], axis));
			}

			Vector3 ClosestPointOnTriangle(const Vector3& point, const Vector3 (&triangle)[3]) {
				const Vector3 ab = Subtract(triangle[1], triangle[0]);
				const Vector3 ac = Subtract(triangle[2], triangle[0]);
				const Vector3 ap = Subtract(point, triangle[0]);
				const float d1 = Dot(ab, ap), d2 = Dot(ac, ap);
				if (d1 <= 0.0f && d2 <= 0.0f) return triangle[0];
				const Vector3 bp = Subtract(point, triangle[1]);
				const float d3 = Dot(ab, bp), d4 = Dot(ac, bp);
				if (d3 >= 0.0f && d4 <= d3) return triangle[1];
				const float vc = d1 * d4 - d3 * d2;
				if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) return Add(triangle[0], Scale(ab, d1 / (d1 - d3)));
				const Vector3 cp = Subtract(point, triangle[2]);
				const float d5 = Dot(ab, cp), d6 = Dot(ac, cp);
				if (d6 >= 0.0f && d5 <= d6) return triangle[2];
				const float vb = d5 * d2 - d1 * d6;
				if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) return Add(triangle[0], Scale(ac, d2 / (d2 - d6)));
				const float va = d3 * d6 - d5 * d4;
				if (va <= 0.0f && d4 - d3 >= 0.0f && d5 - d6 >= 0.0f) return Add(triangle[1], Scale(Subtract(triangle[2], triangle[1]), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
				const float inverse = 1.0f / (va + vb + vc);
				return Add(triangle[0], Add(Scale(ab, vb * inverse), Scale(ac, vc * inverse)));
			}

			void ClosestPointsOnSegments(const Vector3& firstStart, const Vector3& firstEnd, const Vector3& secondStart, const Vector3& secondEnd, Vector3& firstPoint, Vector3& secondPoint) {
				const Vector3 firstDirection = Subtract(firstEnd, firstStart), secondDirection = Subtract(secondEnd, secondStart), between = Subtract(firstStart, secondStart);
				const float a = Dot(firstDirection, firstDirection), e = Dot(secondDirection, secondDirection), f = Dot(secondDirection, between);
				float firstParameter = 0.0f, secondParameter = 0.0f;
				if (a <= axisEpsilon && e <= axisEpsilon) {
					firstPoint = firstStart;
					secondPoint = secondStart;
					return;
				}
				if (a <= axisEpsilon) secondParameter = std::clamp(f / e, 0.0f, 1.0f);
				else {
					const float c = Dot(firstDirection, between);
					if (e <= axisEpsilon) firstParameter = std::clamp(-c / a, 0.0f, 1.0f);
					else {
						const float b = Dot(firstDirection, secondDirection), denominator = a * e - b * b;
						if (denominator != 0.0f) firstParameter = std::clamp((b * f - c * e) / denominator, 0.0f, 1.0f);
						secondParameter = (b * firstParameter + f) / e;
						if (secondParameter < 0.0f) {
							secondParameter = 0.0f;
						firstParameter = std::clamp(-c / a, 0.0f, 1.0f);
						}
						else if (secondParameter > 1.0f) {
							secondParameter = 1.0f;
						firstParameter = std::clamp((b - c) / a, 0.0f, 1.0f);
						}
					}
				}
				firstPoint = Add(firstStart, Scale(firstDirection, firstParameter));
				secondPoint = Add(secondStart, Scale(secondDirection, secondParameter));
			}

			void ClosestPointsOnSegmentTriangle(const Vector3& start, const Vector3& end, const Vector3 (&triangle)[3], Vector3& segmentPoint, Vector3& trianglePoint) {
				const Vector3 direction = Subtract(end, start), edge0 = Subtract(triangle[1], triangle[0]), edge1 = Subtract(triangle[2], triangle[0]);
				const Vector3 cross = Cross(direction, edge1);
				const float determinant = Dot(edge0, cross);
				if (std::abs(determinant) > axisEpsilon) {
					const float inverse = 1.0f / determinant;
					const Vector3 fromStart = Subtract(start, triangle[0]);
					const float u = Dot(fromStart, cross) * inverse;
					const Vector3 q = Cross(fromStart, edge0);
					const float v = Dot(direction, q) * inverse;
					const float t = Dot(edge1, q) * inverse;
					if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f && t >= 0.0f && t <= 1.0f) {
						segmentPoint = Add(start, Scale(direction, t));
						trianglePoint = segmentPoint;
						return;
					}
				}
				segmentPoint = start;
				trianglePoint = ClosestPointOnTriangle(start, triangle);
				float minimum = Dot(Subtract(segmentPoint, trianglePoint), Subtract(segmentPoint, trianglePoint));
				Vector3 candidateSegment = end, candidateTriangle = ClosestPointOnTriangle(end, triangle);
				float distance = Dot(Subtract(candidateSegment, candidateTriangle), Subtract(candidateSegment, candidateTriangle));
				if (distance < minimum) { minimum = distance; segmentPoint = candidateSegment; trianglePoint = candidateTriangle; }
				for (int edge = 0; edge < 3; ++edge) {
					ClosestPointsOnSegments(start, end, triangle[edge], triangle[(edge + 1) % 3], candidateSegment, candidateTriangle);
					distance = Dot(Subtract(candidateSegment, candidateTriangle), Subtract(candidateSegment, candidateTriangle));
					if (distance < minimum) { minimum = distance; segmentPoint = candidateSegment; trianglePoint = candidateTriangle; }
				}
			}

			void GetBoxTriangles(const Box& box, Vector3 (&triangles)[12][3]) {
				Vector3 corners[8];
				for (int corner = 0; corner < 8; ++corner) corners[corner] = Add(box.center, Add(Scale(box.axes[0], corner & 1 ? box.halfSize[0] : -box.halfSize[0]), Add(Scale(box.axes[1], corner & 2 ? box.halfSize[1] : -box.halfSize[1]), Scale(box.axes[2], corner & 4 ? box.halfSize[2] : -box.halfSize[2]))));
				const int indices[12][3] = {{1, 3, 7}, {1, 7, 5}, {0, 4, 6}, {0, 6, 2}, {2, 6, 7}, {2, 7, 3}, {0, 1, 5}, {0, 5, 4}, {4, 5, 7}, {4, 7, 6}, {0, 2, 3}, {0, 3, 1}};
				for (int index = 0; index < 12; ++index) for (int vertex = 0; vertex < 3; ++vertex) triangles[index][vertex] = corners[indices[index][vertex]];
			}

			int BuildNode(MeshTree& tree, const std::vector<Bounds>& bounds, const std::vector<Vector3>& centersData, std::uint32_t first, std::uint32_t count) {
				const int nodeIndex = static_cast<int>(tree.nodes.size());
				tree.nodes.emplace_back();
				BvhNode node;
				node.bounds = EmptyBounds();
				Bounds centers = EmptyBounds();
				for (std::uint32_t index = first; index < first + count; ++index) {
					const auto triangle = tree.order[index];
					Include(node.bounds, bounds[triangle].minimum);
					Include(node.bounds, bounds[triangle].maximum);
					Include(centers, centersData[triangle]);
				}
				if (count <= trianglesPerLeaf) {
					node.first = first;
					node.count = count;
					tree.nodes[nodeIndex] = node;
					return nodeIndex;
				}
				const Vector3 extent{centers.maximum.x - centers.minimum.x, centers.maximum.y - centers.minimum.y, centers.maximum.z - centers.minimum.z};
				int axis = extent.y > extent.x ? 1 : 0;
				if ((axis == 0 ? extent.x : extent.y) < extent.z) axis = 2;
				const std::uint32_t middle = first + count / 2;
				std::nth_element(tree.order.begin() + first, tree.order.begin() + middle, tree.order.begin() + first + count, [&](std::uint32_t a, std::uint32_t b) {
					const Vector3& left = centersData[a];
					const Vector3& right = centersData[b];
					return axis == 0 ? left.x < right.x : axis == 1 ? left.y < right.y : left.z < right.z;
				});
				node.left = BuildNode(tree, bounds, centersData, first, middle - first);
				node.right = BuildNode(tree, bounds, centersData, middle, first + count - middle);
				tree.nodes[nodeIndex] = node;
				return nodeIndex;
			}

			struct PositionHash {
				std::size_t operator()(const std::array<float, 3>& position) const {
					std::size_t hash = 0;
					for (float value : position) hash ^= std::hash<float>{}(value) + 0x9e3779b9u + (hash << 6) + (hash >> 2);
					return hash;
				}
			};

			MeshTree BuildTree(const MeshData& data, const Mesh* ownerMesh) {
				data.Validate();
				MeshTree tree;
				if (ownerMesh) tree.revision.Set(*ownerMesh);
				tree.ownerSource = ownerMesh != nullptr;
				const std::size_t triangleCount = data.indices.size() / 3;
				tree.triangles.reserve(triangleCount);
				tree.order.reserve(triangleCount);
				tree.nodes.reserve(triangleCount / 2 + 1);
				std::vector<Bounds> bounds;
				std::vector<Vector3> centers;
				bounds.reserve(triangleCount);
				centers.reserve(triangleCount);
				float signedVolume = 0.0f;
				std::unordered_map<std::array<float, 3>, std::uint32_t, PositionHash> positionIds;
				positionIds.reserve(data.vertices.size());
				std::vector<std::uint32_t> vertexIds;
				vertexIds.reserve(data.vertices.size());
				for (const auto& vertex : data.vertices) {
					const auto& point = vertex.position;
					vertexIds.push_back(positionIds.try_emplace({point.x, point.y, point.z}, static_cast<std::uint32_t>(positionIds.size())).first->second);
				}
				struct EdgeUse {
					unsigned count = 0;
					int direction = 0;
				};
				std::unordered_map<std::uint64_t, EdgeUse> edges;
				edges.reserve(triangleCount * 2);
				for (std::size_t index = 0; index + 2 < data.indices.size(); index += 3) {
					const Vector3& a = data.vertices[data.indices[index]].position;
					const Vector3& b = data.vertices[data.indices[index + 1]].position;
					const Vector3& c = data.vertices[data.indices[index + 2]].position;
					const Vector3 edgeA = Subtract(b, a);
					const Vector3 edgeB = Subtract(c, a);
					if (Length(Cross(edgeA, edgeB)) <= axisEpsilon * Length(edgeA) * Length(edgeB)) continue;
					for (int corner = 0; corner < 3; ++corner) {
						const auto first = vertexIds[data.indices[index + corner]];
						const auto second = vertexIds[data.indices[index + (corner + 1) % 3]];
						const std::uint64_t key = (static_cast<std::uint64_t>(std::min(first, second)) << 32) | std::max(first, second);
						auto& edge = edges[key];
						++edge.count;
						edge.direction += first < second ? 1 : -1;
					}
					Triangle triangle{};
					triangle.indices[0] = data.indices[index];
					triangle.indices[1] = data.indices[index + 1];
					triangle.indices[2] = data.indices[index + 2];
					Bounds triangleBounds = EmptyBounds();
					Include(triangleBounds, a);
					Include(triangleBounds, b);
					Include(triangleBounds, c);
					bounds.push_back(triangleBounds);
					centers.push_back(Scale(Add(Add(a, b), c), 1.0f / 3.0f));
					tree.triangles.push_back(triangle);
					tree.order.push_back(static_cast<std::uint32_t>(tree.triangles.size() - 1));
					signedVolume += Dot(a, Cross(b, c)) / 6.0f;
				}
				tree.closed = !edges.empty() && std::all_of(edges.begin(), edges.end(), [](const auto& edge) { return edge.second.count == 2 && edge.second.direction == 0; });
				tree.winding = tree.closed && signedVolume < 0.0f ? -1.0f : 1.0f;
				if (!tree.triangles.empty()) BuildNode(tree, bounds, centers, 0, static_cast<std::uint32_t>(tree.triangles.size()));
				return tree;
			}

			std::uint64_t GeometrySignature(const MeshData& data) {
				std::uint64_t hash = 1469598103934665603ull;
				const auto add = [&](std::uint64_t value) {
					hash = (hash ^ value) * 1099511628211ull;
				};
				add(data.vertices.size());
				add(data.indices.size());
				for (const auto& vertex : data.vertices) {
					add(std::hash<float>{}(vertex.position.x));
					add(std::hash<float>{}(vertex.position.y));
					add(std::hash<float>{}(vertex.position.z));
				}
				for (auto index : data.indices) add(index);
				return hash;
			}

			MeshTree& GetTree(const MeshData& data, const Mesh* ownerMesh = nullptr) {
				auto [entry, inserted] = meshTrees.try_emplace(&data);
				if (!ownerMesh && !inserted && !entry->second.ownerSource && entry->second.checkedGeneration == cacheGeneration) return entry->second;
				const std::uint64_t signature = ownerMesh ? 0 : GeometrySignature(data);
				const bool changed = ownerMesh ? !entry->second.ownerSource || !entry->second.revision.Matches(*ownerMesh) : entry->second.ownerSource || entry->second.signature != signature;
				if (inserted || changed) {
					entry->second = BuildTree(data, ownerMesh);
					if (!ownerMesh) entry->second.signature = signature;
				}
				entry->second.generation = cacheGeneration;
				entry->second.checkedGeneration = cacheGeneration;
				return entry->second;
			}

			void QueryNode(const MeshTree& tree, int nodeIndex, const Bounds& queryBounds, std::vector<std::uint32_t>& output) {
				const BvhNode& node = tree.nodes[nodeIndex];
				if (!Overlaps(node.bounds, queryBounds)) return;
				if (node.count) {
					for (std::uint32_t index = node.first; index < node.first + node.count; ++index) output.push_back(tree.order[index]);
					return;
				}
				QueryNode(tree, node.left, queryBounds, output);
				QueryNode(tree, node.right, queryBounds, output);
			}

			void QueryTree(const MeshTree& tree, const Transform& transform, const Bounds& queryBounds, const Vector3 (&axes)[3], std::vector<std::uint32_t>& output) {
				output.clear();
				if (tree.nodes.empty()) return;
				if (std::abs(transform.scale.x) < axisEpsilon || std::abs(transform.scale.y) < axisEpsilon || std::abs(transform.scale.z) < axisEpsilon) {
					output.assign(tree.order.begin(), tree.order.end());
					return;
				}
				const Vector3 center = Subtract(Scale(Add(queryBounds.minimum, queryBounds.maximum), 0.5f), transform.position);
				const Vector3 extent = Scale(Subtract(queryBounds.maximum, queryBounds.minimum), 0.5f);
				const Vector3 localCenter{Dot(center, axes[0]) / transform.scale.x, Dot(center, axes[1]) / transform.scale.y, Dot(center, axes[2]) / transform.scale.z};
				const auto radius = [&](const Vector3& axis, float scale) { return (std::abs(axis.x) * extent.x + std::abs(axis.y) * extent.y + std::abs(axis.z) * extent.z) / std::abs(scale); };
				const Vector3 localExtent{radius(axes[0], transform.scale.x), radius(axes[1], transform.scale.y), radius(axes[2], transform.scale.z)};
				Bounds localBounds{Subtract(localCenter, localExtent), Add(localCenter, localExtent)};
				const Vector3 padding{axisEpsilon, axisEpsilon, axisEpsilon};
				localBounds.minimum = Subtract(localBounds.minimum, padding);
				localBounds.maximum = Add(localBounds.maximum, padding);
				QueryNode(tree, 0, localBounds, output);
			}

			float RayBounds(const Bounds& bounds, const Vector3& origin, const Vector3& direction, float distance) {
				float enter = 0.0f;
				const float positions[3] = {origin.x, origin.y, origin.z};
				const float velocities[3] = {direction.x, direction.y, direction.z};
				const float minimum[3] = {bounds.minimum.x, bounds.minimum.y, bounds.minimum.z};
				const float maximum[3] = {bounds.maximum.x, bounds.maximum.y, bounds.maximum.z};
				for (int axis = 0; axis < 3; ++axis) {
					if (velocities[axis] == 0.0f) {
						if (positions[axis] < minimum[axis] || positions[axis] > maximum[axis]) return -1.0f;
						continue;
					}
					float near = (minimum[axis] - positions[axis]) / velocities[axis];
					float far = (maximum[axis] - positions[axis]) / velocities[axis];
					if (near > far) std::swap(near, far);
					enter = std::max(enter, near);
					distance = std::min(distance, far);
					if (enter > distance) return -1.0f;
				}
				return enter;
			}

			bool RayNode(const MeshTree& tree, const MeshData& data, int index, const Vector3& origin, const Vector3& ray, float rayLength, float& distance, Vector3& normal) {
				const auto& node = tree.nodes[index];
				if (RayBounds(node.bounds, origin, ray, distance) < 0.0f) return false;
				bool found = false;
				if (node.count) {
					for (std::uint32_t entry = node.first; entry < node.first + node.count; ++entry) {
						const auto& triangle = tree.triangles[tree.order[entry]];
						const Vector3 a = data.vertices[triangle.indices[0]].position;
						const Vector3 edgeA = Subtract(data.vertices[triangle.indices[1]].position, a);
						const Vector3 edgeB = Subtract(data.vertices[triangle.indices[2]].position, a);
						const Vector3 perpendicular = Cross(ray, edgeB);
						const float determinant = Dot(edgeA, perpendicular);
						if (std::abs(determinant) <= axisEpsilon * Length(edgeA) * Length(edgeB) * rayLength) continue;
						const Vector3 offset = Subtract(origin, a);
						const float u = Dot(offset, perpendicular) / determinant;
						if (u < 0.0f || u > 1.0f) continue;
						const Vector3 cross = Cross(offset, edgeA);
						const float v = Dot(ray, cross) / determinant;
						if (v < 0.0f || u + v > 1.0f) continue;
						const float candidate = Dot(edgeB, cross) / determinant;
						if (candidate < 0.0f || candidate > distance) continue;
						distance = candidate;
						normal = Cross(edgeA, edgeB);
						found = true;
					}
					return found;
				}
				int first = node.left, second = node.right;
				const float left = RayBounds(tree.nodes[first].bounds, origin, ray, distance);
				const float right = RayBounds(tree.nodes[second].bounds, origin, ray, distance);
				if (left < 0.0f || (right >= 0.0f && right < left)) std::swap(first, second);
				found = RayNode(tree, data, first, origin, ray, rayLength, distance, normal);
				return RayNode(tree, data, second, origin, ray, rayLength, distance, normal) || found;
			}

			bool TestBoxAxis(const Box& a, const Box& b, const Vector3& axis, float& penetration, Vector3& normal) {
				const float axisLength = Length(axis);
				if (axisLength < axisEpsilon) return true;
				const Vector3 unitAxis = Scale(axis, 1.0f / axisLength);
				const Vector3 delta = Subtract(b.center, a.center);
				const float overlap = Radius(a, unitAxis) + Radius(b, unitAxis) - std::abs(Dot(delta, unitAxis));
				if (overlap <= 0.0f) return false;
				if (overlap < penetration) {
					penetration = overlap;
					normal = Dot(delta, unitAxis) >= 0.0f ? unitAxis : Scale(unitAxis, -1.0f);
				}
				return true;
			}

			bool OverlapBoxes(const Box& a, const Box& b, Contact& contact) {
				float penetration = std::numeric_limits<float>::max();
				Vector3 normal;
				for (int axis = 0; axis < 3; ++axis) if (!TestBoxAxis(a, b, a.axes[axis], penetration, normal) || !TestBoxAxis(a, b, b.axes[axis], penetration, normal)) return false;
				for (int axisA = 0; axisA < 3; ++axisA) for (int axisB = 0; axisB < 3; ++axisB) if (!TestBoxAxis(a, b, Cross(a.axes[axisA], b.axes[axisB]), penetration, normal)) return false;
				contact.penetration = penetration;
				contact.normal = normal;
				contact.surfaceNormal = Scale(normal, -1.0f);
				contact.hit = true;
				return true;
			}

			bool TestTriangleAxis(const Box& box, const Vector3 (&triangle)[3], const Vector3& axis, float& penetration, Vector3& normal) {
				const float axisLength = Length(axis);
				if (axisLength < axisEpsilon) return true;
				const Vector3 unitAxis = Scale(axis, 1.0f / axisLength);
				const float boxCenter = Dot(box.center, unitAxis);
				const float boxRadius = Radius(box, unitAxis);
				const float boxMinimum = boxCenter - boxRadius;
				const float boxMaximum = boxCenter + boxRadius;
				const float projection0 = Dot(triangle[0], unitAxis);
				const float projection1 = Dot(triangle[1], unitAxis);
				const float projection2 = Dot(triangle[2], unitAxis);
				const float triangleMinimum = std::min(projection0, std::min(projection1, projection2));
				const float triangleMaximum = std::max(projection0, std::max(projection1, projection2));
				float overlap;
				if (triangleMaximum - triangleMinimum < 0.0001f) {
					if (triangleMinimum < boxMinimum || triangleMinimum > boxMaximum) return false;
					overlap = std::min(boxMaximum - triangleMinimum, triangleMinimum - boxMinimum);
				}
				else {
					overlap = std::min(boxMaximum, triangleMaximum) - std::max(boxMinimum, triangleMinimum);
					if (overlap <= 0.0f) return false;
				}
				if (overlap < penetration) {
					const Vector3 triangleCenter = Scale(Add(Add(triangle[0], triangle[1]), triangle[2]), 1.0f / 3.0f);
					penetration = overlap;
					normal = Dot(Subtract(triangleCenter, box.center), unitAxis) >= 0.0f ? unitAxis : Scale(unitAxis, -1.0f);
				}
				return true;
			}

			bool OverlapTriangleBox(const Vector3 (&triangle)[3], const Box& box, float winding, bool closed, Contact& contact) {
				const Vector3 edge0 = Subtract(triangle[1], triangle[0]);
				const Vector3 edge1 = Subtract(triangle[2], triangle[1]);
				const Vector3 edge2 = Subtract(triangle[0], triangle[2]);
				const Vector3 face = Cross(edge0, Subtract(triangle[2], triangle[0]));
				const float faceLength = Length(face);
				if (faceLength < axisEpsilon) return false;
				const Vector3 outward = Scale(face, winding / faceLength);
				if (closed && Dot(Subtract(box.center, triangle[0]), outward) < -skinWidth) return false;
				float penetration = std::numeric_limits<float>::max();
				Vector3 normal;
				for (const auto& axis : box.axes) if (!TestTriangleAxis(box, triangle, axis, penetration, normal)) return false;
				if (!TestTriangleAxis(box, triangle, face, penetration, normal)) return false;
				const Vector3 edges[3] = {edge0, edge1, edge2};
				for (const auto& boxAxis : box.axes) for (const auto& edge : edges) if (!TestTriangleAxis(box, triangle, Cross(boxAxis, edge), penetration, normal)) return false;
				contact.penetration = penetration;
				contact.normal = normal;
				contact.surfaceNormal = closed ? outward : Scale(normal, -1.0f);
				contact.hit = true;
				return true;
			}

			bool SweepAxis(const Box& box, const Vector3 (&triangle)[3], const Vector3& displacement, const Vector3& axis, float& enter, float& exit, Vector3& enterAxis) {
				const float axisLength = Length(axis);
				if (axisLength < axisEpsilon) return true;
				const Vector3 unitAxis = Scale(axis, 1.0f / axisLength);
				const float boxCenter = Dot(box.center, unitAxis);
				const float boxRadius = Radius(box, unitAxis);
				const float projection0 = Dot(triangle[0], unitAxis);
				const float projection1 = Dot(triangle[1], unitAxis);
				const float projection2 = Dot(triangle[2], unitAxis);
				const float triangleMinimum = std::min(projection0, std::min(projection1, projection2));
				const float triangleMaximum = std::max(projection0, std::max(projection1, projection2));
				const float speed = Dot(displacement, unitAxis);
				const float boxMinimum = boxCenter - boxRadius;
				const float boxMaximum = boxCenter + boxRadius;
				if (std::abs(speed) < axisEpsilon) return boxMaximum >= triangleMinimum && boxMinimum <= triangleMaximum;
				const float first = (triangleMinimum - boxMaximum) / speed;
				const float second = (triangleMaximum - boxMinimum) / speed;
				const float axisEnter = std::min(first, second);
				const float axisExit = std::max(first, second);
				if (axisEnter > enter || (enterAxis.x == 0.0f && enterAxis.y == 0.0f && enterAxis.z == 0.0f && axisEnter >= enter)) {
					enter = axisEnter;
					enterAxis = unitAxis;
				}
				exit = std::min(exit, axisExit);
				return enter <= exit;
			}

			bool SweepTriangleBox(const Vector3 (&triangle)[3], const Box& box, const Vector3& displacement, float winding, bool closed, Contact& contact) {
				const Vector3 edge0 = Subtract(triangle[1], triangle[0]);
				const Vector3 edge1 = Subtract(triangle[2], triangle[1]);
				const Vector3 edge2 = Subtract(triangle[0], triangle[2]);
				const Vector3 face = Cross(edge0, Subtract(triangle[2], triangle[0]));
				const float faceLength = Length(face);
				if (faceLength < axisEpsilon) return false;
				const Vector3 outward = Scale(face, winding / faceLength);
				if (closed && (Dot(Subtract(box.center, triangle[0]), outward) < -skinWidth || Dot(displacement, outward) >= 0.0f)) return false;
				float enter = 0.0f;
				float exit = 1.0f;
				Vector3 enterAxis;
				for (const auto& axis : box.axes) if (!SweepAxis(box, triangle, displacement, axis, enter, exit, enterAxis)) return false;
				if (!SweepAxis(box, triangle, displacement, face, enter, exit, enterAxis)) return false;
				const Vector3 edges[3] = {edge0, edge1, edge2};
				for (const auto& boxAxis : box.axes) for (const auto& edge : edges) if (!SweepAxis(box, triangle, displacement, Cross(boxAxis, edge), enter, exit, enterAxis)) return false;
				if (enterAxis.x == 0.0f && enterAxis.y == 0.0f && enterAxis.z == 0.0f) return false;
				if (enter < 0.0f || enter > 1.0f || exit < 0.0f) return false;
				const Vector3 boxAtImpact = Add(box.center, Scale(displacement, enter));
				const Vector3 triangleCenter = Scale(Add(Add(triangle[0], triangle[1]), triangle[2]), 1.0f / 3.0f);
				contact.time = enter;
				contact.normal = Dot(Subtract(triangleCenter, boxAtImpact), enterAxis) >= 0.0f ? enterAxis : Scale(enterAxis, -1.0f);
				contact.surfaceNormal = closed ? outward : Scale(contact.normal, -1.0f);
				contact.hit = true;
				return true;
			}

			bool OverlapCapsuleTriangle(const Capsule& capsule, const Vector3 (&triangle)[3], float winding, bool closed, Contact& contact) {
				const Vector3 face = Cross(Subtract(triangle[1], triangle[0]), Subtract(triangle[2], triangle[0]));
				const float faceLength = Length(face);
				if (faceLength < axisEpsilon) return false;
				const Vector3 outward = Scale(face, winding / faceLength);
				Vector3 segmentPoint, trianglePoint;
				ClosestPointsOnSegmentTriangle(capsule.start, capsule.end, triangle, segmentPoint, trianglePoint);
				const Vector3 difference = Subtract(trianglePoint, segmentPoint);
				const float distance = Length(difference);
				if (distance >= capsule.radius || (closed && Dot(Subtract(segmentPoint, triangle[0]), outward) < -skinWidth)) return false;
				contact.normal = distance > axisEpsilon ? Scale(difference, 1.0f / distance) : Scale(outward, -1.0f);
				contact.surfaceNormal = !closed && Dot(outward, difference) > 0.0f ? Scale(outward, -1.0f) : outward;
				contact.point = trianglePoint;
				contact.penetration = capsule.radius - distance;
				contact.hit = true;
				return true;
			}

			bool SweepCapsuleTriangle(const Capsule& capsule, const Vector3 (&triangle)[3], const Vector3& displacement, float displacementLength, float winding, bool closed, Contact& contact) {
				const Vector3 face = Cross(Subtract(triangle[1], triangle[0]), Subtract(triangle[2], triangle[0]));
				const float faceLength = Length(face);
				if (faceLength < axisEpsilon) return false;
				const Vector3 outward = Scale(face, winding / faceLength);
				float time = 0.0f;
				for (int iteration = 0; iteration < 12; ++iteration) {
					const Vector3 start = Add(capsule.start, Scale(displacement, time));
					const Vector3 end = Add(capsule.end, Scale(displacement, time));
					Vector3 segmentPoint, trianglePoint;
					ClosestPointsOnSegmentTriangle(start, end, triangle, segmentPoint, trianglePoint);
					const Vector3 difference = Subtract(trianglePoint, segmentPoint);
					const float distance = Length(difference);
					if (closed && Dot(Subtract(segmentPoint, triangle[0]), outward) < -skinWidth) return false;
					if (distance <= capsule.radius + skinWidth + axisEpsilon) {
						if (closed && Dot(displacement, outward) >= -axisEpsilon * displacementLength) return false;
						const Vector3 normal = distance > axisEpsilon ? Scale(difference, 1.0f / distance) : Scale(outward, -1.0f);
						const Vector3 surfaceNormal = !closed && Dot(outward, difference) > 0.0f ? Scale(outward, -1.0f) : outward;
						if (Dot(normal, surfaceNormal) < -0.9999f && Dot(displacement, surfaceNormal) >= -axisEpsilon * displacementLength) return false;
						if (time <= axisEpsilon && Dot(displacement, normal) <= axisEpsilon) return false;
						contact.time = time;
						contact.normal = normal;
						contact.surfaceNormal = surfaceNormal;
						contact.point = trianglePoint;
						contact.hit = true;
						return true;
					}
					const Vector3 normal = Scale(difference, 1.0f / distance);
					const float closingSpeed = Dot(displacement, normal);
					if (closingSpeed <= axisEpsilon) return false;
					time += (distance - capsule.radius - skinWidth) / closingSpeed;
					if (time > 1.0f) return false;
				}
				return false;
			}

			bool SweepBoxAxis(const Box& a, const Box& b, const Vector3& displacement, const Vector3& axis, float& enter, float& exit, Vector3& enterAxis) {
				const float axisLength = Length(axis);
				if (axisLength < axisEpsilon) return true;
				const Vector3 unitAxis = Scale(axis, 1.0f / axisLength);
				const float centerDelta = Dot(Subtract(b.center, a.center), unitAxis);
				const float radius = Radius(a, unitAxis) + Radius(b, unitAxis);
				const float speed = Dot(displacement, unitAxis);
				if (std::abs(speed) < axisEpsilon) return std::abs(centerDelta) <= radius;
				const float first = (centerDelta - radius) / speed;
				const float second = (centerDelta + radius) / speed;
				const float axisEnter = std::min(first, second);
				const float axisExit = std::max(first, second);
				if (axisEnter > enter || (enterAxis.x == 0.0f && enterAxis.y == 0.0f && enterAxis.z == 0.0f && axisEnter >= enter)) {
					enter = axisEnter;
					enterAxis = unitAxis;
				}
				exit = std::min(exit, axisExit);
				return enter <= exit;
			}

			bool SweepBoxes(const Box& a, const Box& b, const Vector3& displacement, Contact& contact) {
				float enter = 0.0f;
				float exit = 1.0f;
				Vector3 enterAxis;
				for (int axis = 0; axis < 3; ++axis) if (!SweepBoxAxis(a, b, displacement, a.axes[axis], enter, exit, enterAxis) || !SweepBoxAxis(a, b, displacement, b.axes[axis], enter, exit, enterAxis)) return false;
				for (int axisA = 0; axisA < 3; ++axisA) for (int axisB = 0; axisB < 3; ++axisB) if (!SweepBoxAxis(a, b, displacement, Cross(a.axes[axisA], b.axes[axisB]), enter, exit, enterAxis)) return false;
				if (enterAxis.x == 0.0f && enterAxis.y == 0.0f && enterAxis.z == 0.0f) return false;
				if (enter < 0.0f || enter > 1.0f || exit < 0.0f) return false;
				const Vector3 centerAtImpact = Add(a.center, Scale(displacement, enter));
				contact.time = enter;
				contact.normal = Dot(Subtract(b.center, centerAtImpact), enterAxis) >= 0.0f ? enterAxis : Scale(enterAxis, -1.0f);
				contact.surfaceNormal = Scale(contact.normal, -1.0f);
				contact.hit = true;
				return true;
			}

			void GetWorldTriangle(const Transform& transform, const MeshData& data, const Triangle& source, const Vector3 (&axes)[3], Vector3 (&triangle)[3]) {
				for (int corner = 0; corner < 3; ++corner) triangle[corner] = TransformPoint(transform, axes, data.vertices[source.indices[corner]].position);
			}

			bool FindMeshOverlap(Object& meshObject, const Box& box, Contact& result, const Vector3* preparedAxes = nullptr) {
				auto* collider = meshObject.GetComponent<Collider>();
				if (!collider) return false;
				const MeshData* data = CollisionMesh(meshObject, *collider);
				if (!data || data->vertices.empty()) return false;
				Mesh* ownerMesh = std::holds_alternative<std::monostate>(collider->mesh) ? meshObject.GetComponent<Mesh>() : nullptr;
				MeshTree& tree = GetTree(*data, ownerMesh);
				const Transform transform = CollisionMeshTransform(meshObject, *collider, *data);
				const Vector3& scale = transform.scale;
				const float winding = tree.winding * ((scale.x < 0.0f) ^ (scale.y < 0.0f) ^ (scale.z < 0.0f) ? -1.0f : 1.0f);
				Vector3 axes[3];
				if (preparedAxes) std::copy_n(preparedAxes, 3, axes);
				else GetAxes(transform, axes);
				auto& candidates = triangleCandidates;
				QueryTree(tree, transform, BoxBounds(box), axes, candidates);
				bool found = false;
				for (const auto index : candidates) {
					Vector3 triangle[3];
					GetWorldTriangle(transform, *data, tree.triangles[index], axes, triangle);
					Contact contact;
					if (OverlapTriangleBox(triangle, box, winding, tree.closed, contact) && (!found || contact.penetration < result.penetration)) {
						result = contact;
						found = true;
					}
				}
				return found;
			}

			bool FindMeshSweep(Object& meshObject, const Box& box, const Vector3& displacement, Contact& result, const Vector3* preparedAxes = nullptr) {
				auto* collider = meshObject.GetComponent<Collider>();
				if (!collider) return false;
				const MeshData* data = CollisionMesh(meshObject, *collider);
				if (!data || data->vertices.empty()) return false;
				Mesh* ownerMesh = std::holds_alternative<std::monostate>(collider->mesh) ? meshObject.GetComponent<Mesh>() : nullptr;
				MeshTree& tree = GetTree(*data, ownerMesh);
				const Transform transform = CollisionMeshTransform(meshObject, *collider, *data);
				const Vector3& scale = transform.scale;
				const float winding = tree.winding * ((scale.x < 0.0f) ^ (scale.y < 0.0f) ^ (scale.z < 0.0f) ? -1.0f : 1.0f);
				Vector3 axes[3];
				if (preparedAxes) std::copy_n(preparedAxes, 3, axes);
				else GetAxes(transform, axes);
				auto& candidates = triangleCandidates;
				QueryTree(tree, transform, SweptBounds(box, displacement), axes, candidates);
				bool found = false;
				for (const auto index : candidates) {
					Vector3 triangle[3];
					GetWorldTriangle(transform, *data, tree.triangles[index], axes, triangle);
					Contact contact;
					if (SweepTriangleBox(triangle, box, displacement, winding, tree.closed, contact) && (!found || contact.time < result.time)) {
						result = contact;
						found = true;
					}
				}
				return found;
			}

			bool FindCapsuleMeshOverlap(Object& meshObject, const Capsule& capsule, Contact& result, const Vector3* preparedAxes = nullptr) {
				auto* collider = meshObject.GetComponent<Collider>();
				if (!collider) return false;
				const MeshData* data = CollisionMesh(meshObject, *collider);
				if (!data || data->vertices.empty()) return false;
				Mesh* ownerMesh = std::holds_alternative<std::monostate>(collider->mesh) ? meshObject.GetComponent<Mesh>() : nullptr;
				MeshTree& tree = GetTree(*data, ownerMesh);
				const Transform transform = CollisionMeshTransform(meshObject, *collider, *data);
				const Vector3& scale = transform.scale;
				const float winding = tree.winding * ((scale.x < 0.0f) ^ (scale.y < 0.0f) ^ (scale.z < 0.0f) ? -1.0f : 1.0f);
				Vector3 axes[3];
				if (preparedAxes) std::copy_n(preparedAxes, 3, axes);
				else GetAxes(transform, axes);
				auto& candidates = triangleCandidates;
				QueryTree(tree, transform, CapsuleBounds(capsule), axes, candidates);
				bool found = false;
				for (const auto index : candidates) {
					Vector3 triangle[3];
					GetWorldTriangle(transform, *data, tree.triangles[index], axes, triangle);
					Contact contact;
					if (OverlapCapsuleTriangle(capsule, triangle, winding, tree.closed, contact) && (!found || contact.penetration < result.penetration)) {
						result = contact;
						found = true;
					}
				}
				return found;
			}

			bool FindCapsuleMeshSweep(Object& meshObject, const Capsule& capsule, const Vector3& displacement, Contact& result, const PhysicsBody* floorBody = nullptr, const Vector3* preparedAxes = nullptr) {
				auto* collider = meshObject.GetComponent<Collider>();
				if (!collider) return false;
				const MeshData* data = CollisionMesh(meshObject, *collider);
				if (!data || data->vertices.empty()) return false;
				Mesh* ownerMesh = std::holds_alternative<std::monostate>(collider->mesh) ? meshObject.GetComponent<Mesh>() : nullptr;
				MeshTree& tree = GetTree(*data, ownerMesh);
				const Transform transform = CollisionMeshTransform(meshObject, *collider, *data);
				const Vector3& scale = transform.scale;
				const float winding = tree.winding * ((scale.x < 0.0f) ^ (scale.y < 0.0f) ^ (scale.z < 0.0f) ? -1.0f : 1.0f);
				Vector3 axes[3];
				if (preparedAxes) std::copy_n(preparedAxes, 3, axes);
				else GetAxes(transform, axes);
				auto& candidates = triangleCandidates;
				QueryTree(tree, transform, SweptBounds(capsule, displacement), axes, candidates);
				const float displacementSquared = Dot(displacement, displacement);
				const float displacementLength = std::sqrt(displacementSquared);
				bool found = false;
				for (const auto index : candidates) {
					Vector3 triangle[3];
					GetWorldTriangle(transform, *data, tree.triangles[index], axes, triangle);
					Contact contact;
					if (SweepCapsuleTriangle(capsule, triangle, displacement, displacementLength, winding, tree.closed, contact) && PreferCapsuleHit(contact, result, capsule, displacementSquared, floorBody)) {
						result = contact;
						found = true;
					}
				}
				return found;
			}

			bool FindCapsuleBoxOverlap(const Box& box, const Capsule& capsule, Contact& result) {
				Vector3 triangles[12][3];
				GetBoxTriangles(box, triangles);
				bool found = false;
				for (const auto& triangle : triangles) {
					Contact contact;
					if (OverlapCapsuleTriangle(capsule, triangle, 1.0f, true, contact) && (!found || contact.penetration < result.penetration)) {
						result = contact;
						found = true;
					}
				}
				return found;
			}

			bool FindCapsuleBoxSweep(const Box& box, const Capsule& capsule, const Vector3& displacement, Contact& result, const PhysicsBody* floorBody = nullptr) {
				Vector3 triangles[12][3];
				GetBoxTriangles(box, triangles);
				const float displacementSquared = Dot(displacement, displacement);
				const float displacementLength = std::sqrt(displacementSquared);
				bool found = false;
				for (const auto& triangle : triangles) {
					Contact contact;
					if (SweepCapsuleTriangle(capsule, triangle, displacement, displacementLength, 1.0f, true, contact) && PreferCapsuleHit(contact, result, capsule, displacementSquared, floorBody)) {
						result = contact;
						found = true;
					}
				}
				return found;
			}

			bool CapsuleContact(const Capsule& moving, const Capsule& obstacle, const Vector3& displacement, bool sweep, Contact& contact) {
				float time = 0.0f;
				const float radius = moving.radius + obstacle.radius;
				for (int iteration = 0; iteration < (sweep ? 12 : 1); ++iteration) {
					Vector3 first, second;
					ClosestPointsOnSegments(Add(moving.start, Scale(displacement, time)), Add(moving.end, Scale(displacement, time)), obstacle.start, obstacle.end, first, second);
					const Vector3 difference = Subtract(second, first);
					const float distance = Length(difference);
					const Vector3 normal = distance > axisEpsilon ? Scale(difference, 1.0f / distance) : Vector3{1.0f, 0.0f, 0.0f};
					if (distance <= radius + (sweep ? skinWidth : 0.0f)) {
						if (sweep && Dot(displacement, normal) <= axisEpsilon) return false;
						contact = {time, std::max(0.0f, radius - distance), normal, Scale(normal, -1.0f), Subtract(second, Scale(normal, obstacle.radius)), true};
						return true;
					}
					if (!sweep) return false;
					const float speed = Dot(displacement, normal);
					if (speed <= axisEpsilon) return false;
					time += (distance - radius - skinWidth) / speed;
					if (time > 1.0f) return false;
				}
				return false;
			}

			bool FindCapsuleHit(Object& object, const Capsule& capsule, const Vector3& displacement, Contact& result, const PhysicsBody* floorBody = nullptr) {
				const float displacementSquared = Dot(displacement, displacement);
				const Bounds sweptBounds = SweptBounds(capsule, displacement);
				for (const auto& record : colliderRecords) {
					Object* obstacle = record.object;
					if (obstacle == &object || record.body) continue;
					auto* collider = record.collider;
					if (!Overlaps(sweptBounds, record.bounds)) continue;
					Contact contact;
					if (record.type == CollisionType::Capsule) {
						if (!CapsuleContact(capsule, record.capsule, displacement, true, contact)) continue;
					}
					else if (record.type == CollisionType::Mesh) {
						if (!FindCapsuleMeshSweep(*obstacle, capsule, displacement, contact, floorBody, record.axes)) continue;
					}
					else {
						const Box& box = record.box;
						if (!FindCapsuleBoxSweep(box, capsule, displacement, contact, floorBody)) continue;
					}
					if (PreferCapsuleHit(contact, result, capsule, displacementSquared, floorBody)) result = contact;
				}
				return result.hit;
			}

			bool FindCapsuleFloor(Scene& scene, Object& object, Collider& collider, float distance, Contact& result) {
				Capsule capsule = WorldCapsule(object, collider);
				capsule.radius = std::max(axisEpsilon, capsule.radius - skinWidth);
				const PhysicsBody& body = *object.GetComponent<PhysicsBody>();
				const Vector3 displacement{0.0f, 0.0f, -distance};
				Contact contact;
				FindCapsuleHit( object, capsule, displacement, contact, &body);
				if (IsCapsuleSupport(capsule, body, contact)) {
					result = contact;
					return true;
				}
				const bool blockingFloor = IsCapsuleFootContact(capsule, contact);
				const float shrink = capsule.radius * 0.25f;
				capsule.radius -= shrink;
				capsule.start.z -= shrink;
				capsule.end.z += shrink;
				Contact narrowContact;
				FindCapsuleHit( object, capsule, displacement, narrowContact, &body);
				if (!IsCapsuleSupport(capsule, body, narrowContact)) return false;
				if (blockingFloor && std::abs(Dot(Subtract(contact.point, narrowContact.point), narrowContact.surfaceNormal)) > skinWidth) return false;
				result = narrowContact;
				if (blockingFloor) result.time = std::min(result.time, contact.time);
				return true;
			}

			bool FindOverlap(Object& movingObject, Object& obstacle, const Collider& movingCollider, const Collider& obstacleCollider, Contact& contact, const Vector3* preparedAxes = nullptr) {
				if (ColliderType(movingObject, movingCollider) == CollisionType::Capsule) {
					const Capsule capsule = WorldCapsule(movingObject, movingCollider);
					if (ColliderType(obstacle, obstacleCollider) == CollisionType::Capsule) return CapsuleContact(capsule, WorldCapsule(obstacle, obstacleCollider), {}, false, contact);
					if (ColliderType(obstacle, obstacleCollider) == CollisionType::Mesh) return FindCapsuleMeshOverlap(obstacle, capsule, contact, preparedAxes);
					return FindCapsuleBoxOverlap(WorldBox(obstacle, obstacleCollider), capsule, contact);
				}
				const Box movingBox = WorldBox(movingObject, movingCollider);
				if (ColliderType(obstacle, obstacleCollider) == CollisionType::Capsule) {
					if (!FindCapsuleBoxOverlap(movingBox, WorldCapsule(obstacle, obstacleCollider), contact)) return false;
					contact.normal = Scale(contact.normal, -1.0f);
					contact.surfaceNormal = Scale(contact.normal, -1.0f);
					return true;
				}
				if (ColliderType(obstacle, obstacleCollider) == CollisionType::Mesh) return FindMeshOverlap(obstacle, movingBox, contact, preparedAxes);
				return OverlapBoxes(movingBox, WorldBox(obstacle, obstacleCollider), contact);
			}

			void RemoveIntoSurface(PhysicsBody& body, const Vector3& normal) {
				const float speed = Dot(body.velocity, normal);
				if (speed > 0.0f) body.velocity += Scale(normal, -speed);
			}

			void ConstrainContact(PhysicsBody& body, Contact& contact) {
				if (IsWalkable(body, contact)) {
					if (Dot(body.velocity, contact.surfaceNormal) <= axisEpsilon) body.grounded = true;
					return;
				}
				const float horizontalLength = std::sqrt(contact.normal.x * contact.normal.x + contact.normal.y * contact.normal.y);
				if (horizontalLength > axisEpsilon) {
					contact.normal = {contact.normal.x / horizontalLength, contact.normal.y / horizontalLength, 0.0f};
					contact.penetration /= horizontalLength;
				}
			}

			void ConstrainCapsuleContact(PhysicsBody& body, Contact& contact, bool allowSlope) {
				if (allowSlope && IsWalkable(body, contact)) return;
				if (contact.surfaceNormal.z >= 0.0f) {
					const float horizontalLength = std::sqrt(contact.normal.x * contact.normal.x + contact.normal.y * contact.normal.y);
					if (horizontalLength > axisEpsilon) {
						contact.normal = {contact.normal.x / horizontalLength, contact.normal.y / horizontalLength, 0.0f};
						contact.penetration /= horizontalLength;
					}
				}
			}

			void ResolveOverlaps(Object& object, PhysicsBody& body, Collider& collider) {
				const bool capsule = ColliderType(object, collider) == CollisionType::Capsule;
				Bounds movingBounds = capsule ? CapsuleBounds(WorldCapsule(object, collider)) : BoxBounds(WorldBox(object, collider));
				for (int iteration = 0; iteration < maxOverlapIterations; ++iteration) {
					bool corrected = false;
					for (const auto& record : colliderRecords) {
						Object* obstacle = record.object;
						if (obstacle == &object || record.body) continue;
						auto* obstacleCollider = record.collider;
						if (!Overlaps(movingBounds, record.bounds)) continue;
						Contact contact;
						if (!FindOverlap(object, *obstacle, collider, *obstacleCollider, contact, record.axes)) continue;
						if (capsule) ConstrainCapsuleContact(body, contact, true);
						else ConstrainContact(body, contact);
						MoveObject(object, Scale(contact.normal, -(contact.penetration + skinWidth)));
						movingBounds = capsule ? CapsuleBounds(WorldCapsule(object, collider)) : BoxBounds(WorldBox(object, collider));
						if (!capsule || !IsWalkable(body, contact)) RemoveIntoSurface(body, contact.normal);
						corrected = true;
					}
					if (!corrected) return;
				}
			}

			bool FindEarliestHit(Object& object, Collider& collider, const Vector3& displacement, Contact& result) {
				if (ColliderType(object, collider) == CollisionType::Capsule) return FindCapsuleHit( object, WorldCapsule(object, collider), displacement, result);
				const Box movingBox = WorldBox(object, collider);
				const Bounds sweptBounds = SweptBounds(movingBox, displacement);
				for (const auto& record : colliderRecords) {
					Object* obstacle = record.object;
					if (obstacle == &object || record.body) continue;
					auto* obstacleCollider = record.collider;
					if (!Overlaps(sweptBounds, record.bounds)) continue;
					Contact contact;
					if (record.type == CollisionType::Capsule) {
						if (!FindCapsuleBoxSweep(movingBox, record.capsule, Scale(displacement, -1.0f), contact)) continue;
						contact.normal = Scale(contact.normal, -1.0f);
						contact.surfaceNormal = Scale(contact.normal, -1.0f);
						contact.point += Scale(displacement, contact.time);
					}
					else if (record.type == CollisionType::Mesh) {
						if (!FindMeshSweep(*obstacle, movingBox, displacement, contact, record.axes)) continue;
					}
					else {
						const Box& obstacleBox = record.box;
						if (!SweepBoxes(movingBox, obstacleBox, displacement, contact)) continue;
					}
					if (!result.hit || contact.time < result.time) result = contact;
				}
				return result.hit;
			}

			void MoveAndSlide(Object& object, PhysicsBody& body, Collider& collider, float deltaTime) {
				Vector3 remaining = Scale(body.velocity, deltaTime);
				for (int iteration = 0; iteration < maxMoveIterations; ++iteration) {
					const float distance = Length(remaining);
					if (distance < axisEpsilon) return;
					Contact contact;
					if (!FindEarliestHit( object, collider, remaining, contact)) {
						MoveObject(object, remaining);
						return;
					}
					const float safeTime = std::max(0.0f, contact.time - skinWidth / distance);
					MoveObject(object, Scale(remaining, safeTime));
					remaining = Scale(remaining, 1.0f - contact.time);
					ConstrainContact(body, contact);
					const float intoSurface = Dot(remaining, contact.normal);
					if (intoSurface > 0.0f) remaining += Scale(contact.normal, -intoSurface);
					RemoveIntoSurface(body, contact.normal);
					if (body.grounded && body.velocity.z < 0.0f) body.velocity.z = 0.0f;
				}
			}

			bool TryCapsuleTransition(Scene& scene, Object& object, PhysicsBody& body, Collider& collider, const Vector3& remaining, const Contact& obstacle, const Contact& support, float riseLimit, bool allowFall) {
				if (riseLimit <= 0.0f || obstacle.point.z > WorldCapsule(object, collider).end.z) return false;
				const Vector3 forward{remaining.x, remaining.y, 0.0f};
				if (Length(forward) <= axisEpsilon) return false;
				const Vector3 original = object.transform.position;
				const Vector3 up{0.0f, 0.0f, riseLimit};
				Contact ceiling;
				float rise = riseLimit;
				if (FindEarliestHit( object, collider, up, ceiling)) rise *= ceiling.time;
				if (rise <= axisEpsilon) return false;
				MoveObject(object, {0.0f, 0.0f, rise});
				Contact ahead;
				if (FindEarliestHit( object, collider, forward, ahead)) {
					object.SetWorldPosition(original);
					return false;
				}
				MoveObject(object, forward);
				const float drop = rise + floorProbeDistance;
				Contact floor;
				if (!FindCapsuleFloor(scene, object, collider, drop, floor)) {
					if (allowFall) {
						body.grounded = false;
						return true;
					}
					object.SetWorldPosition(original);
					return false;
				}
				MoveObject(object, {0.0f, 0.0f, -std::max(0.0f, floor.time * drop - skinWidth)});
				if (object.transform.position.z - original.z > riseLimit + axisEpsilon || floor.point.z - support.point.z > riseLimit + axisEpsilon) {
					object.SetWorldPosition(original);
					return false;
				}
				Contact landing;
				if (!FindCapsuleFloor(scene, object, collider, floorProbeDistance, landing)) {
					object.SetWorldPosition(original);
					return false;
				}
				body.grounded = true;
				return true;
			}

			void MoveCapsuleAndSlide(Scene& scene, Object& object, PhysicsBody& body, Collider& collider, Vector3 remaining, const Contact* floor) {
				const bool walking = floor != nullptr;
				Vector3 previousNormal;
				bool hasPreviousNormal = false;
				for (int iteration = 0; iteration < maxMoveIterations; ++iteration) {
					const float distance = Length(remaining);
					if (distance < axisEpsilon) return;
					Contact contact;
					if (!FindEarliestHit( object, collider, remaining, contact)) {
						MoveObject(object, remaining);
						return;
					}
					MoveObject(object, Scale(remaining, contact.time));
					remaining = Scale(remaining, 1.0f - contact.time);
					const bool walkable = IsWalkable(body, contact);
					if (walking && (!walkable || -contact.normal.z < body.SlopeCosine())) {
						const Capsule capsule = WorldCapsule(object, collider);
						const bool footEdge = contact.point.z <= capsule.start.z && -contact.normal.z >= body.SlopeCosine() && std::abs(Dot(Subtract(contact.point, floor->point), floor->surfaceNormal)) <= skinWidth;
						if (footEdge && TryCapsuleTransition(scene, object, body, collider, remaining, contact, *floor, skinWidth + axisEpsilon * 2.0f, true)) return;
						if (TryCapsuleTransition(scene, object, body, collider, remaining, contact, *floor, body.stepHeight, false)) return;
					}
					const bool descending = remaining.z < 0.0f;
					const bool followSurface = (walkable && (walking || descending)) || (!walkable && contact.surfaceNormal.z > axisEpsilon && descending);
					Contact response = contact;
					if (!followSurface) ConstrainCapsuleContact(body, response, false);
					const float intoSurface = Dot(remaining, response.normal);
					if (intoSurface > 0.0f) remaining += Scale(response.normal, -intoSurface);
					if (hasPreviousNormal && Dot(remaining, previousNormal) > axisEpsilon) {
						const Vector3 crease = Cross(previousNormal, response.normal);
						const float lengthSquared = Dot(crease, crease);
						remaining = lengthSquared > axisEpsilon ? Scale(crease, Dot(remaining, crease) / lengthSquared) : Vector3{};
					}
					previousNormal = response.normal;
					hasPreviousNormal = true;
					if (!walkable && contact.surfaceNormal.z > axisEpsilon && remaining.z > 0.0f) remaining.z = 0.0f;
					if (!followSurface || contact.surfaceNormal.z < 0.0f) RemoveIntoSurface(body, response.normal);
				}
			}

			void SimulateCapsule(Scene& scene, Object& object, PhysicsBody& body, Collider& collider, float deltaTime) {
				ResolveOverlaps( object, body, collider);
				Contact floor;
				const bool walking = body.velocity.z <= 0.0f && FindCapsuleFloor(scene, object, collider, floorProbeDistance, floor);
				body.grounded = walking;
				Vector3 displacement;
				if (walking) {
					if (body.velocity.z < 0.0f) body.velocity.z = 0.0f;
					displacement = {body.velocity.x * deltaTime, body.velocity.y * deltaTime, 0.0f};
					if (floor.surfaceNormal.z > axisEpsilon) displacement.z = -(floor.surfaceNormal.x * displacement.x + floor.surfaceNormal.y * displacement.y) / floor.surfaceNormal.z;
				}
				else {
					if (body.useGravity) body.velocity += scene.gravity * deltaTime;
					displacement = Scale(body.velocity, deltaTime);
				}
				MoveCapsuleAndSlide(scene, object, body, collider, displacement, walking ? &floor : nullptr);
				if (body.velocity.z <= 0.0f) {
					Contact nextFloor;
					const float snapDistance = walking ? std::max(floorProbeDistance, body.stepHeight) : floorProbeDistance;
					if (FindCapsuleFloor(scene, object, collider, snapDistance, nextFloor)) {
						MoveObject(object, {0.0f, 0.0f, -std::max(0.0f, nextFloor.time * snapDistance - skinWidth)});
						body.grounded = true;
						body.velocity.z = 0.0f;
					}
					else body.grounded = false;
				}
				else body.grounded = false;
			}

			void ResolveDynamicPairs() {
				for (std::size_t i = 0; i < colliderRecords.size(); ++i) {
					const auto& a = colliderRecords[i];
					if (!a.body || a.type == CollisionType::Mesh) continue;
					for (std::size_t j = i + 1; j < colliderRecords.size(); ++j) {
						const auto& b = colliderRecords[j];
						if (!b.body || b.type == CollisionType::Mesh) continue;
						const Bounds first = a.type == CollisionType::Capsule ? CapsuleBounds(WorldCapsule(*a.object, *a.collider)) : BoxBounds(WorldBox(*a.object, *a.collider));
						const Bounds second = b.type == CollisionType::Capsule ? CapsuleBounds(WorldCapsule(*b.object, *b.collider)) : BoxBounds(WorldBox(*b.object, *b.collider));
						if (!Overlaps(first, second)) continue;
						Contact contact;
						if (!FindOverlap(*a.object, *b.object, *a.collider, *b.collider, contact)) continue;
						MoveObject(*a.object, Scale(contact.normal, -contact.penetration * 0.5f));
						MoveObject(*b.object, Scale(contact.normal, contact.penetration * 0.5f));
						RemoveIntoSurface(*a.body, contact.normal);
						RemoveIntoSurface(*b.body, Scale(contact.normal, -1.0f));
					}
				}
			}

		}

		bool Raycast(Scene& scene, const Vector3& origin, const Vector3& direction, float maxDistance, RaycastHit& hit) {
			hit = {};
			const float length = Length(direction);
			if (!std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z) || !std::isfinite(length) || length <= axisEpsilon || !std::isfinite(maxDistance) || maxDistance <= 0.0f) return false;
			const Vector3 ray = Scale(direction, 1.0f / length);
			float closest = maxDistance;
			SyncColliders(scene);
			for (const auto& record : colliderRecords) {
				Object* object = record.object;
				Collider* collider = record.collider;
				if (RayBounds(record.bounds, origin, ray, closest) < 0.0f) continue;
				float distance = closest;
				Vector3 normal;
				bool found = false;
				if (record.type == CollisionType::Mesh) {
					MeshTree& tree = GetTree(*record.data, record.ownerMesh);
					const Transform& transform = record.transform;
					if (tree.nodes.empty() || std::abs(transform.scale.x) < axisEpsilon || std::abs(transform.scale.y) < axisEpsilon || std::abs(transform.scale.z) < axisEpsilon) continue;
					Vector3 axes[3];
					std::copy_n(record.axes, 3, axes);
					const Vector3 offset = Subtract(origin, transform.position);
					const Vector3 localOrigin{Dot(offset, axes[0]) / transform.scale.x, Dot(offset, axes[1]) / transform.scale.y, Dot(offset, axes[2]) / transform.scale.z};
					const Vector3 localRay{Dot(ray, axes[0]) / transform.scale.x, Dot(ray, axes[1]) / transform.scale.y, Dot(ray, axes[2]) / transform.scale.z};
					found = RayNode(tree, *record.data, 0, localOrigin, localRay, Length(localRay), distance, normal);
					if (found) {
						normal = TransformNormal(NormalMatrix(transform), normal);
						if (Dot(normal, ray) > 0.0f) normal = Scale(normal, -1.0f);
					}
				}
				else if (record.type == CollisionType::Capsule) {
					const Capsule capsule = record.capsule;
					const Vector3 segment = Subtract(capsule.end, capsule.start);
					const Vector3 offset = Subtract(origin, capsule.start);
					const float segmentLength = Dot(segment, segment);
					const float projection = segmentLength > 0.0f ? std::clamp(Dot(offset, segment) / segmentLength, 0.0f, 1.0f) : 0.0f;
					const Vector3 separation = Subtract(offset, Scale(segment, projection));
					if (Dot(separation, separation) < capsule.radius * capsule.radius) continue;
					auto accept = [&](float candidate) {
						if (candidate < 0.0f || candidate > distance) return;
						const Vector3 point = Add(offset, Scale(ray, candidate));
						const float along = segmentLength > 0.0f ? std::clamp(Dot(point, segment) / segmentLength, 0.0f, 1.0f) : 0.0f;
						normal = Subtract(point, Scale(segment, along));
						const float normalLength = Length(normal);
						if (normalLength <= axisEpsilon) return;
						normal = Scale(normal, 1.0f / normalLength);
						distance = candidate;
						found = true;
					};
					const float alongRay = Dot(segment, ray);
					const float alongOrigin = Dot(segment, offset);
					const float a = segmentLength - alongRay * alongRay;
					const float b = segmentLength * Dot(offset, ray) - alongOrigin * alongRay;
					const float c = segmentLength * (Dot(offset, offset) - capsule.radius * capsule.radius) - alongOrigin * alongOrigin;
					const float discriminant = b * b - a * c;
					if (a > axisEpsilon * segmentLength && discriminant >= 0.0f) {
						const float candidate = (-b - std::sqrt(discriminant)) / a;
						const float along = alongOrigin + candidate * alongRay;
						if (along >= 0.0f && along <= segmentLength) accept(candidate);
					}
					for (const auto& center : {capsule.start, capsule.end}) {
						const Vector3 sphereOffset = Subtract(origin, center);
						const float sphereB = Dot(sphereOffset, ray);
						const float sphereDiscriminant = sphereB * sphereB - Dot(sphereOffset, sphereOffset) + capsule.radius * capsule.radius;
						if (sphereDiscriminant >= 0.0f) accept(-sphereB - std::sqrt(sphereDiscriminant));
					}
				}
				else {
					const Box box = record.box;
					const Vector3 offset = Subtract(origin, box.center);
					float enter = -std::numeric_limits<float>::infinity();
					float exit = closest;
					bool intersects = true;
					for (int axis = 0; axis < 3; ++axis) {
						const float position = Dot(offset, box.axes[axis]);
						const float velocity = Dot(ray, box.axes[axis]);
						if (std::abs(velocity) <= axisEpsilon) {
							if (std::abs(position) > box.halfSize[axis]) intersects = false;
							continue;
						}
						float near = (-box.halfSize[axis] - position) / velocity;
						float far = (box.halfSize[axis] - position) / velocity;
						if (near > far) std::swap(near, far);
						if (near > enter) {
							enter = near;
							normal = Scale(box.axes[axis], velocity > 0.0f ? -1.0f : 1.0f);
						}
						exit = std::min(exit, far);
						if (enter > exit) intersects = false;
					}
					found = intersects && enter >= 0.0f && enter <= closest;
					distance = enter;
				}
				if (found) {
					closest = distance;
					hit = {object->worldPartitionSourceObject ? object->worldPartitionSourceObject : object, collider, Add(origin, Scale(ray, distance)), normal, distance};
				}
			}
			return hit.collider != nullptr;
		}

		void Initialize(Scene& scene) {
			SyncColliders(scene);
			for (const auto& record : colliderRecords) {
				if (record.type == CollisionType::Mesh) GetTree(*record.data, record.ownerMesh);
			}
		}

		void Simulate(Scene& scene, float deltaTime) {
			SyncColliders(scene);
			for (const auto& record : bodyRecords) {
				Object& object = *record.object;
				PhysicsBody& body = *record.body;
				body.previousPosition = object.transform.position;
				if (record.collider && ColliderType(object, *record.collider) == CollisionType::Capsule) SimulateCapsule(scene, object, body, *record.collider, deltaTime);
				else {
					body.grounded = false;
					if (body.useGravity) body.velocity += scene.gravity * deltaTime;
					if (record.collider) {
						ResolveOverlaps( object, body, *record.collider);
						MoveAndSlide( object, body, *record.collider, deltaTime);
					}
					else MoveObject(object, body.velocity * deltaTime);
				}
				if (body.grounded && body.velocity.z < 0.0f) body.velocity.z = 0.0f;
			}
			ResolveDynamicPairs();
		}

		std::vector<Vector3> ColliderDebugLines(Scene& scene) {
			SyncColliders(scene);
			std::vector<Vector3> lines;
			auto line = [&](const Vector3& a, const Vector3& b) { lines.push_back(a); lines.push_back(b); };
			for (const auto& record : colliderRecords) {
				if (record.type == CollisionType::Cube) {
					Vector3 corners[8];
					for (int index = 0; index < 8; ++index) {
						corners[index] = record.box.center;
						for (int axis = 0; axis < 3; ++axis) corners[index] = Add(corners[index], Scale(record.box.axes[axis], record.box.halfSize[axis] * (index & (1 << axis) ? 1.0f : -1.0f)));
					}
					for (int index = 0; index < 8; ++index) for (int axis = 0; axis < 3; ++axis) if (!(index & (1 << axis))) line(corners[index], corners[index | (1 << axis)]);
				}
				else if (record.type == CollisionType::Capsule) {
					const Capsule& capsule = record.capsule;
					const Vector3 direction = Length(Subtract(capsule.end, capsule.start)) > axisEpsilon ? Normalize(Subtract(capsule.end, capsule.start)) : Vector3{0.0f, 0.0f, 1.0f};
					const Vector3 right = Normalize(Cross(direction, std::abs(direction.z) < 0.9f ? Vector3{0.0f, 0.0f, 1.0f} : Vector3{0.0f, 1.0f, 0.0f}));
					const Vector3 forward = Cross(direction, right);
					constexpr int segments = 16;
					constexpr float pi = 3.14159265358979323846f;
					std::array<Vector3, segments> previous{};
					for (int ring = 0; ring <= 9; ++ring) {
						const float angle = (ring <= 4 ? ring - 4 : ring - 5) * pi / 8.0f;
						const Vector3 center = ring <= 4 ? capsule.start : capsule.end;
						const Vector3 ringCenter = Add(center, Scale(direction, std::sin(angle) * capsule.radius));
						const float radius = std::cos(angle) * capsule.radius;
						std::array<Vector3, segments> current{};
						for (int index = 0; index < segments; ++index) {
							const float phase = index * 2.0f * pi / segments;
							current[index] = Add(ringCenter, Add(Scale(right, std::cos(phase) * radius), Scale(forward, std::sin(phase) * radius)));
							if (ring && index % 4 == 0) line(previous[index], current[index]);
						}
						for (int index = 0; index < segments; ++index) line(current[index], current[(index + 1) % segments]);
						previous = current;
					}
				}
				else if (record.type == CollisionType::Mesh) {
					const auto& data = *record.data;
					const Matrix4 model = Model(record.transform);
					std::unordered_set<std::uint64_t> edges;
					for (std::size_t index = 0; index + 2 < data.indices.size(); index += 3) {
						for (int edge = 0; edge < 3; ++edge) {
							const std::uint32_t a = data.indices[index + edge], b = data.indices[index + (edge + 1) % 3];
							if (a >= data.vertices.size() || b >= data.vertices.size()) continue;
							const std::uint64_t key = (static_cast<std::uint64_t>(std::min(a, b)) << 32) | std::max(a, b);
							if (edges.insert(key).second) line(TransformPoint(model, data.vertices[a].position), TransformPoint(model, data.vertices[b].position));
						}
					}
				}
			}
			return lines;
		}

		void Shutdown() {
			std::vector<std::uint32_t>().swap(triangleCandidates);
			std::vector<ColliderRecord>().swap(colliderRecords);
			std::vector<BodyRecord>().swap(bodyRecords);
			decltype(meshBounds)().swap(meshBounds);
			decltype(meshTrees)().swap(meshTrees);
			cacheGeneration = 0;
		}
	}
}
