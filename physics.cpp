#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <unordered_map>
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
	};

	class Collider : public Component {
		public:
			Vector3 center;
			Vector3 size{1.0f, 1.0f, 1.0f};
			bool isMesh = false;
			bool isCapsule = false;
	};
	struct RaycastHit {
		Object* object = nullptr;
		Collider* collider = nullptr;
		Vector3 point;
		Vector3 normal;
		float distance = 0.0f;
	};

	namespace Physics {
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
				Bounds bounds;
				Vector3 center;
			};

			struct BvhNode {
				Bounds bounds;
				std::uint32_t first = 0;
				std::uint32_t count = 0;
				int left = -1;
				int right = -1;
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

			struct MeshBounds {
				Bounds bounds;
				MeshRevision revision;
			};

			struct MeshTree {
				MeshRevision revision;
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
				return contact.hit && contact.surfaceNormal.y >= std::cos(body.slopeLimit * Math::degreesToRadians);
			}

			bool IsCapsuleFootContact(const Capsule& capsule, const Contact& contact) {
				const float x = contact.point.x - capsule.start.x;
				const float z = contact.point.z - capsule.start.z;
				const float radius = std::max(0.0f, capsule.radius - skinWidth);
				return contact.hit && contact.point.y <= capsule.start.y && x * x + z * z <= radius * radius;
			}

			bool IsCapsuleSupport(const Capsule& capsule, const PhysicsBody& body, const Contact& contact) {
				return IsWalkable(body, contact) && IsCapsuleFootContact(capsule, contact);
			}

			bool PreferCapsuleHit(const Contact& candidate, const Contact& current, const Capsule& capsule, const Vector3& displacement, const PhysicsBody* floorBody) {
				if (!current.hit) return true;
				if (floorBody && std::abs(candidate.time - current.time) * Length(displacement) <= axisEpsilon) {
					const bool candidateSupport = IsCapsuleSupport(capsule, *floorBody, candidate);
					const bool currentSupport = IsCapsuleSupport(capsule, *floorBody, current);
					if (candidateSupport != currentSupport) return candidateSupport;
					if (candidateSupport) return candidate.surfaceNormal.y > current.surfaceNormal.y;
				}
				return candidate.time < current.time;
			}

			std::unordered_map<const Mesh*, MeshBounds> meshBounds;
			std::vector<std::uint32_t> triangleCandidates;
			std::vector<const Mesh*> liveMeshes;
			std::vector<const Mesh*> liveTreeMeshes;
			std::unordered_map<const Mesh*, MeshTree> meshTrees;

			const Bounds& GetMeshBounds(const Mesh& mesh) {
				auto [entry, inserted] = meshBounds.try_emplace(&mesh);
				if (inserted || !entry->second.revision.Matches(mesh)) {
					entry->second.bounds = EmptyBounds();
					for (const auto& vertex : mesh.data.vertices) Include(entry->second.bounds, vertex.position);
					entry->second.revision.Set(mesh);
				}
				return entry->second.bounds;
			}

			Box WorldBox(Object& object, const Collider& collider) {
				const Transform& transform = object.transform;
				Box box;
				GetAxes(transform, box.axes);
				Vector3 size = collider.size;
				Vector3 center = collider.center;
				if (!collider.isMesh && !collider.isCapsule && size.x == 1.0f && size.y == 1.0f && size.z == 1.0f && center.x == 0.0f && center.y == 0.0f && center.z == 0.0f) {
					if (auto* mesh = object.GetComponent<Mesh>(); mesh && !mesh->data.vertices.empty()) {
						const Bounds& bounds = GetMeshBounds(*mesh);
						size = {bounds.maximum.x - bounds.minimum.x, bounds.maximum.y - bounds.minimum.y, bounds.maximum.z - bounds.minimum.z};
						center = {(bounds.minimum.x + bounds.maximum.x) * 0.5f, (bounds.minimum.y + bounds.maximum.y) * 0.5f, (bounds.minimum.z + bounds.maximum.z) * 0.5f};
					}
				}
				const Vector3 offset{center.x * transform.scale.x, center.y * transform.scale.y, center.z * transform.scale.z};
				box.center = Add(transform.position, Add(Add(Scale(box.axes[0], offset.x), Scale(box.axes[1], offset.y)), Scale(box.axes[2], offset.z)));
				box.halfSize[0] = std::abs(size.x * transform.scale.x) * 0.5f;
				box.halfSize[1] = std::abs(size.y * transform.scale.y) * 0.5f;
				box.halfSize[2] = std::abs(size.z * transform.scale.z) * 0.5f;
				return box;
			}

			Capsule WorldCapsule(Object& object, const Collider& collider) {
				const Transform& transform = object.transform;
				Vector3 axes[3];
				GetAxes(transform, axes);
				const Vector3 offset{collider.center.x * transform.scale.x, collider.center.y * transform.scale.y, collider.center.z * transform.scale.z};
				const Vector3 center = Add(transform.position, Add(Add(Scale(axes[0], offset.x), Scale(axes[1], offset.y)), Scale(axes[2], offset.z)));
				const float height = std::abs(collider.size.y * transform.scale.y);
				const float radius = std::min(height * 0.5f, std::min(std::abs(collider.size.x * transform.scale.x), std::abs(collider.size.z * transform.scale.z)) * 0.5f);
				const Vector3 segment = Scale(axes[1], std::max(0.0f, height * 0.5f - radius));
				return {Subtract(center, segment), Add(center, segment), radius};
			}

			Bounds BoxBounds(const Box& box) {
				const Vector3 extent{box.halfSize[0] * std::abs(box.axes[0].x) + box.halfSize[1] * std::abs(box.axes[1].x) + box.halfSize[2] * std::abs(box.axes[2].x), box.halfSize[0] * std::abs(box.axes[0].y) + box.halfSize[1] * std::abs(box.axes[1].y) + box.halfSize[2] * std::abs(box.axes[2].y), box.halfSize[0] * std::abs(box.axes[0].z) + box.halfSize[1] * std::abs(box.axes[1].z) + box.halfSize[2] * std::abs(box.axes[2].z)};
				return {{box.center.x - extent.x, box.center.y - extent.y, box.center.z - extent.z}, {box.center.x + extent.x, box.center.y + extent.y, box.center.z + extent.z}};
			}

			Bounds CapsuleBounds(const Capsule& capsule) {
				const Vector3 extent{capsule.radius, capsule.radius, capsule.radius};
				return {{std::min(capsule.start.x, capsule.end.x) - extent.x, std::min(capsule.start.y, capsule.end.y) - extent.y, std::min(capsule.start.z, capsule.end.z) - extent.z}, {std::max(capsule.start.x, capsule.end.x) + extent.x, std::max(capsule.start.y, capsule.end.y) + extent.y, std::max(capsule.start.z, capsule.end.z) + extent.z}};
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

			int BuildNode(MeshTree& tree, std::uint32_t first, std::uint32_t count) {
				const int nodeIndex = static_cast<int>(tree.nodes.size());
				tree.nodes.emplace_back();
				BvhNode node;
				node.bounds = EmptyBounds();
				Bounds centers = EmptyBounds();
				for (std::uint32_t index = first; index < first + count; ++index) {
					const Triangle& triangle = tree.triangles[tree.order[index]];
					Include(node.bounds, triangle.bounds.minimum);
					Include(node.bounds, triangle.bounds.maximum);
					Include(centers, triangle.center);
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
					const Vector3& left = tree.triangles[a].center;
					const Vector3& right = tree.triangles[b].center;
					return axis == 0 ? left.x < right.x : axis == 1 ? left.y < right.y : left.z < right.z;
				});
				node.left = BuildNode(tree, first, middle - first);
				node.right = BuildNode(tree, middle, first + count - middle);
				tree.nodes[nodeIndex] = node;
				return nodeIndex;
			}

			MeshTree BuildTree(const Mesh& mesh) {
				MeshTree tree;
				tree.revision.Set(mesh);
				const std::size_t triangleCount = mesh.data.indices.size() / 3;
				Loading::Timer loading("Collision BVH: " + std::to_string(triangleCount) + " triangles");
				tree.triangles.reserve(triangleCount);
				tree.order.reserve(triangleCount);
				float signedVolume = 0.0f;
				std::map<std::array<float, 3>, std::uint32_t> positionIds;
				std::vector<std::uint32_t> vertexIds;
				vertexIds.reserve(mesh.data.vertices.size());
				for (const auto& vertex : mesh.data.vertices) {
					const auto& point = vertex.position;
					vertexIds.push_back(positionIds.try_emplace({point.x, point.y, point.z}, static_cast<std::uint32_t>(positionIds.size())).first->second);
				}
				struct EdgeUse {
					unsigned count = 0;
					int direction = 0;
				};
				std::unordered_map<std::uint64_t, EdgeUse> edges;
				for (std::size_t index = 0; index + 2 < mesh.data.indices.size(); index += 3) {
					const Vector3& a = mesh.data.vertices[mesh.data.indices[index]].position;
					const Vector3& b = mesh.data.vertices[mesh.data.indices[index + 1]].position;
					const Vector3& c = mesh.data.vertices[mesh.data.indices[index + 2]].position;
					const Vector3 edgeA = Subtract(b, a);
					const Vector3 edgeB = Subtract(c, a);
					if (Length(Cross(edgeA, edgeB)) <= axisEpsilon * Length(edgeA) * Length(edgeB)) continue;
					for (int corner = 0; corner < 3; ++corner) {
						const auto first = vertexIds[mesh.data.indices[index + corner]];
						const auto second = vertexIds[mesh.data.indices[index + (corner + 1) % 3]];
						const std::uint64_t key = (static_cast<std::uint64_t>(std::min(first, second)) << 32) | std::max(first, second);
						auto& edge = edges[key];
						++edge.count;
						edge.direction += first < second ? 1 : -1;
					}
					Triangle triangle{};
					triangle.indices[0] = mesh.data.indices[index];
					triangle.indices[1] = mesh.data.indices[index + 1];
					triangle.indices[2] = mesh.data.indices[index + 2];
					triangle.bounds = EmptyBounds();
					Include(triangle.bounds, a);
					Include(triangle.bounds, b);
					Include(triangle.bounds, c);
					triangle.center = Scale(Add(Add(a, b), c), 1.0f / 3.0f);
					tree.triangles.push_back(triangle);
					tree.order.push_back(static_cast<std::uint32_t>(tree.triangles.size() - 1));
					signedVolume += Dot(a, Cross(b, c)) / 6.0f;
				}
				tree.closed = !edges.empty() && std::all_of(edges.begin(), edges.end(), [](const auto& edge) { return edge.second.count == 2 && edge.second.direction == 0; });
				tree.winding = tree.closed && signedVolume < 0.0f ? -1.0f : 1.0f;
				if (!tree.triangles.empty()) BuildNode(tree, 0, static_cast<std::uint32_t>(tree.triangles.size()));
				return tree;
			}

			MeshTree& GetTree(const Mesh& mesh) {
				auto [entry, inserted] = meshTrees.try_emplace(&mesh);
				if (inserted || !entry->second.revision.Matches(mesh)) entry->second = BuildTree(mesh);
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

			void QueryTree(const MeshTree& tree, Object& meshObject, const Bounds& queryBounds, std::vector<std::uint32_t>& output) {
				output.clear();
				if (tree.nodes.empty()) return;
				const Transform& transform = meshObject.transform;
				if (std::abs(transform.scale.x) < axisEpsilon || std::abs(transform.scale.y) < axisEpsilon || std::abs(transform.scale.z) < axisEpsilon) {
					output.assign(tree.order.begin(), tree.order.end());
					return;
				}
				Vector3 axes[3];
				GetAxes(transform, axes);
				Bounds localBounds = EmptyBounds();
				for (int corner = 0; corner < 8; ++corner) {
					const Vector3 point{corner & 1 ? queryBounds.maximum.x : queryBounds.minimum.x, corner & 2 ? queryBounds.maximum.y : queryBounds.minimum.y, corner & 4 ? queryBounds.maximum.z : queryBounds.minimum.z};
					const Vector3 offset = Subtract(point, transform.position);
					Include(localBounds, {Dot(offset, axes[0]) / transform.scale.x, Dot(offset, axes[1]) / transform.scale.y, Dot(offset, axes[2]) / transform.scale.z});
				}
				const Vector3 padding{axisEpsilon, axisEpsilon, axisEpsilon};
				localBounds.minimum = Subtract(localBounds.minimum, padding);
				localBounds.maximum = Add(localBounds.maximum, padding);
				QueryNode(tree, 0, localBounds, output);
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

			bool SweepCapsuleTriangle(const Capsule& capsule, const Vector3 (&triangle)[3], const Vector3& displacement, float winding, bool closed, Contact& contact) {
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
						if (closed && Dot(displacement, outward) >= -axisEpsilon * Length(displacement)) return false;
						const Vector3 normal = distance > axisEpsilon ? Scale(difference, 1.0f / distance) : Scale(outward, -1.0f);
						const Vector3 surfaceNormal = !closed && Dot(outward, difference) > 0.0f ? Scale(outward, -1.0f) : outward;
						if (Dot(normal, surfaceNormal) < -0.9999f && Dot(displacement, surfaceNormal) >= -axisEpsilon * Length(displacement)) return false;
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

			void GetWorldTriangle(Object& meshObject, const Mesh& mesh, const Triangle& source, const Vector3 (&axes)[3], Vector3 (&triangle)[3]) {
				for (int corner = 0; corner < 3; ++corner) triangle[corner] = TransformPoint(meshObject.transform, axes, mesh.data.vertices[source.indices[corner]].position);
			}

			bool FindMeshOverlap(Object& meshObject, const Box& box, Contact& result) {
				auto* mesh = meshObject.GetComponent<Mesh>();
				if (!mesh) return false;
				MeshTree& tree = GetTree(*mesh);
				Vector3 axes[3];
				GetAxes(meshObject.transform, axes);
				auto& candidates = triangleCandidates;
				QueryTree(tree, meshObject, BoxBounds(box), candidates);
				bool found = false;
				for (const auto index : candidates) {
					Vector3 triangle[3];
					GetWorldTriangle(meshObject, *mesh, tree.triangles[index], axes, triangle);
					Contact contact;
					if (OverlapTriangleBox(triangle, box, tree.winding, tree.closed, contact) && (!found || contact.penetration < result.penetration)) {
						result = contact;
						found = true;
					}
				}
				return found;
			}

			bool FindMeshSweep(Object& meshObject, const Box& box, const Vector3& displacement, Contact& result) {
				auto* mesh = meshObject.GetComponent<Mesh>();
				if (!mesh) return false;
				MeshTree& tree = GetTree(*mesh);
				Vector3 axes[3];
				GetAxes(meshObject.transform, axes);
				auto& candidates = triangleCandidates;
				QueryTree(tree, meshObject, SweptBounds(box, displacement), candidates);
				bool found = false;
				for (const auto index : candidates) {
					Vector3 triangle[3];
					GetWorldTriangle(meshObject, *mesh, tree.triangles[index], axes, triangle);
					Contact contact;
					if (SweepTriangleBox(triangle, box, displacement, tree.winding, tree.closed, contact) && (!found || contact.time < result.time)) {
						result = contact;
						found = true;
					}
				}
				return found;
			}

			bool FindCapsuleMeshOverlap(Object& meshObject, const Capsule& capsule, Contact& result) {
				auto* mesh = meshObject.GetComponent<Mesh>();
				if (!mesh) return false;
				MeshTree& tree = GetTree(*mesh);
				Vector3 axes[3];
				GetAxes(meshObject.transform, axes);
				auto& candidates = triangleCandidates;
				QueryTree(tree, meshObject, CapsuleBounds(capsule), candidates);
				bool found = false;
				for (const auto index : candidates) {
					Vector3 triangle[3];
					GetWorldTriangle(meshObject, *mesh, tree.triangles[index], axes, triangle);
					Contact contact;
					if (OverlapCapsuleTriangle(capsule, triangle, tree.winding, tree.closed, contact) && (!found || contact.penetration < result.penetration)) {
						result = contact;
						found = true;
					}
				}
				return found;
			}

			bool FindCapsuleMeshSweep(Object& meshObject, const Capsule& capsule, const Vector3& displacement, Contact& result, const PhysicsBody* floorBody = nullptr) {
				auto* mesh = meshObject.GetComponent<Mesh>();
				if (!mesh) return false;
				MeshTree& tree = GetTree(*mesh);
				Vector3 axes[3];
				GetAxes(meshObject.transform, axes);
				auto& candidates = triangleCandidates;
				QueryTree(tree, meshObject, SweptBounds(capsule, displacement), candidates);
				bool found = false;
				for (const auto index : candidates) {
					Vector3 triangle[3];
					GetWorldTriangle(meshObject, *mesh, tree.triangles[index], axes, triangle);
					Contact contact;
					if (SweepCapsuleTriangle(capsule, triangle, displacement, tree.winding, tree.closed, contact) && PreferCapsuleHit(contact, result, capsule, displacement, floorBody)) {
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
				bool found = false;
				for (const auto& triangle : triangles) {
					Contact contact;
					if (SweepCapsuleTriangle(capsule, triangle, displacement, 1.0f, true, contact) && PreferCapsuleHit(contact, result, capsule, displacement, floorBody)) {
						result = contact;
						found = true;
					}
				}
				return found;
			}

			bool FindCapsuleHit(Scene& scene, Object& object, const Capsule& capsule, const Vector3& displacement, Contact& result, const PhysicsBody* floorBody = nullptr) {
				const Bounds sweptBounds = SweptBounds(capsule, displacement);
				for (auto& obstacle : scene.Objects()) {
					if (obstacle.get() == &object || obstacle->GetComponent<PhysicsBody>()) continue;
					auto* collider = obstacle->GetComponent<Collider>();
					if (!collider) continue;
					Contact contact;
					if (collider->isMesh) {
						if (!FindCapsuleMeshSweep(*obstacle, capsule, displacement, contact, floorBody)) continue;
					}
					else {
						const Box box = WorldBox(*obstacle, *collider);
						if (!Overlaps(sweptBounds, BoxBounds(box)) || !FindCapsuleBoxSweep(box, capsule, displacement, contact, floorBody)) continue;
					}
					if (PreferCapsuleHit(contact, result, capsule, displacement, floorBody)) result = contact;
				}
				return result.hit;
			}

			bool FindCapsuleFloor(Scene& scene, Object& object, Collider& collider, float distance, Contact& result) {
				Capsule capsule = WorldCapsule(object, collider);
				capsule.radius = std::max(axisEpsilon, capsule.radius - skinWidth);
				const PhysicsBody& body = *object.GetComponent<PhysicsBody>();
				const Vector3 displacement{0.0f, -distance, 0.0f};
				Contact contact;
				FindCapsuleHit(scene, object, capsule, displacement, contact, &body);
				if (IsCapsuleSupport(capsule, body, contact)) {
					result = contact;
					return true;
				}
				const bool blockingFloor = IsCapsuleFootContact(capsule, contact);
				const float shrink = capsule.radius * 0.25f;
				capsule.radius -= shrink;
				capsule.start.y -= shrink;
				capsule.end.y += shrink;
				Contact narrowContact;
				FindCapsuleHit(scene, object, capsule, displacement, narrowContact, &body);
				if (!IsCapsuleSupport(capsule, body, narrowContact)) return false;
				if (blockingFloor && std::abs(Dot(Subtract(contact.point, narrowContact.point), narrowContact.surfaceNormal)) > skinWidth) return false;
				result = narrowContact;
				if (blockingFloor) result.time = std::min(result.time, contact.time);
				return true;
			}

			bool FindOverlap(Object& movingObject, Object& obstacle, const Collider& movingCollider, const Collider& obstacleCollider, Contact& contact) {
				if (movingCollider.isCapsule) {
					const Capsule capsule = WorldCapsule(movingObject, movingCollider);
					if (obstacleCollider.isMesh) return FindCapsuleMeshOverlap(obstacle, capsule, contact);
					return FindCapsuleBoxOverlap(WorldBox(obstacle, obstacleCollider), capsule, contact);
				}
				const Box movingBox = WorldBox(movingObject, movingCollider);
				if (obstacleCollider.isMesh) return FindMeshOverlap(obstacle, movingBox, contact);
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
				const float horizontalLength = std::sqrt(contact.normal.x * contact.normal.x + contact.normal.z * contact.normal.z);
				if (horizontalLength > axisEpsilon) {
					contact.normal = {contact.normal.x / horizontalLength, 0.0f, contact.normal.z / horizontalLength};
					contact.penetration /= horizontalLength;
				}
			}

			void ConstrainCapsuleContact(PhysicsBody& body, Contact& contact, bool allowSlope) {
				if (allowSlope && IsWalkable(body, contact)) return;
				if (contact.surfaceNormal.y >= 0.0f) {
					const float horizontalLength = std::sqrt(contact.normal.x * contact.normal.x + contact.normal.z * contact.normal.z);
					if (horizontalLength > axisEpsilon) {
						contact.normal = {contact.normal.x / horizontalLength, 0.0f, contact.normal.z / horizontalLength};
						contact.penetration /= horizontalLength;
					}
				}
			}

			void ResolveOverlaps(Scene& scene, Object& object, PhysicsBody& body, Collider& collider) {
				auto& objects = scene.Objects();
				for (int iteration = 0; iteration < maxOverlapIterations; ++iteration) {
					bool corrected = false;
					for (auto& obstacle : objects) {
						if (obstacle.get() == &object || obstacle->GetComponent<PhysicsBody>()) continue;
						auto* obstacleCollider = obstacle->GetComponent<Collider>();
						if (!obstacleCollider) continue;
						Contact contact;
						if (!FindOverlap(object, *obstacle, collider, *obstacleCollider, contact)) continue;
						if (collider.isCapsule) ConstrainCapsuleContact(body, contact, true);
						else ConstrainContact(body, contact);
						object.transform.position += Scale(contact.normal, -(contact.penetration + skinWidth));
						if (!collider.isCapsule || !IsWalkable(body, contact)) RemoveIntoSurface(body, contact.normal);
						corrected = true;
					}
					if (!corrected) return;
				}
			}

			bool FindEarliestHit(Scene& scene, Object& object, Collider& collider, const Vector3& displacement, Contact& result) {
				if (collider.isCapsule) return FindCapsuleHit(scene, object, WorldCapsule(object, collider), displacement, result);
				const Box movingBox = WorldBox(object, collider);
				const Bounds sweptBounds = SweptBounds(movingBox, displacement);
				for (auto& obstacle : scene.Objects()) {
					if (obstacle.get() == &object || obstacle->GetComponent<PhysicsBody>()) continue;
					auto* obstacleCollider = obstacle->GetComponent<Collider>();
					if (!obstacleCollider) continue;
					Contact contact;
					if (obstacleCollider->isMesh) {
						if (!FindMeshSweep(*obstacle, movingBox, displacement, contact)) continue;
					}
					else {
						const Box obstacleBox = WorldBox(*obstacle, *obstacleCollider);
						if (!Overlaps(sweptBounds, BoxBounds(obstacleBox)) || !SweepBoxes(movingBox, obstacleBox, displacement, contact)) continue;
					}
					if (!result.hit || contact.time < result.time) result = contact;
				}
				return result.hit;
			}

			void MoveAndSlide(Scene& scene, Object& object, PhysicsBody& body, Collider& collider, float deltaTime) {
				Vector3 remaining = Scale(body.velocity, deltaTime);
				for (int iteration = 0; iteration < maxMoveIterations; ++iteration) {
					const float distance = Length(remaining);
					if (distance < axisEpsilon) return;
					Contact contact;
					if (!FindEarliestHit(scene, object, collider, remaining, contact)) {
						object.transform.position += remaining;
						return;
					}
					const float safeTime = std::max(0.0f, contact.time - skinWidth / distance);
					object.transform.position += Scale(remaining, safeTime);
					remaining = Scale(remaining, 1.0f - contact.time);
					ConstrainContact(body, contact);
					const float intoSurface = Dot(remaining, contact.normal);
					if (intoSurface > 0.0f) remaining += Scale(contact.normal, -intoSurface);
					RemoveIntoSurface(body, contact.normal);
					if (body.grounded && body.velocity.y < 0.0f) body.velocity.y = 0.0f;
				}
			}

			bool TryCapsuleTransition(Scene& scene, Object& object, PhysicsBody& body, Collider& collider, const Vector3& remaining, const Contact& obstacle, const Contact& support, float riseLimit, bool allowFall) {
				if (riseLimit <= 0.0f || obstacle.point.y > WorldCapsule(object, collider).end.y) return false;
				const Vector3 forward{remaining.x, 0.0f, remaining.z};
				if (Length(forward) <= axisEpsilon) return false;
				const Vector3 original = object.transform.position;
				const Vector3 up{0.0f, riseLimit, 0.0f};
				Contact ceiling;
				float rise = riseLimit;
				if (FindEarliestHit(scene, object, collider, up, ceiling)) rise *= ceiling.time;
				if (rise <= axisEpsilon) return false;
				object.transform.position.y += rise;
				Contact ahead;
				if (FindEarliestHit(scene, object, collider, forward, ahead)) {
					object.transform.position = original;
					return false;
				}
				object.transform.position += forward;
				const float drop = rise + floorProbeDistance;
				Contact floor;
				if (!FindCapsuleFloor(scene, object, collider, drop, floor)) {
					if (allowFall) {
						body.grounded = false;
						return true;
					}
					object.transform.position = original;
					return false;
				}
				object.transform.position.y -= std::max(0.0f, floor.time * drop - skinWidth);
				if (object.transform.position.y - original.y > riseLimit + axisEpsilon || floor.point.y - support.point.y > riseLimit + axisEpsilon) {
					object.transform.position = original;
					return false;
				}
				Contact landing;
				if (!FindCapsuleFloor(scene, object, collider, floorProbeDistance, landing)) {
					object.transform.position = original;
					return false;
				}
				body.grounded = true;
				return true;
			}

			void MoveCapsuleAndSlide(Scene& scene, Object& object, PhysicsBody& body, Collider& collider, Vector3 remaining, const Contact* floor) {
				const bool walking = floor != nullptr;
				for (int iteration = 0; iteration < maxMoveIterations; ++iteration) {
					const float distance = Length(remaining);
					if (distance < axisEpsilon) return;
					Contact contact;
					if (!FindEarliestHit(scene, object, collider, remaining, contact)) {
						object.transform.position += remaining;
						return;
					}
					object.transform.position += Scale(remaining, contact.time);
					remaining = Scale(remaining, 1.0f - contact.time);
					const bool walkable = IsWalkable(body, contact);
					if (walking && (!walkable || -contact.normal.y < std::cos(body.slopeLimit * Math::degreesToRadians))) {
						const Capsule capsule = WorldCapsule(object, collider);
						const bool footEdge = contact.point.y <= capsule.start.y && -contact.normal.y >= std::cos(body.slopeLimit * Math::degreesToRadians) && std::abs(Dot(Subtract(contact.point, floor->point), floor->surfaceNormal)) <= skinWidth;
						if (footEdge && TryCapsuleTransition(scene, object, body, collider, remaining, contact, *floor, skinWidth + axisEpsilon * 2.0f, true)) return;
						if (TryCapsuleTransition(scene, object, body, collider, remaining, contact, *floor, body.stepHeight, false)) return;
					}
					const bool descending = remaining.y < 0.0f;
					const bool followSurface = (walkable && (walking || descending)) || (!walkable && contact.surfaceNormal.y > 0.0f && descending);
					Contact response = contact;
					if (!followSurface) ConstrainCapsuleContact(body, response, false);
					const float intoSurface = Dot(remaining, response.normal);
					if (intoSurface > 0.0f) remaining += Scale(response.normal, -intoSurface);
					if (!walkable && contact.surfaceNormal.y > 0.0f && remaining.y > 0.0f) remaining.y = 0.0f;
					if (walking && !walkable && remaining.y > 0.0f) remaining.y = 0.0f;
					if (!followSurface || contact.surfaceNormal.y < 0.0f) RemoveIntoSurface(body, response.normal);
				}
			}

			void SimulateCapsule(Scene& scene, Object& object, PhysicsBody& body, Collider& collider, float deltaTime) {
				ResolveOverlaps(scene, object, body, collider);
				Contact floor;
				const bool walking = body.velocity.y <= 0.0f && FindCapsuleFloor(scene, object, collider, floorProbeDistance, floor);
				body.grounded = walking;
				Vector3 displacement;
				if (walking) {
					if (body.velocity.y < 0.0f) body.velocity.y = 0.0f;
					displacement = {body.velocity.x * deltaTime, 0.0f, body.velocity.z * deltaTime};
					if (floor.surfaceNormal.y > axisEpsilon) displacement.y = -(floor.surfaceNormal.x * displacement.x + floor.surfaceNormal.z * displacement.z) / floor.surfaceNormal.y;
				}
				else {
					if (body.useGravity) body.velocity += scene.gravity * deltaTime;
					displacement = Scale(body.velocity, deltaTime);
				}
				MoveCapsuleAndSlide(scene, object, body, collider, displacement, walking ? &floor : nullptr);
				if (body.velocity.y <= 0.0f) {
					Contact nextFloor;
					const float snapDistance = walking ? std::max(floorProbeDistance, body.stepHeight) : floorProbeDistance;
					if (FindCapsuleFloor(scene, object, collider, snapDistance, nextFloor)) {
						object.transform.position.y -= std::max(0.0f, nextFloor.time * snapDistance - skinWidth);
						body.grounded = true;
						body.velocity.y = 0.0f;
					}
					else body.grounded = false;
				}
				else body.grounded = false;
			}

			void ResolveDynamicPairs(Scene& scene) {
				auto& objects = scene.Objects();
				for (std::size_t i = 0; i < objects.size(); ++i) {
					auto* bodyA = objects[i]->GetComponent<PhysicsBody>();
					auto* colliderA = objects[i]->GetComponent<Collider>();
					if (!bodyA || !colliderA || colliderA->isMesh) continue;
					for (std::size_t j = i + 1; j < objects.size(); ++j) {
						auto* bodyB = objects[j]->GetComponent<PhysicsBody>();
						auto* colliderB = objects[j]->GetComponent<Collider>();
						if (!bodyB || !colliderB || colliderB->isMesh) continue;
						Contact contact;
						if (!OverlapBoxes(WorldBox(*objects[i], *colliderA), WorldBox(*objects[j], *colliderB), contact)) continue;
						objects[i]->transform.position += Scale(contact.normal, -contact.penetration * 0.5f);
						objects[j]->transform.position += Scale(contact.normal, contact.penetration * 0.5f);
						RemoveIntoSurface(*bodyA, contact.normal);
						RemoveIntoSurface(*bodyB, Scale(contact.normal, -1.0f));
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
			for (auto& object : scene.Objects()) {
				auto* collider = object->GetComponent<Collider>();
				if (!collider) continue;
				float distance = closest;
				Vector3 normal;
				bool found = false;
				if (collider->isMesh) {
					auto* mesh = object->GetComponent<Mesh>();
					if (!mesh) continue;
					MeshTree& tree = GetTree(*mesh);
					Bounds bounds = EmptyBounds();
					Include(bounds, origin);
					Include(bounds, Add(origin, Scale(ray, closest)));
					QueryTree(tree, *object, bounds, triangleCandidates);
					Vector3 axes[3];
					GetAxes(object->transform, axes);
					for (const auto index : triangleCandidates) {
						Vector3 triangle[3];
						GetWorldTriangle(*object, *mesh, tree.triangles[index], axes, triangle);
						const Vector3 edgeA = Subtract(triangle[1], triangle[0]);
						const Vector3 edgeB = Subtract(triangle[2], triangle[0]);
						const Vector3 perpendicular = Cross(ray, edgeB);
						const float determinant = Dot(edgeA, perpendicular);
						if (std::abs(determinant) <= axisEpsilon * Length(edgeA) * Length(edgeB)) continue;
						const Vector3 offset = Subtract(origin, triangle[0]);
						const float u = Dot(offset, perpendicular) / determinant;
						if (u < 0.0f || u > 1.0f) continue;
						const Vector3 cross = Cross(offset, edgeA);
						const float v = Dot(ray, cross) / determinant;
						if (v < 0.0f || u + v > 1.0f) continue;
						const float candidate = Dot(edgeB, cross) / determinant;
						if (candidate < 0.0f || candidate > distance) continue;
						normal = Cross(edgeA, edgeB);
						normal = Scale(normal, (Dot(normal, ray) > 0.0f ? -1.0f : 1.0f) / Length(normal));
						distance = candidate;
						found = true;
					}
				}
				else if (collider->isCapsule) {
					const Capsule capsule = WorldCapsule(*object, *collider);
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
					const Box box = WorldBox(*object, *collider);
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
					hit = {object.get(), collider, Add(origin, Scale(ray, distance)), normal, distance};
				}
			}
			return hit.collider != nullptr;
		}

		void Initialize(Scene& scene) {
			Loading::Timer loading("Physics collider initialization");
			meshBounds.clear();
			meshTrees.clear();
			for (auto& object : scene.Objects()) {
				auto* collider = object->GetComponent<Collider>();
				auto* mesh = object->GetComponent<Mesh>();
				if (collider && collider->isMesh && mesh) GetTree(*mesh);
			}
		}

		void Simulate(Scene& scene, float deltaTime) {
			liveMeshes.clear();
			liveTreeMeshes.clear();
			for (auto& object : scene.Objects()) {
				if (auto* collider = object->GetComponent<Collider>()) {
					if (auto* mesh = object->GetComponent<Mesh>()) {
						liveMeshes.push_back(mesh);
						if (collider->isMesh) liveTreeMeshes.push_back(mesh);
					}
				}
			}
			std::sort(liveMeshes.begin(), liveMeshes.end(), std::less<const Mesh*>{});
			std::sort(liveTreeMeshes.begin(), liveTreeMeshes.end(), std::less<const Mesh*>{});
			for (auto entry = meshBounds.begin(); entry != meshBounds.end();) {
				if (!std::binary_search(liveMeshes.begin(), liveMeshes.end(), entry->first, std::less<const Mesh*>{})) entry = meshBounds.erase(entry);
				else ++entry;
			}
			for (auto entry = meshTrees.begin(); entry != meshTrees.end();) {
				if (!std::binary_search(liveTreeMeshes.begin(), liveTreeMeshes.end(), entry->first, std::less<const Mesh*>{})) entry = meshTrees.erase(entry);
				else ++entry;
			}
			for (auto& object : scene.Objects()) {
				if (auto* body = object->GetComponent<PhysicsBody>()) {
					body->previousPosition = object->transform.position;
					if (auto* collider = object->GetComponent<Collider>()) {
						if (collider->isCapsule) SimulateCapsule(scene, *object, *body, *collider, deltaTime);
						else {
							body->grounded = false;
							if (body->useGravity) body->velocity += scene.gravity * deltaTime;
							ResolveOverlaps(scene, *object, *body, *collider);
							MoveAndSlide(scene, *object, *body, *collider, deltaTime);
						}
					}
					else {
							body->grounded = false;
							if (body->useGravity) body->velocity += scene.gravity * deltaTime;
							object->transform.position += body->velocity * deltaTime;
						}
					if (body->grounded && body->velocity.y < 0.0f) body->velocity.y = 0.0f;
				}
			}
			ResolveDynamicPairs(scene);
		}

		void Shutdown() {
			std::vector<std::uint32_t>().swap(triangleCandidates);
			std::vector<const Mesh*>().swap(liveMeshes);
			std::vector<const Mesh*>().swap(liveTreeMeshes);
			meshBounds.clear();
			meshTrees.clear();
		}
	}
}
