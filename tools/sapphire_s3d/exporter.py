import bpy
import math
import os
import struct
from bpy.props import StringProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix, Vector


AXIS = Matrix(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))
BASIS = AXIS.to_4x4()


def write_string(file, value):
    data = value.encode("utf-8")
    file.write(struct.pack("<I", len(data)))
    file.write(data)


def write_vec3(file, value):
    file.write(struct.pack("<3f", value[0], value[1], value[2]))


def converted_transform(matrix):
    converted = BASIS @ matrix @ BASIS.inverted()
    location, rotation, scale = converted.decompose()
    euler = rotation.to_euler("XYZ")
    return (location.x, location.y, location.z), tuple(math.degrees(value) for value in (euler.x, euler.y, euler.z)), (scale.x, scale.y, scale.z)


def object_transform(obj):
    return converted_transform(obj.matrix_basis)


def material_image(material, socket_name):
    if not material or not material.use_nodes or not material.node_tree:
        return None
    principled = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
    if not principled:
        return None
    socket = principled.inputs.get(socket_name)
    if not socket or not socket.is_linked:
        return None
    pending = [link.from_node for link in socket.links]
    visited = set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        if node.type == "TEX_IMAGE":
            return node.image
        pending.extend(link.from_node for input_socket in node.inputs for link in input_socket.links)
    return None


def image_path(image, export_path):
    if not image or not image.filepath:
        return ""
    try:
        path = bpy.path.abspath(image.filepath)
    except Exception:
        return ""
    if not os.path.isfile(path):
        return ""
    return os.path.relpath(path, os.path.dirname(export_path)).replace(os.sep, "/")


def material_record(material, export_path):
    record = {
        "color": (1.0, 1.0, 1.0), "roughness": 1.0, "metallic": 0.0,
        "normal_scale": 1.0, "ao_strength": 1.0, "displacement_scale": 0.05,
        "uv_scale": (1.0, 1.0), "uv_offset": (0.0, 0.0), "flip_normal_y": False,
        "transparent": False, "unlit": False, "opacity": 1.0,
        "textures": ["", "", "", "", "", ""], "modifier": "",
    }
    if not material:
        return record
    record["transparent"] = material.surface_render_method == "BLENDED" if hasattr(material, "surface_render_method") else material.blend_method == "BLEND"
    if material.use_nodes and material.node_tree:
        principled = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
        if principled:
            base = principled.inputs.get("Base Color")
            if base:
                record["color"] = tuple(base.default_value[:3])
                record["opacity"] = float(base.default_value[3])
            metallic = principled.inputs.get("Metallic")
            roughness = principled.inputs.get("Roughness")
            alpha = principled.inputs.get("Alpha")
            if metallic:
                record["metallic"] = float(metallic.default_value)
            if roughness:
                record["roughness"] = float(roughness.default_value)
            if alpha:
                record["opacity"] *= float(alpha.default_value)
            images = [material_image(material, name) for name in ("Base Color", "Normal", "Roughness")]
            record["textures"][:3] = [image_path(image, export_path) for image in images]
    return record


def write_material(file, record):
    write_vec3(file, record["color"])
    file.write(struct.pack("<9f", record["roughness"], record["metallic"], record["normal_scale"], record["ao_strength"], record["displacement_scale"], *record["uv_scale"], *record["uv_offset"]))
    file.write(struct.pack("<3Bf", int(record["flip_normal_y"]), int(record["transparent"]), int(record["unlit"]), record["opacity"]))
    for path in record["textures"]:
        write_string(file, path)
    write_string(file, record["modifier"])


def armature_for(obj):
    for modifier in obj.modifiers:
        if modifier.type == "ARMATURE" and modifier.object:
            return modifier.object
    return None


def bone_records(obj, armature):
    if not armature or armature.type != "ARMATURE":
        return []
    bones = list(armature.data.bones)
    indices = {bone.name: index for index, bone in enumerate(bones)}
    mesh_inverse = obj.matrix_world.inverted()
    bind_matrices = {bone.name: mesh_inverse @ armature.matrix_world @ bone.matrix_local for bone in bones}
    records = []
    for bone in bones:
        local = bind_matrices[bone.name]
        if bone.parent:
            local = bind_matrices[bone.parent.name].inverted() @ local
        position, rotation, scale = converted_transform(local)
        records.append((bone.name, indices[bone.parent.name] if bone.parent else -1, position, rotation, scale))
    return records


def vertex_weights(obj, vertex_index, bone_indices):
    groups = {group.index: group.name for group in obj.vertex_groups}
    weights = []
    for assignment in obj.data.vertices[vertex_index].groups:
        bone_index = bone_indices.get(groups.get(assignment.group, ""))
        if bone_index is not None and assignment.weight > 0.0:
            weights.append((bone_index, assignment.weight))
    weights.sort(key=lambda item: item[1], reverse=True)
    weights = weights[:4]
    total = sum(weight for _, weight in weights)
    if total <= 0.0:
        return ((0, 0, 0, 0), (65535, 0, 0, 0))
    normalized = [weight / total for _, weight in weights]
    quantized = [int(round(weight * 65535.0)) for weight in normalized]
    quantized[0] += 65535 - sum(quantized)
    return (tuple(index for index, _ in weights) + (0,) * (4 - len(weights)), tuple(quantized) + (0,) * (4 - len(weights)))


def mesh_payload(obj, export_path):
    mesh = obj.data
    mesh.calc_loop_triangles()
    materials = list(mesh.materials)
    if not materials:
        materials = [None]
    used = [[] for _ in materials]
    for triangle in mesh.loop_triangles:
        material_index = min(triangle.material_index, len(materials) - 1)
        used[material_index].append(triangle)

    armature = armature_for(obj)
    bones = bone_records(obj, armature)
    bone_indices = {bone[0]: index for index, bone in enumerate(bones)}
    uv_layer = mesh.uv_layers.active
    color_layer = mesh.color_attributes.active_color if hasattr(mesh, "color_attributes") else None
    vertices = []
    indices = []
    skin = []
    sections = []
    material_records = [material_record(material, export_path) for material in materials]
    vertex_lookup = {}

    for material_index, triangles in enumerate(used):
        first_index = len(indices)
        for triangle in triangles:
            for loop_index in triangle.loops:
                loop = mesh.loops[loop_index]
                source = mesh.vertices[loop.vertex_index]
                position = AXIS @ source.co
                normal = AXIS @ mesh.corner_normals[loop_index].vector
                uv = uv_layer.data[loop_index].uv if uv_layer else (0.0, 0.0)
                color = (1.0, 1.0, 1.0, 1.0)
                if color_layer:
                    color_value = color_layer.data[loop_index if color_layer.domain == "CORNER" else loop.vertex_index].color
                    color = tuple(color_value)
                influence = vertex_weights(obj, loop.vertex_index, bone_indices) if bones else ((0, 0, 0, 0), (65535, 0, 0, 0))
                key = (tuple(position), tuple(normal), tuple(uv), tuple(color), influence)
                vertex_index = vertex_lookup.get(key)
                if vertex_index is None:
                    vertex_index = len(vertices)
                    vertex_lookup[key] = vertex_index
                    vertices.append((tuple(position), tuple(normal), tuple(uv), tuple(color)))
                    if bones:
                        skin.append(influence)
                indices.append(vertex_index)
        count = len(indices) - first_index
        if count:
            sections.append((first_index, count, material_index))

    return vertices, indices, sections, material_records, bones, skin


def write_object(file, obj, export_path):
    is_light = obj.type == "LIGHT"
    write_string(file, obj.name)
    file.write(struct.pack("<BB", int(is_light), int(bool(obj.get("isParent", False)))))
    position, rotation, scale = object_transform(obj)
    write_vec3(file, position)
    write_vec3(file, rotation)
    write_vec3(file, scale)
    if is_light:
        light = obj.data
        light_types = {"SUN": 0, "POINT": 1, "SPOT": 2}
        light_type = light_types.get(light.type)
        if light_type is None:
            raise ValueError("Unsupported Blender light type: " + light.type)
        color = tuple(light.color)
        outer = math.degrees(light.spot_size) if light.type == "SPOT" else 30.0
        inner = outer * (1.0 - light.spot_blend) if light.type == "SPOT" else 25.0
        file.write(struct.pack("<B3f4fBf", light_type, *color, light.energy, light.cutoff_distance if light.type != "SUN" else 1000.0, inner, outer, int(light.use_shadow), 0.02))
        return

    vertices, indices, sections, materials, bones, skin = mesh_payload(obj, export_path)
    file.write(struct.pack("<5I", len(vertices), len(indices), len(sections), len(materials), len(bones)))
    for position, normal, uv, color in vertices:
        file.write(struct.pack("<12f", *position, *normal, *uv, *color))
    if indices:
        file.write(struct.pack("<%dI" % len(indices), *indices))
    for section in sections:
        file.write(struct.pack("<3I", *section))
    for material in materials:
        write_material(file, material)
    for name, parent_index, position, rotation, scale in bones:
        write_string(file, name)
        file.write(struct.pack("<i", parent_index))
        write_vec3(file, position)
        write_vec3(file, rotation)
        write_vec3(file, scale)
    file.write(struct.pack("<I", len(skin)))
    for indices4, weights4 in skin:
        file.write(struct.pack("<8H", *indices4, *weights4))


class ExportSapphireS3D(bpy.types.Operator, ExportHelper):
    bl_idname = "export_scene.sapphire_s3d"
    bl_label = "Sapphire (.s3d)"
    bl_options = {"PRESET"}

    filename_ext = ".s3d"
    filter_glob: StringProperty(default="*.s3d", options={"HIDDEN"})

    def execute(self, context):
        objects = [obj for obj in context.selected_objects if obj.type in {"MESH", "LIGHT"}]
        if not objects:
            self.report({"ERROR"}, "Select at least one mesh or light object")
            return {"CANCELLED"}
        if sum(bool(obj.get("isParent", False)) for obj in objects) > 1:
            self.report({"ERROR"}, "Only one selected object may have isParent enabled")
            return {"CANCELLED"}
        objects.sort(key=lambda obj: not bool(obj.get("isParent", False)))
        try:
            with open(self.filepath, "wb") as file:
                file.write(b"S3D\0")
                file.write(struct.pack("<I", len(objects)))
                for obj in objects:
                    write_object(file, obj, self.filepath)
        except (OSError, ValueError, struct.error) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "scene.s3d"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


def menu_export(self, context):
    self.layout.operator(ExportSapphireS3D.bl_idname, text="Sapphire (.s3d)")


