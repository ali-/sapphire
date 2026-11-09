import bpy
import hashlib
import json
import math
import os
import struct
from collections.abc import Mapping, Sequence
from bpy.props import BoolProperty, StringProperty
from bpy_extras.io_utils import ExportHelper


def write_string(file, value):
    data = value.encode("utf-8")
    file.write(struct.pack("<I", len(data)))
    file.write(data)


def write_vec3(file, value):
    file.write(struct.pack("<3f", value[0], value[1], value[2]))


def export_transform(matrix):
    location, rotation, scale = matrix.decompose()
    euler = rotation.to_euler("XYZ")
    return (location.x, location.y, location.z), tuple(math.degrees(value) for value in (euler.x, euler.y, euler.z)), (scale.x, scale.y, scale.z)


def parameter_value(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite custom property value")
        return str(value)
    if isinstance(value, str):
        return value
    def json_value(item):
        if item is None or isinstance(item, (bool, int, str)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("Non-finite custom property value")
            return item
        if isinstance(item, Mapping) or hasattr(item, "items"):
            return {str(key): json_value(entry) for key, entry in item.items()}
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            return [json_value(entry) for entry in item]
        if hasattr(item, "to_list"):
            return json_value(item.to_list())
        raise ValueError("Unsupported custom property value: " + type(item).__name__)
    return json.dumps(json_value(value), separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def export_objects(context, objects):
    depths = {}
    def visit(collection, depth):
        depths[collection] = max(depth, depths.get(collection, -1))
        for child in collection.children:
            visit(child, depth + 1)
    visit(context.scene.collection, 0)
    groups = {}
    for obj in objects:
        memberships = [(depths[collection], collection) for collection in obj.users_collection if collection in depths]
        if not memberships:
            raise ValueError("Object is not in the active scene: " + obj.name)
        depth = max(item[0] for item in memberships)
        closest = [collection for level, collection in memberships if level == depth]
        if len(closest) != 1:
            raise ValueError("Ambiguous deepest collection for " + obj.name)
        if "isParent" in obj and not isinstance(obj["isParent"], bool):
            raise ValueError("isParent must be boolean on " + obj.name)
        groups.setdefault(closest[0], []).append(obj)
    if len({obj.name for obj in objects}) != len(objects):
        raise ValueError("Exported object names must be unique")
    entries = []
    for collection in sorted(groups, key=lambda item: item.name):
        members = groups[collection]
        designated = [obj for obj in collection.objects if obj.get("isParent") is True]
        if len(designated) > 1:
            raise ValueError("Multiple parents in collection " + collection.name)
        parent = designated[0] if designated else None
        if parent and parent not in members:
            raise ValueError("Collection parent must be selected for export: " + parent.name)
        if parent and "parent" in parent:
            raise ValueError("Parent object has a parent property: " + parent.name)
        ordered = ([parent] if parent else []) + sorted((obj for obj in members if obj != parent), key=lambda obj: obj.name)
        for obj in ordered:
            actual_parent = parent if obj != parent else None
            if "parent" in obj and (not actual_parent or parameter_value(obj["parent"]) != actual_parent.name):
                raise ValueError("Conflicting parent property on " + obj.name)
            hierarchy = ("parent", parent.name) if actual_parent else ("isParent", "True" if obj == parent else "False")
            parameters = [hierarchy]
            parameters.extend((key, parameter_value(obj[key])) for key in sorted(obj.keys()) if key not in {"isParent", "parent"})
            entries.append((obj, actual_parent, parameters))
    return entries


def material_record(material):
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
    return record


def linked_input(node, name):
    socket = node.inputs.get(name) if node else None
    return socket if socket and socket.is_linked else None


def bake_channels(material):
    if not material or not material.use_nodes or not material.node_tree:
        return set()
    nodes = material.node_tree.nodes
    principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output), None)
    channels = set()
    if linked_input(principled, "Base Color"):
        channels.add("base")
    if linked_input(principled, "Alpha"):
        channels.add("alpha")
        channels.add("base")
    if linked_input(principled, "Normal"):
        channels.add("normal")
    if linked_input(principled, "Roughness"):
        channels.add("roughness")
    if any(node.type == "AMBIENT_OCCLUSION" for node in nodes):
        channels.add("ao")
    if linked_input(output, "Displacement"):
        channels.add("displacement")
    return channels


def bake_size(material):
    dimensions = [(node.image.size[0], node.image.size[1]) for node in material.node_tree.nodes if node.type == "TEX_IMAGE" and node.image and any(socket.is_linked for socket in node.outputs) and node.image.size[0] and node.image.size[1]]
    return max(dimensions, key=lambda size: size[0] * size[1]) if dimensions else (1024, 1024)


def emit_socket(material, channel):
    tree = material.node_tree
    nodes = tree.nodes
    output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output), None)
    principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if not output:
        raise ValueError("Material has no active output: " + material.name)
    if channel == "ao":
        ao = next((node for node in nodes if node.type == "AMBIENT_OCCLUSION"), None)
        source = ao.outputs.get("AO") if ao else None
    elif channel == "displacement":
        connection = linked_input(output, "Displacement")
        displacement = connection.links[0].from_node if connection else None
        if not displacement or displacement.type != "DISPLACEMENT":
            raise ValueError("Unsupported displacement graph in " + material.name)
        source = displacement.inputs.get("Height")
        scale = displacement.inputs.get("Scale")
        midlevel = displacement.inputs.get("Midlevel")
        if not source or not scale or not midlevel:
            raise ValueError("Invalid displacement node in " + material.name)
        subtract = nodes.new("ShaderNodeMath")
        subtract.operation = "SUBTRACT"
        subtract.inputs[1].default_value = midlevel.default_value
        if source.is_linked:
            tree.links.new(source.links[0].from_socket, subtract.inputs[0])
        else:
            subtract.inputs[0].default_value = source.default_value
        if midlevel.is_linked:
            tree.links.new(midlevel.links[0].from_socket, subtract.inputs[1])
        multiply = nodes.new("ShaderNodeMath")
        multiply.operation = "MULTIPLY"
        tree.links.new(subtract.outputs[0], multiply.inputs[0])
        if scale.is_linked:
            tree.links.new(scale.links[0].from_socket, multiply.inputs[1])
        else:
            multiply.inputs[1].default_value = scale.default_value
        add = nodes.new("ShaderNodeMath")
        add.operation = "ADD"
        add.inputs[1].default_value = 0.5
        tree.links.new(multiply.outputs[0], add.inputs[0])
        source = add.outputs[0]
    else:
        socket_name = {"base": "Base Color", "alpha": "Alpha", "roughness": "Roughness"}[channel]
        source = principled.inputs.get(socket_name) if principled else None
    if not source:
        raise ValueError("Material has no " + channel + " output: " + material.name)
    emission = nodes.new("ShaderNodeEmission")
    if source.is_output:
        tree.links.new(source, emission.inputs[0])
    elif source.is_linked:
        tree.links.new(source.links[0].from_socket, emission.inputs[0])
    else:
        value = source.default_value
        emission.inputs[0].default_value = (value, value, value, 1.0) if isinstance(value, float) else value
    for link in list(output.inputs["Surface"].links):
        tree.links.remove(link)
    tree.links.new(emission.outputs[0], output.inputs["Surface"])
    return emission


def bake_materials(context, obj, materials, records, export_path, texture_dir):
    channels = [bake_channels(material) for material in materials]
    if not any(channels):
        return
    if not obj.data.uv_layers.active:
        raise ValueError("Mesh needs a UV map for material baking: " + obj.name)
    scene = context.scene
    selected = list(context.selected_objects)
    active = context.view_layer.objects.active
    active_mode = active.mode if active else "OBJECT"
    engine = scene.render.engine
    original_mesh = obj.data
    temporary_mesh = original_mesh.copy()
    images = []
    base_images = {}
    digest = hashlib.sha256(obj.name.encode("utf-8")).hexdigest()[:16]
    prefix = bpy.path.clean_name(obj.name) + "_" + digest
    os.makedirs(texture_dir, exist_ok=True)
    try:
        if active and active_mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        obj.data = temporary_mesh
        for selected_obj in selected:
            selected_obj.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        scene.render.engine = "CYCLES"
        for channel in ("base", "alpha", "normal", "roughness", "ao", "displacement"):
            if not any(channel in needed for needed in channels):
                continue
            temporary_materials = []
            targets = []
            try:
                temporary_mesh.materials.clear()
                for index, source in enumerate(materials):
                    material = source.copy() if source else bpy.data.materials.new("S3D Bake")
                    material.use_nodes = True
                    temporary_materials.append(material)
                    temporary_mesh.materials.append(material)
                    width, height = bake_size(material) if source and channel in channels[index] else (1, 1)
                    image = bpy.data.images.new("S3D " + channel, width=width, height=height, alpha=True)
                    if channel != "base":
                        image.colorspace_settings.name = "Non-Color"
                    images.append(image)
                    targets.append(image)
                    target = material.node_tree.nodes.new("ShaderNodeTexImage")
                    target.image = image
                    if channel != "normal":
                        if channel in channels[index]:
                            emit_socket(material, channel)
                        else:
                            emit_socket_constant(material)
                    for node in material.node_tree.nodes:
                        node.select = False
                    target.select = True
                    material.node_tree.nodes.active = target
                result = bpy.ops.object.bake(type="NORMAL" if channel == "normal" else "EMIT", use_clear=True, normal_space="TANGENT", uv_layer=temporary_mesh.uv_layers.active.name)
                if "FINISHED" not in result:
                    raise ValueError("Could not bake " + channel + " for " + obj.name)
                for index, image in enumerate(targets):
                    if channel not in channels[index]:
                        continue
                    if channel == "alpha":
                        base = base_images[index]
                        pixels = list(base.pixels[:])
                        alpha_pixels = image.pixels[:]
                        for offset in range(0, len(pixels), 4):
                            pixels[offset + 3] = alpha_pixels[offset]
                        base.pixels[:] = pixels
                        records[index]["opacity"] = 1.0
                        continue
                    if channel == "base":
                        base_images[index] = image
                        records[index]["color"] = (1.0, 1.0, 1.0)
                        if "alpha" in channels[index]:
                            continue
                    elif channel == "roughness":
                        records[index]["roughness"] = 1.0
                    elif channel == "normal":
                        records[index]["normal_scale"] = 1.0
                    elif channel == "displacement":
                        records[index]["displacement_scale"] = 1.0
                    path = os.path.join(texture_dir, f"{prefix}_{index}_{channel}.png")
                    image.filepath_raw = path
                    image.file_format = "PNG"
                    image.save()
                    slot = {"base": 0, "normal": 1, "roughness": 2, "ao": 3, "displacement": 4}[channel]
                    records[index]["textures"][slot] = os.path.relpath(path, os.path.dirname(export_path)).replace(os.sep, "/")
            finally:
                temporary_mesh.materials.clear()
                for material in temporary_materials:
                    bpy.data.materials.remove(material)
        for index, image in base_images.items():
            if "alpha" not in channels[index]:
                continue
            path = os.path.join(texture_dir, f"{prefix}_{index}_base.png")
            image.filepath_raw = path
            image.file_format = "PNG"
            image.save()
            records[index]["textures"][0] = os.path.relpath(path, os.path.dirname(export_path)).replace(os.sep, "/")
    finally:
        obj.data = original_mesh
        scene.render.engine = engine
        for image in images:
            bpy.data.images.remove(image)
        bpy.data.meshes.remove(temporary_mesh)
        for selected_obj in selected:
            selected_obj.select_set(True)
        context.view_layer.objects.active = active
        if active and active_mode != "OBJECT":
            bpy.ops.object.mode_set(mode=active_mode)


def emit_socket_constant(material):
    tree = material.node_tree
    output = next(node for node in tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output)
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.name = "S3D Bake Emit"
    for link in list(output.inputs["Surface"].links):
        tree.links.remove(link)
    tree.links.new(emission.outputs[0], output.inputs["Surface"])
    return emission


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
        position, rotation, scale = export_transform(local)
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


def mesh_payload(obj, mesh, apply_modifiers, material_records):
    mesh.calc_loop_triangles()
    materials = list(mesh.materials)
    if not materials:
        materials = [None]
    used = [[] for _ in materials]
    for triangle in mesh.loop_triangles:
        material_index = min(triangle.material_index, len(materials) - 1)
        used[material_index].append(triangle)

    armature = None if apply_modifiers else armature_for(obj)
    bones = bone_records(obj, armature)
    bone_indices = {bone[0]: index for index, bone in enumerate(bones)}
    uv_layer = mesh.uv_layers.active
    color_layer = mesh.color_attributes.active_color if hasattr(mesh, "color_attributes") else None
    vertices = []
    indices = []
    skin = []
    sections = []
    vertex_lookup = {}

    for material_index, triangles in enumerate(used):
        first_index = len(indices)
        for triangle in triangles:
            for loop_index in triangle.loops:
                loop = mesh.loops[loop_index]
                source = mesh.vertices[loop.vertex_index]
                position = source.co
                normal = mesh.corner_normals[loop_index].vector
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


def write_object(file, obj, parent, parameters, source):
    is_light = obj.type == "LIGHT"
    write_string(file, obj.name)
    file.write(struct.pack("<BI", int(is_light), len(parameters)))
    for key, value in parameters:
        write_string(file, key)
        write_string(file, value)
    matrix = parent.matrix_world.inverted() @ obj.matrix_world if parent else obj.matrix_world
    position, rotation, scale = export_transform(matrix)
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
        file.write(struct.pack("<B3f4fBf", light_type, *color, light.energy * 2.0 ** light.exposure, light.cutoff_distance if light.type != "SUN" else 1000.0, inner, outer, int(light.use_shadow), 0.02))
        return

    write_string(file, source)


def write_model(file, context, obj, export_path, texture_dir, apply_modifiers, bake_textures, depsgraph):
    write_string(file, obj.name)
    source_materials = list(obj.data.materials) or [None]
    material_records = [material_record(material) for material in source_materials]
    if bake_textures:
        bake_materials(context, obj, source_materials, material_records, export_path, texture_dir)
    if apply_modifiers:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            if mesh is None:
                raise ValueError("Could not evaluate mesh: " + obj.name)
            vertices, indices, sections, materials, bones, skin = mesh_payload(obj, mesh, True, material_records)
        finally:
            evaluated.to_mesh_clear()
    else:
        vertices, indices, sections, materials, bones, skin = mesh_payload(obj, obj.data, False, material_records)
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


def source_name(name):
    root, separator, suffix = name.rpartition(".")
    return root if separator and suffix and all("0" <= digit <= "9" for digit in suffix) else name


class ExportSapphireS3D(bpy.types.Operator, ExportHelper):
    bl_idname = "export_scene.sapphire_s3d"
    bl_label = "Sapphire (.s3d)"
    bl_options = {"PRESET"}

    filename_ext = ".s3d"
    filter_glob: StringProperty(default="*.s3d", options={"HIDDEN"})
    apply_modifiers: BoolProperty(name="Apply Modifiers", default=False)
    bake_textures: BoolProperty(name="Bake Textures", default=False)
    exported_textures_directory: StringProperty(name="Exported Textures Directory", default="")

    def execute(self, context):
        objects = [obj for obj in context.selected_objects if obj.type in {"MESH", "LIGHT"}]
        if not objects:
            self.report({"ERROR"}, "Select at least one mesh or light object")
            return {"CANCELLED"}
        depsgraph = context.evaluated_depsgraph_get() if self.apply_modifiers else None
        try:
            entries = export_objects(context, objects)
            meshes = {obj.name: obj for obj, _, _ in entries if obj.type == "MESH"}
            sources = {name: source_name(name) for name in meshes}
            for name, source in sources.items():
                if source not in meshes:
                    raise ValueError("Missing source mesh " + source + " for " + name)
            models = [obj for obj, _, _ in entries if obj.type == "MESH" and sources[obj.name] == obj.name]
            export_dir = os.path.dirname(os.path.abspath(self.filepath))
            texture_dir = os.path.abspath(os.path.join(export_dir, os.path.expanduser(self.exported_textures_directory))) if self.bake_textures else ""
            with open(self.filepath, "wb") as file:
                file.write(b"S3D\0")
                file.write(struct.pack("<I", len(models)))
                for obj in models:
                    write_model(file, context, obj, self.filepath, texture_dir, self.apply_modifiers, self.bake_textures, depsgraph)
                file.write(struct.pack("<I", len(entries)))
                for obj, parent, parameters in entries:
                    write_object(file, obj, parent, parameters, sources[obj.name] if obj.type == "MESH" else "")
        except (OSError, ValueError, RuntimeError, struct.error) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}

    def draw(self, context):
        self.layout.prop(self, "apply_modifiers")
        self.layout.prop(self, "bake_textures")
        texture_options = self.layout.column()
        texture_options.enabled = self.bake_textures
        texture_options.label(text="Exported Textures Directory")
        texture_options.prop(self, "exported_textures_directory", text="")

    def invoke(self, context, event):
        if not self.filepath:
            blend_path = context.blend_data.filepath
            self.filepath = os.path.splitext(blend_path)[0] + self.filename_ext if blend_path else "untitled.s3d"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


def menu_export(self, context):
    self.layout.operator(ExportSapphireS3D.bl_idname, text="Sapphire (.s3d)")


