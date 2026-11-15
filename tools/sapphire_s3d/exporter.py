import bpy
import json
import math
import os
import struct
from array import array
from collections.abc import Mapping, Sequence
from bpy.props import BoolProperty, EnumProperty, StringProperty
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
        "textures": ["", "", "", "", "", ""], "base_alpha": False, "modifier": "",
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


def material_is_baked(material):
    if not material:
        return False
    values = [owner["isBaked"] for owner in (material, material.node_tree) if owner and "isBaked" in owner]
    if any(not isinstance(value, bool) for value in values):
        raise ValueError("isBaked must be boolean on material " + material.name)
    return any(values)


def limited_size(width, height, maximum):
    if not maximum or max(width, height) <= maximum:
        return width, height
    ratio = maximum / max(width, height)
    return max(1, round(width * ratio)), max(1, round(height * ratio))


def bake_size(material, maximum):
    dimensions = [(node.image.size[0], node.image.size[1]) for node in material.node_tree.nodes if node.type == "TEX_IMAGE" and node.image and any(socket.is_linked for socket in node.outputs) and node.image.size[0] and node.image.size[1]]
    width, height = max(dimensions, key=lambda size: size[0] * size[1]) if dimensions else (1024, 1024)
    return limited_size(width, height, maximum)


def save_png(image, path, scene, channel, alpha=False, red_only=False):
    scalar = channel in {"roughness", "ao", "displacement", "displacement_mask"}
    settings = scene.render.image_settings
    original_mode = settings.color_mode
    temporary = None
    try:
        if scalar and red_only:
            width, height = image.size
            pixels = array("f", [0.0]) * (width * height * 4)
            image.pixels.foreach_get(pixels)
            red = pixels[::4]
            pixels[1::4] = red
            pixels[2::4] = red
            pixels[3::4] = array("f", [1.0]) * (width * height)
            temporary = bpy.data.images.new("S3D Grayscale", width=width, height=height, alpha=False, float_buffer=True)
            temporary.colorspace_settings.name = "Non-Color"
            temporary.pixels.foreach_set(pixels)
            temporary.update()
        mode = "BW" if scalar else "RGBA" if channel == "base" and alpha else "RGB"
        settings.color_mode = mode
        (temporary or image).save_render(path, scene=scene)
        if png_info(path)[2] != {"BW": 0, "RGB": 2, "RGBA": 6}[mode]:
            raise ValueError("PNG channel format was not saved correctly: " + path)
    finally:
        settings.color_mode = original_mode
        if temporary:
            bpy.data.images.remove(temporary)


def png_info(path):
    with open(path, "rb") as file:
        header = file.read(26)
    if len(header) != 26 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("Invalid PNG texture: " + path)
    return *struct.unpack(">II", header[16:24]), header[25]


def normalize_references(records, export_path, maximum, scene):
    normalized = set()
    for record in records:
        for index, reference in enumerate(record["textures"]):
            if not reference or not reference.lower().endswith(".png"):
                continue
            source = os.path.abspath(os.path.join(os.path.dirname(export_path), reference))
            if source not in normalized:
                width, height, color_type = png_info(source)
                target_width, target_height = limited_size(width, height, maximum)
                channel = ("base", "normal", "roughness", "ao", "displacement", "displacement_mask")[index]
                expected_type = 0 if channel in {"roughness", "ao", "displacement", "displacement_mask"} else 6 if channel == "base" and record["base_alpha"] else 2
                if (target_width, target_height) != (width, height) or color_type != expected_type:
                    image = bpy.data.images.load(source, check_existing=False)
                    try:
                        if (target_width, target_height) != (width, height):
                            image.scale(target_width, target_height)
                        save_png(image, source, scene, channel, alpha=expected_type == 6, red_only=True)
                    finally:
                        bpy.data.images.remove(image)
                normalized.add(source)
            record["textures"][index] = texture_reference(source, export_path)


def texture_path(material, channel, texture_dir):
    return os.path.join(texture_dir, f"{bpy.path.clean_name(material.name)}_{channel}.png")


def texture_reference(path, export_path):
    return os.path.relpath(path, os.path.dirname(os.path.abspath(export_path))).replace(os.sep, "/")


def bake_uv_transform(mesh):
    source_uv = mesh.uv_layers.active
    if not source_uv or not source_uv.data:
        raise ValueError("Mesh needs a UV map for material baking")
    minimum_u = min(loop.uv.x for loop in source_uv.data)
    maximum_u = max(loop.uv.x for loop in source_uv.data)
    minimum_v = min(loop.uv.y for loop in source_uv.data)
    maximum_v = max(loop.uv.y for loop in source_uv.data)
    if minimum_u == maximum_u or minimum_v == maximum_v:
        raise ValueError("Mesh UV map has no area")
    margin = 0.02
    scale_u = (1.0 - margin * 2.0) / (maximum_u - minimum_u)
    scale_v = (1.0 - margin * 2.0) / (maximum_v - minimum_v)
    offset_u = margin - minimum_u * scale_u
    offset_v = margin - minimum_v * scale_v
    return scale_u, scale_v, offset_u, offset_v


def reuse_baked_materials(obj, materials, records, export_path, texture_dir):
    channels = [bake_channels(material) for material in materials]
    existing = {index for index, (material, record) in enumerate(zip(materials, records))
                if any(record["textures"]) or (material and material.use_nodes and material.node_tree
                and any(node.type == "TEX_IMAGE" and node.image for node in material.node_tree.nodes))}
    available = [(index, channel, texture_path(materials[index], channel, texture_dir))
                 for index, needed in enumerate(channels) if index not in existing for channel in needed if channel != "alpha"]
    available = [(index, channel, path) for index, channel, path in available if os.path.isfile(path)]
    if not available:
        return False
    scale_u, scale_v, offset_u, offset_v = bake_uv_transform(obj.data)
    for index, channel, path in available:
        record = records[index]
        record["uv_scale"] = (scale_u, scale_v)
        record["uv_offset"] = (offset_u, 1.0 - scale_v - offset_v)
        slot = {"base": 0, "normal": 1, "roughness": 2, "ao": 3, "displacement": 4}[channel]
        record["textures"][slot] = texture_reference(path, export_path)
        if channel == "base":
            record["color"] = (1.0, 1.0, 1.0)
            if "alpha" in channels[index]:
                record["opacity"] = 1.0
                record["base_alpha"] = True
        elif channel == "roughness":
            record["roughness"] = 1.0
        elif channel == "normal":
            record["normal_scale"] = 1.0
            record["flip_normal_y"] = True
        elif channel == "displacement":
            record["displacement_scale"] = 1.0
    return True


def source_link(socket):
    while socket and len(socket.links) == 1:
        link = socket.links[0]
        if link.from_node.type != "REROUTE":
            return link
        socket = link.from_node.inputs[0]
    return None


def direct_image(socket):
    link = source_link(socket)
    if not link or link.from_node.type != "TEX_IMAGE" or link.from_socket.name != "Color" or not link.from_node.image:
        return None
    return link.from_node.image


def node_image(node):
    return node.image if node.type == "TEX_IMAGE" else None


def export_direct_image(image, material, channel, export_path, texture_dir, maximum, scene, alpha=False):
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("Image has no pixels: " + image.name)
    target_width, target_height = limited_size(width, height, maximum)
    os.makedirs(texture_dir, exist_ok=True)
    destination = texture_path(material, channel, texture_dir)
    if (target_width, target_height) != (width, height):
        resized = image.copy()
        try:
            if image.is_dirty:
                pixels = array("f", [0.0]) * (width * height * 4)
                image.pixels.foreach_get(pixels)
                resized.pixels.foreach_set(pixels)
                resized.update()
            resized.scale(target_width, target_height)
            save_png(resized, destination, scene, channel, alpha=alpha, red_only=True)
        finally:
            bpy.data.images.remove(resized)
    else:
        save_png(image, destination, scene, channel, alpha=alpha, red_only=True)
    return texture_reference(destination, export_path)


def reference_direct_images(materials, records, export_path, texture_dir, maximum, scene):
    for index, material in enumerate(materials):
        if not material or not material.use_nodes or not material.node_tree:
            continue
        named = {}
        referenced = set()
        extra_index = 1
        def reference(image, channel, alpha=False):
            nonlocal extra_index
            pointer = image.as_pointer()
            if channel in named and named[channel][0] != pointer:
                while "image" + str(extra_index) in named:
                    extra_index += 1
                channel = "image" + str(extra_index)
                extra_index += 1
            if channel not in named:
                path = export_direct_image(image, material, channel, export_path, texture_dir, maximum, scene, alpha=alpha)
                named[channel] = (pointer, path)
            referenced.add(pointer)
            return named[channel][1]
        principled = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
        if principled:
            base = linked_input(principled, "Base Color")
            alpha = linked_input(principled, "Alpha")
            base_link = source_link(base)
            alpha_link = source_link(alpha)
            base_alpha = bool(base_link and alpha_link and base_link.from_node == alpha_link.from_node and alpha_link.from_socket.name == "Alpha")
            for socket_name, slot, channel in (("Base Color", 0, "base"), ("Roughness", 2, "roughness")):
                image = direct_image(linked_input(principled, socket_name))
                if image:
                    records[index]["textures"][slot] = reference(image, channel, alpha=base_alpha if channel == "base" else False)
                    records[index]["color" if slot == 0 else "roughness"] = (1.0, 1.0, 1.0) if slot == 0 else 1.0
            if base_alpha and records[index]["textures"][0]:
                records[index]["opacity"] = 1.0
                records[index]["base_alpha"] = True
            normal = linked_input(principled, "Normal")
            if normal:
                link = source_link(normal)
                normal_node = link.from_node if link else None
                if normal_node:
                    strength = normal_node.inputs.get("Strength") if normal_node.type == "NORMAL_MAP" else None
                    color = linked_input(normal_node, "Color") if strength and not strength.is_linked else None
                    image = direct_image(color) if strength else direct_image(normal)
                    if image:
                        records[index]["textures"][1] = reference(image, "normal")
                        records[index]["normal_scale"] = float(strength.default_value) if strength else 1.0
                        records[index]["flip_normal_y"] = True
        output = next((node for node in material.node_tree.nodes if node.type == "OUTPUT_MATERIAL" and node.is_active_output), None)
        displacement = linked_input(output, "Displacement")
        if displacement:
            node = displacement.links[0].from_node
            height = linked_input(node, "Height") if node.type == "DISPLACEMENT" else None
            scale = node.inputs.get("Scale") if height else None
            midlevel = node.inputs.get("Midlevel") if height else None
            image = direct_image(height)
            if image and scale and not scale.is_linked and midlevel and not midlevel.is_linked and abs(midlevel.default_value - 0.5) <= 0.0001:
                records[index]["textures"][4] = reference(image, "displacement")
                records[index]["displacement_scale"] = float(scale.default_value)
        for node in material.node_tree.nodes:
            image = node_image(node)
            if not image:
                continue
            label = (node.label or node.name).strip().lower().replace("_", " ")
            if label in {"ao", "ambient occlusion"}:
                records[index]["textures"][3] = reference(image, "ao")
            elif label in {"displacement mask", "height mask"}:
                records[index]["textures"][5] = reference(image, "displacement_mask")
            elif image.as_pointer() not in referenced:
                channel = bpy.path.clean_name(node.label.strip()).lower() if node.label.strip() else ""
                if not channel or channel in named:
                    while "image" + str(extra_index) in named:
                        extra_index += 1
                    channel = "image" + str(extra_index)
                    extra_index += 1
                reference(image, channel)


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


def bake_materials(context, obj, materials, records, export_path, texture_dir, maximum):
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
    scale_u, scale_v, offset_u, offset_v = bake_uv_transform(original_mesh)
    images = []
    base_images = {}
    os.makedirs(texture_dir, exist_ok=True)
    temporary_mesh = original_mesh.copy()
    try:
        source_uv = temporary_mesh.uv_layers.active
        render_uv = temporary_mesh.uv_layers.active_render
        bake_uv = temporary_mesh.uv_layers.new(name="S3D Bake", do_init=True)
        for loop in bake_uv.data:
            loop.uv = (loop.uv.x * scale_u + offset_u, loop.uv.y * scale_v + offset_v)
        source_uv.active = True
        if render_uv:
            render_uv.active_render = True
        for index, record in enumerate(records):
            if channels[index]:
                record["uv_scale"] = (scale_u, scale_v)
                record["uv_offset"] = (offset_u, 1.0 - scale_v - offset_v)
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
                    width, height = bake_size(material, maximum) if source and channel in channels[index] else (1, 1)
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
                result = bpy.ops.object.bake(type="NORMAL" if channel == "normal" else "EMIT", use_clear=True, normal_space="TANGENT", uv_layer=bake_uv.name)
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
                            records[index]["base_alpha"] = True
                            continue
                    elif channel == "roughness":
                        records[index]["roughness"] = 1.0
                    elif channel == "normal":
                        records[index]["normal_scale"] = 1.0
                        records[index]["flip_normal_y"] = True
                    elif channel == "displacement":
                        records[index]["displacement_scale"] = 1.0
                    path = texture_path(materials[index], channel, texture_dir)
                    save_png(image, path, scene, channel, alpha=channel == "base" and "alpha" in channels[index])
                    slot = {"base": 0, "normal": 1, "roughness": 2, "ao": 3, "displacement": 4}[channel]
                    records[index]["textures"][slot] = texture_reference(path, export_path)
            finally:
                temporary_mesh.materials.clear()
                for material in temporary_materials:
                    bpy.data.materials.remove(material)
        for index, image in base_images.items():
            if "alpha" not in channels[index]:
                continue
            path = texture_path(materials[index], "base", texture_dir)
            save_png(image, path, scene, "base", alpha=True)
            records[index]["textures"][0] = texture_reference(path, export_path)
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
    normal_sums = []
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
                position = tuple(source.co)
                normal = mesh.corner_normals[loop_index].vector
                uv = (uv_layer.data[loop_index].uv.x, 1.0 - uv_layer.data[loop_index].uv.y) if uv_layer else (0.0, 0.0)
                color = (1.0, 1.0, 1.0, 1.0)
                if color_layer:
                    color_value = color_layer.data[loop_index if color_layer.domain == "CORNER" else loop.vertex_index].color
                    color = tuple(color_value)
                influence = vertex_weights(obj, loop.vertex_index, bone_indices) if bones else ((0, 0, 0, 0), (65535, 0, 0, 0))
                key = (position, tuple(uv), tuple(color), influence)
                vertex_index = vertex_lookup.get(key)
                if vertex_index is None:
                    vertex_index = len(vertices)
                    vertex_lookup[key] = vertex_index
                    vertices.append((position, tuple(normal), tuple(uv), tuple(color)))
                    normal_sums.append(normal.copy())
                    if bones:
                        skin.append(influence)
                else:
                    normal_sums[vertex_index] += normal
                indices.append(vertex_index)
        count = len(indices) - first_index
        if count:
            sections.append((first_index, count, material_index))

    for index, (position, normal, uv, color) in enumerate(vertices):
        average = normal_sums[index]
        if average.length_squared:
            average.normalize()
            vertices[index] = (position, tuple(average), uv, color)

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


def write_model(file, context, obj, export_path, texture_dir, apply_modifiers, depsgraph, maximum):
    write_string(file, obj.name)
    source_materials = list(obj.data.materials) or [None]
    material_records = [material_record(material) for material in source_materials]
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
    baked_materials = [material if material_is_baked(material) else None for material in source_materials]
    direct_materials = [material if baked is None else None for material, baked in zip(source_materials, baked_materials)]
    reference_direct_images(direct_materials, material_records, export_path, texture_dir, maximum, context.scene)
    reuse_baked_materials(obj, direct_materials, material_records, export_path, texture_dir)
    bake_materials(context, obj, baked_materials, material_records, export_path, texture_dir, maximum)
    normalize_references(material_records, export_path, maximum, context.scene)
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
    exported_textures_directory: StringProperty(name="Exported Textures Directory", default="")
    max_texture_size: EnumProperty(name="Size", items=[("NONE", "None", "No size limit")] + [(str(size), str(size), f"Limit longest side to {size} pixels") for size in (4096, 2048, 1024, 512, 256, 128)], default="NONE")

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
            texture_dir = os.path.abspath(os.path.join(export_dir, os.path.expanduser(self.exported_textures_directory)))
            settings = context.scene.render.image_settings
            original_settings = (settings.file_format, settings.color_mode, settings.color_depth, settings.compression)
            try:
                settings.file_format = "PNG"
                settings.color_depth = "8"
                settings.compression = 0
                maximum = 0 if self.max_texture_size == "NONE" else int(self.max_texture_size)
                with open(self.filepath, "wb") as file:
                    file.write(b"S3D\0")
                    file.write(struct.pack("<I", len(models)))
                    for obj in models:
                        write_model(file, context, obj, self.filepath, texture_dir, self.apply_modifiers, depsgraph, maximum)
                    file.write(struct.pack("<I", len(entries)))
                    for obj, parent, parameters in entries:
                        write_object(file, obj, parent, parameters, sources[obj.name] if obj.type == "MESH" else "")
            finally:
                settings.file_format, settings.color_mode, settings.color_depth, settings.compression = original_settings
        except (OSError, ValueError, RuntimeError, struct.error) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"FINISHED"}

    def draw(self, context):
        self.layout.prop(self, "apply_modifiers")
        texture_options = self.layout.column()
        texture_options.label(text="Exported Textures Directory")
        texture_options.prop(self, "exported_textures_directory", text="")
        self.layout.prop(self, "max_texture_size")

    def invoke(self, context, event):
        if not self.filepath:
            blend_path = context.blend_data.filepath
            self.filepath = os.path.splitext(blend_path)[0] + self.filename_ext if blend_path else "untitled.s3d"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


def menu_export(self, context):
    self.layout.operator(ExportSapphireS3D.bl_idname, text="Sapphire (.s3d)")


