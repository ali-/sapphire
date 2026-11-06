import bpy

from .exporter import ExportSapphireS3D, menu_export


bl_info = {
	"name": "Sapphire S3D Exporter",
	"author": "Sapphire",
	"version": (1, 0, 0),
	"blender": (5, 2, 0),
	"location": "File > Export",
	"description": "Export selected mesh and light objects to Sapphire S3D",
	"category": "Import-Export",
}


CLASSES = (ExportSapphireS3D,)


def register():
	for cls in CLASSES:
		bpy.utils.register_class(cls)
	bpy.types.TOPBAR_MT_file_export.append(menu_export)


def unregister():
	bpy.types.TOPBAR_MT_file_export.remove(menu_export)
	for cls in reversed(CLASSES):
		bpy.utils.unregister_class(cls)


if __name__ == "__main__":
	register()
