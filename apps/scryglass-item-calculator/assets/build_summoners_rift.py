import bpy
import math
import os
from mathutils import Vector

ASSET_DIR = os.path.dirname(__file__)
OUT = os.path.join(ASSET_DIR, "summoners-rift-bg.png")
CONCEPT_BG = os.path.join(ASSET_DIR, "summoners-rift-concept-bg.png")

# A camera-ready Rift diorama.  The render is intentionally more detailed than
# the UI needs; the calculator applies a dark veil to keep the map atmospheric
# and the numbers legible.
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
    for block in list(datablocks):
        if block.users == 0:
            datablocks.remove(block)


def material(name, color, roughness=0.82, metallic=0.0, emission=None, emission_strength=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission_strength or 0.4
    mat.diffuse_color = (*color, 1.0)
    return mat


def assign(obj, mat):
    obj.data.materials.append(mat)
    return obj


def smooth(obj):
    if hasattr(obj.data, "polygons"):
        for poly in obj.data.polygons:
            poly.use_smooth = True
    return obj


def bevel(obj, width=0.12, segments=3):
    mod = obj.modifiers.new("hand-finished edges", "BEVEL")
    mod.width = width
    mod.segments = segments
    return obj


def cube(name, loc, scale, mat, rotation=0.0, edge=0.0):
    bpy.ops.mesh.primitive_cube_add(location=loc, rotation=(0, 0, rotation))
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if edge:
        bevel(obj, edge, 3)
    return assign(obj, mat)


def cylinder(name, loc, radius, depth, mat, vertices=16, rotation=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=loc)
    obj = bpy.context.object
    obj.name = name
    if rotation:
        obj.rotation_euler = rotation
    bevel(obj, min(radius * 0.16, 0.08), 2)
    return assign(smooth(obj), mat)


def cone(name, loc, radius1, radius2, depth, mat, vertices=16):
    bpy.ops.mesh.primitive_cone_add(vertices=vertices, radius1=radius1, radius2=radius2, depth=depth, location=loc)
    obj = bpy.context.object
    obj.name = name
    bevel(obj, min(radius1 * 0.12, 0.06), 2)
    return assign(smooth(obj), mat)


def sphere(name, loc, radius, mat, scale=(1, 1, 1), ico=False):
    if ico:
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius, location=loc)
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12, radius=radius, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return assign(smooth(obj), mat)


def curve(name, points, width, mat, resolution=3):
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    data.resolution_u = resolution
    data.bevel_depth = width
    data.bevel_resolution = 3
    spline = data.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for point, co in zip(spline.bezier_points, points):
        point.co = co
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return assign(obj, mat)


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def point_light(name, loc, color, energy, radius=1.2):
    bpy.ops.object.light_add(type="POINT", location=loc)
    light = bpy.context.object
    light.name = name
    light.data.energy = energy
    light.data.color = color
    light.data.shadow_soft_size = radius
    return light


# Palette: cool stone and river, with restrained blue/red team light.
void = material("void backdrop", (0.006, 0.009, 0.011), roughness=1.0)
frame = material("basalt frame", (0.018, 0.028, 0.030), roughness=0.58, metallic=0.18)
stone = material("weathered slate", (0.105, 0.135, 0.132), roughness=0.9)
stone_light = material("cut slate", (0.19, 0.23, 0.21), roughness=0.78)
soil = material("deep forest soil", (0.028, 0.073, 0.052), roughness=1.0)
grass = material("moss green", (0.055, 0.17, 0.105), roughness=0.94)
grass_hi = material("leaf highlight", (0.11, 0.30, 0.16), roughness=0.88)
road = material("lane stone", (0.25, 0.27, 0.24), roughness=0.72)
road_edge = material("lane moss edge", (0.08, 0.20, 0.14), roughness=0.88)
water = material("rift water", (0.018, 0.14, 0.18), roughness=0.2, metallic=0.28, emission=(0.01, 0.10, 0.14), emission_strength=0.32)
water_hi = material("water highlights", (0.05, 0.34, 0.38), roughness=0.15, metallic=0.22, emission=(0.02, 0.18, 0.20), emission_strength=0.45)
blue = material("blue side", (0.04, 0.15, 0.34), roughness=0.46, metallic=0.14, emission=(0.03, 0.18, 0.72), emission_strength=1.7)
red = material("red side", (0.34, 0.045, 0.035), roughness=0.46, metallic=0.14, emission=(0.78, 0.035, 0.02), emission_strength=1.7)
gold = material("objective brass", (0.38, 0.23, 0.065), roughness=0.36, metallic=0.42)
crystal = material("objective crystal", (0.07, 0.31, 0.30), roughness=0.18, metallic=0.14, emission=(0.04, 0.55, 0.50), emission_strength=2.4)
mist = material("mist", (0.12, 0.19, 0.18), roughness=1.0, emission=(0.04, 0.10, 0.09), emission_strength=0.16)

# A substantial beveled plinth makes the render feel staged rather than like a
# viewport screenshot.  The raised map surface leaves room for foreground depth.
cube("diorama plinth", (0, 0, -1.05), (14.5, 10.5, 0.8), frame, edge=0.38)
cube("diorama inset", (0, 0, -0.28), (13.8, 9.8, 0.22), soil, edge=0.26)

# The Rift surface: soft banks, raised plateaus, and a winding river.
cube("rift terrain", (0, 0, 0.02), (13.5, 9.3, 0.22), soil, edge=0.36)
for x, y, sx, sy, h in [(-8.5, 5.6, 3.0, 1.6, 0.5), (7.8, 5.3, 3.7, 1.5, 0.42), (-8.5, -4.8, 3.6, 1.7, 0.48), (8.2, -4.9, 3.5, 1.7, 0.5), (-2.2, 5.3, 2.0, 1.25, 0.38), (2.2, -5.0, 2.1, 1.1, 0.42)]:
    cube("raised jungle shelf", (x, y, 0.34), (sx, sy, h), grass, edge=0.22)

river_points = [(-12.6, -7.8, 0.55), (-9.1, -5.1, 0.57), (-5.4, -2.6, 0.56), (-1.6, -0.3, 0.55), (2.4, 1.9, 0.56), (6.8, 4.7, 0.58), (12.5, 7.9, 0.58)]
curve("river shadow bank", river_points, 1.28, road_edge)
curve("river", river_points, 0.98, water)
curve("river light", [(-11.6, -7.1, 0.73), (-7.8, -4.9, 0.73), (-4.8, -2.65, 0.72), (-1.4, -0.45, 0.72), (2.8, 1.8, 0.74), (6.9, 4.45, 0.74), (11.6, 7.4, 0.75)], 0.07, water_hi)

# Three recognizable lanes with a narrow central inlay.  The fourth curves
# around the river to make the map read as a game space, not a wiring diagram.
lanes = [
    [(-12.4, -7.5, 0.62), (-9.2, -5.3, 0.64), (-5.4, -2.4, 0.65), (-2.0, -0.25, 0.66)],
    [(-12.0, 7.4, 0.62), (-8.7, 5.15, 0.64), (-5.0, 2.65, 0.65), (-1.9, 0.55, 0.66)],
    [(12.2, -7.35, 0.62), (8.6, -5.0, 0.64), (5.2, -2.2, 0.65), (1.9, 0.45, 0.66)],
    [(12.0, 7.3, 0.62), (8.6, 5.0, 0.64), (5.2, 2.7, 0.65), (1.9, 1.95, 0.66)],
]
for idx, lane_points in enumerate(lanes):
    curve(f"lane bank {idx}", lane_points, 0.39, road_edge)
    curve(f"lane {idx}", lane_points, 0.26, road)
    curve(f"lane center inlay {idx}", lane_points, 0.025, stone_light)


def tree(name, x, y, size=1.0, accent_mat=None):
    trunk = cylinder(f"{name} trunk", (x, y, 0.72 * size), 0.13 * size, 1.3 * size, stone, vertices=9)
    for i, (dx, dy, z) in enumerate([(0, 0, 1.45), (-0.22, 0.04, 1.15), (0.20, -0.06, 1.2)]):
        sphere(f"{name} canopy {i}", (x + dx * size, y + dy * size, z * size), 0.55 * size, accent_mat or grass, scale=(1.0, 0.92, 0.82), ico=True)
    cone(f"{name} crown", (x, y, 2.02 * size), 0.52 * size, 0.08 * size, 1.0 * size, grass_hi, vertices=8)
    return trunk


# Dense, intentional jungle clusters rather than isolated identical blobs.
tree_specs = [
    (-9.3, 3.1, 1.20), (-7.8, 2.55, 0.84), (-6.6, 3.15, 1.0), (-8.4, 1.7, 0.74),
    (7.5, 2.85, 1.1), (8.8, 3.5, 0.82), (6.25, 3.9, 0.7), (7.2, 1.65, 0.9),
    (-5.8, -4.4, 0.92), (-4.5, -5.0, 1.15), (-3.6, -3.65, 0.72),
    (4.5, -4.5, 1.08), (5.8, -3.7, 0.78), (6.5, -4.8, 0.68),
    (-2.8, 3.6, 0.7), (3.0, -3.0, 0.68),
]
for idx, (x, y, size) in enumerate(tree_specs):
    tree(f"jungle tree {idx}", x, y, size, grass_hi if idx % 4 == 0 else grass)


def rock_field(name, x, y, scale, count=4):
    for i in range(count):
        angle = i * 2.27
        radius = scale * (0.35 + 0.14 * (i % 3))
        sphere(f"{name} rock {i}", (x + math.cos(angle) * radius, y + math.sin(angle) * radius, 0.55 + 0.12 * (i % 2)), scale * (0.25 + 0.08 * (i % 3)), stone_light if i == 0 else stone, scale=(1.0, 0.76, 0.72), ico=True)


for idx, (x, y, s) in enumerate([(-4.2, 1.4, 1.25), (4.4, 1.1, 1.1), (-2.0, -2.8, 0.85), (2.5, 3.0, 0.88), (0.0, 5.15, 0.95)]):
    rock_field(f"stone outcrop {idx}", x, y, s)


def tower(name, x, y, team_mat):
    cylinder(f"{name} foundation", (x, y, 0.52), 0.72, 0.28, stone_light, vertices=12)
    cylinder(f"{name} base", (x, y, 1.02), 0.48, 0.72, stone, vertices=12)
    cylinder(f"{name} shaft", (x, y, 1.78), 0.30, 0.86, stone_light, vertices=12)
    cone(f"{name} crown", (x, y, 2.36), 0.57, 0.16, 0.52, team_mat, vertices=8)
    cylinder(f"{name} beacon", (x, y, 2.66), 0.13, 0.24, team_mat, vertices=10)
    point_light(f"{name} glow", (x, y, 2.7), (0.08, 0.3, 1.0) if team_mat == blue else (1.0, 0.08, 0.03), 26, 1.4)


for i, (x, y) in enumerate([(-10.7, -6.6), (-7.7, -5.1), (-9.4, -3.9)]):
    tower(f"blue tower {i}", x, y, blue)
for i, (x, y) in enumerate([(10.7, 6.6), (7.7, 5.1), (9.4, 3.9)]):
    tower(f"red tower {i}", x, y, red)


def base(name, x, y, team_mat):
    cylinder(f"{name} platform", (x, y, 0.62), 1.42, 0.34, stone_light, vertices=12)
    cylinder(f"{name} ring", (x, y, 0.84), 1.02, 0.12, team_mat, vertices=12)
    cylinder(f"{name} nexus pedestal", (x, y, 1.04), 0.62, 0.42, stone, vertices=10)
    cone(f"{name} nexus crystal", (x, y, 1.78), 0.48, 0.08, 1.18, team_mat, vertices=6)
    point_light(f"{name} light", (x, y, 1.9), (0.05, 0.2, 1.0) if team_mat == blue else (1.0, 0.05, 0.02), 60, 2.4)


base("blue nexus", -11.1, -6.9, blue)
base("red nexus", 11.1, 6.9, red)


def objective_pit(name, x, y, team_mat):
    cylinder(f"{name} outer rim", (x, y, 0.45), 1.58, 0.18, stone, vertices=18)
    cylinder(f"{name} inner pit", (x, y, 0.54), 1.24, 0.18, soil, vertices=18)
    cylinder(f"{name} gold rim", (x, y, 0.65), 0.92, 0.12, gold, vertices=18)
    sphere(f"{name} objective", (x, y, 1.12), 0.44, crystal, scale=(1.0, 1.0, 1.28), ico=True)
    point_light(f"{name} objective glow", (x, y, 1.1), (0.04, 0.8, 0.7), 44, 2.0)


objective_pit("dragon pit", 4.15, 0.95, red)
objective_pit("baron pit", -4.05, 0.85, blue)

# Small rune stones along the river make the middle read as a lived-in space.
for idx, (x, y) in enumerate([(-1.6, -0.2), (0.2, 0.9), (1.9, 1.8), (-2.1, -1.15)]):
    cylinder(f"river rune {idx}", (x, y, 0.85), 0.22, 0.48, stone_light, vertices=8, rotation=(math.radians(8), math.radians(-10), math.radians(idx * 15)))
    cone(f"river rune cap {idx}", (x, y, 1.16), 0.17, 0.03, 0.32, crystal, vertices=6)

# Foreground stones, banners, and warm practical lights create the depth cue the
# old overhead blockout lacked, while remaining quiet behind the product UI.
for idx, x in enumerate([-10.8, -7.3, -3.3, 3.4, 7.4, 10.8]):
    rock_field(f"foreground stone {idx}", x, -8.7, 0.65 + (idx % 2) * 0.18, count=3)
    point_light(f"foreground ember {idx}", (x, -8.1, 1.2), (0.25, 0.07, 0.02), 10, 0.9)

# A few vertical map markers frame the scene and keep the diagonal composition
# legible even when the UI dark veil is applied.
for idx, (x, y, team_mat) in enumerate([(-12.4, 3.5, blue), (12.4, -3.5, red)]):
    cube(f"banner pole {idx}", (x, y, 2.2), (0.06, 0.06, 2.0), stone_light, edge=0.04)
    cube(f"banner {idx}", (x + (0.35 if idx == 0 else -0.35), y, 3.05), (0.38, 0.04, 0.52), team_mat, rotation=math.radians(-8 if idx == 0 else 8), edge=0.04)

# Camera and studio lighting.  Perspective, a shallow depth of field, and a
# large soft key produce a composed render rather than a diagnostic viewport.
bpy.ops.object.camera_add(location=(18.8, -20.6, 17.2))
camera = bpy.context.object
camera.data.type = "PERSP"
camera.data.lens = 52
camera.data.dof.use_dof = True
camera.data.dof.focus_object = bpy.data.objects.get("dragon pit objective")
camera.data.dof.aperture_fstop = 5.0
look_at(camera, (0, 0.3, 0.65))
bpy.context.scene.camera = camera

# The generated plate is the hero image.  Keep the hand-modeled Rift pieces in
# the .blend for future art direction, but do not let the current blockout
# compete with the high-fidelity environment in the shipped background.
for scene_object in bpy.context.scene.objects:
    if scene_object.type in {"MESH", "CURVE"}:
        scene_object.hide_render = True

# Use a generated Rift environment as a photographic-quality depth plate.  The
# modeled map remains in front of it, so Blender still owns the final render and
# the scene keeps real geometry at the focal plane.
if os.path.exists(CONCEPT_BG):
    forward = camera.rotation_euler.to_matrix() @ Vector((0, 0, -1))
    plate_location = camera.location + forward * 33.0
    bpy.ops.mesh.primitive_plane_add(size=2, location=plate_location)
    plate = bpy.context.object
    plate.name = "Rift environment depth plate"
    plate.rotation_euler = (camera.rotation_euler.x, camera.rotation_euler.y, camera.rotation_euler.z)
    plate.scale = (15.5, 9.7, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    plate_data = bpy.data.materials.new("Rift environment plate")
    plate_data.use_nodes = True
    nodes = plate_data.node_tree.nodes
    links = plate_data.node_tree.links
    nodes.clear()
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(CONCEPT_BG, check_existing=True)
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 0.72
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(tex.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    plate.data.materials.append(plate_data)

bpy.ops.object.light_add(type="AREA", location=(-8.0, -7.0, 17.5))
key = bpy.context.object
key.name = "large cool key"
key.data.energy = 1550
key.data.shape = "DISK"
key.data.size = 10.0
key.data.color = (0.58, 0.78, 0.85)
look_at(key, (0, 0, 0))

bpy.ops.object.light_add(type="AREA", location=(8.0, 8.0, 13.0))
fill = bpy.context.object
fill.name = "warm rim light"
fill.data.energy = 1000
fill.data.shape = "DISK"
fill.data.size = 8.0
fill.data.color = (1.0, 0.27, 0.12)
look_at(fill, (0, 1, 1))

bpy.ops.object.light_add(type="AREA", location=(0, 4.0, 7.0))
top = bpy.context.object
top.name = "soft top light"
top.data.energy = 420
top.data.size = 6.0
top.data.color = (0.32, 0.58, 0.52)
look_at(top, (0, 0, 0))

scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = 1600
scene.render.resolution_y = 1000
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.filepath = OUT
scene.render.film_transparent = False
scene.world.color = (0.002, 0.004, 0.005)
scene.view_settings.look = "AgX - Medium High Contrast"
bpy.ops.wm.save_as_mainfile(filepath=OUT.replace(".png", ".blend"))
bpy.ops.render.render(write_still=True)
