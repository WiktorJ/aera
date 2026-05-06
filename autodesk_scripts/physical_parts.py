"""Fusion 360 script: builds physical print parts matching the MuJoCo sim.

Dimension provenance
--------------------
MuJoCo's <geom type="box" size="..."> is HALF-EXTENTS in meters. Physical
full-extent in millimeters = size * 2 * 1000.

From `ar4_mk3.xml`:
    object0           size="0.012 0.012 0.012"   -> 24 x 24 x 24 mm cube
    object_distractor size="0.012 0.012 0.012"   -> 24 x 24 x 24 mm cube
    target_visual     size="0.025 0.025 0.001"   -> 50 x 50 x  2 mm plate
    distractor_visual size="0.025 0.025 0.001"   -> 50 x 50 x  2 mm plate

From `domain_rand_config_generator.py` (object size randomization range):
    half-extents [0.010, 0.010, 0.010] .. [0.015, 0.015, 0.015]
    -> cube edge 20 mm .. 30 mm

The CUBOID_VARIANTS / TARGET_VARIANTS lists below cover that range plus the
nominal sim defaults. Edit them to add/remove parts before running.

How to run inside Fusion 360
----------------------------
1. Utilities -> Add-Ins -> Scripts and Add-Ins -> Scripts -> "+" (My Scripts)
2. Point it at this file, Run.
3. A new untitled document opens with one component per variant.
4. If STL_OUTPUT_DIR is set below, one STL per component is written there.
5. File -> Save As ... .f3d to keep the parametric design.
"""

import os
import traceback

import adsk.core
import adsk.fusion


# Where to write one STL per component. Set to "" or None to skip STL export.
# The directory is created if it doesn't exist.
STL_OUTPUT_DIR = os.path.expanduser("~/Desktop/aera_physical_parts_stl")

# STL mesh refinement: Low / Medium / High. High is recommended for print.
STL_REFINEMENT = adsk.fusion.MeshRefinementSettings.MeshRefinementHigh


# (component_name, length_mm, width_mm, height_mm)
CUBOID_VARIANTS = [
    ("Cube_24mm", 24.0, 24.0, 24.0),
    ("Cube_20mm", 20.0, 20.0, 20.0),
    ("Cube_28mm", 28.0, 28.0, 28.0),
    ("Cuboid_20x24x28", 20.0, 24.0, 28.0),
]

# (component_name, length_mm, width_mm, thickness_mm)
TARGET_VARIANTS = [
    ("Target_50mm", 50.0, 50.0, 2.0),
    ("Target_40mm", 40.0, 40.0, 2.0),
    ("Target_60mm", 60.0, 60.0, 2.0),
]


# Fusion's internal length unit is centimeters.
MM_TO_CM = 0.1


def _create_box_component(parent_comp, name, length_mm, width_mm, height_mm):
    """Create a sub-component containing a single box body, centered on origin.

    The body is centered on the XY plane in X/Y and symmetric about that plane
    in Z, so its body frame matches the MuJoCo geom frame (which sits at the
    geometric center of the box).
    """
    occurrence = parent_comp.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    component = occurrence.component
    component.name = name

    sketch = component.sketches.add(component.xYConstructionPlane)
    sketch.name = f"{name}_base"

    half_l_cm = (length_mm / 2.0) * MM_TO_CM
    half_w_cm = (width_mm / 2.0) * MM_TO_CM
    center = adsk.core.Point3D.create(0, 0, 0)
    corner = adsk.core.Point3D.create(half_l_cm, half_w_cm, 0)
    sketch.sketchCurves.sketchLines.addCenterPointRectangle(center, corner)

    profile = sketch.profiles.item(0)
    extrudes = component.features.extrudeFeatures
    ext_input = extrudes.createInput(
        profile, adsk.fusion.FeatureOperations.NewBodyFeatureOperation
    )
    # Symmetric extrude: distance is the FULL extent when isFullLength=True.
    full_height = adsk.core.ValueInput.createByString(f"{height_mm} mm")
    ext_input.setSymmetricExtent(full_height, True)
    feature = extrudes.add(ext_input)
    body = feature.bodies.item(0)
    body.name = name

    return component


def _export_component_stl(export_mgr, component, out_dir):
    """Export the given component's bodies to a single STL file in out_dir."""
    path = os.path.join(out_dir, f"{component.name}.stl")
    stl_opts = export_mgr.createSTLExportOptions(component, path)
    stl_opts.meshRefinement = STL_REFINEMENT
    stl_opts.sendToPrintUtility = False
    export_mgr.execute(stl_opts)
    return path


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface

        doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
        design = adsk.fusion.Design.cast(
            doc.products.itemByProductType("DesignProductType")
        )
        if design is None:
            raise RuntimeError("Failed to acquire Fusion Design product.")

        design.designType = adsk.fusion.DesignTypes.ParametricDesignType
        root = design.rootComponent

        components = []
        for name, length_mm, width_mm, height_mm in CUBOID_VARIANTS:
            components.append(
                _create_box_component(root, name, length_mm, width_mm, height_mm)
            )

        for name, length_mm, width_mm, thickness_mm in TARGET_VARIANTS:
            components.append(
                _create_box_component(root, name, length_mm, width_mm, thickness_mm)
            )

        stl_msg = "STL export skipped (STL_OUTPUT_DIR not set)."
        if STL_OUTPUT_DIR:
            os.makedirs(STL_OUTPUT_DIR, exist_ok=True)
            export_mgr = design.exportManager
            for comp in components:
                _export_component_stl(export_mgr, comp, STL_OUTPUT_DIR)
            stl_msg = f"Wrote {len(components)} STL file(s) to:\n{STL_OUTPUT_DIR}"

        if ui:
            ui.messageBox(
                f"Created {len(CUBOID_VARIANTS)} cuboid(s) and "
                f"{len(TARGET_VARIANTS)} target(s).\n\n"
                f"{stl_msg}\n\n"
                "Save the document as .f3d to keep the parametric design."
            )
    except Exception:
        if ui:
            ui.messageBox(f"Script failed:\n{traceback.format_exc()}")
        raise
