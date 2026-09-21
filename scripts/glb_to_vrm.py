#!/usr/bin/env python3
"""Convert a MetaPerson-style rigged GLB (ARKit blendshapes + visemes, Mixamo
skeleton) into a VRM 0.x file — pure container surgery, no Blender needed.

Injects the VRM extension block into the glTF JSON chunk:
  - humanoid: Mixamo -> VRM bone-name mapping (incl. fingers)
  - blendShapeMaster: blink / a-i-u-e-o viseme presets bound to the existing
    ARKit + MetaPerson-viseme morph targets
  - meta: minimal author/title block

The binary chunk is copied byte-for-byte. Usage:
    python3 glb_to_vrm.py <in.glb> <out.vrm>
"""
import json
import struct
import sys
from pathlib import Path

# Mixamo bone name -> VRM 0.x humanoid bone id
BONE_MAP = {
    "Hips": "hips", "Spine": "spine", "Spine1": "chest", "Spine2": "upperChest",
    "Neck": "neck", "Head": "head",
    "LeftEye": "leftEye", "RightEye": "rightEye",
    "LeftShoulder": "leftShoulder", "RightShoulder": "rightShoulder",
    "LeftArm": "leftUpperArm", "RightArm": "rightUpperArm",
    "LeftForeArm": "leftLowerArm", "RightForeArm": "rightLowerArm",
    "LeftHand": "leftHand", "RightHand": "rightHand",
    "LeftUpLeg": "leftUpperLeg", "RightUpLeg": "rightUpperLeg",
    "LeftLeg": "leftLowerLeg", "RightLeg": "rightLowerLeg",
    "LeftFoot": "leftFoot", "RightFoot": "rightFoot",
    "LeftToeBase": "leftToes", "RightToeBase": "rightToes",
}
for side, l, r in (("Thumb", "leftThumb", "rightThumb"),
                   ("Index", "leftIndex", "rightIndex"),
                   ("Middle", "leftMiddle", "rightMiddle"),
                   ("Ring", "leftRing", "rightRing"),
                   ("Pinky", "leftPinky", "rightPinky")):
    for n, part in ((1, "Proximal"), (2, "Intermediate"), (3, "Distal")):
        BONE_MAP[f"LeftHand{side}{n}"] = f"{l}{part}"
        BONE_MAP[f"RightHand{side}{n}"] = f"{r}{part}"

# VRM 0.x display names (J_Bip_*) for the mapped nodes — three-vrm reads the
# humanoid map regardless, but Emma Studio's A-pose matcher greps bone NAMES.
_FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")
_SEGMENTS = {"Proximal": "1", "Intermediate": "2", "Distal": "3"}

def vrm_display_name(vrm_id: str) -> str:
    side = "L" if vrm_id.startswith("left") else "R" if vrm_id.startswith("right") else "C"
    # fingers first: leftThumbProximal -> J_Bip_L_Thumb1
    for finger in _FINGERS:
        if finger not in vrm_id:
            continue
        for seg, num in _SEGMENTS.items():
            if vrm_id.endswith(seg):
                return f"J_Bip_{side}_{finger}{num}"
        return f"J_Bip_{side}_{finger}"
    simple = {"hips": "Hips", "spine": "Spine", "chest": "Chest",
              "upperChest": "UpperChest", "neck": "Neck", "head": "Head",
              "leftEye": "Eye", "rightEye": "Eye"}
    base = vrm_id.replace("left", "").replace("right", "")
    return f"J_Bip_{side}_{simple.get(vrm_id, base)}"

# VRM preset -> morph-target names it should bind (per mesh containing them)
SHAPE_PRESETS = {
    "blink": ["eyeBlinkLeft", "eyeBlinkRight"],
    "a": ["aa"],
    "i": ["ih"],
    "u": ["ou"],
    "e": ["E"],
    "o": ["oh"],
}


def morph_index(names: list[str], want: str) -> int | None:
    return names.index(want) if want in names else None


def main(src: str, dst: str) -> None:
    data = Path(src).read_bytes()
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF", "not a GLB"
    js_len, js_type = struct.unpack_from("<I4s", data, 12)
    assert js_type == b"JSON"
    js = json.loads(data[20:20 + js_len])

    # ---- locate morph targets per mesh ----
    mesh_targets: dict[int, list[str]] = {}
    for mi, mesh in enumerate(js.get("meshes", [])):
        for prim in mesh.get("primitives", []):
            if prim.get("targets"):
                mesh_targets[mi] = mesh.get("extras", {}).get("targetNames", [])
                break

    # ---- blendShapeMaster ----
    groups = []
    for preset, shapes in SHAPE_PRESETS.items():
        binds = []
        for mi, names in mesh_targets.items():
            for shape in shapes:
                idx = morph_index(names, shape)
                if idx is not None:
                    binds.append({"mesh": mi, "index": idx, "weight": 100})
        if binds:
            groups.append({
                "name": preset, "presetName": preset,
                "binds": binds,
                "isBinary": preset == "blink",
            })
    print(f"blendShapeGroups: {[(g['presetName'], len(g['binds'])) for g in groups]}")

    # ---- humanoid ----
    node_by_name = {}
    for ni, node in enumerate(js.get("nodes", [])):
        if node.get("name"):
            node_by_name[node["name"]] = ni
    human_bones = []
    missing = []
    for mixamo, vrm_id in BONE_MAP.items():
        ni = node_by_name.get(mixamo)
        if ni is None:
            missing.append(mixamo)
        else:
            # rename the node to VRM0 display convention so name-based
            # matchers (A-pose) work without engine changes
            js["nodes"][ni]["name"] = vrm_display_name(vrm_id)
            human_bones.append({"bone": vrm_id, "node": ni})
    print(f"humanoid bones mapped: {len(human_bones)} (missing: {missing or 'none'})")

    js.setdefault("extensionsUsed", []).append("VRM")
    js["extensions"] = js.get("extensions") or {}
    js["extensions"]["VRM"] = {
        "exporterVersion": "Emma Studio glb_to_vrm",
        "specVersion": "0.0",
        "meta": {
            "title": "Emma",
            "version": "2",
            "author": "Basti",
            "allowedUserName": "Everyone",
            "violentUssageName": "Disallow",
            "sexualUssageName": "Disallow",
            "commercialUssageName": "Disallow",
        },
        "humanoid": {"humanBones": human_bones},
        "firstPerson": {"firstPersonBone": node_by_name.get("Head", 0),
                        "firstPersonBoneOffset": {"x": 0, "y": 0.06, "z": 0},
                        "meshAnnotations": []},
        "blendShapeMaster": {"blendShapeGroups": groups},
        "secondaryAnimation": {"boneGroups": [], "colliderGroups": []},
        "materialProperties": [],
    }

    out_js = json.dumps(js, separators=(",", ":"), ensure_ascii=False).encode()
    pad = (4 - len(out_js) % 4) % 4
    out_js += b" " * pad

    # original binary chunk starts after the JSON chunk
    bin_hdr = 20 + js_len
    bin_len, bin_type = struct.unpack_from("<I4s", data, bin_hdr)
    assert bin_type == b"BIN\0"
    bin_data = data[bin_hdr + 8: bin_hdr + 8 + bin_len]

    blob = out_js + struct.pack("<I4s", len(bin_data), b"BIN\0") + bin_data
    out = struct.pack("<4sII", b"glTF", 2, 12 + len(blob)) + \
          struct.pack("<I4s", len(out_js), b"JSON") + blob
    Path(dst).write_bytes(out)
    print(f"wrote {dst}: {len(out)//1024//1024} MB ({len(out)} bytes)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
