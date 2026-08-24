#!/usr/bin/env python3
"""Build the shared G1+PIKA URDF from the upstream G1 description."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


PI = "1.5708"
# Equivalent URDF rpy for the verified MuJoCo transform:
# Rx(1.5708) followed by MuJoCo euler(1.57, 1.5708, 0).
RIGHT_PIKA_MOUNT_RPY = "1.56616231 -1.57000366 1.57543035"
# PIKAsense tracking axes measured on the physical right gripper:
# tracker +X -> TCP +Z, tracker +Y -> TCP -Y, tracker +Z -> TCP +X.
RIGHT_PIKA_TRACKING_RPY = "3.141592653589793 -1.5707963267948966 0"


def add_origin(parent: ET.Element, xyz: str = "0 0 0", rpy: str = "0 0 0") -> None:
    ET.SubElement(parent, "origin", {"xyz": xyz, "rpy": rpy})


def add_inertial(link: ET.Element, mass: str, origin: str, inertia: dict[str, str]) -> None:
    inertial = ET.SubElement(link, "inertial")
    add_origin(inertial, origin)
    ET.SubElement(inertial, "mass", {"value": mass})
    ET.SubElement(inertial, "inertia", inertia)


def add_mesh_link(robot: ET.Element, name: str, mesh: str, mass: str, origin: str,
                  inertia: dict[str, str]) -> None:
    link = ET.SubElement(robot, "link", {"name": name})
    add_inertial(link, mass, origin, inertia)
    visual = ET.SubElement(link, "visual")
    add_origin(visual)
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "mesh", {"filename": mesh})
    material = ET.SubElement(visual, "material", {"name": "pika_gray"})
    ET.SubElement(material, "color", {"rgba": "0.792 0.820 0.933 1"})


def add_fixed_joint(robot: ET.Element, name: str, parent: str, child: str,
                    xyz: str = "0 0 0", rpy: str = "0 0 0") -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "fixed"})
    add_origin(joint, xyz, rpy)
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})


def add_prismatic_joint(robot: ET.Element, name: str, parent: str, child: str,
                        rpy: str, axis: str, lower: str, upper: str) -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "prismatic"})
    add_origin(joint, "0 0 0.1358", rpy)
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "axis", {"xyz": axis})
    ET.SubElement(
        joint,
        "limit",
        {"lower": lower, "upper": upper, "effort": "10", "velocity": "1"},
    )


def add_pika(robot: ET.Element, side: str) -> None:
    wrist = f"{side}_wrist_yaw_link"
    mount = f"{side}_pika_mount"
    base = f"{side}_pika_gripper_base"
    finger_a = f"{side}_pika_finger_a"
    finger_b = f"{side}_pika_finger_b"

    ET.SubElement(robot, "link", {"name": mount})
    mount_rpy = RIGHT_PIKA_MOUNT_RPY if side == "right" else f"0 {PI} 0"
    add_fixed_joint(robot, f"{side}_pika_mount_joint", wrist, mount, "0.0415 0 0", mount_rpy)

    add_mesh_link(
        robot,
        base,
        "package://g1_pika_description/meshes/pika/gripper_base.STL",
        "0.145318531013916",
        "-0.000183807 0.000080503 0.032143669",
        {
            "ixx": "0.0001017403", "ixy": "-0.00000014396", "ixz": "-0.00000008724",
            "iyy": "0.0000416518", "iyz": "0.00000003277", "izz": "0.0001186913",
        },
    )
    add_fixed_joint(robot, f"{side}_pika_base_joint", mount, base)

    finger_inertia = {
        "ixx": "0.0000113870", "ixy": "0.00000042853", "ixz": "-0.00000006452",
        "iyy": "0.0000062611", "iyz": "0.00000157290", "izz": "0.0000157822",
    }
    add_mesh_link(
        robot, finger_a, "package://g1_pika_description/meshes/pika/link7.STL",
        "0.0303534921", "0.000651232 -0.049192987 0.009722588", finger_inertia,
    )
    add_mesh_link(
        robot, finger_b, "package://g1_pika_description/meshes/pika/link8.STL",
        "0.0303534921", "0.000651232 -0.049192987 0.009722588", finger_inertia,
    )
    add_prismatic_joint(
        robot, f"{side}_pika_finger_a_joint", base, finger_a,
        f"{PI} 0 0", "0 0 1", "0", "0.035",
    )
    add_prismatic_joint(
        robot, f"{side}_pika_finger_b_joint", base, finger_b,
        f"{PI} 0 -3.1416", "0 0 -1", "-0.035", "0",
    )

    tcp = f"{side}_pika_tcp"
    ET.SubElement(robot, "link", {"name": tcp})
    add_fixed_joint(robot, f"{side}_pika_tcp_joint", wrist, tcp, "0.18 0 0", mount_rpy)

    if side == "right":
        tracking = "right_pika_tracking_frame"
        ET.SubElement(robot, "link", {"name": tracking})
        add_fixed_joint(
            robot,
            "right_pika_tracking_joint",
            tcp,
            tracking,
            rpy=RIGHT_PIKA_TRACKING_RPY,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    tree = ET.parse(args.source)
    robot = tree.getroot()
    robot.set("name", "g1_29dof_pika")

    for element in list(robot):
        name = element.get("name", "")
        if name.startswith("left_hand_") or name.startswith("right_hand_"):
            robot.remove(element)

    for mesh in robot.findall(".//mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith("meshes/"):
            mesh.set(
                "filename",
                f"package://g1_pika_description/meshes/g1/{filename.removeprefix('meshes/')}",
            )

    add_pika(robot, "left")
    add_pika(robot, "right")
    ET.indent(tree, space="  ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
