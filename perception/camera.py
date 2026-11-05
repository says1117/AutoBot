from pathlib import Path

import cv2
import mujoco

SCENE = Path(__file__).parent.parent / "sim" / "scene.xml"
WIDTH, HEIGHT = 640, 480


def capture(model, data, renderer, camera="overhead"):
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera)
    depth = renderer.render()
    renderer.disable_depth_rendering()
    return rgb, depth


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)

    rgb, depth = capture(model, data, renderer, "overhead")
    print("rgb:", rgb.shape, "| depth:", depth.min(), "to", depth.max(), "meters")
    cv2.imwrite("scene.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))