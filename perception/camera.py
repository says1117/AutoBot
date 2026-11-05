from pathlib import Path

import cv2
import mujoco
import numpy as np

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

    # testing
    from perception.locate import find_pixels, locate
    debug = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for name in ["red_cube", "green_cube", "blue_cube"]:
        xs, ys = find_pixels(rgb, name)
        print(name, "pixels:", len(xs), "center:", xs.mean(), ys.mean())
        estimate = locate(model, data, rgb, depth, name)
        truth = data.body(name).xpos
        error_mm = np.linalg.norm(estimate - truth) * 1000
        print(name, "estimate:", estimate.round(3), "truth:", truth.round(3), "error:", round(error_mm, 1), "mm")

        center = (int(xs.mean()), int(ys.mean()))
        cv2.rectangle(debug, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (255, 255, 255), 1)
        cv2.drawMarker(debug, center, (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
        label = f"{name} ({center[0]}, {center[1]})"
        cv2.putText(debug, label, (center[0] + 20, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite("debug.png", debug)