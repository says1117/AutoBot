import mujoco

from perception.camera import HEIGHT, SCENE, WIDTH, capture
from perception.locate import COLOR_RANGES, locate


def observe():
    # build the scene and calculate where everything starts
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, HEIGHT, WIDTH)

    # take the photos, then release the renderer's memory
    rgb, depth = capture(model, data, renderer)
    renderer.close()

    # record every cube the camera can see; hidden cubes are left out
    objects = {}
    for name in COLOR_RANGES:
        position = locate(model, data, rgb, depth, name)
        if position is not None:
            objects[name] = position.round(3).tolist()
    return objects


if __name__ == "__main__":
    print(observe())
