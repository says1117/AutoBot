from pathlib import Path

import yaml

from skills.contract import SkillSpec
# path of the parent of the file
SKILLS_DIR = Path(__file__).parent

# list of everything the robot can do
# register discovers new skills when folder dropped in, no code changes
def load_registry() -> dict[str, SkillSpec]:
    registry = {}
    # finds skill.yaml inside each skill folder
    for path in sorted(SKILLS_DIR.glob('*/skill.yaml')):

        # yaml data extracted from each skill.yaml file
        data = yaml.safe_load(open(path))
        # returns SkillSpec object with yaml data
        spec = SkillSpec.model_validate(data)
        registry[spec.name] = spec
    return registry